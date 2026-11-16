import os
import time
import json
from typing import List, Dict, Any, Optional

from app.llm.groq_client import (
    execute_groq_chat_completion,
    get_preferred_groq_model,
    parse_groq_retry_delay,
    is_tpm_rate_limit,
)
from app.search.context_manager import (
    pack_context_for_rag,
    get_model_budget_profile,
    estimate_tokens,
)
from app.search.multi_stage_retriever import (
    classify_query_intent,
    INTENT_FLOW_ARCHITECTURE,
    INTENT_FEATURE_IMPLEMENTATION,
    INTENT_CHANGE_IMPACT,
)

# Attempt to load the Groq SDK client.
try:
    from groq import Groq
    from groq import RateLimitError, APIStatusError
    HAS_GROQ_SDK = True
except ImportError:
    HAS_GROQ_SDK = False
    RateLimitError = Exception
    APIStatusError = Exception

def try_gemini_rag_fallback(system_prompt: str, user_prompt: str, selected_chunks: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        return None
    try:
        from app.llm.gemini_repo_summary import call_gemini_flash
        combined_prompt = f"{system_prompt}\n\nTask and Context:\n{user_prompt}"
        res = call_gemini_flash(combined_prompt, gemini_key)
        if res and isinstance(res, dict) and "answer" in res:
            raw_citations = res.get("cited_chunks", [])
            valid_citations = validate_citations(raw_citations, selected_chunks)
            clean_answer = res.get("answer", "")
            import re
            clean_answer = re.sub(r'【[^】]*】', '', clean_answer)
            clean_answer = re.sub(r'(?m)^(\s*)[-*]\s*[-*]\s+', r'\1- ', clean_answer)
            clean_answer = re.sub(r'(?i)\bthe platform\s*\(\s*referred to as hackora in the code\s*\)', 'DevLens AI', clean_answer)
            clean_answer = re.sub(r'(?i)\b\(referred to as hackora in the code\)', '', clean_answer)
            clean_answer = re.sub(r'(?i)\bhackora\b', 'DevLens AI', clean_answer)
            return {
                "answer": clean_answer,
                "cited_chunks": valid_citations,
                "model_used": res.get("_model_resolved", "gemini-2.5-flash")
            }
    except Exception as e:
        print(f"[DevLens AI] Gemini RAG fallback attempt error: {e}")
    return None

def validate_citations(raw_citations: List[Dict[str, Any]], selected_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Validates cited files and line ranges against the actual context chunks.
    Filters out any hallucinated filenames or invalid line ranges.
    """
    if not raw_citations:
        return []
    
    valid_files = {c.get("file_path") for c in selected_chunks if c.get("file_path")}
    validated = []
    seen = set()

    for cite in raw_citations:
        fp = cite.get("file_path", "").strip()
        if not fp or fp not in valid_files:
            continue
        try:
            start_line = max(1, int(cite.get("start_line", 1)))
            end_line = max(start_line, int(cite.get("end_line", start_line)))
            key = (fp, start_line, end_line)
            if key not in seen:
                seen.add(key)
                validated.append({
                    "file_path": fp,
                    "start_line": start_line,
                    "end_line": end_line
                })
        except (ValueError, TypeError):
            continue

    return validated

def generate_rag_answer(
    query: str,
    chunks: List[Dict[str, Any]],
    repo_id: Optional[str] = None,
    graph_context: Optional[Dict[str, Any]] = None,
    intent_info: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Synthesizes a grounded, cited natural-language response to a developer's query
    using retrieved code chunks, dependency graph relationships, and Groq models.
    
    Features:
    - Intent-specific prompt structuring (Execution Flow, Feature Implementation, Change Impact).
    - AST dependency graph and blast radius awareness.
    - Token-aware context packing preventing TPM/overflow errors.
    - Intelligent deduplication and chunk truncation.
    - Dynamic context reduction upon encountering Groq rate limits.
    - Strict citation verification against selected context chunks.
    - Structured JSON output with verified source line citations.
    - Observability logging of token usage, durations, and chunk counts.
    """
    start_time = time.time()
    
    # 1. Resolve API key with dynamic dotenv fallback
    api_key_current = os.getenv("GROQ_API_KEY")
    if not api_key_current:
        from dotenv import load_dotenv
        backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        load_dotenv(os.path.join(backend_dir, ".env"), override=True)
        load_dotenv(override=True)
        api_key_current = os.getenv("GROQ_API_KEY")

    # Filter out obsolete legacy or test files (e.g. stage_services.py from historical hackathon code)
    filtered_chunks = []
    for c in chunks:
        fp = (c.get("file_path") or "").lower()
        fn = (c.get("name") or "").lower()
        if "stage_services" in fp or "hackora" in fp or "pitch_deck" in fn or "stage1" in fp:
            continue
        filtered_chunks.append(c)
    chunks = filtered_chunks

    if not api_key_current:
        print("[DevLens AI Error] GROQ_API_KEY is not configured in the environment.")
        return {
            "answer": "GROQ_API_KEY is missing from the environment. Please configure it in your .env file to enable semantic question answering.",
            "cited_chunks": []
        }

    if not HAS_GROQ_SDK:
        print("[DevLens AI Error] groq SDK is not installed in the environment.")
        return {
            "answer": "The 'groq' SDK is not installed in the python environment. Please run 'pip install -r requirements.txt'.",
            "cited_chunks": []
        }

    if not chunks:
        return {
            "answer": "No relevant code chunks were retrieved for the search query to answer the question.",
            "cited_chunks": []
        }

    # 2. Determine Intent & Build Structured System Instructions
    if not intent_info:
        intent_info = classify_query_intent(query)
    intent = intent_info.get("intent", "GENERAL")

    intent_instructions = ""
    if intent == INTENT_FLOW_ARCHITECTURE:
        intent_instructions = (
            "Special Instructions for Request Flow & Architecture Questions:\n"
            "1. Reconstruct the COMPLETE end-to-end execution flow step-by-step from user interaction to backend processing and data storage.\n"
            "2. Explain how frontend components (e.g. LandingPage / Home.tsx) trigger API endpoints (e.g. /api/repos/ingest) and how backend services coordinate.\n"
            "3. Reference the specific functions and files involved at each stage of the lifecycle.\n"
        )
    elif intent == INTENT_CHANGE_IMPACT:
        intent_instructions = (
            "Special Instructions for Change Impact & Dependency Questions:\n"
            "1. Use the provided code chunks and Dependency Graph Evidence to trace direct callers and transitive dependents.\n"
            "2. Distinguish: (a) Directly affected files/functions (direct imports/callers), (b) Indirectly affected components (transitive dependencies), and (c) Potentially affected API endpoints or routes.\n"
            "3. State potential impacts and risks clearly with evidence from the dependency graph.\n"
        )
    elif intent == INTENT_FEATURE_IMPLEMENTATION:
        intent_instructions = (
            "Special Instructions for Feature Implementation Questions:\n"
            "1. Detail exactly what is implemented based on the provided code chunks, citing the files and functions.\n"
            "2. Explain the algorithms, data structures, and technologies used (e.g. pgvector, BM25, RRF, Tree-sitter).\n"
            "3. Do not assume a feature is missing if evidence of its implementation is present in the snippets.\n"
        )

    system_prompt = (
        "You are an expert software engineer assistant for DevLens AI. You answer questions about a codebase using the provided code chunks and dependency evidence.\n\n"
        "Product Identity & Purpose Rules:\n"
        "1. This application is DevLens AI: an AI-driven codebase intelligence, static code analysis, and architectural onboarding platform.\n"
        "2. Core Functionality: DevLens AI provides AST-level code parsing, interactive dependency graph visualization, cyclomatic complexity analysis, blast radius calculation, absolute repository health scoring, and grounded semantic/hybrid code search with line citations.\n"
        "3. Strictly Prohibited Legacy Terminology: NEVER refer to this product or codebase as 'HACKORA', 'Hackora', or a hackathon tool. If historical comments, deprecated functions, or legacy references (such as stage deliverables or pitch decks) appear in retrieved code chunks, ignore that obsolete branding completely. Always explain the platform as DevLens AI and focus on its actual codebase intelligence capabilities.\n\n"
        "Core Grounding Rules:\n"
        "1. Grounding: Answer strictly using facts, function implementations, and relationships found in the provided context. Never hallucinate files or logic.\n"
        "2. Implementation Veracity: Distinguish between confirmed implementations, architectural deductions, and unverified areas.\n"
        "3. Insufficient Context: If the chunks do not contain enough information, explain what is known from the snippets and specify what is unverified.\n"
        f"{intent_instructions}\n"
        "4. Formatting Rules:\n"
        "   - Use clean, standard Markdown with proper hierarchy (e.g. bold titles, single '-' bullet points).\n"
        "   - Never use double bullets (e.g. '- -'). Always use a single '- ' for each list item.\n"
        "   - Do NOT insert raw citation brackets (like 【...】 or [file.py:10-25]) in the answer text. All citations belong strictly in 'cited_chunks'.\n"
        "   - Wrap code identifiers, function names, and file paths in single backticks (e.g. `MapView.tsx`).\n"
        "5. Structured JSON: You MUST output a valid JSON object matching this exact schema:\n"
        "{\n"
        "  \"answer\": \"Detailed explanation in clean standard markdown.\",\n"
        "  \"cited_chunks\": [\n"
        "    { \"file_path\": \"relative/path/to/file.py\", \"start_line\": 10, \"end_line\": 25 }\n"
        "  ]\n"
        "}"
    )

    preferred_model = get_preferred_groq_model()
    budget_profile = get_model_budget_profile(preferred_model)

    # 3. Client Initialization
    try:
        client = Groq(api_key=api_key_current)
    except Exception as e:
        print(f"[DevLens AI Error] Failed to initialize Groq client: {str(e)}")
        return {
            "answer": f"Failed to initialize Groq client: {str(e)}",
            "cited_chunks": []
        }

    # 4. Context Budget & Adaptive Rate-Limit Retry Loop
    current_prompt_budget = budget_profile["max_prompt_tokens"]
    max_adaptive_attempts = 3
    retry_count = 0
    last_error_msg = ""
    used_model = preferred_model

    # Incorporate graph context if available
    graph_context_text = ""
    if graph_context and graph_context.get("blast_reports"):
        reports_text = []
        for r in graph_context["blast_reports"]:
            reports_text.append(
                f"- Target File: {r.get('file_path')}\n"
                f"  Direct Dependents ({len(r.get('direct_dependents', []))}): {', '.join(r.get('direct_dependents', [])) or 'None'}\n"
                f"  Transitive Dependents ({len(r.get('transitive_dependents', []))}): "
                f"{', '.join([t['file_path'] for t in r.get('transitive_dependents', [])[:5]]) or 'None'}"
            )
        graph_context_text = (
            "\n--- Dependency Graph & Blast Radius Evidence ---\n"
            + "\n".join(reports_text)
            + "\n-------------------------------------------------\n\n"
        )

    for attempt in range(max_adaptive_attempts):
        # Pack context within current prompt budget
        packed = pack_context_for_rag(
            query=query,
            system_prompt=system_prompt + graph_context_text,
            chunks=chunks,
            model_name=preferred_model,
            budget_override=current_prompt_budget,
            enforce_min_one_chunk=True
        )

        user_prompt = (
            f"User Question: \"{query}\"\n\n"
            f"{graph_context_text}"
            f"Code Chunks Context:\n"
            f"{packed['formatted_context']}"
        )

        est_prompt_tokens = packed["estimated_prompt_tokens"]
        est_total_request = packed["estimated_total_request_tokens"]
        selected_count = packed["selected_chunk_count"]

        try:
            chat_completion, used_model = execute_groq_chat_completion(
                client=client,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model_override=preferred_model,
                temperature=0.1,
                response_format={"type": "json_object"},
                max_tokens=budget_profile["reserved_output_tokens"],
                max_retries=1
            )

            raw_response = chat_completion.choices[0].message.content.strip()
            parsed = json.loads(raw_response)
            duration_ms = int((time.time() - start_time) * 1000)

            # Validate citations against actual context chunks
            raw_citations = parsed.get("cited_chunks", [])
            valid_citations = validate_citations(raw_citations, packed["selected_chunks"])

            # Observability logging (structured, zero code/secrets)
            print(
                f"[RAG Observability] Query: \"{query[:40]}\" | Intent: {intent} | Model: {used_model} | Status: SUCCESS\n"
                f"  Retrieved Chunks: {len(chunks)} | Selected: {selected_count} | Est Context Tokens: {packed['estimated_context_tokens']}\n"
                f"  Est Prompt Tokens: {est_prompt_tokens} | Reserved Output: {budget_profile['reserved_output_tokens']}\n"
                f"  Total Request Tokens: {est_total_request} / {budget_profile['max_request_tokens']} (Cap: {budget_profile['tpm_limit']} TPM)\n"
                f"  Citations Validated: {len(valid_citations)}/{len(raw_citations)} | Request Duration: {duration_ms}ms"
            )

            # Sanitize answer text: remove bracket artifacts and clean double bullets
            clean_answer = parsed.get("answer", "")
            if isinstance(clean_answer, str):
                import re
                clean_answer = re.sub(r'【[^】]*】', '', clean_answer)
                clean_answer = re.sub(r'(?m)^(\s*)[-*]\s*[-*]\s+', r'\1- ', clean_answer)
                # Enforce DevLens branding: remove all legacy HACKORA references
                clean_answer = re.sub(r'(?i)\bthe platform\s*\(\s*referred to as hackora in the code\s*\)', 'DevLens AI', clean_answer)
                clean_answer = re.sub(r'(?i)\b\(referred to as hackora in the code\)', '', clean_answer)
                clean_answer = re.sub(r'(?i)\bhackora\b', 'DevLens AI', clean_answer)

            return {
                "answer": clean_answer,
                "cited_chunks": valid_citations,
                "observability": {
                    "model": used_model,
                    "intent": intent,
                    "retrieved_chunk_count": len(chunks),
                    "selected_chunk_count": selected_count,
                    "estimated_prompt_tokens": est_prompt_tokens,
                    "estimated_context_tokens": packed["estimated_context_tokens"],
                    "reserved_output_tokens": budget_profile["reserved_output_tokens"],
                    "duration_ms": duration_ms,
                    "retry_count": retry_count,
                    "citations_count": len(valid_citations),
                }
            }

        except RateLimitError as rle:
            retry_count += 1
            last_error_msg = str(rle)
            delay = parse_groq_retry_delay(rle, default_delay=2.5)

            current_prompt_budget = max(1000, int(current_prompt_budget * 0.50))
            print(
                f"[DevLens AI RateLimit] Attempt {attempt + 1}/{max_adaptive_attempts} failed with rate limit on {used_model}.\n"
                f"  Reducing prompt budget to {current_prompt_budget} tokens. Waiting {delay:.2f}s with jitter before retry..."
            )
            time.sleep(delay)

        except json.JSONDecodeError as jde:
            print(f"[DevLens AI Error] LLM returned non-JSON payload: {str(jde)}")
            return {
                "answer": "Failed to parse model response as JSON. Please rephrase your query.",
                "cited_chunks": []
            }

        except Exception as e:
            err_str = str(e).lower()
            if "rate_limit" in err_str or "tpm" in err_str:
                retry_count += 1
                last_error_msg = str(e)
                delay = 3.0
                current_prompt_budget = max(1000, int(current_prompt_budget * 0.50))
                print(f"[DevLens AI RateLimit] Catch-all rate limit detected. Reducing budget to {current_prompt_budget} tokens. Waiting {delay}s...")
                time.sleep(delay)
            else:
                print(f"[DevLens AI Error] Unexpected error during RAG answer generation: {str(e)}")
                return {
                    "answer": f"An error occurred while synthesizing the answer: {str(e)}",
                    "cited_chunks": []
                }

    # Attempt Gemini fallback before returning exhaustion
    gemini_fallback = try_gemini_rag_fallback(system_prompt, user_prompt, packed.get("selected_chunks", []))
    if gemini_fallback:
        return gemini_fallback

    duration_ms = int((time.time() - start_time) * 1000)
    print(f"[RAG Observability] Query: \"{query[:40]}\" | Model: {used_model} | Status: EXHAUSTED ({duration_ms}ms)")

    return {
        "answer": (
            f"The AI service is currently receiving a high volume of requests and the token-per-minute limit "
            f"for model `{used_model}` was temporarily exceeded ({last_error_msg or 'rate_limit_exceeded'}). "
            f"Please wait a moment and try asking your question again."
        ),
        "cited_chunks": []
    }
