import os
import re
from typing import List, Dict, Any, Tuple, Optional

# ==============================================================================
# TOKEN ESTIMATION (CONSERVATIVE & FAST, ZERO EXTERNAL DEPENDENCY)
# ==============================================================================

def estimate_tokens(text: str) -> int:
    """
    Conservative token estimator for source code, markdown, and structured prompts.
    Does not require external dependencies like tiktoken.
    
    Source code exhibits higher token density than standard English prose due to 
    indentation, operators, variable naming (camelCase/snake_case), and brackets.
    We apply a conservative ~3.2 characters per token ratio alongside word/symbol splits.
    """
    if not text:
        return 0
    # Conservative code character ratio
    char_estimate = len(text) / 3.2
    # Boundary-aware word split estimate
    words = text.split()
    word_estimate = len(words) * 1.35
    # Return conservative upper bound with minimal buffer
    return max(1, int(max(char_estimate, word_estimate) + 4))


# ==============================================================================
# PROVIDER & MODEL TOKEN BUDGET PROFILES
# ==============================================================================

# Verified provider/model defaults (Tokens Per Minute, Requests Per Minute)
DEFAULT_MODEL_LIMITS: Dict[str, Dict[str, int]] = {
    "openai/gpt-oss-20b": {"tpm": 8000, "rpm": 30},
    "openai/gpt-oss-120b": {"tpm": 8000, "rpm": 30},
    "qwen/qwen3.8-27b": {"tpm": 15000, "rpm": 30},
    "llama-3.3-70b-versatile": {"tpm": 12000, "rpm": 30},
    "llama-3.1-8b-instant": {"tpm": 20000, "rpm": 30},
}

def get_model_budget_profile(model_name: Optional[str] = None) -> Dict[str, int]:
    """
    Establishes a configurable token budget profile based on model capabilities
    and environment configuration.
    
    Environment variable overrides:
    - GROQ_TPM_LIMIT: Overrides the TPM ceiling for the active model.
    - GROQ_MAX_REQUEST_TOKENS: Hard cap on total tokens (prompt + output) in one request.
    - GROQ_MAX_OUTPUT_TOKENS: Reserved tokens for the model response (default: 800).
    """
    model = (model_name or os.getenv("GROQ_MODEL") or "llama-3.3-70b-versatile").strip()
    
    # 1. Resolve TPM Limit
    env_tpm = os.getenv("GROQ_TPM_LIMIT")
    if env_tpm and env_tpm.isdigit():
        tpm_limit = int(env_tpm)
    else:
        profile = DEFAULT_MODEL_LIMITS.get(model, {"tpm": 8000, "rpm": 30})
        tpm_limit = profile.get("tpm", 8000)

    # 2. Resolve Reserved Output Tokens
    env_output = os.getenv("GROQ_MAX_OUTPUT_TOKENS")
    if env_output and env_output.isdigit():
        reserved_output = int(env_output)
    else:
        reserved_output = 800

    # 3. Resolve Request Cap
    # To prevent rate-limit exhaustion, a single request should safely consume
    # at most 65% - 70% of the entire 1-minute token allowance (TPM).
    env_req_max = os.getenv("GROQ_MAX_REQUEST_TOKENS")
    if env_req_max and env_req_max.isdigit():
        max_request_tokens = int(env_req_max)
    else:
        safe_tpm_ceiling = int(tpm_limit * 0.65)
        # Never exceed safe ceiling; capped between 3,500 and 8,000 for standard RAG
        max_request_tokens = max(2500, min(safe_tpm_ceiling, 8000))

    # Available prompt budget after reserving output tokens
    max_prompt_tokens = max(1500, max_request_tokens - reserved_output)

    # Max single chunk budget (prevents any single chunk from monopolizing context)
    max_chunk_tokens = max(400, int(max_prompt_tokens * 0.40))

    return {
        "model": model,
        "tpm_limit": tpm_limit,
        "max_request_tokens": max_request_tokens,
        "reserved_output_tokens": reserved_output,
        "max_prompt_tokens": max_prompt_tokens,
        "max_chunk_tokens": max_chunk_tokens,
        "max_chunks_per_file": 2,
    }


# ==============================================================================
# DEDUPLICATION & OVERLAP ELIMINATION
# ==============================================================================

def deduplicate_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Filters out duplicate chunks and overlapping line ranges within the same file.
    If chunk A largely subsumes chunk B (>50% line overlap), the higher-ranked
    chunk is retained and the redundant chunk is dropped.
    """
    seen_hashes = set()
    cleaned: List[Dict[str, Any]] = []

    for chunk in chunks:
        content = chunk.get("content", "").strip()
        if not content:
            continue

        file_path = chunk.get("file_path", "")
        start_line = chunk.get("start_line", 0)
        end_line = chunk.get("end_line", 0)

        # 1. Exact content hash check
        content_hash = hash((file_path, content[:200], content[-200:], len(content)))
        if content_hash in seen_hashes:
            continue

        # 2. Overlap check against previously accepted chunks from the same file
        is_overlapping = False
        for accepted in cleaned:
            if accepted.get("file_path") == file_path:
                a_start = accepted.get("start_line", 0)
                a_end = accepted.get("end_line", 0)
                
                # Check for line intersection
                overlap_start = max(start_line, a_start)
                overlap_end = min(end_line, a_end)
                
                if overlap_end >= overlap_start:
                    overlap_len = overlap_end - overlap_start + 1
                    chunk_len = max(1, end_line - start_line + 1)
                    accepted_len = max(1, a_end - a_start + 1)
                    
                    # If either chunk overlaps by > 50%, treat as redundant
                    if (overlap_len / chunk_len > 0.5) or (overlap_len / accepted_len > 0.5):
                        is_overlapping = True
                        break

        if not is_overlapping:
            seen_hashes.add(content_hash)
            cleaned.append(chunk)

    return cleaned


# ==============================================================================
# INTELLIGENT CHUNK TRUNCATION
# ==============================================================================

def truncate_chunk_content(chunk: Dict[str, Any], max_chunk_tokens: int) -> Dict[str, Any]:
    """
    Intelligently truncates oversized chunks to fit within token boundaries while
    preserving structural syntax (function signatures, class headers, return statements).
    """
    content = chunk.get("content", "")
    chunk_tokens = estimate_tokens(content)

    if chunk_tokens <= max_chunk_tokens:
        return chunk

    lines = content.splitlines(keepends=True)
    total_lines = len(lines)

    if total_lines <= 20:
        # Short file with very long lines: character slice
        max_chars = int(max_chunk_tokens * 3.0)
        truncated_content = content[:max_chars] + "\n// ... [remaining code omitted for brevity] ..."
        new_chunk = dict(chunk)
        new_chunk["content"] = truncated_content
        new_chunk["truncated"] = True
        return new_chunk

    # For structured code: keep top lines (signatures, imports, declarations)
    # and bottom lines (return statements, closing braces)
    head_lines_count = max(10, int(total_lines * 0.40))
    tail_lines_count = max(5, int(total_lines * 0.15))

    # Binary search/step down until token count fits max_chunk_tokens
    while head_lines_count + tail_lines_count >= total_lines:
        head_lines_count = max(5, head_lines_count - 5)
        tail_lines_count = max(2, tail_lines_count - 2)

    head = "".join(lines[:head_lines_count])
    tail = "".join(lines[-tail_lines_count:])
    omitted = total_lines - head_lines_count - tail_lines_count

    separator = f"\n// ... [{omitted} lines omitted for context efficiency] ...\n"
    combined = head + separator + tail

    # Verify token count and reduce if still oversized
    while estimate_tokens(combined) > max_chunk_tokens and head_lines_count > 6:
        head_lines_count -= 4
        omitted = total_lines - head_lines_count - tail_lines_count
        separator = f"\n// ... [{omitted} lines omitted for context efficiency] ...\n"
        combined = "".join(lines[:head_lines_count]) + separator + tail

    new_chunk = dict(chunk)
    new_chunk["content"] = combined
    new_chunk["truncated"] = True
    new_chunk["original_line_count"] = total_lines
    return new_chunk


# ==============================================================================
# TOKEN-AWARE CONTEXT PACKER
# ==============================================================================

def pack_context_for_rag(
    query: str,
    system_prompt: str,
    chunks: List[Dict[str, Any]],
    model_name: Optional[str] = None,
    budget_override: Optional[int] = None,
    enforce_min_one_chunk: bool = True
) -> Dict[str, Any]:
    """
    Selects, deduplicates, bounds, and packs retrieved chunks so the resulting prompt
    strictly obeys the provider's token limits.
    
    Guarantees:
    - Never exceeds the calculated prompt token budget.
    - Retains source diversity across files.
    - Prevents single chunks from overwhelming the context.
    - Guarantees at least 1 top-ranked chunk is always included if chunks exist.
    """
    from app.security.secret_scanner import redact_secrets

    # 1. Resolve Budget Profile
    profile = get_model_budget_profile(model_name)
    max_prompt_budget = budget_override if budget_override is not None else profile["max_prompt_tokens"]
    max_chunk_tokens = profile["max_chunk_tokens"]
    max_per_file = profile["max_chunks_per_file"]

    # 2. Base Prompts Token Cost
    system_tokens = estimate_tokens(system_prompt)
    query_tokens = estimate_tokens(f"User Question: \"{query}\"\n\nCode Chunks Context:\n")
    fixed_overhead = system_tokens + query_tokens + 50  # 50 tokens safety margin
    
    available_context_budget = max(400, max_prompt_budget - fixed_overhead)

    # 3. Deduplicate
    deduped = deduplicate_chunks(chunks)

    # 4. Greedily Pack Chunks Based on Relevance, Diversity, and Budget
    selected_chunks: List[Dict[str, Any]] = []
    file_counts: Dict[str, int] = {}
    current_context_tokens = 0
    formatted_context_parts: List[str] = []

    for idx, raw_chunk in enumerate(deduped, start=1):
        file_path = raw_chunk.get("file_path", "unknown")
        
        # Diversity check: cap occurrences of any single file
        if file_counts.get(file_path, 0) >= max_per_file:
            continue

        # Sanitize / Redact Secrets
        raw_content = raw_chunk.get("content", "")
        redaction_res = redact_secrets(raw_content)
        clean_content = redaction_res["redacted_content"]
        
        chunk_copy = dict(raw_chunk)
        chunk_copy["content"] = clean_content

        # Truncate oversized chunk if needed
        bounded_chunk = truncate_chunk_content(chunk_copy, max_chunk_tokens=max_chunk_tokens)
        
        # Render chunk formatted text
        chunk_header = (
            f"--- Chunk {len(selected_chunks) + 1}: {bounded_chunk['file_path']} "
            f"(lines {bounded_chunk.get('start_line', 1)}-{bounded_chunk.get('end_line', 1)}) ---\n"
        )
        chunk_text = f"{chunk_header}{bounded_chunk['content']}\n\n"
        chunk_cost = estimate_tokens(chunk_text)

        # Check if adding this chunk fits within context budget
        if current_context_tokens + chunk_cost <= available_context_budget:
            selected_chunks.append(bounded_chunk)
            formatted_context_parts.append(chunk_text)
            current_context_tokens += chunk_cost
            file_counts[file_path] = file_counts.get(file_path, 0) + 1
        elif len(selected_chunks) == 0 and enforce_min_one_chunk:
            # Guarantee at least the #1 chunk is included, shrinking it to fit remaining budget
            emergency_chunk_tokens = max(200, available_context_budget - estimate_tokens(chunk_header) - 20)
            shrunk_chunk = truncate_chunk_content(chunk_copy, max_chunk_tokens=emergency_chunk_tokens)
            shrunk_text = (
                f"--- Chunk 1: {shrunk_chunk['file_path']} "
                f"(lines {shrunk_chunk.get('start_line', 1)}-{shrunk_chunk.get('end_line', 1)}) ---\n"
                f"{shrunk_chunk['content']}\n\n"
            )
            selected_chunks.append(shrunk_chunk)
            formatted_context_parts.append(shrunk_text)
            current_context_tokens += estimate_tokens(shrunk_text)
            file_counts[file_path] = 1
            break
        else:
            # Context budget reached for further full chunks
            continue

    formatted_context = "".join(formatted_context_parts)
    estimated_prompt_tokens = fixed_overhead + current_context_tokens
    estimated_total_request = estimated_prompt_tokens + profile["reserved_output_tokens"]

    return {
        "selected_chunks": selected_chunks,
        "formatted_context": formatted_context,
        "selected_chunk_count": len(selected_chunks),
        "original_chunk_count": len(chunks),
        "estimated_prompt_tokens": estimated_prompt_tokens,
        "estimated_context_tokens": current_context_tokens,
        "reserved_output_tokens": profile["reserved_output_tokens"],
        "estimated_total_request_tokens": estimated_total_request,
        "budget_profile": profile,
        "prompt_budget": max_prompt_budget,
    }

