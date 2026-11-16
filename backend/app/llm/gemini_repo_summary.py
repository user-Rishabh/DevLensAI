import os
import json
import re
import time
from typing import Dict, Any, Optional, List

import httpx

from app.db import get_repo_analysis, get_file_content, CACHE_DIR, supabase

# In-memory cache for fast repeated reads
_REPO_SUMMARY_MEMORY_CACHE: Dict[str, Dict[str, Any]] = {}

def get_gemini_api_key() -> Optional[str]:
    """
    Retrieves the Gemini / Google API key from the environment or .env file.
    """
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        from dotenv import load_dotenv
        backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        load_dotenv(os.path.join(backend_dir, ".env"), override=True)
        load_dotenv(override=True)
        key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    return key.strip() if key and key.strip() else None


def clean_markdown_text(text: str) -> str:
    """
    Strips raw HTML tags, image tags, SVGs, badges, and legacy references
    to ensure clean textual comprehension.
    """
    if not text:
        return ""
    # Strip HTML tags
    t = re.sub(r'<[^>]+>', ' ', text)
    # Strip markdown images ![alt](url)
    t = re.sub(r'!\[.*?\]\(.*?\)', ' ', t)
    # Strip badge markdown links
    t = re.sub(r'\[!\[.*?\]\(.*?\)\]\(.*?\)', ' ', t)
    # Normalize multiple newlines/spaces
    t = re.sub(r'\n{3,}', '\n\n', t)
    # Clean legacy HACKORA phrasing
    t = re.sub(r'(?i)\bhackora\b', 'DevLens AI', t)
    t = re.sub(r'(?i)\bthe platform\s*\(\s*referred to as hackora in the code\s*\)', 'DevLens AI', t)
    t = re.sub(r'(?i)\(referred to as hackora in the code\)', '', t)
    return t.strip()


def gather_repo_context(repo_id: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
    """
    Gathers informative context about the repository to allow Gemini Flash
    to explain what the website/project is about with high fidelity.
    """
    from app.analysis.dependency_graph import get_all_project_files
    
    file_tree = analysis.get("file_tree", {})
    all_files = list(get_all_project_files(file_tree))
    dependencies = analysis.get("dependencies", [])
    hotspots = analysis.get("hotspots", [])
    
    # 1. Look for README
    readme_content = ""
    for f in all_files:
        fname = os.path.basename(f).lower()
        if fname in ("readme.md", "readme.markdown", "readme.txt", "readme"):
            c = get_file_content(repo_id, f)
            if c:
                readme_content = clean_markdown_text(c[:10000])
                break

    # 2. Look for package.json or pyproject.toml / requirements.txt
    manifest_snippet = ""
    for f in all_files:
        fname = os.path.basename(f).lower()
        if fname in ("package.json", "pyproject.toml", "requirements.txt", "cargo.toml", "go.mod", "pom.xml", "gemfile"):
            c = get_file_content(repo_id, f)
            if c:
                manifest_snippet += f"\n--- {f} ---\n{clean_markdown_text(c[:2500])}\n"

    # 3. Look for core entrypoint files
    entrypoint_snippets = ""
    for f in all_files:
        fname = os.path.basename(f).lower()
        if fname in ("main.py", "app.py", "index.html", "app.tsx", "main.tsx", "server.ts", "server.js", "train.py", "predict.py", "model.py", "manage.py", "index.py", "api.py"):
            c = get_file_content(repo_id, f)
            if c:
                entrypoint_snippets += f"\n--- {f} ---\n{c[:2500]}\n"

    # 4. Folder structure highlights
    top_level_folders = set()
    for f in all_files:
        parts = f.replace("\\", "/").split("/")
        if len(parts) > 1:
            top_level_folders.add(parts[0])

    return {
        "file_count": len(all_files),
        "folder_count": len(top_level_folders),
        "top_level_folders": sorted(list(top_level_folders)),
        "dependencies_count": len(dependencies),
        "hotspots_count": len(hotspots),
        "sample_files": all_files[:30],
        "readme_content": readme_content,
        "manifest_snippet": manifest_snippet,
        "entrypoint_snippets": entrypoint_snippets,
    }


def call_gemini_flash(prompt: str, api_key: str) -> Dict[str, Any]:
    """
    Calls Google Gemini Flash via REST API.
    Attempts models in order: gemini-2.5-flash, gemini-flash-latest, gemini-2.0-flash, gemini-2.5-pro.
    """
    candidate_models = [
        os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip(),
        "gemini-2.5-flash",
        "gemini-flash-lite-latest",
        "gemini-flash-latest"
    ]
    # Deduplicate while preserving order
    seen = set()
    models_to_try = [m for m in candidate_models if m and not (m in seen or seen.add(m))]

    last_error = ""
    headers = {"Content-Type": "application/json"}
    
    payload = {
        "contents": [
          {
            "role": "user",
            "parts": [
              {"text": prompt}
            ]
          }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 2500,
            "responseMimeType": "application/json"
        }
    }

    with httpx.Client(timeout=45.0) as client:
        for model in models_to_try:
            endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
            try:
                response = client.post(endpoint, json=payload, headers=headers)
                if response.status_code == 200:
                    data = response.json()
                    candidates = data.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        if parts:
                            raw_text = parts[0].get("text", "").strip()
                            # Clean code fence wrapper if present
                            json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
                            if json_match:
                                parsed = json.loads(json_match.group(0))
                                parsed["_model_resolved"] = model
                                return parsed
                            parsed = json.loads(raw_text)
                            parsed["_model_resolved"] = model
                            return parsed
                elif response.status_code in (404, 400):
                    last_error = f"{model} returned {response.status_code}: {response.text[:200]}"
                    continue
                else:
                    last_error = f"{model} error {response.status_code}: {response.text[:200]}"
            except Exception as e:
                last_error = f"{model} request failed: {str(e)}"
                continue

    raise RuntimeError(f"All Gemini Flash models failed. Last error: {last_error}")


def fallback_with_groq(prompt: str) -> Optional[Dict[str, Any]]:
    """
    Fallback using the active Groq model if Gemini is not reachable.
    """
    try:
        from app.llm.groq_client import execute_groq_chat_completion
        from groq import Groq
        
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return None
            
        client = Groq(api_key=api_key)
        completion, model_used = execute_groq_chat_completion(
            client=client,
            messages=[
                {"role": "system", "content": "You are an expert software architect analyzing codebases. Always respond strictly in valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=2500,
            response_format={"type": "json_object"},
            max_retries=2
        )
        if completion.choices:
            raw = completion.choices[0].message.content.strip()
            json_match = re.search(r'\{.*\}', raw, re.DOTALL)
            parsed = json.loads(json_match.group(0) if json_match else raw)
            parsed["_fallback_model"] = model_used
            return parsed
    except Exception as e:
        print(f"[DevLens AI Warning] Groq fallback summary failed: {str(e)}")
    return None


def generate_heuristic_summary(repo_name: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clean, human-readable baseline summary generated dynamically from the repository's
    actual files, manifests, and README when external AI APIs are offline.
    """
    readme = context.get("readme_content", "")
    lines = [clean_markdown_text(line) for line in readme.split("\n") if line.strip() and not line.startswith("<")]
    readable_intro = " ".join(lines[:4]) if lines else f"{repo_name} is a software project comprising {context.get('file_count', 0)} files."

    top_folders = context.get("top_level_folders", [])
    sample_files = context.get("sample_files", [])

    tech_detected = []
    if any(str(f).endswith((".tsx", ".jsx", ".ts", ".js")) for f in sample_files):
        tech_detected.append("TypeScript/JavaScript")
    if any(str(f).endswith(".py") for f in sample_files):
        tech_detected.append("Python")
    if any(str(f).endswith(".ipynb") for f in sample_files):
        tech_detected.append("Jupyter Notebook / Data Science")
    if any(str(f).endswith(".go") for f in sample_files):
        tech_detected.append("Go")
    if any(str(f).endswith(".rs") for f in sample_files):
        tech_detected.append("Rust")
    if any(str(f).endswith(".java") for f in sample_files):
        tech_detected.append("Java")
    if any(str(f).endswith((".cpp", ".c", ".h")) for f in sample_files):
        tech_detected.append("C/C++")

    tagline = f"{repo_name}: {readable_intro[:120].strip()}" if readable_intro else f"{repo_name} software project."

    return {
        "tagline": tagline,
        "overview": readable_intro,
        "core_purpose": f"The primary purpose of {repo_name} is to provide functionality implemented using {', '.join(tech_detected) or 'modern software techniques'}.",
        "target_audience": "Developers, technical users, and engineering teams interested in this domain.",
        "key_features": [
            f"Modular project architecture organized across {len(top_folders)} main directory modules",
            f"Automated execution and source assets across {context.get('file_count', 0)} project files",
            f"Implemented using {', '.join(tech_detected) or 'core languages'}"
        ],
        "tech_stack": {
            "languages": tech_detected or ["Polyglot"],
            "structure": top_folders[:5] or ["root"]
        },
        "architecture_summary": (
            f"The project is structured with {context.get('file_count', 0)} files across "
            f"directories ({', '.join(top_folders[:5]) or 'root'})."
        ),
        "model_used": "deterministic-heuristic"
    }


def generate_repo_summary(
    repo_id: str,
    repo_name: str,
    force_regenerate: bool = False
) -> Dict[str, Any]:
    """
    Generates or retrieves a comprehensive repository summary explaining what the website/project is about.
    Uses Gemini Flash (`gemini-2.5-flash` / `gemini-flash-latest`) as the primary model.
    """
    # 1. Check in-memory cache (unless force regenerating or cached item was a broken heuristic)
    if not force_regenerate and repo_id in _REPO_SUMMARY_MEMORY_CACHE:
        cached = _REPO_SUMMARY_MEMORY_CACHE[repo_id]
        if cached.get("model_used") != "deterministic-heuristic":
            return cached

    # 2. Check disk cache
    disk_path = os.path.join(CACHE_DIR, f"{repo_id}_repo_summary.json")
    if not force_regenerate and os.path.exists(disk_path):
        try:
            with open(disk_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
                # Only use cached disk if not broken heuristic
                if cached.get("model_used") != "deterministic-heuristic" and "<div" not in cached.get("overview", ""):
                    _REPO_SUMMARY_MEMORY_CACHE[repo_id] = cached
                    return cached
        except Exception:
            pass

    # 3. Retrieve repo analysis
    analysis = get_repo_analysis(repo_id) or {}
    context = gather_repo_context(repo_id, analysis)

    # 4. Construct prompt for Gemini Flash
    prompt = f"""
You are an expert principal software architect and technical lead.
Your task is to write a comprehensive, crystal-clear, and engaging executive repository summary explaining:
WHAT IS THIS WEBSITE / APPLICATION ABOUT?

Repository Name: {repo_name}
Total Files: {context['file_count']}
Key Directories: {', '.join(context['top_level_folders'])}
Detected Dependencies: {context['dependencies_count']}

=== CLEANED README CONTENT ===
{context['readme_content'] or 'No README provided in root.'}

=== PACKAGE & DEPENDENCY MANIFESTS ===
{context['manifest_snippet'] or 'None found.'}

=== CORE ENTRYPOINT & SOURCE SNIPPETS ===
{context['entrypoint_snippets'] or 'None found.'}

INSTRUCTIONS:
1. Base your summary STRICTLY and EXCLUSIVELY on the specific repository '{repo_name}' using its README, manifests, files, and code snippets provided above.
   - For example:
     * If '{repo_name}' is a Machine Learning or Data Science project (e.g. house price prediction, sentiment classification, image recognition), explain the ML models, dataset, prediction goals, and features.
     * If '{repo_name}' is a Web or Mobile Application (e.g. e-commerce, social, dashboard, tool), explain what users do on the platform.
     * If '{repo_name}' is a Developer Tool or Platform (e.g. linter, compiler, DevLens AI), explain its developer workflow.
2. What does THIS specific project '{repo_name}' do? What real-world user or business problem does it solve?
3. Who is the target audience for THIS project?
4. What are the key features and capabilities implemented in THIS repository?
5. What is the technical architecture and tech stack of THIS repository (languages, frameworks, libraries, databases)?
6. Strict Grounding Rules:
   - Do NOT assume that every repository is DevLens AI. You are using DevLens AI to review and explain '{repo_name}'.
   - Do NOT output boilerplate or hallucinated features. Every sentence must describe '{repo_name}'.
   - NEVER refer to this product or repo as 'HACKORA'.
7. Return ONLY a valid JSON object matching this schema:
{{
  "tagline": "A single compelling, punchy sentence capturing the project's essence.",
  "overview": "A rich, thorough explanation (2 to 3 paragraphs) detailing what the website/project is, why it exists, what it enables users to do, and the primary problem it solves.",
  "core_purpose": "2 to 4 sentences clearly defining the core mission and value proposition.",
  "target_audience": "1 to 2 sentences describing who this application was designed for (e.g., developers, enterprise teams, etc.).",
  "key_features": [
    "Feature 1: Clear description of capability and impact",
    "Feature 2: Clear description of capability and impact",
    "Feature 3: Clear description of capability and impact",
    "Feature 4: Clear description of capability and impact",
    "Feature 5: Clear description of capability and impact"
  ],
  "tech_stack": {{
    "frontend": ["Technology 1", "Technology 2"],
    "backend": ["Technology 1", "Technology 2"],
    "database_storage": ["Technology 1"],
    "ai_services": ["Gemini 2.5 Flash", "SentenceTransformers"]
  }},
  "architecture_summary": "1 to 2 paragraphs explaining how the pieces fit together: how requests flow from user interface to backend logic and databases."
}}
"""

    gemini_key = get_gemini_api_key()
    summary_data = None
    model_used = "gemini-2.5-flash"

    # 5. Execute with Gemini Flash if API key is present
    if gemini_key:
        try:
            print(f"[DevLens AI] Requesting repository summary for '{repo_name}' using Gemini Flash...")
            summary_data = call_gemini_flash(prompt, gemini_key)
            model_used = summary_data.pop("_model_resolved", "gemini-2.5-flash")
            print(f"[DevLens AI] Successfully generated repository summary with {model_used}!")
        except Exception as e:
            print(f"[DevLens AI Warning] Gemini Flash request encountered error: {str(e)}. Attempting fallback...")
            summary_data = None

    # 6. Fallback to Groq if Gemini key not set or failed
    if not summary_data:
        groq_res = fallback_with_groq(prompt)
        if groq_res:
            model_used = f"groq ({groq_res.pop('_fallback_model', 'openai/gpt-oss-120b')})"
            summary_data = groq_res
            print(f"[DevLens AI] Repository summary generated via fallback model: {model_used}")

    # 7. Fallback to heuristic if all LLMs are offline
    if not summary_data or not isinstance(summary_data, dict) or "overview" not in summary_data:
        summary_data = generate_heuristic_summary(repo_name, context)
        model_used = summary_data.get("model_used", "deterministic-heuristic")

    # Final sanitization
    summary_data["overview"] = clean_markdown_text(summary_data.get("overview", ""))
    summary_data["tagline"] = clean_markdown_text(summary_data.get("tagline", ""))
    summary_data["core_purpose"] = clean_markdown_text(summary_data.get("core_purpose", ""))
    summary_data["target_audience"] = clean_markdown_text(summary_data.get("target_audience", ""))
    summary_data["architecture_summary"] = clean_markdown_text(summary_data.get("architecture_summary", ""))
    if "key_features" in summary_data and isinstance(summary_data["key_features"], list):
        summary_data["key_features"] = [
            clean_markdown_text(str(feat)) for feat in summary_data["key_features"]
        ]

    summary_data["model_used"] = model_used
    summary_data["generated_at"] = time.time()
    summary_data["repo_id"] = repo_id
    summary_data["repo_name"] = repo_name

    # 8. Persist to memory and disk cache
    _REPO_SUMMARY_MEMORY_CACHE[repo_id] = summary_data
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(disk_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2)
    except Exception as e:
        print(f"[DevLens AI Warning] Failed to write summary to disk cache: {str(e)}")

    # 9. Sync to Supabase if table exists
    if supabase is not None:
        try:
            supabase.table("repo_summaries").upsert({
                "repo_id": repo_id,
                "summary": summary_data,
                "model_used": model_used
            }).execute()
        except Exception:
            pass

    return summary_data
