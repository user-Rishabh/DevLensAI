import os
import re
from typing import List, Dict, Any, Optional

from app.search.hybrid_search import hybrid_search
from app.analysis.blast_radius import compute_blast_radius
from app.db import get_repo_analysis, get_file_content, supabase
from app.search.logger import log_debug

# ==============================================================================
# STAGE 1: INTENT & ENTITY REASONING
# ==============================================================================

INTENT_FLOW_ARCHITECTURE = "FLOW_ARCHITECTURE"
INTENT_FEATURE_IMPLEMENTATION = "FEATURE_IMPLEMENTATION"
INTENT_CHANGE_IMPACT = "CHANGE_IMPACT"
INTENT_GENERAL = "GENERAL"

def classify_query_intent(query: str) -> Dict[str, Any]:
    """
    Analyzes the user's question to identify underlying intent and target concepts.
    """
    q_lower = query.lower()

    # 1. Change impact / blast radius intent
    impact_keywords = [
        "modify", "change", "affect", "impact", "blast radius", "break", 
        "depend", "downstream", "callers", "refactor", "if i update", "if i delete"
    ]
    if any(k in q_lower for k in impact_keywords):
        return {
            "intent": INTENT_CHANGE_IMPACT,
            "target": _extract_target_symbol_or_module(q_lower),
        }

    # 2. Complete flow / lifecycle / architecture intent
    flow_keywords = [
        "request flow", "flow from", "frontend to backend", "lifecycle", 
        "step by step", "when a user submits", "execution flow", "end to end", 
        "architecture overview", "how does the system process"
    ]
    if any(k in q_lower for k in flow_keywords):
        return {
            "intent": INTENT_FLOW_ARCHITECTURE,
            "target": _extract_target_symbol_or_module(q_lower),
        }

    # 3. Technical feature implementation intent
    implementation_keywords = [
        "how does", "how is", "explain", "implement", "algorithm", 
        "vector search", "bm25", "rrf", "embedding", "chunking", "hotspot"
    ]
    if any(k in q_lower for k in implementation_keywords):
        return {
            "intent": INTENT_FEATURE_IMPLEMENTATION,
            "target": _extract_target_symbol_or_module(q_lower),
        }

    return {"intent": INTENT_GENERAL, "target": None}

def _extract_target_symbol_or_module(q_lower: str) -> Optional[str]:
    """Extracts likely target concepts such as authentication, search, ingestion, etc."""
    patterns = [
        ("auth", "authentication"),
        ("search", "search"),
        ("vector", "vector_search"),
        ("embed", "embeddings"),
        ("bm25", "keyword_search"),
        ("rrf", "hybrid_search"),
        ("chunk", "chunker"),
        ("ingest", "ingestion"),
        ("hotspot", "git_hotspots"),
        ("quality", "quality_score"),
        ("blast", "blast_radius"),
    ]
    for key, name in patterns:
        if key in q_lower:
            return name
    return None


# ==============================================================================
# STAGE 2 & 3: MULTI-STAGE RETRIEVAL WITH CODE RELATIONSHIP EXPANSION
# ==============================================================================

def multistage_code_retrieval(repo_id: str, query: str, top_k: int = 15) -> Dict[str, Any]:
    """
    Executes an intelligent multi-stage retrieval:
    1. Intent classification.
    2. Hybrid vector + full-text search (RRF).
    3. Structural & dependency graph expansion based on intent:
       - FLOW_ARCHITECTURE: Bridges frontend trigger to backend controller & internal engine steps.
       - CHANGE_IMPACT: Traverses AST dependency graph and calculates blast radius with exact call chains.
       - FEATURE_IMPLEMENTATION: Ensures core implementation files for requested concepts are retrieved.
    4. Gathers diagnostics for observability.
    """
    intent_info = classify_query_intent(query)
    intent = intent_info["intent"]
    target = intent_info.get("target")

    # 1. Base Hybrid Retrieval
    base_results = hybrid_search(repo_id, query, top_k=top_k)
    
    # Track statistics
    retrieved_chunk_ids = {c.get("id") for c in base_results if c.get("id")}
    expanded_chunks: List[Dict[str, Any]] = list(base_results)
    dependency_graph_context: Dict[str, Any] = {}
    expansion_count = 0

    analysis = get_repo_analysis(repo_id) or {}
    dependencies = analysis.get("dependencies", [])

    # 2. Stage-Specific Relationship Expansion
    if intent == INTENT_CHANGE_IMPACT and dependencies:
        # Trace direct and indirect reverse dependencies for target module/files
        target_files = _find_matching_files_in_repo(repo_id, target, query)
        blast_reports = []
        for tf in target_files[:3]:
            blast = compute_blast_radius(repo_id, tf, max_depth=3, dependencies=dependencies)
            blast_reports.append(blast)
            
            # Fetch chunks from directly affected dependent files
            all_affected = set(blast.get("direct_dependents", []))
            for trans in blast.get("transitive_dependents", []):
                all_affected.add(trans.get("file_path"))

            for dep_file in list(all_affected)[:5]:
                extra_chunks = _get_chunks_for_file(repo_id, dep_file, limit=2)
                for echunk in extra_chunks:
                    if echunk.get("id") not in retrieved_chunk_ids:
                        retrieved_chunk_ids.add(echunk.get("id"))
                        expanded_chunks.append(echunk)
                        expansion_count += 1

        dependency_graph_context = {
            "target_files": target_files,
            "blast_reports": blast_reports,
            "dependencies_count": len(dependencies),
        }

    elif intent == INTENT_FLOW_ARCHITECTURE:
        # Trace flow: If frontend and API routes are involved, bridge the connected backend handlers
        relevant_flow_modules = [
            "frontend/src/pages/Home.tsx",
            "frontend/src/components/LandingPage.tsx",
            "backend/app/api/routes.py",
            "backend/app/ingestion/clone.py",
            "backend/app/ingestion/filter.py",
            "backend/app/analysis/dependency_graph.py",
            "backend/app/analysis/git_hotspots.py",
            "backend/app/db.py",
            "backend/app/search/chunker.py",
            "backend/app/search/embeddings.py",
            "frontend/src/components/LoadingScreen.tsx",
        ]
        for fpath in relevant_flow_modules:
            f_chunks = _get_chunks_for_file(repo_id, fpath, limit=2)
            for ch in f_chunks:
                if ch.get("id") not in retrieved_chunk_ids:
                    retrieved_chunk_ids.add(ch.get("id"))
                    expanded_chunks.append(ch)
                    expansion_count += 1

    elif intent == INTENT_FEATURE_IMPLEMENTATION:
        # If query asks about semantic search, embeddings, vector search, BM25, RRF:
        feature_module_map = {
            "vector": ["backend/app/search/vector_search.py", "backend/app/search/embeddings.py"],
            "search": ["backend/app/search/hybrid_search.py", "backend/app/search/keyword_search.py"],
            "bm25": ["backend/app/search/keyword_search.py"],
            "rrf": ["backend/app/search/hybrid_search.py"],
            "embed": ["backend/app/search/embeddings.py"],
            "rag": ["backend/app/llm/rag_answer.py", "backend/app/llm/groq_client.py"],
            "hotspot": ["backend/app/analysis/git_hotspots.py"],
            "blast": ["backend/app/analysis/blast_radius.py"],
        }
        for kw, paths in feature_module_map.items():
            if kw in query.lower():
                for p in paths:
                    extra = _get_chunks_for_file(repo_id, p, limit=3)
                    for echunk in extra:
                        if echunk.get("id") not in retrieved_chunk_ids:
                            retrieved_chunk_ids.add(echunk.get("id"))
                            expanded_chunks.append(echunk)
                            expansion_count += 1

    # Diagnostics
    diagnostics = {
        "intent": intent,
        "base_results_count": len(base_results),
        "expanded_results_count": len(expanded_chunks),
        "dependency_expansions": expansion_count,
        "has_dependency_graph": bool(dependencies),
    }

    return {
        "chunks": expanded_chunks,
        "intent_info": intent_info,
        "dependency_graph_context": dependency_graph_context,
        "diagnostics": diagnostics,
    }

def _find_matching_files_in_repo(repo_id: str, target: Optional[str], query: str) -> List[str]:
    """Finds matching files in the repository for a given target topic."""
    from app.analysis.dependency_graph import get_all_project_files
    analysis = get_repo_analysis(repo_id) or {}
    all_files = list(get_all_project_files(analysis.get("file_tree", {})))
    
    matches = []
    keywords = [target] if target else []
    for word in query.lower().split():
        if len(word) > 3 and word not in ("which", "files", "functions", "could", "affected", "modify"):
            keywords.append(word)

    for f in all_files:
        f_lower = f.lower()
        if any(k in f_lower for k in keywords):
            matches.append(f)

    return matches[:5]

def _get_chunks_for_file(repo_id: str, file_path: str, limit: int = 2) -> List[Dict[str, Any]]:
    """Retrieves chunks for a specific file directly from Supabase code_chunks or generates fallback."""
    if supabase is not None:
        try:
            res = supabase.table("code_chunks")\
                .select("id, repo_id, file_path, chunk_type, name, parent_class, start_line, end_line, content")\
                .eq("repo_id", repo_id)\
                .eq("file_path", file_path)\
                .limit(limit)\
                .execute()
            if res.data:
                return res.data
        except Exception:
            pass

    # Fallback to local content chunking if DB does not yet have this file indexed
    content = get_file_content(repo_id, file_path)
    if not content:
        return []

    lines = content.splitlines()
    total_lines = len(lines)
    head_content = "\n".join(lines[:min(100, total_lines)])
    return [{
        "id": f"dyn_{repo_id}_{file_path}_1",
        "repo_id": repo_id,
        "file_path": file_path,
        "chunk_type": "module_segment",
        "name": file_path.split("/")[-1],
        "start_line": 1,
        "end_line": min(100, total_lines),
        "content": head_content,
        "found_by": ["structural_graph_expansion"],
        "rrf_score": 0.85
    }]

