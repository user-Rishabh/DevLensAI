import re
from app.db import supabase
from app.search.logger import log_debug

def sanitize_fts_query(raw_query: str) -> str:
    """
    Cleans and prepares a user query for PostgreSQL full-text search.
    Preserves meaningful code identifiers (like `compute_blast_radius`, `app.main`, `UserProfile`),
    while safely stripping noisy punctuation (such as `?`, `!`, `,`, `;`, `(`, `)`, `{`, `}`, `[`, `]`).
    Decomposes snake_case identifiers into both full identifier and component words.
    """
    if not raw_query or not raw_query.strip():
        return ""

    # Replace syntax/noise punctuation with spaces, while keeping underscores, hyphens, and periods
    cleaned = re.sub(r'[?!,;:(){}\[\]\\<>\'"`*^~#$@|/=+\&%]', ' ', raw_query)
    tokens = cleaned.split()
    if not tokens:
        return ""

    expanded_tokens = []
    for t in tokens:
        expanded_tokens.append(t)
        # If token contains underscores (e.g. compute_blast_radius), add decomposed words
        if "_" in t:
            parts = [p for p in t.split("_") if p]
            if len(parts) > 1:
                expanded_tokens.extend(parts)

    return " ".join(expanded_tokens)

def keyword_search(repo_id: str, query: str, top_k: int = 10) -> list[dict]:
    """
    Executes PostgreSQL full-text search over indexed codebase chunks.
    
    Primary path: Invokes the database RPC `search_code_chunks_fts` (or legacy `match_code_chunks_keyword`),
    leveraging a GIN index and `ts_rank_cd` cover-density ranking directly within PostgreSQL.
    Eliminates downloading full repository chunk datasets into Python.
    
    Fallback path: If database RPCs are not yet provisioned or network is unavailable,
    falls back gracefully to client-side token relevance scoring.
    """
    if not repo_id or not query or not query.strip():
        return []

    if supabase is None:
        log_debug("[DevLens AI Error] Supabase client offline. Cannot execute keyword search.")
        return []

    clean_query = sanitize_fts_query(query)
    if not clean_query:
        return []

    log_debug(f"[KEYWORD SEARCH FTS] repo_id: {repr(repo_id)}, query: {repr(clean_query[:60])}, top_k: {top_k}")

    # =========================================================================
    # PRIMARY PATH: PostgreSQL Full-Text Search RPC (`search_code_chunks_fts`)
    # =========================================================================
    try:
        rpc_params = {
            "query_text": clean_query,
            "filter_repo_id": repo_id,
            "match_count": max(1, min(top_k, 100))
        }
        rpc_res = supabase.rpc("search_code_chunks_fts", rpc_params).execute()
        if rpc_res.data is not None and len(rpc_res.data) > 0:
            results = []
            for row in rpc_res.data:
                results.append({
                    "id": str(row["id"]),
                    "repo_id": row["repo_id"],
                    "file_path": row["file_path"],
                    "chunk_type": row["chunk_type"],
                    "name": row["name"],
                    "parent_class": row.get("parent_class"),
                    "start_line": int(row["start_line"]),
                    "end_line": int(row["end_line"]),
                    "content": row["content"],
                    "rank_score": float(row.get("rank_score", 0.0))
                })
            log_debug(f"[KEYWORD SEARCH FTS] RPC returned {len(results)} chunks from PostgreSQL.")
            return results
    except Exception as rpc_err:
        log_debug(f"[DevLens AI Info] search_code_chunks_fts RPC call failed or pending: {str(rpc_err)}. Trying legacy RPC...")

    # =========================================================================
    # SECONDARY PATH: Legacy Database RPC (`match_code_chunks_keyword`)
    # =========================================================================
    try:
        legacy_params = {
            "query_text": clean_query,
            "filter_repo_id": repo_id,
            "match_count": max(1, min(top_k, 100))
        }
        legacy_res = supabase.rpc("match_code_chunks_keyword", legacy_params).execute()
        if legacy_res.data is not None and len(legacy_res.data) > 0:
            results = []
            for row in legacy_res.data:
                results.append({
                    "id": str(row["id"]),
                    "repo_id": row["repo_id"],
                    "file_path": row["file_path"],
                    "chunk_type": row["chunk_type"],
                    "name": row["name"],
                    "parent_class": row.get("parent_class"),
                    "start_line": int(row["start_line"]),
                    "end_line": int(row["end_line"]),
                    "content": row["content"],
                    "rank_score": float(row.get("rank_score", 0.0))
                })
            log_debug(f"[KEYWORD SEARCH LEGACY RPC] Returned {len(results)} chunks.")
            return results
    except Exception as legacy_err:
        log_debug(f"[DevLens AI Info] Legacy RPC not available: {str(legacy_err)}. Falling back to client-side matching.")

    # =========================================================================
    # FALLBACK PATH: Client-side Token & Identifier Matching
    # =========================================================================
    try:
        response = supabase.table("code_chunks")\
            .select("id, repo_id, file_path, chunk_type, name, parent_class, start_line, end_line, content")\
            .eq("repo_id", repo_id)\
            .execute()
            
        chunks = response.data if response.data else []
        log_debug(f"[KEYWORD SEARCH FALLBACK] Fetched {len(chunks)} chunks for client-side search.")
        
        query_terms = [t.lower() for t in clean_query.split() if len(t) > 1]
        if not query_terms:
            query_terms = [t.lower() for t in clean_query.split() if t]
        if not query_terms:
            return []

        scored_chunks = []
        for chunk in chunks:
            content_lower = chunk.get("content", "").lower()
            name_lower = (chunk.get("name") or "").lower()
            path_lower = (chunk.get("file_path") or "").lower()
            
            score = 0.0
            
            # Exact symbol name match (Highest priority)
            for term in query_terms:
                if term == name_lower:
                    score += 15.0
                elif term in name_lower:
                    score += 8.0
                if term in path_lower:
                    score += 5.0
                if term in content_lower:
                    # Count term frequency with diminishing returns
                    count = min(content_lower.count(term), 10)
                    score += 1.0 + (count * 0.2)
                    
            if score > 0:
                chunk_copy = dict(chunk)
                chunk_copy["rank_score"] = score
                scored_chunks.append(chunk_copy)
                
        scored_chunks.sort(key=lambda c: c["rank_score"], reverse=True)
        results = scored_chunks[:top_k]
        return results
    except Exception as e:
        log_debug(f"[DevLens AI Error] Keyword search fallback lookup failed: {str(e)}")
        return []
