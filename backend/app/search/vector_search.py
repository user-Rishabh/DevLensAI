import json
from app.db import supabase
from app.search.embeddings import embed_chunk
from app.search.logger import log_debug

def dot_product(v1, v2):
    return sum(x * y for x, y in zip(v1, v2))

def vector_search(repo_id: str, query: str, top_k: int = 10) -> list[dict]:
    """
    Executes an index-accelerated PostgreSQL vector similarity search via pgvector RPC.
    
    Primary path: Invokes the database function `match_code_chunks()` which performs
    cosine distance ordering via HNSW index inside PostgreSQL and returns only the top K chunks,
    completely avoiding the massive bandwidth and latency cost of transferring full table vectors.
    
    Fallback path: If the database RPC is unavailable or encounters an error, gracefully falls back
    to client-side dot product calculation over available chunks.
    """
    if not repo_id or not query or not query.strip():
        return []

    if supabase is None:
        log_debug("[DevLens AI Error] Supabase client offline. Cannot execute vector search.")
        return []

    try:
        query_vector = embed_chunk(query)
    except Exception as e:
        log_debug(f"[DevLens AI Error] Query embedding generation failed: {str(e)}")
        return []

    log_debug(f"[VECTOR SEARCH] repo_id: {repr(repo_id)}, query: {repr(query[:50])}, top_k: {top_k}")

    # =========================================================================
    # PRIMARY PATH: PostgreSQL pgvector RPC (Database-Side HNSW Index Scan)
    # =========================================================================
    try:
        rpc_params = {
            "query_embedding": query_vector,
            "match_threshold": -1.0,
            "match_count": max(1, min(top_k, 100)),
            "filter_repo_id": repo_id
        }
        rpc_response = supabase.rpc("match_code_chunks", rpc_params).execute()
        
        if rpc_response.data is not None:
            results = []
            for row in rpc_response.data:
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
                    "similarity": float(row.get("similarity", 0.0))
                })
            log_debug(f"[VECTOR SEARCH RPC] Success: Returned {len(results)} chunks from pgvector RPC.")
            return results
    except Exception as rpc_err:
        log_debug(f"[DevLens AI Warning] pgvector RPC execution failed or not yet deployed: {str(rpc_err)}. Using client fallback.")

    # =========================================================================
    # FALLBACK PATH: Client-side dot product (for unmigrated or offline setups)
    # =========================================================================
    try:
        response = supabase.table("code_chunks")\
            .select("id, repo_id, file_path, chunk_type, name, parent_class, start_line, end_line, content, embedding")\
            .eq("repo_id", repo_id)\
            .execute()

        chunks = response.data if response.data else []
        log_debug(f"[VECTOR SEARCH FALLBACK] Fetched {len(chunks)} chunks for client-side dot product.")

        scored_chunks = []
        for chunk in chunks:
            emb = chunk.get("embedding")
            if not emb:
                continue
            if isinstance(emb, str):
                try:
                    emb = json.loads(emb)
                except Exception:
                    continue
            if not isinstance(emb, list):
                continue

            similarity = dot_product(query_vector, emb)
            chunk_copy = dict(chunk)
            chunk_copy["similarity"] = similarity
            chunk_copy.pop("embedding", None)  # Clean out heavy embedding vector
            scored_chunks.append(chunk_copy)

        scored_chunks.sort(key=lambda c: c["similarity"], reverse=True)
        results = scored_chunks[:top_k]
        return results
    except Exception as e:
        log_debug(f"[DevLens AI Error] Vector similarity fallback lookup failed: {str(e)}")
        return []
