-- ==============================================================================
-- DevLens 2.0 — Database-Side Hybrid Retrieval Migration
-- 1. HNSW Vector Index on code_chunks.embedding (vector_cosine_ops)
-- 2. GIN Full-Text Search on code_chunks.search_tsv (with code identifier & underscore support)
-- 3. Vector Similarity Search RPC: match_code_chunks
-- 4. PostgreSQL Full-Text Search RPC: search_code_chunks_fts (with ts_rank_cd)
-- ==============================================================================

-- 1. Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Verify or create HNSW index for cosine distance
-- HNSW provides sub-linear O(log N) nearest neighbor search without downloading vectors to client
CREATE INDEX IF NOT EXISTS idx_code_chunks_embedding_hnsw 
ON code_chunks USING hnsw (embedding vector_cosine_ops);

-- 3. Create composite TSVector column incorporating file_path, name, and content with underscore decomposition
-- In codebases, identifiers like 'compute_blast_radius' must match both exact identifier queries
-- and decomposed words ('compute', 'blast', 'radius').
-- Weight hierarchy: 'A' for function/class names, 'B' for file paths, 'C' for code content.
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns 
    WHERE table_name = 'code_chunks' AND column_name = 'search_tsv'
  ) THEN
    ALTER TABLE code_chunks 
    ADD COLUMN search_tsv tsvector 
    GENERATED ALWAYS AS (
      setweight(to_tsvector('simple', COALESCE(name, '') || ' ' || replace(COALESCE(name, ''), '_', ' ')), 'A') ||
      setweight(to_tsvector('simple', COALESCE(file_path, '') || ' ' || replace(COALESCE(file_path, ''), '/', ' ')), 'B') ||
      setweight(to_tsvector('simple', COALESCE(content, '') || ' ' || replace(COALESCE(content, ''), '_', ' ')), 'C')
    ) STORED;
  END IF;
END $$;

-- 4. Create GIN index on search_tsv for fast full-text lookup
CREATE INDEX IF NOT EXISTS idx_code_chunks_search_tsv 
ON code_chunks USING gin (search_tsv);

-- 5. Ensure metadata indexes exist for scoped repository lookups
CREATE INDEX IF NOT EXISTS idx_code_chunks_repo_id ON code_chunks (repo_id);
CREATE INDEX IF NOT EXISTS idx_code_chunks_file_path ON code_chunks (file_path);

-- 6. PostgreSQL Vector Search RPC function: match_code_chunks
-- Replaces client-side dot product by executing cosine distance directly in PostgreSQL/pgvector.
-- Returns only the requested top K chunks, completely eliminating embedding vector transmission over the network.
CREATE OR REPLACE FUNCTION match_code_chunks (
  query_embedding vector(384),
  match_threshold float DEFAULT -1.0,
  match_count int DEFAULT 10,
  filter_repo_id text DEFAULT ''
)
RETURNS TABLE (
  id uuid,
  repo_id text,
  file_path text,
  chunk_type text,
  name text,
  parent_class text,
  start_line int,
  end_line int,
  content text,
  similarity float
)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
BEGIN
  -- Safe parameter validation: prevent empty queries or cross-repo leaks
  IF query_embedding IS NULL OR filter_repo_id IS NULL OR filter_repo_id = '' THEN
    RETURN;
  END IF;

  RETURN QUERY
  SELECT
    c.id,
    c.repo_id,
    c.file_path,
    c.chunk_type,
    c.name,
    c.parent_class,
    c.start_line,
    c.end_line,
    c.content,
    -- Cosine similarity is computed as 1.0 - cosine distance (operator <=>)
    (1.0 - (c.embedding <=> query_embedding))::float AS similarity
  FROM code_chunks c
  WHERE c.repo_id = filter_repo_id
    AND (1.0 - (c.embedding <=> query_embedding)) >= COALESCE(match_threshold, -1.0)
  ORDER BY c.embedding <=> query_embedding ASC
  LIMIT GREATEST(1, LEAST(COALESCE(match_count, 10), 100));
END;
$$;

-- 7. PostgreSQL Full-Text Search RPC function: search_code_chunks_fts
-- Replaces naive Python string matching with index-backed PostgreSQL full-text search.
-- Uses websearch_to_tsquery for natural multi-word and quoted queries,
-- and ts_rank_cd (Cover Density Ranking) to prioritize chunks where query terms appear in close proximity.
CREATE OR REPLACE FUNCTION search_code_chunks_fts (
  query_text text,
  filter_repo_id text,
  match_count int DEFAULT 10
)
RETURNS TABLE (
  id uuid,
  repo_id text,
  file_path text,
  chunk_type text,
  name text,
  parent_class text,
  start_line int,
  end_line int,
  content text,
  rank_score float
)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
DECLARE
  parsed_query tsquery;
  clean_q text;
BEGIN
  -- Safe parameter validation
  IF query_text IS NULL OR filter_repo_id IS NULL OR filter_repo_id = '' THEN
    RETURN;
  END IF;

  clean_q := trim(query_text);
  IF clean_q = '' THEN
    RETURN;
  END IF;

  -- 1. First parse with websearch_to_tsquery using simple configuration (handles quotes and phrases)
  parsed_query := websearch_to_tsquery('simple', clean_q);
  
  -- 2. Fallback to plainto_tsquery if websearch returned empty (e.g. only operators passed)
  IF parsed_query IS NULL OR parsed_query = ''::tsquery THEN
    parsed_query := plainto_tsquery('simple', clean_q);
  END IF;

  IF parsed_query IS NULL OR parsed_query = ''::tsquery THEN
    RETURN;
  END IF;

  RETURN QUERY
  SELECT
    c.id,
    c.repo_id,
    c.file_path,
    c.chunk_type,
    c.name,
    c.parent_class,
    c.start_line,
    c.end_line,
    c.content,
    -- ts_rank_cd computes cover density ranking with weight scaling
    -- Flag 32 normalizes rank score as score / (score + 1)
    ts_rank_cd(COALESCE(c.search_tsv, c.content_tsv), parsed_query, 32)::float AS rank_score
  FROM code_chunks c
  WHERE c.repo_id = filter_repo_id
    AND COALESCE(c.search_tsv, c.content_tsv) @@ parsed_query
  ORDER BY rank_score DESC, c.start_line ASC
  LIMIT GREATEST(1, LEAST(COALESCE(match_count, 10), 100));
END;
$$;

-- 8. Grant execute permissions on RPC functions
GRANT EXECUTE ON FUNCTION match_code_chunks TO anon, authenticated, service_role;
GRANT EXECUTE ON FUNCTION search_code_chunks_fts TO anon, authenticated, service_role;
