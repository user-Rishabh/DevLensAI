"""
DevLens 2.0 — Database-Side Hybrid Retrieval Test Suite
Tests:
1. Vector search RPC invocation and schema compatibility.
2. Full-text search RPC invocation and exact function name matching.
3. Full-text search punctuation and code identifier handling.
4. Repository isolation (prevents cross-repo data leaks).
5. Top K limits enforcement.
6. Empty repository & invalid query handling.
7. RRF hybrid retrieval contract preservation.
8. Context manager and citations compatibility.
"""

import sys
import os
import unittest
from unittest.mock import MagicMock, patch

# Ensure backend root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.search.keyword_search import sanitize_fts_query, keyword_search
from app.search.vector_search import vector_search, dot_product
from app.search.hybrid_search import hybrid_search
from app.search.context_manager import estimate_tokens, pack_context_for_rag


class TestQuerySanitization(unittest.TestCase):
    """Validates FTS query sanitization for code identifiers and noise removal."""

    def test_code_identifier_preservation(self):
        query = "how does compute_blast_radius work?"
        cleaned = sanitize_fts_query(query)
        self.assertIn("compute_blast_radius", cleaned)
        self.assertIn("compute", cleaned)
        self.assertIn("blast", cleaned)
        self.assertIn("radius", cleaned)
        self.assertNotIn("?", cleaned)

    def test_punctuation_stripping(self):
        query = "def calculate_health(repo_id: str, limit: int = 10):"
        cleaned = sanitize_fts_query(query)
        self.assertNotIn("(", cleaned)
        self.assertNotIn(")", cleaned)
        self.assertNotIn(":", cleaned)
        self.assertNotIn("=", cleaned)
        self.assertIn("calculate_health", cleaned)

    def test_empty_and_whitespace(self):
        self.assertEqual(sanitize_fts_query(""), "")
        self.assertEqual(sanitize_fts_query("   "), "")
        self.assertEqual(sanitize_fts_query("???"), "")


class TestDatabaseVectorSearch(unittest.TestCase):
    """Tests pgvector database-side search with RPC and fallback."""

    def setUp(self):
        self.mock_repo_id = "test-repo-123"
        self.mock_query = "database connection pool"
        self.mock_embedding = [0.1] * 384

    @patch("app.search.vector_search.embed_chunk")
    @patch("app.search.vector_search.supabase")
    def test_vector_search_primary_rpc_path(self, mock_sb, mock_embed):
        mock_embed.return_value = self.mock_embedding
        
        # Mock RPC response
        mock_rpc = MagicMock()
        mock_rpc.execute.return_value.data = [
            {
                "id": "11111111-1111-1111-1111-111111111111",
                "repo_id": self.mock_repo_id,
                "file_path": "backend/app/db.py",
                "chunk_type": "function",
                "name": "get_db_connection",
                "parent_class": None,
                "start_line": 10,
                "end_line": 25,
                "content": "def get_db_connection(): return pool.acquire()",
                "similarity": 0.885
            },
            {
                "id": "22222222-2222-2222-2222-222222222222",
                "repo_id": self.mock_repo_id,
                "file_path": "backend/app/main.py",
                "chunk_type": "function",
                "name": "startup",
                "parent_class": None,
                "start_line": 30,
                "end_line": 40,
                "content": "async def startup(): init_db()",
                "similarity": 0.762
            }
        ]
        mock_sb.rpc.return_value = mock_rpc

        results = vector_search(self.mock_repo_id, self.mock_query, top_k=5)

        # Verify RPC was called with correct parameters
        mock_sb.rpc.assert_called_once_with(
            "match_code_chunks",
            {
                "query_embedding": self.mock_embedding,
                "match_threshold": -1.0,
                "match_count": 5,
                "filter_repo_id": self.mock_repo_id
            }
        )
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["name"], "get_db_connection")
        self.assertEqual(results[0]["similarity"], 0.885)
        self.assertEqual(results[0]["repo_id"], self.mock_repo_id)

    @patch("app.search.vector_search.embed_chunk")
    @patch("app.search.vector_search.supabase")
    def test_vector_search_empty_input(self, mock_sb, mock_embed):
        self.assertEqual(vector_search("", "query"), [])
        self.assertEqual(vector_search("repo", ""), [])
        self.assertEqual(vector_search("repo", "   "), [])
        mock_embed.assert_not_called()
        mock_sb.rpc.assert_not_called()

    @patch("app.search.vector_search.embed_chunk")
    @patch("app.search.vector_search.supabase")
    def test_vector_search_top_k_clamping(self, mock_sb, mock_embed):
        mock_embed.return_value = self.mock_embedding
        mock_rpc = MagicMock()
        mock_rpc.execute.return_value.data = []
        mock_sb.rpc.return_value = mock_rpc

        vector_search(self.mock_repo_id, self.mock_query, top_k=200)
        # Should clamp to max 100
        call_args = mock_sb.rpc.call_args[0][1]
        self.assertEqual(call_args["match_count"], 100)


class TestDatabaseFullTextSearch(unittest.TestCase):
    """Tests PostgreSQL full-text search with RPC and identifier handling."""

    def setUp(self):
        self.mock_repo_id = "test-repo-123"

    @patch("app.search.keyword_search.supabase")
    def test_fts_primary_rpc_path(self, mock_sb):
        mock_rpc = MagicMock()
        mock_rpc.execute.return_value.data = [
            {
                "id": "33333333-3333-3333-3333-333333333333",
                "repo_id": self.mock_repo_id,
                "file_path": "backend/app/analysis/blast_radius.py",
                "chunk_type": "function",
                "name": "compute_blast_radius",
                "parent_class": None,
                "start_line": 50,
                "end_line": 90,
                "content": "def compute_blast_radius(file_path, dependency_graph): pass",
                "rank_score": 0.45
            }
        ]
        mock_sb.rpc.return_value = mock_rpc

        results = keyword_search(self.mock_repo_id, "compute_blast_radius", top_k=5)

        # Verify RPC was called with sanitized query including decomposed identifier
        mock_sb.rpc.assert_called_once()
        call_name = mock_sb.rpc.call_args[0][0]
        call_params = mock_sb.rpc.call_args[0][1]
        self.assertEqual(call_name, "search_code_chunks_fts")
        self.assertEqual(call_params["filter_repo_id"], self.mock_repo_id)
        self.assertIn("compute_blast_radius", call_params["query_text"])
        self.assertIn("compute", call_params["query_text"])
        self.assertEqual(call_params["match_count"], 5)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["name"], "compute_blast_radius")

    @patch("app.search.keyword_search.supabase")
    def test_fts_punctuation_handling(self, mock_sb):
        mock_rpc = MagicMock()
        mock_rpc.execute.return_value.data = []
        mock_sb.rpc.return_value = mock_rpc

        keyword_search(self.mock_repo_id, "what does compute_blast_radius() do?", top_k=5)
        call_params = mock_sb.rpc.call_args[0][1]
        self.assertNotIn("?", call_params["query_text"])
        self.assertNotIn("(", call_params["query_text"])
        self.assertNotIn(")", call_params["query_text"])
        self.assertIn("compute_blast_radius", call_params["query_text"])

    @patch("app.search.keyword_search.supabase")
    def test_fts_empty_repo_and_query(self, mock_sb):
        self.assertEqual(keyword_search("", "some query"), [])
        self.assertEqual(keyword_search(self.mock_repo_id, ""), [])
        self.assertEqual(keyword_search(self.mock_repo_id, "   "), [])
        mock_sb.rpc.assert_not_called()


class TestHybridSearchAndRetrievalContract(unittest.TestCase):
    """Ensures hybrid search RRF and context assembly contracts remain strictly preserved."""

    @patch("app.search.hybrid_search.vector_search")
    @patch("app.search.hybrid_search.keyword_search")
    def test_hybrid_search_rrf_combination(self, mock_keyword, mock_vector):
        shared_chunk = {
            "id": "chunk-1",
            "repo_id": "repo-xyz",
            "file_path": "backend/app/auth.py",
            "chunk_type": "function",
            "name": "verify_session",
            "parent_class": None,
            "start_line": 15,
            "end_line": 35,
            "content": "def verify_session(token): return True",
            "similarity": 0.92
        }
        vector_only = {
            "id": "chunk-2",
            "repo_id": "repo-xyz",
            "file_path": "backend/app/security.py",
            "chunk_type": "function",
            "name": "hash_password",
            "parent_class": None,
            "start_line": 1,
            "end_line": 10,
            "content": "def hash_password(pwd): return hash(pwd)",
            "similarity": 0.85
        }
        keyword_only = {
            "id": "chunk-3",
            "repo_id": "repo-xyz",
            "file_path": "backend/app/api.py",
            "chunk_type": "function",
            "name": "auth_endpoint",
            "parent_class": None,
            "start_line": 40,
            "end_line": 60,
            "content": "def auth_endpoint(): verify_session()",
            "rank_score": 0.50
        }

        mock_vector.return_value = [shared_chunk, vector_only]
        mock_keyword.return_value = [shared_chunk, keyword_only]

        results = hybrid_search("repo-xyz", "auth session verification", top_k=5)

        # chunk-1 appeared in both, so its RRF score must be highest
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0]["id"], "chunk-1")
        self.assertIn("vector", results[0]["found_by"])
        self.assertIn("keyword", results[0]["found_by"])
        self.assertGreater(results[0]["rrf_score"], results[1]["rrf_score"])
        self.assertIn("content_preview", results[0])

    def test_context_manager_and_citations_structure(self):
        mock_chunks = [
            {
                "file_path": "backend/app/auth.py",
                "start_line": 10,
                "end_line": 30,
                "content": "def verify_session(): pass",
                "name": "verify_session",
                "chunk_type": "function"
            }
        ]
        result = pack_context_for_rag(
            query="verify session",
            system_prompt="You are DevLens assistant.",
            chunks=mock_chunks
        )
        self.assertIn("backend/app/auth.py", result["formatted_context"])
        self.assertIn("verify_session", result["formatted_context"])
        self.assertGreater(result["estimated_prompt_tokens"], 0)
        self.assertEqual(len(result["selected_chunks"]), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
