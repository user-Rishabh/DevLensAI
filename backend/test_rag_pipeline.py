import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Ensure backend root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.search.context_manager import (
    estimate_tokens,
    get_model_budget_profile,
    deduplicate_chunks,
    truncate_chunk_content,
    pack_context_for_rag,
)
from app.llm.groq_client import (
    parse_groq_retry_delay,
    is_tpm_rate_limit,
)
from app.llm.rag_answer import generate_rag_answer


class TestRAGContextManager(unittest.TestCase):

    def test_token_estimation_conservative(self):
        """Scenario: Verify token estimation is conservative and fast."""
        sample_code = "def authenticate_user(token: str) -> dict:\n    return {'user_id': token}\n"
        est = estimate_tokens(sample_code)
        self.assertGreater(est, 5)
        # Empty string should yield 0
        self.assertEqual(estimate_tokens(""), 0)

    def test_model_budget_profile_defaults_and_overrides(self):
        """Scenario: Configurable token budget based on provider/model limits."""
        # Default for openai/gpt-oss-20b has 8000 TPM
        profile = get_model_budget_profile("openai/gpt-oss-20b")
        self.assertEqual(profile["tpm_limit"], 8000)
        self.assertLessEqual(profile["max_request_tokens"], 8000)
        self.assertEqual(profile["reserved_output_tokens"], 800)
        self.assertGreaterEqual(profile["max_prompt_tokens"], 1500)

        # Environment variable override test
        with patch.dict(os.environ, {"GROQ_TPM_LIMIT": "16000", "GROQ_MAX_REQUEST_TOKENS": "6000"}):
            custom_profile = get_model_budget_profile("openai/gpt-oss-20b")
            self.assertEqual(custom_profile["tpm_limit"], 16000)
            self.assertEqual(custom_profile["max_request_tokens"], 6000)

    def test_deduplication_of_overlapping_chunks(self):
        """Scenario: Avoid duplicate or overlapping code chunks."""
        chunks = [
            {"id": "1", "file_path": "app/auth.py", "start_line": 10, "end_line": 50, "content": "line\n" * 40},
            {"id": "2", "file_path": "app/auth.py", "start_line": 15, "end_line": 45, "content": "line\n" * 30}, # >50% overlap with #1
            {"id": "3", "file_path": "app/routes.py", "start_line": 1, "end_line": 20, "content": "def index(): pass\n"},
            {"id": "4", "file_path": "app/routes.py", "start_line": 1, "end_line": 20, "content": "def index(): pass\n"}, # Exact duplicate
        ]
        deduped = deduplicate_chunks(chunks)
        # Should drop chunk 2 (overlapping) and chunk 4 (duplicate)
        self.assertEqual(len(deduped), 2)
        self.assertEqual(deduped[0]["id"], "1")
        self.assertEqual(deduped[1]["id"], "3")

    def test_intelligent_truncation_of_large_source_file(self):
        """Scenario 4: Query involving a large source file."""
        lines = [f"    # line {i}: logic\n" for i in range(1, 201)]
        large_content = "def massive_module_executor():\n" + "".join(lines) + "    return True\n"
        chunk = {
            "file_path": "app/massive.py",
            "start_line": 1,
            "end_line": 202,
            "content": large_content,
        }
        max_chunk_tokens = 250
        truncated = truncate_chunk_content(chunk, max_chunk_tokens)
        self.assertTrue(truncated.get("truncated"))
        self.assertIn("lines omitted for context efficiency", truncated["content"])
        self.assertIn("def massive_module_executor", truncated["content"])
        self.assertIn("return True", truncated["content"])
        self.assertLessEqual(estimate_tokens(truncated["content"]), max_chunk_tokens + 50)

    def test_context_budget_enforcement_many_chunks(self):
        """Scenario 3 & 5: Query retrieving many chunks exceeding budget."""
        chunks = []
        for i in range(20):
            chunks.append({
                "file_path": f"app/module_{i % 5}.py",
                "start_line": i * 10,
                "end_line": (i + 1) * 10,
                "content": f"def func_{i}():\n    x = {i}\n    print(x)\n    return x\n" * 15,
            })
        
        packed = pack_context_for_rag(
            query="find functions",
            system_prompt="System instructions...",
            chunks=chunks,
            model_name="openai/gpt-oss-20b",
            budget_override=2000
        )
        
        # Verify prompt tokens strictly obey budget limit
        self.assertLessEqual(packed["estimated_prompt_tokens"], 2000)
        self.assertGreater(packed["selected_chunk_count"], 0)
        # Source diversity check: no file appears more than max_chunks_per_file (2)
        counts = {}
        for c in packed["selected_chunks"]:
            fp = c["file_path"]
            counts[fp] = counts.get(fp, 0) + 1
            self.assertLessEqual(counts[fp], 2)

    def test_insufficient_or_empty_chunks(self):
        """Scenario 7: Query where insufficient or empty code context is available."""
        # Empty chunk list
        res = generate_rag_answer("how is auth handled?", [])
        self.assertIn("No relevant code chunks were retrieved", res["answer"])
        self.assertEqual(res["cited_chunks"], [])

    def test_parse_groq_retry_delay_and_tpm_detection(self):
        """Scenario 6: Simulated Groq rate-limit detection and retry-after parsing."""
        # Mock error with retry message
        err_msg = "Rate limit reached for model 'openai/gpt-oss-20b' on tokens per minute (TPM): Limit 8000, Used 668, Requested 8668. Please try again in 4.5s."
        err = Exception(err_msg)
        self.assertTrue(is_tpm_rate_limit(err))
        delay = parse_groq_retry_delay(err)
        # Should be parsed ~4.5 + jitter (between 4.5 and 5.5)
        self.assertGreaterEqual(delay, 4.5)
        self.assertLessEqual(delay, 6.0)

    def test_adaptive_context_reduction_on_rate_limit(self):
        """Scenario 6: Test adaptive context reduction during rate limit loop."""
        from groq import RateLimitError

        dummy_chunks = [
            {"file_path": "app/auth.py", "start_line": 1, "end_line": 10, "content": "def auth(): pass\n" * 10},
            {"file_path": "app/db.py", "start_line": 1, "end_line": 10, "content": "def get_db(): pass\n" * 10},
        ]

        # Simulate first attempt failing with RateLimitError, second succeeding
        mock_completion = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = '{"answer": "Auth is implemented in app/auth.py", "cited_chunks": [{"file_path": "app/auth.py", "start_line": 1, "end_line": 10}]}'
        mock_completion.choices = [mock_choice]

        rate_limit_err = RateLimitError(
            message="Rate limit reached on tokens per minute (TPM). Please try again in 0.1s.",
            response=MagicMock(),
            body={"error": {"message": "Rate limit reached"}}
        )

        with patch("app.llm.rag_answer.execute_groq_chat_completion") as mock_exec, \
             patch("time.sleep") as mock_sleep:
            # First call raises RateLimitError, second succeeds
            mock_exec.side_effect = [rate_limit_err, (mock_completion, "openai/gpt-oss-20b")]

            res = generate_rag_answer("how does auth work?", dummy_chunks)
            
            self.assertEqual(mock_exec.call_count, 2)
            self.assertIn("Auth is implemented", res["answer"])
            self.assertEqual(len(res["cited_chunks"]), 1)
            self.assertEqual(res["observability"]["retry_count"], 1)


if __name__ == "__main__":
    unittest.main()

