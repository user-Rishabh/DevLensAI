import os
import time
import re
import random
from threading import Lock
from typing import Any, List, Dict, Tuple, Optional
from groq import Groq, RateLimitError, APIConnectionError, APIStatusError

# Cached working model discovered during runtime
_RESOLVED_GROQ_MODEL: str | None = None

# Concurrency guard: prevents multiple simultaneous requests from blowing TPM limits
_GROQ_REQUEST_LOCK = Lock()

# Verified active Groq models in order of reasoning capability & availability
DEFAULT_FALLBACK_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.8-27b",
]

def get_preferred_groq_model() -> str:
    """
    Returns the preferred Groq model name, respecting GROQ_MODEL environment variable.
    """
    global _RESOLVED_GROQ_MODEL
    if _RESOLVED_GROQ_MODEL:
        return _RESOLVED_GROQ_MODEL
    env_model = os.getenv("GROQ_MODEL")
    if env_model and env_model.strip():
        return env_model.strip()
    return "llama-3.3-70b-versatile"

def parse_groq_retry_delay(error: Exception, default_delay: float = 2.5) -> float:
    """
    Extracts recommended retry-after delay from Groq HTTP headers or error message.
    Adds jitter to prevent synchronized retry spikes.
    """
    delay = default_delay
    
    # 1. Attempt extracting from response headers
    if hasattr(error, "response") and error.response is not None:
        headers = getattr(error.response, "headers", {}) or {}
        retry_header = headers.get("retry-after")
        if retry_header:
            try:
                delay = float(retry_header)
            except (ValueError, TypeError):
                pass

    # 2. Attempt extracting from error message string (e.g., "Please try again in 5.2s")
    err_str = str(error)
    match = re.search(r"try again in (\d+(\.\d+)?)s", err_str, re.IGNORECASE)
    if match:
        try:
            delay = float(match.group(1))
        except (ValueError, TypeError):
            pass

    # Add 200ms - 600ms random jitter
    jitter = random.uniform(0.2, 0.6)
    return max(1.0, delay + jitter)

def is_tpm_rate_limit(error: Exception) -> bool:
    """
    Checks if an error specifically indicates a Tokens-Per-Minute (TPM) limit exhaustion.
    """
    err_str = str(error).lower()
    return (
        "tokens per minute" in err_str
        or "(tpm)" in err_str
        or "requested tokens" in err_str
        or "tpm limit" in err_str
    )

def execute_groq_chat_completion(
    client: Groq,
    messages: List[Dict[str, str]],
    model_override: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 800,
    response_format: Any = None,
    max_retries: int = 2,
    backoffs: List[float] | None = None,
) -> Tuple[Any, str]:
    """
    Executes a Groq chat completion with automatic model fallback for deprecated, decommissioned,
    or plan-restricted models, and exponential backoff with jitter for transient rate limits.
    Returns a tuple of (chat_completion, resolved_model_name).
    """
    global _RESOLVED_GROQ_MODEL
    if backoffs is None:
        backoffs = [2.0, 4.0]

    primary_model = model_override or get_preferred_groq_model()
    candidate_models = [primary_model]
    for m in DEFAULT_FALLBACK_MODELS:
        if m not in candidate_models:
            candidate_models.append(m)

    last_error = None
    for model_name in candidate_models:
        for attempt in range(max_retries + 1):
            try:
                kwargs: Dict[str, Any] = {
                    "messages": messages,
                    "model": model_name,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                if response_format is not None:
                    kwargs["response_format"] = response_format

                # Concurrency lock to prevent multiple rapid requests from exhausting TPM in parallel
                with _GROQ_REQUEST_LOCK:
                    chat_completion = client.chat.completions.create(**kwargs)

                _RESOLVED_GROQ_MODEL = model_name
                return chat_completion, model_name

            except RateLimitError as rle:
                last_error = rle
                # If TPM limit exceeded, fail fast out of this model's retry loop so caller
                # can dynamically reduce prompt context size before retrying.
                if is_tpm_rate_limit(rle):
                    print(f"[DevLens AI] Groq TPM limit exceeded on {model_name}: {rle}. Passing to caller for context reduction.")
                    raise rle

                if attempt < max_retries:
                    sleep_time = parse_groq_retry_delay(rle, default_delay=backoffs[attempt] if attempt < len(backoffs) else 3.0)
                    print(f"[DevLens AI] Groq rate limited on model {model_name}. Retrying in {sleep_time:.2f}s... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(sleep_time)
                else:
                    break

            except Exception as e:
                err_str = str(e).lower()
                # If model is not found, deprecated, decommissioned, or plan-restricted, fail fast to next candidate model
                is_model_unavailable = any(
                    token in err_str for token in [
                        "model_not_found",
                        "model_decommissioned",
                        "decommissioned",
                        "does not exist",
                        "not have access",
                        "no longer supported",
                        "unavailable",
                        "404",
                    ]
                )
                if is_model_unavailable:
                    print(f"[DevLens AI Warning] Groq model '{model_name}' is unavailable ({e}). Trying fallback model...")
                    last_error = e
                    break
                else:
                    last_error = e
                    if attempt < max_retries:
                        time.sleep(1.0)
                    else:
                        break

    if last_error:
        raise last_error
    raise RuntimeError("All candidate Groq models failed to complete request.")
