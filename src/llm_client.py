"""
llm_client.py
-------------
Thin wrapper around the Groq chat-completions API (free tier friendly). If
GROQ_API_KEY is not set, falls back to a MOCK mode that returns a
clearly-labeled placeholder answer, so the rest of the pipeline (retrieval,
credibility, evaluation) can still be exercised end-to-end without live API
calls.

Model is chosen via the GROQ_MODEL env var (default: llama-3.1-8b-instant;
see https://console.groq.com/docs/models for options). Every strategy in
strategies.py only depends on call_llm(prompt), so swapping providers means
replacing this module's internals only.
"""
import os
import time

MODEL = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
MAX_TOKENS = 700
# Free-tier rate limits are ~30 requests/min; keep a small gap between calls.
_MIN_SECONDS_BETWEEN_CALLS = 2.1
_last_call_time = 0.0
_MOCK_MODE = os.environ.get("GROQ_API_KEY") is None

if not _MOCK_MODE:
    from groq import Groq
    _client = Groq()


def call_llm(prompt, model=MODEL, max_tokens=MAX_TOKENS, retries=3):
    if _MOCK_MODE:
        return _mock_response(prompt)

    _throttle()
    last_err = None
    for attempt in range(retries):
        try:
            resp = _client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.choices[0].message.content
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(2 ** attempt)
    raise RuntimeError(
        f"LLM call failed after {retries} attempts: {last_err}\n"
        f"Check that GROQ_MODEL={model!r} is a valid model id "
        f"(https://console.groq.com/docs/models) and that your free-tier "
        f"rate limit is not exhausted."
    )


def _throttle():
    """Space out requests to stay under Groq's free-tier rate limits."""
    global _last_call_time
    wait = _MIN_SECONDS_BETWEEN_CALLS - (time.monotonic() - _last_call_time)
    if wait > 0:
        time.sleep(wait)
    _last_call_time = time.monotonic()


def _mock_response(prompt):
    """Deterministic placeholder so the pipeline is testable without an API key.
    Alternates Yes/No based on prompt length parity -- NOT a real answer,
    only for wiring/debugging. Real experiments require GROQ_API_KEY."""
    answer = "Yes" if len(prompt) % 2 == 0 else "No"
    return (
        f"[MOCK MODE -- no GROQ_API_KEY set, this is not a real answer]\n"
        f"Reasoning: (mock) evaluated {len(prompt)} chars of context.\n"
        f"Answer: {answer}"
    )
