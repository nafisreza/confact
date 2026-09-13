"""
llm_client.py
-------------
Thin wrapper around the Anthropic Messages API. If ANTHROPIC_API_KEY is not
set, falls back to a MOCK mode that returns a clearly-labeled placeholder
answer, so the rest of the pipeline (retrieval, credibility, evaluation) can
still be exercised end-to-end without live API calls or cost.

Swap MODEL below (or pass model= to call_llm) to use a different Claude
model, or replace this module's internals to call OpenAI/local models instead
-- every strategy in strategies.py only depends on call_llm(prompt).
"""
import os
import time

MODEL = "claude-sonnet-4-5"
MAX_TOKENS = 700
_MOCK_MODE = os.environ.get("ANTHROPIC_API_KEY") is None

if not _MOCK_MODE:
    import anthropic
    _client = anthropic.Anthropic()


def call_llm(prompt, model=MODEL, max_tokens=MAX_TOKENS, retries=3):
    if _MOCK_MODE:
        return _mock_response(prompt)

    last_err = None
    for attempt in range(retries):
        try:
            resp = _client.messages.create(
                model=model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"LLM call failed after {retries} attempts: {last_err}")


def _mock_response(prompt):
    """Deterministic placeholder so the pipeline is testable without an API key.
    Alternates Yes/No based on prompt length parity -- NOT a real answer,
    only for wiring/debugging. Real experiments require ANTHROPIC_API_KEY."""
    answer = "Yes" if len(prompt) % 2 == 0 else "No"
    return (
        f"[MOCK MODE -- no ANTHROPIC_API_KEY set, this is not a real answer]\n"
        f"Reasoning: (mock) evaluated {len(prompt)} chars of context.\n"
        f"Answer: {answer}"
    )
