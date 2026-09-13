"""
llm_client.py
-------------
Thin wrapper around the Groq chat-completions API (free tier friendly). If
GROQ_API_KEY is not set, falls back to a MOCK mode that returns a
clearly-labeled placeholder answer, so the rest of the pipeline (retrieval,
credibility, evaluation) can still be exercised end-to-end without live API
calls.

Model is chosen via the GROQ_MODEL env var (default: qwen/qwen3.8-27b;
see https://console.groq.com/docs/models for options). Every strategy in
strategies.py only depends on call_llm(prompt), so swapping providers means
replacing this module's internals only.

GROQ_API_KEY / GROQ_MODEL can also be put in a .env file at the repo root
(gitignored); it is loaded here at import time without overriding variables
already set in the environment.
"""
import os
import time


def _load_dotenv():
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"
    )
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()

MODEL = os.environ.get("GROQ_MODEL", "qwen/qwen3.8-27b")
# Cap chosen against two constraints: chain-of-thought answers need room for
# the final "Answer: Yes/No" line (700 truncated them), while Groq's free tier
# rejects any request whose max_tokens exceeds the per-minute output limit
# (1000 for the qwen3 models). Keep reasoning blocks disabled via
# GROQ_REASONING_EFFORT=none so answers fit comfortably.
MAX_TOKENS = 1000
# Free-tier rate limits are ~30 requests/min; keep a small gap between calls.
_MIN_SECONDS_BETWEEN_CALLS = 2.1
_last_call_time = 0.0
_MOCK_MODE = os.environ.get("GROQ_API_KEY") is None

if not _MOCK_MODE:
    from groq import Groq
    _client = Groq()

# For reasoning models (e.g. Qwen-3), GROQ_REASONING_EFFORT=none suppresses
# <think> blocks, which otherwise burn the token budget before the final
# "Answer:" line. Left unset, the parameter is not sent at all.
_EXTRA_PARAMS = {}
if os.environ.get("GROQ_REASONING_EFFORT"):
    _EXTRA_PARAMS["reasoning_effort"] = os.environ["GROQ_REASONING_EFFORT"]


def call_llm(prompt, model=MODEL, max_tokens=MAX_TOKENS, retries=4):
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
                **_EXTRA_PARAMS,
            )
            return resp.choices[0].message.content
        except Exception as e:  # noqa: BLE001
            last_err = e
            # Per-minute rate-limit windows need patient waits, not seconds.
            time.sleep(20 * (attempt + 1))
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
