# CLAUDE.md

Context file for Claude Code working in this repo.

## What this project is

A course project implementing a scoped-down version of:

> Ge, Wu, Chin, Lee & Cao, **"Resolving Conflicting Evidence in Automated
> Fact-Checking: A Study on Retrieval-Augmented LLMs"** (IJCAI-25).
> Original paper repo: https://github.com/zoeyyes/CONFACT

The paper studies RAG-based fact-checking when retrieved evidence documents
conflict, and shows that injecting **source credibility** information (from
Media Bias/Fact Check, MBFC) helps LLMs resolve those conflicts — especially
when the credibility info is added at answer-generation time combined with
chain-of-thought reasoning.

This project reproduces that core finding at small scale: 30 real claims
from the paper's actual CONFACT dataset, 1 LLM (Qwen-3 27B via the Groq
free-tier API), 5 answering strategies (2 baselines, 3 source-aware). Full scope/limitations
are documented in `README.md` — read that first for the "why" behind design
decisions; this file is about the "what's where" for making code changes.

## Architecture (data flows top to bottom)

```
data/confact_subset.json  (30 claims: claim text, question, gold label, evidence docs)
data/mbfc_lookup.json     (MBFC credibility rating per source domain)
        │
        ▼
src/retrieval.py     -- chunk_paragraphs(), rank_passages(): TF-IDF cosine
        │                similarity of question vs. paragraph chunks -> top-K
        ▼
src/credibility.py   -- get_background(domain): MBFC label -> credibility
        │                score in (0,1); filter_low_credibility() for SF strategy
        ▼
src/strategies.py    -- 5 prompt-building functions in STRATEGIES dict:
        │                DirA, CoT (no credibility info)
        │                SF, SBA_dir, SBA_CoT (credibility-aware)
        │                each calls llm_client.call_llm() and returns raw text
        ▼
src/llm_client.py    -- call_llm(prompt): wraps the Groq chat-completions
        │                API (with free-tier rate-limit throttling). Falls
        │                back to MOCK MODE (deterministic placeholder
        │                answers) if GROQ_API_KEY is unset -- this lets
        │                the whole pipeline be tested without API access.
        ▼
src/pipeline.py       -- run_claim(claim): runs retrieval -> credibility
        │                annotation -> all 5 strategies for one claim
        ▼
run_experiment.py      -- loops over all 30 claims, writes results/results.csv
        │                (per-claim, per-strategy predictions) and
        │                results/raw_responses.jsonl (full LLM outputs)
        ▼
src/evaluate.py        -- accuracy + macro-F1 per strategy from results.csv
src/analysis.py         -- bar chart (results/strategy_comparison.png) +
                           qualitative error examples (results/error_examples.md)
src/build_demo.py       -- builds results/demo.html: a self-contained,
                           offline HTML page embedding claims, passages,
                           credibility badges, and every strategy's
                           prediction + reasoning, for live presentations.
                           Rerun this after any new run_experiment.py run --
                           it does not auto-refresh.
```

`src/build_dataset.py` is a one-time extraction script (not part of the
runtime pipeline) that pulled `data/confact_subset.json` and
`data/mbfc_lookup.json` from the original CONFACT release
(`HumC.pkl.gz`, `ModC.pkl.gz`, `mbfc_media_data.pkl`). Those large raw files
are NOT in this repo (only the small extracted subset is) — don't re-run
`build_dataset.py` unless the raw CONFACT release is present locally, and
note it uses a fixed `SEED` so re-running should reproduce the same subset.

## Key conventions

- Every strategy function in `strategies.py` has the signature
  `(question, passages) -> raw_response_text` and ends its prompt asking the
  model for a final `Answer: Yes` / `Answer: No` line, parsed by
  `parse_answer()`. If you add a new strategy, follow this exact contract and
  register it in the `STRATEGIES` dict — `pipeline.py` and
  `run_experiment.py` iterate over that dict generically, so nothing else
  needs to change.
- Gold labels: `Supported` → `"Yes"`, `Refuted` → `"No"` (see
  `trim_claim()` in `build_dataset.py` and the `answer` field in
  `confact_subset.json`).
- `llm_client.py`'s mock mode is intentional scaffolding, not a bug — it lets
  the pipeline run and be tested without `GROQ_API_KEY` set. Don't
  remove it; if you change the mock's behavior, keep it clearly labeled as
  mock output in both the returned text and any docs.
- Model selection lives in one place: `llm_client.MODEL`, read from the
  `GROQ_MODEL` env var (default `qwen/qwen3.8-27b`). Swapping models (or
  providers) should only require editing this file. `llm_client` also loads
  a repo-root `.env` file at import time (without overriding real env vars).
- Credibility scores map MBFC's categorical labels (`high`, `mostly
  factual`, `mixed`, `low`, `very low`, `unknown`) to numbers in
  `credibility.CREDIBILITY_SCORE_MAP` — this is the paper's "GT-MB" setting
  (ground-truth MBFC ratings), not the paper's trained Hybrid-MB predictor,
  which this project does not implement.

## Running it

```bash
pip install -r requirements.txt
export GROQ_API_KEY=gsk_...   # or put it in .env; omit entirely to run in mock mode
python run_experiment.py
python src/evaluate.py --csv results/results.csv
PYTHONPATH=src python src/analysis.py
PYTHONPATH=src python src/build_demo.py   # then open results/demo.html
```

## Things to be careful about

- `results/results.csv`, `results/raw_responses.jsonl`,
  `results/strategy_comparison.png`, `results/error_examples.md`, and
  `results/demo.html` are all **generated output**, currently reflecting a
  mock-mode run (placeholder, not real answers) — `demo.html` even shows an
  in-page warning banner for this. Don't treat their current contents as
  real experimental findings when writing docs/slides — they need to be
  regenerated with a real `GROQ_API_KEY` first (`run_experiment.py` →
  `evaluate.py` → `analysis.py` → `build_demo.py`, in that order).
- This is a small (30-claim), single-model project by design — see
  `README.md` §2 "What we implemented (and why we scoped it down)" before
  suggesting scale-up changes; that section explains the deliberate
  trade-offs against the full paper.
- Don't hardcode an API key anywhere, don't commit `.env` files (already in
  `.gitignore`).
