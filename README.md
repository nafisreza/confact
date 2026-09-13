# Source-Aware Fact-Checking with Conflicting Evidence

A course project implementation based on:

> Ge, Wu, Chin, Lee & Cao. **"Resolving Conflicting Evidence in Automated
> Fact-Checking: A Study on Retrieval-Augmented LLMs."** IJCAI-25, Special
> Track on AI and Social Good.
> Paper's own repo/dataset: https://github.com/zoeyyes/CONFACT

## 1. What the paper does

The paper studies a specific failure mode of Retrieval-Augmented Generation
(RAG) for fact-checking: when retrieved documents **disagree** with each
other, and the disagreement traces back to **source credibility** (e.g. a
BBC article vs. a source flagged untrustworthy by Media Bias/Fact Check).
They:

1. Build **CONFACT**, a dataset of claims paired with conflicting evidence
   documents and MBFC credibility ratings, split into **ModC** (conflicts a
   GPT-4 judge flags) and **HumC** (conflicts human annotators flag).
2. Show vanilla RAG baselines (Direct Answer, Majority Vote, Discern-and-
   Answer, Chain-of-Thought) struggle on these conflicts.
3. Propose 3 ways to inject media-source-credibility information into the
   RAG pipeline: **Source Filtering** (drop low-credibility docs before
   ranking), **Credibility-Weighted Ranking**, and **Source Background
   Augmentation** (attach credibility info to passages at the answer stage).
4. Find that credibility info injected **at the answer-generation stage**,
   combined with **Chain-of-Thought reasoning**, works best — and that
   filtering/ranking-stage approaches can backfire by discarding needed
   counter-evidence.

## 2. What we implemented (and why we scoped it down)

The full paper evaluates 3 open-source LLMs × ~12 method variants × 2 dataset
splits (3,180 claims total), plus a human evaluation study and a trained
credibility-prediction model. That's a multi-author research project, not a
one-course-project undertaking. We kept the **research question and pipeline
architecture identical** but scoped the *scale*:

| Paper | This project |
|---|---|
| 3 LLMs (LLaMA-3.1, Qwen-2, Mistral) | 1 LLM (Qwen-3 27B via Groq's free API — successor of the paper's Qwen-2, swappable via `GROQ_MODEL`) |
| ModC (611) + HumC (287) claims | 30-claim curated subset of the **real** CONFACT data |
| 7 baseline + 11 source-aware method variants | 2 baselines (DirA, CoT) + 3 source-aware (SF, SBA_dir, SBA_CoT) |
| Trained credibility-score predictor (Hybrid-MB) | Direct MBFC rating → score mapping (GT-MB only) |
| LLM-as-judge stance/credibility annotation pipeline | Reused paper's existing annotations (skipped re-annotating) |

We use the **actual CONFACT dataset** (real claims, real scraped evidence
documents, real MBFC credibility ratings) rather than synthetic data — this
is a genuine subset of a published benchmark, not a toy example.

### Pipeline (mirrors the paper's Figure 2)

```
claim + question
      │
      ▼
[Retrieval]   (given: CONFACT's pre-scraped evidence docs, reproducibility per paper)
      │
      ▼
[Ranking]     chunk into paragraphs → TF-IDF cosine similarity to question → top-K
      │
      ▼
[Media Source Background Provider]   MBFC lookup → credibility label + score
      │
      ▼
[Answer Generation]   one of 5 strategies:
   - DirA      : baseline, no credibility info, direct answer
   - CoT       : baseline, no credibility info, chain-of-thought
   - SF        : source filtering (drop low-credibility passages) + direct answer
   - SBA_dir   : passages tagged with credibility background + direct answer
   - SBA_CoT   : passages tagged with credibility background + chain-of-thought
      │
      ▼
Yes/No prediction → compare to gold label (Supported→Yes, Refuted→No)
```

## 3. Repo structure

```
confact_project/
├── data/
│   ├── confact_subset.json   # 30 curated real claims + evidence (built by build_dataset.py)
│   └── mbfc_lookup.json      # MBFC credibility ratings for domains referenced above
├── src/
│   ├── build_dataset.py      # extracts the subset from the original CONFACT release
│   ├── retrieval.py           # chunking + TF-IDF ranking (Stage 1-2)
│   ├── credibility.py         # MBFC-backed source background provider (GT-MB)
│   ├── strategies.py          # prompt templates for DirA / CoT / SF / SBA_dir / SBA_CoT
│   ├── llm_client.py          # Groq API wrapper (+ mock mode w/o API key)
│   ├── pipeline.py            # orchestrates one claim end-to-end
│   ├── evaluate.py            # Accuracy + Macro-F1 per strategy
│   ├── analysis.py            # bar chart + qualitative error examples
│   └── build_demo.py          # builds results/demo.html, a self-contained demo page
├── results/
│   ├── results.csv             # per-claim, per-strategy predictions
│   ├── results_summary.csv     # accuracy/F1 per strategy
│   ├── strategy_comparison.png # bar chart
│   ├── error_examples.md       # qualitative cases
│   ├── raw_responses.jsonl     # full LLM responses for inspection
│   └── demo.html               # interactive offline demo -- open in any browser
├── run_experiment.py
├── requirements.txt
└── README.md
```

## 4. How to run

```bash
pip install -r requirements.txt

# Get a free key (no card) at console.groq.com, then either export it:
export GROQ_API_KEY=gsk_...
# ...or put `GROQ_API_KEY=gsk_...` in a .env file at the repo root (gitignored)

# (Optional) rebuild the dataset subset from scratch — already included in data/
python src/build_dataset.py

# Run all 5 strategies on all 30 claims
python run_experiment.py

# Compute accuracy / macro-F1 per strategy
python src/evaluate.py --csv results/results.csv

# Generate chart + qualitative error examples for the report/slides
PYTHONPATH=src python src/analysis.py

# Generate the interactive HTML demo for presenting (open results/demo.html in any browser)
PYTHONPATH=src python src/build_demo.py
```

If `GROQ_API_KEY` is not set, everything still runs in **mock mode** so
you can verify the pipeline works, but predictions are placeholders, not real
results — you need a real key for actual experiment numbers.

Cost: 30 claims × 5 strategies = 150 API calls — comfortably within Groq's
free tier (the client throttles itself to stay under the free-tier rate
limit, so a full run takes a few minutes). The model defaults to
`qwen/qwen3.8-27b`; set `GROQ_MODEL` to any model id from
https://console.groq.com/docs/models to try another.

### Actual run configuration behind `results/`

The committed results were produced with `GROQ_MODEL=qwen/qwen3.6-27b`
and `GROQ_REASONING_EFFORT=none` (visible `<think>` blocks otherwise consume
the output-token budget before the final `Answer:` line). Groq's free tier
caps both output tokens per minute (1,000 — which also bounds `max_tokens`
per request) and tokens per day (200,000 — less than a full 150-call run),
so the full 30-claim run was completed across two days via
`python run_experiment.py --resume`, which re-runs exactly the missing and
unparsed (claim, strategy) pairs once quota is available. A few long
source-aware responses (3/30 for SBA_CoT) still truncate at the per-request
token cap and score as unparsed/incorrect — this systematically penalizes
the CoT-style strategies and is flagged in the limitations.

## 5. Demoing the project

`results/demo.html` is a self-contained, offline page for presenting —
no server, no internet needed once generated. It shows:
- a sidebar listing all 30 claims, filterable to just the ones where the
  source-aware strategy (SBA_CoT) fixed or broke a baseline (DirA) answer
- for the selected claim: the retrieved passages with their MBFC credibility
  badges, and a side-by-side card per strategy showing its Yes/No prediction,
  whether it matched the gold label, and its full reasoning text (click
  "show reasoning" to expand)
- an overall accuracy summary bar at the top of the sidebar

It shows a orange warning banner if the embedded data is still from a mock
run — regenerate it (`build_demo.py`) after a real `run_experiment.py` run
before presenting. Just double-click `results/demo.html` to open it in any
browser.

## 5. Challenges & Limitations (for the report)

- **Scale**: 30 claims is enough to see directional patterns, not to draw
  statistically robust conclusions the way the paper's 611/287-claim splits can.
- **Single LLM**: we can't test whether findings vary by model capacity/context
  length the way the paper does across LLaMA/Qwen/Mistral.
- **Ranking**: TF-IDF cosine similarity is a much simpler ranker than the
  paper's trained reranking model — it may retrieve less relevant passages
  in some cases, which could disadvantage credibility-aware strategies that
  depend on having the right conflicting evidence in view.
- **GT-MB only**: we use MBFC ratings directly rather than reproducing the
  paper's Hybrid-MB LLM-based credibility estimator for sources missing from
  MBFC (Section 4.3) — so we can't compare GT-MB vs. Hybrid-MB as the paper does.
- **Answer parsing**: the model occasionally hedges instead of giving a clean
  Yes/No; unparseable answers are scored as incorrect (matches the paper's
  treatment of ambiguous outputs), but this is worth flagging as a limitation.

## 6. Extending this project

- Swap in `SBA_ens` (paper's ensemble strategy: per-passage rationale, then a
  second LLM call to reconcile) — the scaffolding in `strategies.py` makes
  this a straightforward addition.
- Add `CW_soft`/`CW_hard` (credibility-weighted ranking) by modifying
  `retrieval.rank_passages` to blend relevance and credibility score.
- Run on the full ModC/HumC splits (`data/dataset/*.pkl.gz` in the original
  repo) for more statistically meaningful results.
