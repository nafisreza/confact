"""
pipeline.py
-----------
Runs the full 3-stage RAG pipeline (Figure 2a/2d in the paper) for one claim,
across every strategy in strategies.STRATEGIES, and returns predictions.
"""
from retrieval import get_relevant_passages
from credibility import annotate_passages
from strategies import STRATEGIES, parse_answer

TOP_K = 6


def run_claim(claim, strategies=None, top_k=TOP_K):
    """Returns a dict: {strategy_name: {"raw": ..., "prediction": "Yes"/"No"/None}}"""
    strategies = strategies or list(STRATEGIES.keys())

    # Stage 1 (retrieval, given) + Stage 2 (ranking)
    passages = get_relevant_passages(claim, top_k=top_k)
    # Media Source Background Provider (GT-MB)
    passages = annotate_passages(passages)

    results = {}
    for name in strategies:
        fn = STRATEGIES[name]
        raw = fn(claim["question"], passages)
        results[name] = {"raw": raw, "prediction": parse_answer(raw)}
    return results, passages
