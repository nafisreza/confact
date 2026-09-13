"""
run_experiment.py
------------------
Main entry point. Runs every strategy (DirA, CoT, SF, SBA_dir, SBA_CoT) on
every claim in data/confact_subset.json, and writes per-claim predictions to
results/results.csv.

Usage:
    export GROQ_API_KEY=gsk_...
    python run_experiment.py
    python src/evaluate.py --csv results/results.csv

If a run dies partway (e.g. free-tier daily token cap), rerun with
--resume to keep existing rows and continue from the first unfinished claim.
"""
import argparse
import json
import os
import sys
import csv
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from pipeline import run_claim  # noqa: E402
from strategies import STRATEGIES  # noqa: E402

DATA_PATH = "data/confact_subset.json"
OUT_PATH = "results/results.csv"
RAW_OUT_PATH = "results/raw_responses.jsonl"


def _completed_claim_ids():
    """Claim ids that already have a row for every strategy in results.csv."""
    if not os.path.exists(OUT_PATH):
        return set()
    counts = {}
    with open(OUT_PATH, newline="") as f:
        for row in csv.DictReader(f):
            counts[row["claim_id"]] = counts.get(row["claim_id"], 0) + 1
    return {cid for cid, n in counts.items() if n >= len(STRATEGIES)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true",
                        help="keep existing results and continue from the "
                             "first claim without a full set of predictions")
    args = parser.parse_args()

    with open(DATA_PATH) as f:
        claims = json.load(f)

    if os.environ.get("GROQ_API_KEY") is None:
        print("WARNING: GROQ_API_KEY is not set -- running in MOCK mode. "
              "Predictions will be placeholders, not real experiment results. "
              "Set the key to get real results.\n")

    done = _completed_claim_ids() if args.resume else set()
    if done:
        claims = [c for c in claims if str(c["id"]) not in done]
        print(f"Resuming: {len(done)} claims already complete, "
              f"{len(claims)} remaining.")
    if not claims:
        print("Nothing to do -- all claims already have predictions.")
        return

    mode = "a" if done else "w"
    fieldnames = ["claim_id", "claim", "gold_answer", "strategy", "prediction", "correct"]
    with open(OUT_PATH, mode, newline="") as out_f, open(RAW_OUT_PATH, mode) as raw_f:
        writer = csv.DictWriter(out_f, fieldnames=fieldnames)
        if not done:
            writer.writeheader()

        for claim in tqdm(claims, desc="Claims"):
            results, passages = run_claim(claim, strategies=list(STRATEGIES.keys()))
            for strategy_name, r in results.items():
                pred = r["prediction"]
                writer.writerow({
                    "claim_id": claim["id"],
                    "claim": claim["claim"],
                    "gold_answer": claim["answer"],
                    "strategy": strategy_name,
                    "prediction": pred,
                    "correct": (pred == claim["answer"]) if pred else False,
                })
                raw_f.write(json.dumps({
                    "claim_id": claim["id"],
                    "strategy": strategy_name,
                    "raw_response": r["raw"],
                    "passages_used": [
                        {"domain": p["domain"], "credibility": p["background"]["credibility_label"]}
                        for p in passages
                    ],
                }) + "\n")

    print(f"\nDone. Wrote {OUT_PATH} and {RAW_OUT_PATH}")


if __name__ == "__main__":
    main()
