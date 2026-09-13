"""
run_experiment.py
------------------
Main entry point. Runs every strategy (DirA, CoT, SF, SBA_dir, SBA_CoT) on
every claim in data/confact_subset.json, and writes per-claim predictions to
results/results.csv.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python run_experiment.py
    python src/evaluate.py --csv results/results.csv
"""
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


def main():
    with open(DATA_PATH) as f:
        claims = json.load(f)

    if os.environ.get("ANTHROPIC_API_KEY") is None:
        print("WARNING: ANTHROPIC_API_KEY is not set -- running in MOCK mode. "
              "Predictions will be placeholders, not real experiment results. "
              "Set the key to get real results.\n")

    fieldnames = ["claim_id", "claim", "gold_answer", "strategy", "prediction", "correct"]
    with open(OUT_PATH, "w", newline="") as out_f, open(RAW_OUT_PATH, "w") as raw_f:
        writer = csv.DictWriter(out_f, fieldnames=fieldnames)
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
