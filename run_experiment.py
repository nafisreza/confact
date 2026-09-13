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
--resume: it keeps every (claim, strategy) row that produced a parseable
prediction and re-runs only missing or unparsed ones.
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


def _load_good_rows():
    """Existing (claim_id, strategy) pairs whose prediction parsed, plus the
    matching csv rows and raw jsonl entries, so a resumed run can keep them
    and redo everything else (missing claims AND unparsed/truncated answers)."""
    good_rows, good_raw = [], []
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH, newline="") as f:
            for row in csv.DictReader(f):
                if row["prediction"]:
                    good_rows.append(row)
    good_keys = {(r["claim_id"], r["strategy"]) for r in good_rows}
    if os.path.exists(RAW_OUT_PATH):
        with open(RAW_OUT_PATH) as f:
            for line in f:
                d = json.loads(line)
                if (str(d["claim_id"]), d["strategy"]) in good_keys:
                    good_raw.append(line)
    return good_rows, good_raw, good_keys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true",
                        help="keep (claim, strategy) rows that already have a "
                             "parseable prediction; re-run only the rest")
    args = parser.parse_args()

    with open(DATA_PATH) as f:
        claims = json.load(f)

    if os.environ.get("GROQ_API_KEY") is None:
        print("WARNING: GROQ_API_KEY is not set -- running in MOCK mode. "
              "Predictions will be placeholders, not real experiment results. "
              "Set the key to get real results.\n")

    good_rows, good_raw, good_keys = _load_good_rows() if args.resume else ([], [], set())

    todo = [(claim, s) for claim in claims for s in STRATEGIES
            if (str(claim["id"]), s) not in good_keys]
    if good_keys:
        print(f"Resuming: keeping {len(good_keys)} good predictions, "
              f"re-running {len(todo)} (claim, strategy) pairs.")
    if not todo:
        print("Nothing to do -- all claims already have parseable predictions.")
        return

    fieldnames = ["claim_id", "claim", "gold_answer", "strategy", "prediction", "correct"]
    with open(OUT_PATH, "w", newline="") as out_f, open(RAW_OUT_PATH, "w") as raw_f:
        writer = csv.DictWriter(out_f, fieldnames=fieldnames)
        writer.writeheader()
        for row in good_rows:
            writer.writerow(row)
        for line in good_raw:
            raw_f.write(line)
        out_f.flush()
        raw_f.flush()

        # One (claim, strategy) pair per iteration, flushed immediately, so a
        # mid-run crash (rate limit) never loses completed API calls.
        for claim, strategy_name in tqdm(todo, desc="Calls"):
            results, passages = run_claim(claim, strategies=[strategy_name])
            r = results[strategy_name]
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
            out_f.flush()
            raw_f.flush()

    print(f"\nDone. Wrote {OUT_PATH} and {RAW_OUT_PATH}")


if __name__ == "__main__":
    main()
