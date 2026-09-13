"""
evaluate.py
-----------
Computes Accuracy and Macro-F1 per strategy (the paper's two metrics,
Section 4.1), from the predictions CSV produced by run_experiment.py.
"""
import argparse
import os

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


def evaluate(csv_path):
    df = pd.read_csv(csv_path)
    strategies = sorted(df["strategy"].unique())
    rows = []
    for s in strategies:
        sub = df[df["strategy"] == s].copy()
        # unparseable predictions count as wrong (matches the paper's treatment
        # of failure to answer as an incorrect prediction)
        sub["pred_norm"] = sub["prediction"].fillna("Unparsed")
        y_true = sub["gold_answer"]
        y_pred = sub["pred_norm"]
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro", labels=["Yes", "No"], zero_division=0)
        n_unparsed = (sub["prediction"].isna()).sum()
        rows.append({
            "strategy": s,
            "n": len(sub),
            "accuracy": round(acc, 4),
            "macro_f1": round(f1, 4),
            "unparsed": n_unparsed,
        })
    return pd.DataFrame(rows).sort_values("macro_f1", ascending=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    _default_csv = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", "results.csv",
    )
    parser.add_argument("--csv", default=_default_csv)
    args = parser.parse_args()
    summary = evaluate(args.csv)
    print(summary.to_string(index=False))
    summary.to_csv(args.csv.replace(".csv", "_summary.csv"), index=False)
