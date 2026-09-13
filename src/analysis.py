"""
analysis.py
-----------
Produces:
  1. results/strategy_comparison.png -- bar chart of Accuracy & Macro-F1 per strategy
  2. results/error_examples.md -- a few claims where baseline (DirA) got it wrong
     but a source-aware strategy (SBA_CoT) got it right, and vice versa --
     useful for the "Results" and "Challenges & Limitations" slides.
"""
import os

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from evaluate import evaluate

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_CSV = os.path.join(_ROOT, "results", "results.csv")
SUMMARY_PNG = os.path.join(_ROOT, "results", "strategy_comparison.png")
ERROR_MD = os.path.join(_ROOT, "results", "error_examples.md")


def plot_comparison(summary_df):
    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(summary_df))
    width = 0.35
    ax.bar([i - width / 2 for i in x], summary_df["accuracy"], width, label="Accuracy")
    ax.bar([i + width / 2 for i in x], summary_df["macro_f1"], width, label="Macro-F1")
    ax.set_xticks(list(x))
    ax.set_xticklabels(summary_df["strategy"], rotation=20)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Fact-Checking Performance by Strategy (CONFACT subset)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(SUMMARY_PNG, dpi=150)
    print(f"Saved {SUMMARY_PNG}")


def find_error_examples(df, baseline="DirA", best="SBA_CoT", n=3):
    pivot = df.pivot(index="claim_id", columns="strategy", values="correct")
    # One claim_id -> (claim, gold) table, instead of re-scanning the whole
    # frame twice per example below.
    by_claim = df.drop_duplicates("claim_id").set_index("claim_id")
    lines = ["# Qualitative Error Examples\n"]

    improved = pivot[(pivot[baseline] == False) & (pivot[best] == True)].index.tolist()  # noqa: E712
    lines.append(f"## Cases where {best} fixed a {baseline} mistake ({len(improved)} found)\n")
    for cid in improved[:n]:
        row = by_claim.loc[cid]
        lines.append(f"- Claim #{cid}: \"{row['claim']}\" (gold: {row['gold_answer']})")

    regressed = pivot[(pivot[baseline] == True) & (pivot[best] == False)].index.tolist()  # noqa: E712
    lines.append(f"\n## Cases where {best} introduced an error {baseline} didn't have "
                  f"({len(regressed)} found)\n")
    for cid in regressed[:n]:
        row = by_claim.loc[cid]
        lines.append(f"- Claim #{cid}: \"{row['claim']}\" (gold: {row['gold_answer']})")

    with open(ERROR_MD, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved {ERROR_MD}")


if __name__ == "__main__":
    df = pd.read_csv(RESULTS_CSV)
    summary = evaluate(RESULTS_CSV, df=df)  # reuse the frame, don't re-read
    plot_comparison(summary)
    find_error_examples(df)
