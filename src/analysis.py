"""
analysis.py
-----------
Produces:
  1. results/strategy_comparison.png -- bar chart of Accuracy & Macro-F1 per strategy
  2. results/error_examples.md -- a few claims where baseline (DirA) got it wrong
     but a source-aware strategy (SBA_CoT) got it right, and vice versa --
     useful for the "Results" and "Challenges & Limitations" slides.
"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from evaluate import evaluate

RESULTS_CSV = "/home/claude/confact_project/results/results.csv"
SUMMARY_PNG = "/home/claude/confact_project/results/strategy_comparison.png"
ERROR_MD = "/home/claude/confact_project/results/error_examples.md"


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
    lines = ["# Qualitative Error Examples\n"]

    improved = pivot[(pivot[baseline] == False) & (pivot[best] == True)].index.tolist()  # noqa: E712
    lines.append(f"## Cases where {best} fixed a {baseline} mistake ({len(improved)} found)\n")
    for cid in improved[:n]:
        claim_text = df[df["claim_id"] == cid]["claim"].iloc[0]
        gold = df[df["claim_id"] == cid]["gold_answer"].iloc[0]
        lines.append(f"- Claim #{cid}: \"{claim_text}\" (gold: {gold})")

    regressed = pivot[(pivot[baseline] == True) & (pivot[best] == False)].index.tolist()  # noqa: E712
    lines.append(f"\n## Cases where {best} introduced an error {baseline} didn't have "
                  f"({len(regressed)} found)\n")
    for cid in regressed[:n]:
        claim_text = df[df["claim_id"] == cid]["claim"].iloc[0]
        gold = df[df["claim_id"] == cid]["gold_answer"].iloc[0]
        lines.append(f"- Claim #{cid}: \"{claim_text}\" (gold: {gold})")

    with open(ERROR_MD, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved {ERROR_MD}")


if __name__ == "__main__":
    df = pd.read_csv(RESULTS_CSV)
    summary = evaluate(RESULTS_CSV)
    plot_comparison(summary)
    find_error_examples(df)
