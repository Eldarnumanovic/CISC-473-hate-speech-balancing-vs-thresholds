import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)

parser = argparse.ArgumentParser()
parser.add_argument("run_directory")
args = parser.parse_args()
run_dir = Path(args.run_directory)

# Use the checkpoint already selected during training.
epoch = json.loads((run_dir / "best_epoch.json").read_text())["epoch"]
data = pd.read_csv(run_dir / f"validation_epoch{epoch}.csv")
labels = data["label"].to_numpy(dtype=int)
scores = data["hate_score"].to_numpy()

assert set(np.unique(labels)) == {0, 1}, "Expected both classes."
assert np.isfinite(scores).all(), "Scores contain invalid values."
assert ((scores >= 0) & (scores <= 1)).all(), "Scores must be probabilities."

def evaluate(threshold):
    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        labels, predictions, labels=[0, 1]
    ).ravel()
    return {
        "threshold": float(threshold),
        "macro_f1": float(f1_score(
            labels, predictions, average="macro", zero_division=0
        )),
        "hate_precision": float(precision_score(
            labels, predictions, zero_division=0
        )),
        "hate_recall": float(recall_score(
            labels, predictions, zero_division=0
        )),
        "false_positive_rate": float(fp / (fp + tn)),
        "false_negative_rate": float(fn / (fn + tp)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

rows = [evaluate(t) for t in np.linspace(0, 1, 1001)]
best_f1 = max(row["macro_f1"] for row in rows)
ties = [
    row for row in rows
    if abs(row["macro_f1"] - best_f1) <= 1e-12
]
tuned = min(ties, key=lambda row: (
    abs(row["threshold"] - 0.5), row["threshold"]
))
default = evaluate(0.5)
ap = float(average_precision_score(labels, scores))

metric_names = [
    "macro_f1", "hate_precision", "hate_recall",
    "false_positive_rate", "false_negative_rate",
]
deltas = {
    name: 100 * (tuned[name] - default[name])
    for name in metric_names
}

summary = {
    "split": "validation",
    "selected_epoch": epoch,
    "threshold_search": "0 to 1 inclusive, step 0.001",
    "tie_break": "closest to 0.5, then lower threshold",
    "default": default,
    "tuned": tuned,
    "delta_percentage_points": deltas,
    "average_precision": ap,
}
(run_dir / "threshold_summary.json").write_text(
    json.dumps(summary, indent=2)
)
pd.DataFrame(rows).to_csv(run_dir / "threshold_search.csv", index=False)

precision, recall, _ = precision_recall_curve(labels, scores)
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(recall, precision, label=f"BERT, epoch {epoch}; AP={ap:.3f}")
ax.axhline(
    labels.mean(), linestyle="--", color="gray",
    label="Hate-class prevalence"
)
for result, name, marker in [
    (default, "Default", "o"), (tuned, "Tuned", "s")
]:
    ax.scatter(
        result["hate_recall"], result["hate_precision"],
        s=70, marker=marker, zorder=3,
        label=f"{name}: threshold={result['threshold']:.3f}",
    )
ax.set(
    xlabel="Hate-class recall",
    ylabel="Hate-class precision",
    title="Validation precision–recall curve",
    xlim=(0, 1), ylim=(0, 1.02),
)
ax.legend()
ax.grid(alpha=0.2)
fig.tight_layout()
fig.savefig(run_dir / "validation_pr_curve.png", dpi=200)
plt.close(fig)

print(f"\nSelected epoch: {epoch}")
print(f"Default threshold: {default['threshold']:.3f}")
print(f"Tuned threshold:   {tuned['threshold']:.3f}\n")
print(f"{'Metric':<25} {'Default':>9} {'Tuned':>9} {'Delta (pp)':>12}")
for name in metric_names:
    print(
        f"{name:<25} {100*default[name]:>8.2f}% "
        f"{100*tuned[name]:>8.2f}% {deltas[name]:>+12.2f}"
    )
print(f"\nAverage precision: {ap:.4f} (unchanged by threshold)")
print(f"Default: FP={default['fp']}, FN={default['fn']}")
print(f"Tuned:   FP={tuned['fp']}, FN={tuned['fn']}")
print(f"\nSaved summary, threshold table, and PR plot in {run_dir}")
