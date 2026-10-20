"""Regenerate benchmark evidence and figures from shipped per-image predictions."""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cnn_v2.metrics import classification_metrics  # noqa: E402

RUN = ROOT / "artifacts/v2"


def read_predictions(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main(check=False):
    metadata = json.loads((RUN / "metadata.json").read_text())
    audit = json.loads((RUN / "audit.json").read_text())
    manifest_bytes = (RUN / "manifest.json").read_bytes()
    assert hashlib.sha256(manifest_bytes).hexdigest() == audit["manifest_sha256"]
    assert metadata["dataset"] == audit
    manifest = json.loads(manifest_bytes)
    assert len({row["id"] for row in manifest}) == len(manifest)
    assert len({row["sha256"] for row in manifest}) == len(manifest)
    for split, counts in audit["counts"].items():
        actual = Counter(
            metadata["class_names"][row["label"]]
            for row in manifest
            if row["split"] == split
        )
        assert dict(actual) == counts
    assert (
        hashlib.sha256((RUN / "best.pt").read_bytes()).hexdigest()
        == metadata["checkpoint_sha256"]
    )
    current = read_predictions(RUN / "test_predictions.csv")
    old = read_predictions(RUN / "legacy_fer_predictions.csv")
    assert [row["id"] for row in current] == [row["id"] for row in old], (
        "Different test examples"
    )
    assert [row["label"] for row in current] == [row["label"] for row in old], (
        "Different test labels"
    )
    labels = np.array([int(row["label"]) for row in current])
    scores = np.array([float(row["probability_sad"]) for row in current])
    old_scores = np.array([float(row["probability_sad"]) for row in old])
    measured = classification_metrics(labels, scores)
    previous = classification_metrics(labels, old_scores)
    saved = json.loads((RUN / "metrics.json").read_text())
    for key in ["accuracy", "f1_sad", "roc_auc", "log_loss"]:
        assert abs(saved["fer2013_test"][key] - measured[key]) < 1e-8, key
    # Paired resampling keeps the two predictions for an image together.
    differences = ((scores > 0.5) == labels).astype(float) - (
        (old_scores > 0.5) == labels
    )
    rng = np.random.default_rng(42)
    bootstrap = np.array(
        [rng.choice(differences, len(labels)).mean() for _ in range(2000)]
    )
    benchmark = {
        "evaluation_date": "2026-10-20",
        "scope": "Audited FER2013 happy/sad face-crop test subset; not seven-class FER2013",
        "new_model": measured,
        "original_model_same_test": previous,
        "majority_baseline": saved["majority_baseline"],
        "accuracy_gain_percentage_points": float(differences.mean() * 100),
        "paired_bootstrap_gain_ci95_percentage_points": (
            np.quantile(bootstrap, [0.025, 0.975]) * 100
        ).tolist(),
        "bootstrap": {"seed": 42, "resamples": 2000, "unit": "image, not subject"},
        "legacy_30_full_photos": {
            "original_accuracy": 0.8,
            "new_accuracy": saved["legacy_full_image_test"]["accuracy"],
        },
    }
    if check:
        existing = json.loads((RUN / "benchmark.json").read_text())
        assert existing == benchmark, "Rebuild the report after changing evidence."
        print(
            "Checkpoint checksum, prediction alignment, and benchmark arithmetic verified."
        )
        return
    (RUN / "benchmark.json").write_text(
        json.dumps(benchmark, indent=2, allow_nan=False) + "\n"
    )
    figures = ROOT / "docs/figures"
    figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelcolor": "#31483f",
            "text.color": "#17342f",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#ccd7cb",
            "xtick.color": "#63736d",
            "ytick.color": "#63736d",
            "figure.facecolor": "#f8faf6",
            "axes.facecolor": "#f8faf6",
            "savefig.facecolor": "#f8faf6",
        }
    )
    teal, gray, gold = "#167660", "#81908b", "#bb8c48"
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.6), layout="constrained")
    fig.suptitle(
        "CNN v2  /  held-out evidence",
        fontsize=23,
        fontweight="bold",
        x=0.035,
        ha="left",
    )
    ax = axes[0, 0]
    values = [
        previous["accuracy"],
        saved["majority_baseline"]["accuracy"],
        measured["accuracy"],
    ]
    names = ["Original CNN", "Always happy", "MobileNetV3 · v2"]
    bars = ax.barh(names, np.array(values) * 100, color=[gray, gold, teal], height=0.5)
    for bar, value in zip(bars, values):
        ax.text(
            value * 100 + 1,
            bar.get_y() + 0.25,
            f"{value:.1%}",
            va="center",
            fontsize=11,
        )
    ax.set(
        xlim=(0, 105),
        xlabel="Accuracy (%)",
        title="Same 1,469 test images · binary classification",
    )
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.invert_yaxis()
    ax = axes[0, 1]
    matrix = np.array(measured["confusion_matrix"])
    ax.imshow(matrix, cmap="Greens", vmin=0, vmax=900)
    for (i, j), value in np.ndenumerate(matrix):
        ax.text(
            j,
            i,
            str(value),
            ha="center",
            va="center",
            fontsize=23,
            color="white" if value > 450 else "#17342f",
        )
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Happy", "Sad"],
        yticklabels=["Happy", "Sad"],
        xlabel="Predicted label",
        ylabel="Dataset label",
        title="126 errors · inspect the confusion matrix",
    )
    ax = axes[1, 0]
    history = json.loads((RUN / "history.json").read_text())
    epochs = [row["epoch"] for row in history]
    ax.plot(
        epochs,
        [row["train_accuracy"] for row in history],
        label="Train (augmented)",
        color=gray,
    )
    ax.plot(
        epochs,
        [row["val_accuracy"] for row in history],
        label="Validation",
        color=teal,
        marker=".",
        markersize=4,
    )
    ax.axvline(
        metadata["best_epoch"], linestyle=":", color=gold, label="Best validation loss"
    )
    ax.set(
        xlabel="Epoch",
        ylabel="Accuracy",
        ylim=(0.5, 1),
        title="Selected on validation loss; stopped at epoch 16",
    )
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax = axes[1, 1]
    for probabilities, name, color in [
        (old_scores, "Original CNN", gray),
        (scores, "MobileNetV3 · v2", teal),
    ]:
        fpr, tpr, _ = roc_curve(labels, probabilities)
        ax.plot(fpr, tpr, label=name, color=color)
    ax.plot([0, 1], [0, 1], "--", color=gold, linewidth=1, label="Chance")
    ax.set(
        xlabel="False positive rate",
        ylabel="True positive rate",
        xlim=(0, 1),
        ylim=(0, 1.02),
        title=f"Sad-class ROC · v2 AUC {measured['roc_auc']:.3f}",
    )
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.savefig(figures / "benchmark.png", dpi=180)
    plt.close(fig)
    print(json.dumps(benchmark, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    main(parser.parse_args().check)
