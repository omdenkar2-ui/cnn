import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    log_loss,
    brier_score_loss,
)


def classification_metrics(labels, scores):
    labels = np.asarray(labels, dtype=int).ravel()
    scores = np.asarray(scores, dtype=float).ravel()
    if len(labels) != len(scores) or len(labels) == 0 or not np.isfinite(scores).all():
        raise ValueError("Labels and finite scores must have matching, nonzero lengths")
    if not set(labels).issubset({0, 1}) or np.any((scores < 0) | (scores > 1)):
        raise ValueError("Expected binary labels and probabilities between 0 and 1")
    decisions = scores > 0.5
    accuracy = float(accuracy_score(labels, decisions))
    n = len(labels)
    z = 1.96
    center = (accuracy + z * z / (2 * n)) / (1 + z * z / n)
    half = (
        z
        * np.sqrt(accuracy * (1 - accuracy) / n + z * z / (4 * n * n))
        / (1 + z * z / n)
    )
    confidence = np.maximum(scores, 1 - scores)
    correct = decisions == labels
    bins = np.minimum((confidence * 10).astype(int), 9)
    ece = sum(
        np.mean(bins == i)
        * abs(np.mean(correct[bins == i]) - np.mean(confidence[bins == i]))
        for i in range(10)
        if np.any(bins == i)
    )
    return {
        "examples": n,
        "accuracy": accuracy,
        "accuracy_ci95_wilson": [float(center - half), float(center + half)],
        "balanced_accuracy": float(balanced_accuracy_score(labels, decisions)),
        "precision_sad": float(precision_score(labels, decisions, zero_division=0)),
        "recall_sad": float(recall_score(labels, decisions, zero_division=0)),
        "f1_sad": float(f1_score(labels, decisions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, scores))
        if len(set(labels)) == 2
        else None,
        "log_loss": float(log_loss(labels, scores, labels=[0, 1])),
        "brier_score": float(brier_score_loss(labels, scores)),
        "ece_10_bins": float(ece),
        "confusion_matrix": confusion_matrix(labels, decisions, labels=[0, 1]).tolist(),
    }
