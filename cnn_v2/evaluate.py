"""Evaluate a frozen checkpoint; write per-example evidence and summary metrics."""

from pathlib import Path
import argparse
import csv
import json
import time
import numpy as np
import torch
from .model import preprocess
from .predict import Predictor
from .metrics import classification_metrics
from .train import predict_logits, write_json


def save_predictions(path, ids, labels, scores):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "label", "probability_sad", "prediction"])
        for identifier, label, score in zip(ids, labels, scores):
            writer.writerow([identifier, int(label), float(score), int(score > 0.5)])


def evaluate(data, run, legacy_data=None):
    data = Path(data)
    run = Path(run)
    if (run / "metrics.json").exists():
        raise FileExistsError("Evaluation already exists; preserve it for audit.")
    torch.set_num_threads(4)
    predictor = Predictor(run, device="cuda" if torch.cuda.is_available() else "cpu")
    with np.load(data / "dataset.npz", allow_pickle=False) as bundle:
        images = torch.from_numpy(bundle["test_images"][:, None].copy())
        labels = bundle["test_labels"].copy()
        ids = bundle["test_ids"].copy()
        train_labels = bundle["train_labels"].copy()
    logits = predict_logits(predictor.model, images, predictor.device)
    raw_scores = logits.softmax(1)[:, 1].numpy()
    scores = (logits / predictor.metadata["temperature"]).softmax(1)[:, 1].numpy()
    result = {
        "fer2013_test": classification_metrics(labels, scores),
        "fer2013_test_uncalibrated": classification_metrics(labels, raw_scores),
        "majority_baseline": classification_metrics(
            labels, np.full(len(labels), float(np.bincount(train_labels).argmax()))
        ),
    }
    save_predictions(run / "test_predictions.csv", ids, labels, scores)
    if legacy_data:
        legacy = Path(legacy_data)
        manifest = json.loads(
            (Path("artifacts/reference-run") / "manifest.json").read_text()
        )
        test = [row for row in manifest if row["split"] == "test"]
        legacy_scores = [
            predictor.predict(legacy / row["path"])["scores"]["sad"] for row in test
        ]
        result["legacy_full_image_test"] = classification_metrics(
            [r["label"] for r in test], legacy_scores
        )
        save_predictions(
            run / "legacy_predictions.csv",
            [r["path"] for r in test],
            [r["label"] for r in test],
            legacy_scores,
        )
    # CPU latency for preprocessed one-image forward passes, after warmup.
    predictor.model.cpu()
    sample = preprocess(images[:1])
    with torch.inference_mode():
        for _ in range(10):
            predictor.model(sample)
        elapsed = []
        for _ in range(50):
            started = time.perf_counter()
            predictor.model(sample)
            elapsed.append((time.perf_counter() - started) * 1000)
    result["latency_cpu"] = {
        "median_ms": float(np.median(elapsed)),
        "p95_ms": float(np.percentile(elapsed, 95)),
        "threads": 4,
        "batch": 1,
        "warmup": 10,
        "repeats": 50,
        "scope": "preprocessed forward pass; excludes image decoding",
    }
    write_json(run / "metrics.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/fer2013-v2"))
    parser.add_argument("--run", type=Path, default=Path("runs/v2"))
    parser.add_argument("--legacy-data", type=Path)
    args = parser.parse_args()
    evaluate(args.data, args.run, args.legacy_data)
