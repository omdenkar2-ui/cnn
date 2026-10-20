"""Run in the TensorFlow environment to benchmark the original released model."""

from pathlib import Path
import argparse
import csv
import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "4")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
import numpy as np
from PIL import Image
import tensorflow as tf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/fer2013-v2"))
    parser.add_argument(
        "--output", type=Path, default=Path("runs/legacy-fer-predictions.csv")
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with np.load(args.data / "dataset.npz", allow_pickle=False) as bundle:
        images = bundle["test_images"]
        labels = bundle["test_labels"]
        ids = bundle["test_ids"]
    model = tf.keras.models.load_model(
        "artifacts/reference-run/best.keras", compile=False
    )
    scores = []
    for start in range(0, len(images), 32):
        batch = np.stack(
            [
                np.asarray(
                    Image.fromarray(x)
                    .convert("RGB")
                    .resize((128, 128), Image.Resampling.BILINEAR)
                )
                for x in images[start : start + 32]
            ]
        )
        scores.extend(model(batch, training=False).numpy().ravel().tolist())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "label", "probability_sad", "prediction"])
        for identifier, label, score in zip(ids, labels, scores):
            writer.writerow([identifier, int(label), score, int(score > 0.5)])
    print(
        json.dumps(
            {
                "examples": len(labels),
                "accuracy": float(np.mean((np.array(scores) > 0.5) == labels)),
            }
        )
    )


if __name__ == "__main__":
    main()
