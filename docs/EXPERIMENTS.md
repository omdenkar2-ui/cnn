# Experiments and reproduction

## Question

Can a compact pretrained CNN trained on substantially more face data outperform the original small-data tutorial checkpoint on happy/sad face crops?

The result is **yes for the audited face-crop benchmark**, with a measured regression on the original full-photo domain. This is a project-level upgrade, not an architecture ablation study.

## Recorded experiments

| Run | Data | Initialization | Selection | Held-out outcome |
|---|---|---|---|---|
| Original | 212 training full photos | Random CNN weights | Original validation loss | 80% on original 30 photos |
| v2 | 11,684 FER face crops | ImageNet MobileNetV3 Small | Minimum validation loss; epoch 10 | 91.42% on 1,469 audited FER face crops |
| Original, transferred | No retraining | Shipped original weights | No selection | 41.12% on the same 1,469 FER crops |
| v2, transferred | No retraining | Shipped v2 weights | No selection | 46.67% on the original 30 photos |

The dataset and transfer-learning recipe were chosen before test evaluation. One v2 training run was performed. Its checkpoint and calibration were selected only on validation data. There was no test-based search for architecture, seed, temperature, or decision threshold. The full-photo regression is preserved rather than tuned away.

## Recipe

- Preserve source split membership; audit exact and near duplicates before training.
- Convert grayscale crops to three channels, resize to 128×128, apply ImageNet normalization.
- Replace MobileNetV3's classifier with 576→128→2, Hardswish and dropout 0.25.
- Train the head for one epoch, then fine-tune the full network.
- Batch 128; seed 42; AdamW (backbone 2e-4, head 1e-3, decay 1e-4).
- Cosine schedule over 20 requested epochs; minimum learning rate 1e-6.
- Random horizontal flips, affine rotation approximately ±11.5°, affine scale 0.9–1.1, normalized translations ±0.06, brightness multiplier 0.85–1.15.
- Cross-entropy with 0.03 label smoothing; validation loss without smoothing.
- Mixed precision on CUDA; gradient norm clipped at 1.
- Stop after six epochs without a validation-loss improvement.
- Fit one scalar temperature on validation logits; threshold fixed at 0.5.

The data are moderately imbalanced. This run uses ordinary shuffled sampling and no class reweighting. Balanced accuracy and per-class metrics are reported alongside accuracy.

## Reproduce v2

Follow the environment setup in [the README](../README.md), then:

```bash
python -m cnn_v2.dataset
python -m cnn_v2.train --output runs/reproduction --epochs 20 --seed 42
python -m cnn_v2.evaluate --run runs/reproduction
```

Dataset download pins revision `9c0267afc93fa4d4bfc05d7ff1108a04f72e8392` and verifies all three parquet checksums. The manifest records original row IDs and decoded-pixel hashes; compare its SHA-256 to the shipped audit. NPZ compression or package differences may affect the container checksum; inspect arrays and manifest when diagnosing a difference.

The trainer reads train and validation arrays only. Evaluation is a separate command and refuses to overwrite an existing evaluation. Reproduction means running the documented procedure, not expecting identical floating-point weights on every platform.

`training_state.pt` contains optimizer/scheduler/scaler and Torch RNG states for manual experiment continuation. Automatic resume is not implemented. These private run files are ignored by Git; inference needs only `best.pt` and `metadata.json`.

## Reproduce the original-model comparison

Use a **separate Python 3.12 environment** with `requirements-legacy.txt` to avoid mixing TensorFlow and PyTorch dependency requirements.

After preparing FER with the v2 environment:

```bash
# Run with the TensorFlow environment:
python scripts/evaluate_legacy_on_fer.py
```

This writes `runs/legacy-fer-predictions.csv`. It uses the original saved checkpoint and its own RGB/Pillow preprocessing. It does not retrain or calibrate the original model.

To check full-photo transfer, first run `scripts/download_reference.py` in the legacy environment. Evaluate a fresh v2 run using `--legacy-data data`; that adds the original fixed 30-photo test set.

## Rebuild and audit the published report

```bash
python scripts/build_report.py --check
python scripts/build_report.py
```

The first command verifies the released checkpoint hash, original/v2 test ID alignment, labels, metric arithmetic, and saved benchmark. The second regenerates `benchmark.json` and the figure from the published CSVs. Neither command selects weights or fits anything on test data.

The original run's 30-photo results are in [its preserved model card](LEGACY_MODEL_CARD.md). Model checkpoint sizes, dataset counts, software versions, and class mapping are recorded in metadata rather than inferred from filenames.

## What remains untested

- Identity-disjoint and demographic subgroup performance.
- Robustness to pose, occlusion, lighting, skin tone, age, and camera differences.
- Other expressions, multiple people, and reliable rejection of out-of-distribution inputs.
- Multiple-seed variance and architecture-only ablations on equal training data.
- Face detection followed by end-to-end evaluation on unseen full photos.

These are meaningful next experiments. The current test set has now been inspected; use a new external holdout when making decisions informed by these results.
