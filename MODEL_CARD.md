# Model card · CNN v2

Release date: **2026-10-20**. Maintainer: **Om Denkar**.

## Intended use and boundaries

An educational, local classifier for **closely cropped faces** with visual expressions labeled happy or sad. It is not a face detector, a general scene classifier, or an assessment of internal emotion. It always emits one of two labels; other expressions, non-faces, multiple faces, and unusual photos can receive confident but meaningless results. Do not use it to make consequential decisions about people.

The upgrade is specific to face crops. It improves the audited FER2013 binary test result from **41.1% to 91.4%**, but regresses on the old 30 full-photo examples from **80.0% to 46.7%**. The old checkpoint and code remain available for that original experiment.

## Model and input contract

| Property | Value |
|---|---|
| Backbone | TorchVision MobileNetV3 Small, ImageNet-1K V1 initialization |
| Head | Linear 576→128, Hardswish, dropout 0.25, Linear 128→2 |
| Parameters | 1,001,122 |
| Input | EXIF-corrected single image, grayscale, replicated into 3 channels |
| Resize | PyTorch bilinear to 128×128, antialias enabled, align_corners=False |
| Normalize | ImageNet mean `[.485,.456,.406]`, standard deviation `[.229,.224,.225]` |
| Output | Softmax of logits divided by temperature 1.150000095 |
| Decision | Sad if P(sad)>0.5, otherwise happy |
| Low confidence | Maximum score <0.65; UI heuristic, not an OOD detector |
| Artifact | `artifacts/v2/best.pt`, state_dict only, loaded with `weights_only=True` |
| SHA-256 | `306c7c867d2e8f19cf83e86162e0f47a22085cacb3da67bf60b2c7e8cac02a5c` |

Image decode and resize are shared between inference and training. Training additionally applies random affine transformations, horizontal flipping, and brightness perturbation. Prediction verifies the checkpoint checksum before loading.

## Dataset and audit

Source: [AutumnQiu/fer2013](https://huggingface.co/datasets/AutumnQiu/fer2013/tree/9c0267afc93fa4d4bfc05d7ff1108a04f72e8392), revision `9c0267afc93fa4d4bfc05d7ff1108a04f72e8392`.
Source labels **3=happy**, **4=sad** map to output labels **0**, **1**. Other labels are excluded. Inputs are 48×48 grayscale face crops.

| Split | Happy | Sad | Total |
|---|---:|---:|---:|
| Train | 7,025 | 4,659 | 11,684 |
| Validation | 889 | 648 | 1,537 |
| Test | 878 | 591 | 1,469 |
| **Total** | **8,792** | **5,898** | **14,690** |

From 15,066 selected examples, the audit removes **374 redundant cluster members and 2 conflicting-label examples**. Exact decoded-pixel duplicates are clustered. Near duplicates require both 64-bit dHash distance ≤2 and mean absolute grayscale pixel difference ≤8. Connected components with conflicting labels are completely excluded.

One representative remains per nonconflicting cluster, prioritizing test, then validation, then train. Surviving examples retain the source split: no training example is moved into a holdout. This examines all splits for data hygiene before training; checkpoint and temperature selection never use test scores. Source file checksums, stable source-row IDs, retained membership, and removed records are published in [the artifact directory](artifacts/v2/).

**Limitations:** similarity matching is a heuristic, not identity separation. The same person or source may appear across splits. Labels can be subjective or wrong. Demographic metadata is unavailable in this mirror; demographic and identity-independent performance has not been measured. Removing examples changes the canonical test set, so these numbers are not directly comparable with standard FER2013 leaderboards.

## Training and selection

One recorded training run, seed 42, batch size 128. Epoch 1 trains the new head with the backbone and its batch normalization frozen; later epochs fine-tune all parameters. AdamW uses learning rates 2e-4 (backbone) and 1e-3 (head), weight decay 1e-4, cosine decay over 20 requested epochs, cross-entropy label smoothing 0.03, gradient clipping at norm 1, and CUDA automatic mixed precision.

Checkpoint selection minimizes **unsmoothed validation cross-entropy**. Early stopping patience is six epochs. Best epoch: **10**; completed epochs: **16**. Temperature is selected over 101 values from 0.5 to 3.0 using validation negative log likelihood. Test threshold stays at 0.5.

Training used Python 3.13.5, PyTorch 2.6.0+cu124, TorchVision 0.21.0+cu124, and an NVIDIA RTX 4070 Laptop GPU. Training plus validation calibration took **263.8 seconds**, excluding data preparation and evaluation. Seeds and cuDNN deterministic settings are recorded; cross-platform bitwise reproducibility is not guaranteed.

## Held-out results

| Metric | v2 | Original checkpoint, same test |
|---|---:|---:|
| Accuracy | **91.42%** (1,343/1,469) | 41.12% (604/1,469) |
| Balanced accuracy | **91.05%** | 49.86% |
| Precision, sad | 89.47% | 40.16% |
| Recall, sad | 89.17% | 94.59% |
| F1, sad | **0.8932** | 0.5638 |
| ROC AUC, sad | **0.9675** | 0.5119 |
| Log loss | **0.2256** | 0.8139 |

An always-happy baseline achieves **59.77% accuracy**. V2 accuracy has a 95% Wilson interval of **89.88–92.75%**. The paired accuracy gain is **50.31 percentage points**, with an image-bootstrap 95% interval of **47.31–53.23 points** (2,000 resamples, seed 42). These intervals assume independent images and do not capture uncertainty from repeated subjects or retraining.

Confusion matrix, rows=true labels and columns=predictions:

| | Predicted happy | Predicted sad |
|---|---:|---:|
| True happy | 816 | 62 |
| True sad | 64 | 527 |

Validation-fitted temperature scaling reduces test log loss from **0.2294 to 0.2256**, Brier score from **0.06636 to 0.06584**, and ten-bin confidence ECE from **0.01922 to 0.01196**. Accuracy is unchanged. Calibration has not been established outside this test distribution.

The original model used RGB Pillow bilinear preprocessing at 128×128, matching its own saved input contract. Its poor FER result demonstrates a domain mismatch, not a controlled finding about small CNNs. This upgrade changes the data, pretraining, architecture, regularization, and optimization together.

### Cross-domain check

On the unchanged original 30-image full-photo holdout, the original model gets **24/30** correct and v2 gets **14/30**. Full-scene images should not be passed to v2 without an appropriate face-cropping workflow; no automatic cropping is implemented here. The test result was reported without further tuning on these images.

### Performance

Measured CPU forward-pass latency: **24.6 ms median**, **30.8 ms p95**, batch 1, four PyTorch threads, ten warmup calls and 50 measurements on the development laptop. This excludes decoding and preprocessing, is not a throughput benchmark, and varies across machines.

## Evidence, rights, and reproducibility

- [Metadata and checksums](artifacts/v2/metadata.json)
- [Exact split manifest](artifacts/v2/manifest.json) and [removal log](artifacts/v2/removed.json)
- [Training history](artifacts/v2/history.json), [metrics](artifacts/v2/metrics.json), [paired comparison](artifacts/v2/benchmark.json)
- [Individual v2 predictions](artifacts/v2/test_predictions.csv), [original predictions on FER](artifacts/v2/legacy_fer_predictions.csv), [v2 predictions on original photos](artifacts/v2/legacy_predictions.csv)
- [Commands and experimental limitations](docs/EXPERIMENTS.md)

The Hugging Face mirror does not supply an explicit image license. Raw images are excluded from this repository. Availability of a dataset or checkpoint does not establish rights for commercial deployment or redistribution. See [NOTICE](NOTICE.md) for source attribution.
