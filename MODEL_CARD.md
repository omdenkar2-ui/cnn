# Model card: reference-style happy/sad CNN

## Training and provenance

Trained from randomly initialized weights on the reference repository's images.
Source: https://github.com/nicknochnack/ImageClassification
Source commit: `ecbc2b219606c4153bdec47cd07f6ab00652c6d6`. All 305 downloaded files were
verified against their Git blob hashes. The upstream model weights were not used.

RGB input is 128 x 128; the network has
813,041 parameters. Conv16/Conv32/Conv16 blocks, max pooling,
Flatten, Dense256, and a sigmoid output mirror the tutorial pattern. Horizontal
flip and dropout are added for training. Seed 42, batch size
16, Adam learning rate 0.001, binary cross-entropy.

Training stopped after 14 epochs; validation loss
selected epoch 7. No test-based threshold tuning was used.
The inference checkpoint omits optimizer state. `train.py` saves complete training
checkpoints when you run a new experiment.

## Data audit and splits

- Downloaded source files: 305.
- Unreadable files: 0.
- Within-class duplicate files omitted: 1.
- Conflicting-label pairs: 1; all copies of those
  identical model inputs were excluded.
- Retained unique, non-conflicting inputs: 302.
- Train: 212 images (107 happy, 105 sad).
- Validation: 60 images (30 happy, 30 sad).
- Test: 30 images (15 happy, 15 sad).

Exact resized-input hashes are deduplicated. Difference-hash clusters (Hamming
distance <=3) stay within one split. `manifest.json` records every path, label,
cluster, checksum, and split. This heuristic is not a complete subject/source
identity audit. Repeated people or undetected near duplicates may remain.

## Measured final test results

Positive class: sad. Predict sad when score >0.5.

| Metric | Value |
| --- | --- |
| Accuracy | 80.0% (24/30 correct) |
| Majority baseline accuracy | 50.0% |
| Precision, sad | 73.7% |
| Recall, sad | 93.3% |
| F1, sad | 0.8235 |

Confusion matrix, rows=true and columns=predicted, order happy/sad:

| | Predicted happy | Predicted sad |
| --- | --- | --- |
| True happy | 10 | 5 |
| True sad | 1 | 14 |

Approximate 95% Wilson interval for accuracy: 62.7% to 90.5%.
That interval assumes independent examples from the same distribution. It does
not account for selection bias, repeated subjects, or dataset/domain shift.
Raw model scores are not calibrated probabilities.

## Intended use and practical limits

An educational image-classification demonstration. Labels describe this
dataset's visual categories and do not establish a person's internal emotions.
The small, web-sourced dataset mixes subjects, backgrounds, and styles. The model
can use those correlations rather than facial-expression features. No independent
external test, fairness assessment, or deployment certification was performed.

Before any application, evaluate representative images with known rights and
labels, split by source/person where appropriate, and inspect failure cases.
One-image inference accepts static formats Pillow can decode; EXIF orientation is
normalized, RGB conversion and bilinear resizing precede embedded rescaling.

## Verification

Seven automated tests passed, including real synthetic training/save/reload/
prediction. The reference run's checkpoint reload predictions agree with stored
test scores within rtol=1e-5, atol=1e-6. Test metrics were recomputed from those
scores. No duplicate cluster intersects two splits. Notebook code cells compile.
Exact authoring environment is in `requirements-lock.txt`.

Checkpoint SHA-256: `ef339d95bb1adfbf27047936cae03159768f6fdf5bf14e2b7eab78efc9ebdd31`.
