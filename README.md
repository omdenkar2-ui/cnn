# CNN Image Classifier

A TensorFlow/Keras classifier inspired by Nicholas Renotte's
[ImageClassification tutorial](https://github.com/nicknochnack/ImageClassification)
and [video](https://www.youtube.com/watch?v=jztwpsIzEGc).
The code here is a new implementation of a similar architecture.
It trains on two image folders and ships with a trained happy/sad model.

The included checkpoint scored **80% accuracy (24/30)** on the held-out reference
split, versus a **50%** majority baseline. This is a small educational test;
see [the model card](MODEL_CARD.md) for precision, recall, uncertainty, and data
limitations. [Training curves](artifacts/reference-run/learning_curves.png) and
[raw test metrics](artifacts/reference-run/metrics.json) are included.

## Architecture

RGB input -> rescale -> training-only horizontal flip -> Conv2D(16) + pool ->
Conv2D(32) + pool -> Conv2D(16) + pool -> Flatten -> Dense(256) ->
training-only Dropout(0.3) -> Dense(1, sigmoid).

Default input resolution is 128 x 128 for practical CPU training. Use `--size 256`
to match the reference's input dimensions. All convolutions use 3 x 3 kernels,
valid padding and ReLU. The positive class is the second sorted folder name:
`happy=0`, `sad=1` for the reference images.

## Setup

Use Python 3.12. On Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

On macOS/Linux, use `python3.12 -m venv .venv` and replace subsequent
`.venv\Scripts\python.exe` commands with `.venv/bin/python`.
TensorFlow 2.20 is pinned. Windows native training uses the CPU; supported NVIDIA
GPU training requires a suitable Linux/WSL2 TensorFlow environment.

## Predict with the included model

```powershell
.venv\Scripts\python.exe -m image_classifier.predict --image "C:\path\to\photo.jpg"
```

The command loads `artifacts/reference-run/best.keras` and its preprocessing
metadata, then prints the predicted label and both class scores. Scores are model
outputs, not independently calibrated confidence estimates. This educational
dataset's happy/sad visual labels do not establish a person's internal emotions.

## Reproduce training

```powershell
.venv\Scripts\python.exe scripts/download_reference.py
.venv\Scripts\python.exe -m image_classifier.train --data data --output runs/my-run --epochs 30
.venv\Scripts\python.exe -m image_classifier.predict --run runs/my-run --image data/happy/example.jpg
```

Replace `example.jpg` with an actual filename. The downloader retrieves images
from an immutable upstream commit and verifies their Git blob checksums.
The dataset is downloaded locally rather than redistributed in this repository.
The upstream repository does not provide a dataset license; check the image
rights before using it outside this learning exercise.

Training writes a checkpoint, metadata, metrics, learning curves, a fixed split
manifest, a data audit, and test predictions into the selected output directory.
An existing output directory is rejected to preserve prior runs. Best epoch is
selected using validation loss; the test split is evaluated afterward at a fixed
threshold of 0.5.

## Use your own images

Place files under exactly two class folders:

```text
data/
  class_a/
    image1.jpg
  class_b/
    image2.jpg
```

Use at least 10 readable unique images per class; hundreds of varied examples are
more useful. Corrupt files are reported and skipped. Identical resized RGB inputs
are removed within classes. Training excludes all copies of inputs with
contradictory labels and records them in `data_audit.json`; the low-level scanner
also supports a strict mode that fails instead. The upstream dataset contains
such conflicts, so they are not silently assigned an arbitrary label.
EXIF orientation, RGB conversion, and Pillow bilinear resize are shared by
training and inference. Rescaling is embedded in the saved model.

Before the target 70/20/10 split, perceptually similar images are clustered using
64-bit difference hashes (default Hamming distance <=3). Clusters stay together,
so actual ratios can vary. This heuristic cannot detect every repeated person,
crop, or source. For a subject-independent claim, add source/subject metadata and
split by those identities before training. This implementation deliberately
fails if a class is absent from a partition instead of reporting misleading scores.

## Tests

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.venv\Scripts\python.exe -m pytest -q
```

Tests cover RGB decoding, corrupt-file handling, conflicting duplicates, stable
cluster memberships, metric arithmetic, and actual one-epoch training with
checkpoint reload and inference on synthetic images. CI runs the same tests.

## Notebook and results

`ImageClassification.ipynb` provides an interactive walkthrough of download,
training, saved metrics, and prediction. `MODEL_CARD.md` records the trained
model's measured results and evaluation limitations. `requirements-lock.txt`
records the environment used for the shipped model.

To use the notebook, install `jupyterlab` and `ipykernel` in this environment,
register it with `python -m ipykernel install --user --name cnn-classifier`, and
select that kernel in JupyterLab or your editor. The command-line workflow needs
neither package.

References: [TensorFlow installation](https://www.tensorflow.org/install/pip),
[Keras Conv2D](https://keras.io/api/layers/convolution_layers/convolution2d/),
[Keras model persistence](https://keras.io/guides/serialization_and_saving/).
