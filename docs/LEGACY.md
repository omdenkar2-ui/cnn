# Original TensorFlow CNN

The original tutorial-style implementation remains in `image_classifier/`, with its trained checkpoint and evidence in [`artifacts/reference-run/`](../artifacts/reference-run/).
It achieved 24/30 correct (80%) on its small original test split. See the [preserved model card](LEGACY_MODEL_CARD.md).

Use a separate **Python 3.12** environment:

```bash
python -m venv .venv-legacy
# Activate .venv-legacy, then:
python -m pip install -r requirements-legacy.txt
python -m image_classifier.predict --image path/to/photo.jpg
```

The old architecture is a three-block Conv2D/MaxPool CNN (16/32/16 filters), Flatten, Dense(256), Dropout(0.3), and one sigmoid output. Its default resolution is 128×128 RGB. `sad` is the positive class.

Reproduce training:

```bash
python scripts/download_reference.py
python -m image_classifier.train --data data --output runs/legacy-reproduction --epochs 30
python -m image_classifier.predict --run runs/legacy-reproduction --image path/to/photo.jpg
```

The downloader pins an upstream commit and verifies Git blob checksums. The original image source provides no dataset license; images are not redistributed here. The pipeline reports corrupt files, excludes conflicting inputs, clusters near duplicates, and creates a deterministic split.

Test the original implementation:

```bash
python -m pip install -r requirements-legacy-dev.txt
python -m pytest tests/test_pipeline.py -q
```

The [legacy notebook](../notebooks/LegacyImageClassification.ipynb) contains the original walkthrough. Start it with the repository root as working directory, using the legacy environment. The original training environment is preserved in [`requirements-legacy-lock.txt`](../requirements-legacy-lock.txt).
