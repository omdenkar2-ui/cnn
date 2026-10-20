import hashlib
from io import BytesIO
import json
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image
import pytest
import torch

from cnn_v2.dataset import BKTree, SOURCE_SHA256, deduplicate
from cnn_v2.demo import MAX_BYTES, make_server
from cnn_v2.metrics import classification_metrics
from cnn_v2.model import ARCHITECTURE, build_model, preprocess
from cnn_v2.predict import DEFAULT_RUN, Predictor, decode_image

torch.set_num_threads(2)


def png_bytes(array):
    buffer = BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


def test_hamming_index_matches_brute_force():
    rng = np.random.default_rng(1)
    values = rng.integers(0, 256, 200).tolist()
    tree = BKTree()
    for index, value in enumerate(values):
        tree.insert(value, index)
    for query in [0, 71, 255]:
        assert set(tree.query(query, 2)) == {
            i for i, value in enumerate(values) if (query ^ value).bit_count() <= 2
        }


def test_duplicate_audit_preserves_holdout_and_excludes_label_conflicts():
    rng = np.random.default_rng(9)
    a, b, c = rng.integers(0, 255, (3, 48, 48), dtype=np.uint8)
    images = np.stack([a, a, a, b, b, c])
    labels = [0, 0, 0, 0, 1, 1]
    splits = ["train", "val", "test", "train", "test", "train"]
    rows = [
        {
            "id": str(i),
            "label": label,
            "split": split,
            "sha256": hashlib.sha256(image.tobytes()).hexdigest(),
        }
        for i, (image, label, split) in enumerate(zip(images, labels, splits))
    ]
    retained, removed = deduplicate(images, rows)
    assert retained == [2, 5]
    assert {row["id"] for row in removed if row["reason"] == "conflicting_cluster"} == {
        "3",
        "4",
    }
    assert all(
        row["kept_id"] == "2" for row in removed if row["reason"] == "duplicate_cluster"
    )
    assert all(len(value) == 64 for value in SOURCE_SHA256.values())


def test_preprocessing_decode_and_augmentation_contract():
    image = np.random.default_rng(3).integers(0, 256, (48, 48), dtype=np.uint8)
    decoded = np.asarray(decode_image(png_bytes(image)))
    np.testing.assert_array_equal(image, decoded)
    batch = torch.from_numpy(image)[None, None]
    expected = preprocess(batch)
    assert expected.shape == (1, 3, 128, 128)
    torch.testing.assert_close(expected, preprocess(batch))
    torch.testing.assert_close(expected, preprocess(batch.expand(-1, 3, -1, -1)))
    assert torch.isfinite(preprocess(batch, augment=True)).all()
    with pytest.raises(ValueError):
        preprocess(batch[0])


def test_metrics_and_probability_validation():
    result = classification_metrics([0, 0, 1, 1], [0.1, 0.5, 0.8, 0.3])
    assert result["confusion_matrix"] == [[2, 0], [1, 1]]
    assert result["accuracy"] == 0.75
    assert result["f1_sad"] == pytest.approx(2 / 3)
    assert result["accuracy_ci95_wilson"][0] < 0.75 < result["accuracy_ci95_wilson"][1]
    for labels, scores in [([], []), ([0], [float("nan")]), ([2], [0.2]), ([0], [1.2])]:
        with pytest.raises(ValueError):
            classification_metrics(labels, scores)


def test_actual_training_step_checkpoint_and_prediction_parity(tmp_path):
    torch.manual_seed(42)
    model = build_model().train()
    images = torch.randint(0, 256, (4, 1, 48, 48), dtype=torch.uint8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    before = model.classifier[-1].weight.detach().clone()
    loss = torch.nn.functional.cross_entropy(
        model(preprocess(images)), torch.tensor([0, 1, 0, 1])
    )
    loss.backward()
    optimizer.step()
    assert not torch.equal(before, model.classifier[-1].weight)
    weights = tmp_path / "best.pt"
    torch.save(
        {"state_dict": model.state_dict(), "architecture": ARCHITECTURE}, weights
    )
    metadata = {
        "class_names": ["happy", "sad"],
        "temperature": 1.15,
        "checkpoint_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
    }
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    predictor = Predictor(tmp_path)
    result = predictor.predict(png_bytes(images[0, 0].numpy()))
    with torch.inference_mode():
        expected = (model.eval()(preprocess(images[:1])) / 1.15).softmax(1)[0]
    assert list(result["scores"].values()) == pytest.approx(expected.tolist(), abs=1e-6)
    weights.write_bytes(weights.read_bytes() + b"corruption")
    with pytest.raises(ValueError, match="checksum"):
        Predictor(tmp_path)


def test_released_checkpoint_is_loadable_and_reports_probabilities():
    predictor = Predictor(DEFAULT_RUN)
    result = predictor.predict(png_bytes(np.full((48, 48), 128, dtype=np.uint8)))
    assert result["label"] in ["happy", "sad"]
    assert sum(result["scores"].values()) == pytest.approx(1)
    assert predictor.metadata["parameters"] == sum(
        p.numel() for p in predictor.model.parameters()
    )


def test_demo_http_prediction_and_invalid_uploads():
    predictor = Predictor(DEFAULT_RUN)
    server = make_server(predictor, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    address = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(address) as response:
            assert b"Expression lab" in response.read()
        payload = png_bytes(np.full((48, 48), 128, dtype=np.uint8))
        with urlopen(Request(address + "/api/predict", data=payload)) as response:
            result = json.load(response)
        assert result["label"] in ["happy", "sad"]
        for request, status in [
            (Request(address + "/api/predict", data=b"bad image"), 400),
            (
                Request(
                    address + "/api/predict",
                    data=b"x",
                    headers={"Origin": "https://example.com"},
                ),
                403,
            ),
            (
                Request(
                    address + "/api/predict",
                    data=b"x",
                    headers={"Content-Length": str(MAX_BYTES + 1)},
                ),
                413,
            ),
            (Request(address + "/../README.md"), 404),
        ]:
            with pytest.raises(HTTPError) as error:
                urlopen(request)
            assert error.value.code == status
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
