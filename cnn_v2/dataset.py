"""Prepare the Happy/Sad subset without changing the source split membership."""

from collections import Counter, defaultdict
from io import BytesIO
from pathlib import Path
import argparse
import hashlib
import json
import time

import numpy as np
from PIL import Image
import pyarrow.parquet as pq
import requests

SOURCE = "AutumnQiu/fer2013"
REVISION = "9c0267afc93fa4d4bfc05d7ff1108a04f72e8392"
LABELS = {3: 0, 4: 1}
CLASS_NAMES = ["happy", "sad"]
SOURCE_SHA256 = {
    "train": "f02d2f0013d424f087325ca802914b955f12d8a4bbb0d0b929defb2fecdf6f9b",
    "valid": "3577f5721214e41710069bfeb55768f8e0abe877499f7652c1fa085dcd80f110",
    "test": "bed4a9621b660159a30305dace4042b7d602817939e2277d462c63de94f46c14",
}


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url, target):
    target = Path(target)
    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".partial")
    for attempt in range(4):
        try:
            with requests.get(url, stream=True, timeout=(30, 180)) as response:
                response.raise_for_status()
                with partial.open("wb") as handle:
                    for chunk in response.iter_content(1024 * 1024):
                        handle.write(chunk)
            partial.replace(target)
            return
        except requests.RequestException:
            if attempt == 3:
                raise
            time.sleep(2**attempt)


def dhash(image):
    thumb = np.asarray(Image.fromarray(image).resize((9, 8), Image.Resampling.BILINEAR))
    return sum(
        int(v) << i for i, v in enumerate((thumb[:, 1:] > thumb[:, :-1]).ravel())
    )


class BKTree:
    """Hamming-distance index; avoids a quadratic all-pairs duplicate scan."""

    def __init__(self):
        self.root = None

    def insert(self, value, index):
        if self.root is None:
            self.root = [value, [index], {}]
            return
        node = self.root
        while True:
            distance = (value ^ node[0]).bit_count()
            if distance == 0:
                node[1].append(index)
                return
            if distance not in node[2]:
                node[2][distance] = [value, [index], {}]
                return
            node = node[2][distance]

    def query(self, value, radius):
        stack = [self.root] if self.root else []
        while stack:
            node = stack.pop()
            distance = (value ^ node[0]).bit_count()
            if distance <= radius:
                yield from node[1]
            stack.extend(
                child
                for edge, child in node[2].items()
                if distance - radius <= edge <= distance + radius
            )


def deduplicate(images, rows):
    """Remove label conflicts and duplicate clusters, preferring test then val.

    Near duplicates require dHash distance <=2 AND mean pixel error <=8/255.
    This conservative pixel heuristic does not identify repeated subjects.
    """
    parents = list(range(len(rows)))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    tree = BKTree()
    exact = {}
    for i, (image, row) in enumerate(zip(images, rows)):
        digest = row["sha256"]
        if digest in exact:
            parents[find(i)] = find(exact[digest])
        else:
            exact[digest] = i
        key = dhash(image)
        for j in tree.query(key, 2):
            if np.abs(image.astype(np.int16) - images[j].astype(np.int16)).mean() <= 8:
                parents[find(i)] = find(j)
        tree.insert(key, i)
    groups = defaultdict(list)
    for i in range(len(rows)):
        groups[find(i)].append(i)
    priority = {"test": 0, "val": 1, "train": 2}
    retained, removed = [], []
    for members in groups.values():
        labels = {rows[i]["label"] for i in members}
        if len(labels) > 1:
            removed.extend(dict(rows[i], reason="conflicting_cluster") for i in members)
            continue
        keep = min(members, key=lambda i: (priority[rows[i]["split"]], rows[i]["id"]))
        retained.append(keep)
        removed.extend(
            dict(rows[i], reason="duplicate_cluster", kept_id=rows[keep]["id"])
            for i in members
            if i != keep
        )
    retained.sort()
    return retained, removed


def prepare(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "dataset.npz").exists():
        raise FileExistsError(
            "Prepared dataset exists; choose a new output or reuse it."
        )
    rows, images, sources = [], [], []
    for source_split, split in [("train", "train"), ("valid", "val"), ("test", "test")]:
        filename = f"data/{source_split}-00000-of-00001.parquet"
        target = output / "raw" / Path(filename).name
        download(
            f"https://huggingface.co/datasets/{SOURCE}/resolve/{REVISION}/{filename}",
            target,
        )
        if sha256(target) != SOURCE_SHA256[source_split]:
            raise ValueError(f"Source checksum mismatch: {target}")
        table = pq.read_table(target)
        print(source_split, table.num_rows, "source rows", flush=True)
        sources.append(
            {"file": filename, "sha256": sha256(target), "rows": table.num_rows}
        )
        for i, record in enumerate(table.to_pylist()):
            if record["label"] not in LABELS:
                continue
            image = np.asarray(
                Image.open(BytesIO(record["image"]["bytes"])).convert("L")
            )
            if image.shape != (48, 48):
                raise ValueError(f"Unexpected source shape: {image.shape}")
            digest = hashlib.sha256(image.tobytes()).hexdigest()
            rows.append(
                {
                    "id": f"fer2013/{source_split}/{i:05d}",
                    "source_row": i,
                    "source_split": source_split,
                    "split": split,
                    "label": LABELS[record["label"]],
                    "sha256": digest,
                }
            )
            images.append(image)
    images = np.stack(images)
    retained, removed = deduplicate(images, rows)
    manifest = [rows[i] for i in retained]
    values = {}
    for split in ["train", "val", "test"]:
        ids = [i for i in retained if rows[i]["split"] == split]
        values[f"{split}_images"] = images[ids]
        values[f"{split}_labels"] = np.array([rows[i]["label"] for i in ids], np.int64)
        values[f"{split}_ids"] = np.array([rows[i]["id"] for i in ids])
    np.savez_compressed(output / "dataset.npz", **values)
    write_json(output / "manifest.json", manifest)
    write_json(output / "removed.json", removed)
    audit = {
        "source": SOURCE,
        "revision": REVISION,
        "source_files": sources,
        "source_label_mapping": {"3": "happy=0", "4": "sad=1"},
        "selected_before_audit": len(rows),
        "retained": len(retained),
        "removed": len(removed),
        "removed_reasons": dict(Counter(r["reason"] for r in removed)),
        "counts": {
            split: dict(Counter(CLASS_NAMES[int(v)] for v in values[f"{split}_labels"]))
            for split in ["train", "val", "test"]
        },
        "dataset_sha256": sha256(output / "dataset.npz"),
        "manifest_sha256": sha256(output / "manifest.json"),
        "deduplication": "dHash<=2 AND mean pixel error<=8; conflicting clusters excluded; test>val>train",
    }
    write_json(output / "audit.json", audit)
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/fer2013-v2"))
    prepare(parser.parse_args().output)
