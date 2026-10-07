"""One image decoding contract shared by training and prediction."""
from pathlib import Path
import hashlib
import json

import numpy as np
from PIL import Image, ImageOps


def read_rgb(path, size):
    with Image.open(path) as source:
        if getattr(source, 'n_frames', 1) != 1:
            raise ValueError('Animated images are unsupported')
        image = ImageOps.exif_transpose(source).convert('RGB')
        image = image.resize((size, size), Image.Resampling.BILINEAR)
        return np.asarray(image, dtype=np.uint8).copy()


def difference_hash(image):
    gray = Image.fromarray(image).convert('L').resize((9, 8))
    pixels = np.asarray(gray)
    bits = (pixels[:, 1:] > pixels[:, :-1]).ravel()
    return sum(int(bit) << i for i, bit in enumerate(bits))


def scan_dataset(data_dir, size, conflict_policy='error'):
    if conflict_policy not in {'error','exclude'}:
        raise ValueError('conflict_policy must be error or exclude')
    root = Path(data_dir).resolve()
    classes = sorted(p.name for p in root.iterdir() if p.is_dir())
    if len(classes) != 2:
        raise ValueError('Expected exactly two class folders in the data directory')
    rows, images, seen, skipped, duplicates = [], [], {}, [], []
    conflicts, excluded = [], set()
    for label, name in enumerate(classes):
        for path in sorted((root / name).rglob('*')):
            if not path.is_file():
                continue
            try:
                image = read_rgb(path, size)
            except (OSError, ValueError) as error:
                skipped.append({'path':str(path.relative_to(root)), 'reason':str(error)})
                continue
            digest = hashlib.sha256(image.tobytes()).hexdigest()
            relative = path.relative_to(root).as_posix()
            if digest in seen:
                prior = seen[digest]
                if prior['label'] != label:
                    if conflict_policy == 'error':
                        raise ValueError(f'Conflicting labels for identical inputs: {relative}')
                    excluded.add(digest)
                    conflicts.append({'path':relative,'other_path':prior['path'],
                                      'input_sha256':digest})
                    continue
                duplicates.append({'path':relative,'same_as':prior['path']})
                continue
            row = {'path':relative, 'label':label, 'input_sha256':digest,
                   'dhash':f'{difference_hash(image):016x}'}
            rows.append(row); images.append(image); seen[digest] = row
    filtered = [(row,image) for row,image in zip(rows,images)
                if row['input_sha256'] not in excluded]
    if not filtered:
        raise ValueError('No readable images found')
    rows, images = zip(*filtered)
    return np.stack(images), list(rows), classes, {
        'skipped':skipped,'duplicates':duplicates,'conflicting_inputs':conflicts,
        'conflict_policy':conflict_policy}


def split_rows(rows, seed=42, near_distance=3):
    """Keep connected near-duplicate clusters in a single partition.

    A dHash heuristic is not a substitute for subject/source group metadata.
    """
    if not 0 <= near_distance <= 64:
        raise ValueError('near_distance must be between 0 and 64')
    parent = list(range(len(rows)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    hashes = [int(r['dhash'],16) for r in rows]
    for i in range(len(rows)):
        for j in range(i):
            if (hashes[i] ^ hashes[j]).bit_count() <= near_distance:
                parent[find(i)] = find(j)
    groups = {}
    for i in range(len(rows)):
        groups.setdefault(find(i),[]).append(i)
    rng = np.random.default_rng(seed)
    grouped = list(groups.values())
    rng.shuffle(grouped)
    grouped.sort(key=len,reverse=True)
    totals = np.bincount([r['label'] for r in rows],minlength=2)
    if np.any(totals < 10) or len(grouped) < 3:
        raise ValueError('Need >=10 unique inputs per class and >=3 independent clusters')
    targets = np.array([.7,.2,.1])[:,None] * totals[None,:]
    counts = np.zeros((3,2),dtype=int)
    assignments = {}
    for group_id, group in enumerate(grouped):
        addition = np.bincount([rows[i]['label'] for i in group],minlength=2)
        costs = []
        for split in range(3):
            candidate = counts.copy(); candidate[split] += addition
            costs.append(np.sum((candidate-targets)**2 / np.maximum(targets,1)))
        chosen = int(np.argmin(costs)); counts[chosen] += addition
        for i in group:
            assignments[i] = (['train','val','test'][chosen],group_id)
    for split in range(3):
        if np.any(counts[split] == 0):
            raise ValueError('Cannot populate every split/class; inspect near-duplicate groups')
    result = [dict(row,split=assignments[i][0],group=assignments[i][1])
              for i,row in enumerate(rows)]
    return result


def save_json(path, value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
