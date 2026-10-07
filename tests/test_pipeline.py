from pathlib import Path
import json
import numpy as np
import pytest
from PIL import Image
from image_classifier.data import read_rgb, scan_dataset, split_rows
from image_classifier.train import binary_metrics, train
from image_classifier.predict import predict
from image_classifier.model import build_model


def test_rgb_and_orientation(tmp_path):
    image = Image.new('RGB',(10,20),(255,0,0))
    image.save(tmp_path/'red.png')
    decoded = read_rgb(tmp_path/'red.png',32)
    assert decoded.shape == (32,32,3)
    np.testing.assert_array_equal(decoded[0,0],[255,0,0])
    assert decoded.dtype == np.uint8


def test_duplicate_conflicting_labels_rejected(tmp_path):
    for name in ['happy','sad']:
        (tmp_path/name).mkdir()
        Image.new('RGB',(20,20),'red').save(tmp_path/name/'same.png')
    with pytest.raises(ValueError,match='Conflicting labels'):
        scan_dataset(tmp_path,32)
    Image.new('RGB',(20,20),'green').save(tmp_path/'happy/unique.png')
    _,rows,_,audit = scan_dataset(tmp_path,32,conflict_policy='exclude')
    assert [r['path'] for r in rows] == ['happy/unique.png']
    assert len(audit['conflicting_inputs']) == 1


def test_corrupt_files_audited_and_duplicates_removed(tmp_path):
    for name in ['a','b']:
        (tmp_path/name).mkdir()
    Image.new('RGB',(20,20),'red').save(tmp_path/'a/first.png')
    Image.new('RGB',(20,20),'red').save(tmp_path/'a/second.png')
    Image.new('RGB',(20,20),'blue').save(tmp_path/'b/third.png')
    (tmp_path/'b/broken.jpg').write_text('broken')
    images,rows,classes,audit = scan_dataset(tmp_path,32)
    assert len(rows) == 2 and len(images) == 2
    assert classes == ['a','b']
    assert len(audit['duplicates']) == len(audit['skipped']) == 1


def test_group_membership_fixed_and_all_examples_retained():
    rng = np.random.default_rng(11)
    rows = []
    for label in range(2):
        for i in range(20):
            digest = int.from_bytes(rng.bytes(8),'little')
            rows.append({'path':f'{label}/{i}','label':label,
                         'input_sha256':f'{label}-{i}','dhash':f'{digest:016x}'})
    rows.append(dict(rows[0],path='near-copy',input_sha256='copy',
                     dhash=f"{int(rows[0]['dhash'],16)^1:016x}"))
    result = split_rows(rows,42,3)
    assert result == split_rows(rows,42,3)
    assert len(result) == len(rows)
    assert result[0]['split'] == result[-1]['split']
    for name in ['train','val','test']:
        assert {r['label'] for r in result if r['split']==name} == {0,1}
    memberships = {}
    for row in result:
        memberships.setdefault(row['group'],set()).add(row['split'])
    assert all(len(v)==1 for v in memberships.values())


def test_metrics_and_strict_threshold():
    metrics = binary_metrics([1]*18+[0]*18,[.9]*12+[.1]*6+[.9]*3+[.1]*15)
    assert [metrics[k] for k in ['TP','TN','FP','FN']] == [12,15,3,6]
    assert metrics['accuracy'] == .75
    assert metrics['f1'] == pytest.approx(24/33)
    assert binary_metrics([0],[.5])['TN'] == 1
    assert binary_metrics([0],[.1])['precision'] is None


def test_reference_architecture_shape_and_parameter_count():
    model = build_model(size=256)
    assert model.input_shape == (None,256,256,3)
    assert model.output_shape == (None,1)
    assert model.count_params() == 3696625


def test_training_save_reload_and_real_prediction(tmp_path):
    rng = np.random.default_rng(8)
    data = tmp_path/'data'
    for label,name in enumerate(['happy','sad']):
        (data/name).mkdir(parents=True)
        for i in range(12):
            image = rng.integers(0,60,(32,32,3),dtype=np.uint8)
            image[:,:,label] += 190
            Image.fromarray(image).save(data/name/f'{i}.png')
    run = tmp_path/'run'
    metrics = train(data,run,epochs=1,size=32,batch=4,near_distance=0)
    assert 0 <= metrics['accuracy'] <= 1
    assert (run/'best.keras').is_file()
    metadata = json.loads((run/'metadata.json').read_text())
    assert metadata['class_names'] == ['happy','sad']
    result = predict(data/'happy/0.png',run)
    assert result['class'] in metadata['class_names']
    assert sum(result['scores'].values()) == pytest.approx(1)
    with pytest.raises(FileExistsError):
        train(data,run,epochs=1,size=32)
