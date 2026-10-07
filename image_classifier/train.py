import argparse
from collections import Counter
from pathlib import Path
import platform
import time

import numpy as np
import tensorflow as tf
from .data import scan_dataset, split_rows, save_json
from .model import build_model


def binary_metrics(labels, scores, threshold=.5):
    labels = np.asarray(labels).ravel().astype(int)
    decisions = np.asarray(scores).ravel() > threshold
    tp = int(np.sum((labels == 1) & decisions))
    tn = int(np.sum((labels == 0) & ~decisions))
    fp = int(np.sum((labels == 0) & decisions))
    fn = int(np.sum((labels == 1) & ~decisions))
    def ratio(a,b):
        return float(a/b) if b else None
    return {'TP':tp,'TN':tn,'FP':fp,'FN':fn,
            'accuracy':ratio(tp+tn,len(labels)), 'precision':ratio(tp,tp+fp),
            'recall':ratio(tp,tp+fn), 'f1':ratio(2*tp,2*tp+fp+fn)}


def make_dataset(images, labels, indices, batch, training, seed):
    ds = tf.data.Dataset.from_tensor_slices((images[indices],labels[indices,None]))
    if training:
        ds = ds.shuffle(len(indices),seed=seed,reshuffle_each_iteration=True)
    options = tf.data.Options()
    options.threading.private_threadpool_size = 2
    ds = ds.with_options(options)
    return ds.batch(batch).prefetch(1)


def plot_results(history, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,2,figsize=(10,4))
    for axis,key in zip(axes,['loss','accuracy']):
        axis.plot(np.arange(1,len(history[key])+1),history[key],label='Training')
        axis.plot(np.arange(1,len(history[key])+1),history['val_'+key],label='Validation')
        axis.set(xlabel='Epoch',ylabel=key.title()); axis.legend(); axis.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(destination,dpi=160); plt.close(fig)


def train(data, output, epochs=30, size=128, batch=16, seed=42, near_distance=3):
    if epochs < 1 or batch < 1:
        raise ValueError('epochs and batch must be positive')
    if size < 32:
        raise ValueError('Image size must be at least 32')
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Choose a fresh output directory: {output}')
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    images, rows, classes, audit = scan_dataset(data,size,conflict_policy='exclude')
    manifest = split_rows(rows,seed,near_distance)
    indices = {name:np.array([i for i,r in enumerate(manifest) if r['split']==name])
               for name in ['train','val','test']}
    labels = np.array([r['label'] for r in manifest],dtype=np.float32)
    output.mkdir(parents=True)
    save_json(output/'manifest.json',manifest)
    save_json(output/'data_audit.json',audit)
    split_counts = {name:dict(Counter(classes[int(labels[i])] for i in ix))
                    for name,ix in indices.items()}
    print('Class mapping:',classes,flush=True)
    print('Split counts:',split_counts,flush=True)
    datasets = {name:make_dataset(images,labels,ix,batch,name=='train',seed)
                for name,ix in indices.items()}
    model = build_model(size,seed)
    model.summary()
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=7),
                 tf.keras.callbacks.ModelCheckpoint(str(output/'best.keras'),
                     monitor='val_loss',save_best_only=True)]
    started = time.monotonic()
    history = model.fit(datasets['train'],validation_data=datasets['val'],
                        epochs=epochs,callbacks=callbacks,verbose=2,shuffle=False)
    save_json(output/'history.json',history.history)
    best = tf.keras.models.load_model(output/'best.keras')
    scores = best.predict(datasets['test'],verbose=0).ravel()
    test_labels = labels[indices['test']]
    metrics = binary_metrics(test_labels,scores)
    majority = Counter(labels[indices['train']].astype(int)).most_common(1)[0][0]
    metrics['majority_baseline_accuracy'] = float(np.mean(test_labels == majority))
    metrics['test_examples'] = len(test_labels)
    metrics['confusion_matrix'] = [[metrics['TN'],metrics['FP']],[metrics['FN'],metrics['TP']]]
    save_json(output/'metrics.json',metrics)
    best_epoch = int(np.argmin(history.history['val_loss']))+1
    metadata = {'class_names':classes,'positive_class':classes[1],
                'threshold':.5,'decision_rule':'score > threshold',
                'image_size':size,'input_dtype':'uint8','color_mode':'RGB',
                'resize':'Pillow bilinear','exif_orientation':'transpose',
                'normalization':'Rescaling(1/255) inside model',
                'seed':seed,'batch_size':batch,'max_epochs':epochs,
                'epochs_completed':len(history.history['loss']),'best_epoch':best_epoch,
                'parameters':best.count_params(),'near_duplicate_distance':near_distance,
                'split_counts':split_counts,'python':platform.python_version(),
                'tensorflow':tf.__version__,'numpy':np.__version__,
                'training_seconds':round(time.monotonic()-started,2)}
    save_json(output/'metadata.json',metadata)
    save_json(output/'test_predictions.json',[
        dict(manifest[int(i)],score=float(p),prediction=classes[int(p>.5)])
        for i,p in zip(indices['test'],scores)])
    probe = images[indices['val'][:3]]
    np.testing.assert_allclose(best(probe,training=False).numpy(),
        tf.keras.models.load_model(output/'best.keras')(probe,training=False).numpy(),
        rtol=1e-5,atol=1e-6)
    plot_results(history.history,output/'learning_curves.png')
    print('Final test metrics:',metrics,flush=True)
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Train a reference-style binary CNN')
    parser.add_argument('--data',type=Path,default=Path('data'))
    parser.add_argument('--output',type=Path,default=Path('runs/experiment-01'))
    parser.add_argument('--epochs',type=int,default=30)
    parser.add_argument('--size',type=int,default=128)
    parser.add_argument('--batch',type=int,default=16)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--near-distance',type=int,default=3)
    args = parser.parse_args()
    train(args.data,args.output,args.epochs,args.size,args.batch,args.seed,args.near_distance)


if __name__ == '__main__':
    main()
