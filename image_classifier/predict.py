import argparse
import json
from pathlib import Path
import tensorflow as tf
from .data import read_rgb


def predict(image, run):
    run = Path(run)
    metadata = json.loads((run/'metadata.json').read_text(encoding='utf-8'))
    pixels = read_rgb(image,metadata['image_size'])
    model = tf.keras.models.load_model(run/'best.keras',compile=False)
    score = float(model(pixels[None,...],training=False).numpy()[0,0])
    classes = metadata['class_names']
    label = classes[int(score > metadata['threshold'])]
    return {'class':label,'score_for_positive_class':score,
            'positive_class':classes[1],'threshold':metadata['threshold'],
            'scores':{classes[0]:1-score,classes[1]:score}}


def main():
    parser = argparse.ArgumentParser(description='Predict one image')
    parser.add_argument('--image',type=Path,required=True)
    parser.add_argument('--run',type=Path,default=Path('artifacts/reference-run'))
    args = parser.parse_args()
    print(json.dumps(predict(args.image,args.run),indent=2))


if __name__ == '__main__':
    main()
