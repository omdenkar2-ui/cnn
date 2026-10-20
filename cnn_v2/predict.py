"""Predict a cropped face with the released v2 CNN. No network access needed."""

from io import BytesIO
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
from PIL import Image, ImageOps
import torch
from .model import build_model, preprocess, ARCHITECTURE

DEFAULT_RUN = Path(__file__).resolve().parents[1] / "artifacts/v2"


def decode_image(source):
    if isinstance(source, bytes):
        source = BytesIO(source)
    with Image.open(source) as image:
        if image.width * image.height > 20_000_000:
            raise ValueError("Please use an image smaller than 20 megapixels.")
        if getattr(image, "n_frames", 1) != 1:
            raise ValueError("Please use a static image, not an animation.")
        return ImageOps.exif_transpose(image).convert("L").copy()


class Predictor:
    def __init__(self, run=DEFAULT_RUN, device="cpu"):
        run = Path(run)
        self.metadata = json.loads((run / "metadata.json").read_text(encoding="utf-8"))
        weights = run / "best.pt"
        actual = hashlib.sha256(weights.read_bytes()).hexdigest()
        if actual != self.metadata["checkpoint_sha256"]:
            raise ValueError("Checkpoint checksum does not match metadata.")
        checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
        if checkpoint["architecture"] != ARCHITECTURE:
            raise ValueError("Unsupported checkpoint architecture.")
        self.device = torch.device(device)
        self.model = build_model(pretrained=False).to(self.device).eval()
        self.model.load_state_dict(checkpoint["state_dict"])

    @torch.inference_mode()
    def predict(self, source):
        image = decode_image(source)
        tensor = torch.from_numpy(np.asarray(image).copy())[None, None].to(self.device)
        logits = self.model(preprocess(tensor)) / self.metadata["temperature"]
        scores = logits.softmax(1)[0].cpu().tolist()
        index = int(scores[1] > 0.5)
        return {
            "label": self.metadata["class_names"][index],
            "scores": dict(zip(self.metadata["class_names"], scores)),
            "confidence": scores[index],
            "uncertain": scores[index] < 0.65,
            "model": "MobileNetV3 / v2",
            "input": "cropped face",
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(4)
    print(json.dumps(Predictor(args.run, args.device).predict(args.image), indent=2))


if __name__ == "__main__":
    main()
