"""Fine-tune MobileNetV3 using training and validation data only."""

from pathlib import Path
import argparse
import json
import random
import time
import platform
import hashlib

import numpy as np
import torch
import torchvision
from torch import nn
from .model import build_model, preprocess, ARCHITECTURE


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(4)


@torch.inference_mode()
def predict_logits(model, images, device, batch=128):
    model.eval()
    outputs = []
    for start in range(0, len(images), batch):
        outputs.append(
            model(preprocess(images[start : start + batch].to(device))).cpu()
        )
    return torch.cat(outputs)


def train(data, output, epochs=20, batch=128, seed=42, device=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Use a fresh run directory: {output}")
    if epochs < 2 or batch < 2:
        raise ValueError("epochs and batch must be >=2")
    set_seed(seed)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    # Test arrays are intentionally never loaded during model selection.
    with np.load(Path(data) / "dataset.npz", allow_pickle=False) as bundle:
        x = torch.from_numpy(bundle["train_images"][:, None].copy())
        y = torch.from_numpy(bundle["train_labels"].copy())
        xv = torch.from_numpy(bundle["val_images"][:, None].copy())
        yv = torch.from_numpy(bundle["val_labels"].copy())
    output.mkdir(parents=True)
    model = build_model(pretrained=True).to(device)
    # Warm up the new head for one epoch, then fine-tune the entire backbone.
    optimizer = torch.optim.AdamW(
        [
            {"params": model.features.parameters(), "lr": 2e-4},
            {"params": model.classifier.parameters(), "lr": 1e-3},
        ],
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.03)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history = []
    best_loss = float("inf")
    stale = 0
    started = time.monotonic()
    for epoch in range(1, epochs + 1):
        model.train()
        if epoch == 1:
            model.features.requires_grad_(False)
            model.features.eval()
        else:
            model.features.requires_grad_(True)
        order = torch.randperm(len(x))
        total_loss = 0.0
        correct = 0
        seen = 0
        for start in range(0, len(x), batch):
            ix = order[start : start + batch]
            if len(ix) == 1:  # BatchNorm needs more than one example.
                continue
            xb = x[ix].to(device)
            yb = y[ix].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(preprocess(xb, augment=True))
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach()) * len(ix)
            correct += int((logits.argmax(1) == yb).sum())
            seen += len(ix)
        scheduler.step()
        val_logits = predict_logits(model, xv, device, batch)
        val_loss = float(nn.functional.cross_entropy(val_logits, yv))
        record = {
            "epoch": epoch,
            "train_loss": total_loss / seen,
            "train_accuracy": correct / seen,
            "val_loss": val_loss,
            "val_accuracy": float((val_logits.argmax(1) == yv).float().mean()),
            "elapsed_seconds": round(time.monotonic() - started, 2),
        }
        history.append(record)
        write_json(output / "history.json", history)
        print(json.dumps(record), flush=True)
        # A full checkpoint supports continuing an experiment from this epoch.
        if val_loss < best_loss:
            best_loss = val_loss
            stale = 0
            torch.save(
                {
                    "state_dict": {
                        k: v.detach().cpu() for k, v in model.state_dict().items()
                    },
                    "architecture": ARCHITECTURE,
                    "epoch": epoch,
                },
                output / "best.pt",
            )
            torch.save(
                {
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "epoch": epoch,
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all()
                    if device.type == "cuda"
                    else [],
                },
                output / "training_state.pt",
            )
        else:
            stale += 1
        if stale >= 6:
            break
    checkpoint = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    val_logits = predict_logits(model, xv, device, batch)
    # Temperature selection uses validation only; the decision threshold stays .5.
    temperatures = torch.linspace(0.5, 3.0, 101)
    losses = torch.stack(
        [nn.functional.cross_entropy(val_logits / t, yv) for t in temperatures]
    )
    temperature = float(temperatures[losses.argmin()])
    audit = json.loads((Path(data) / "audit.json").read_text())
    metadata = {
        "architecture": ARCHITECTURE,
        "class_names": ["happy", "sad"],
        "positive_class": "sad",
        "image_size": 128,
        "temperature": temperature,
        "threshold": 0.5,
        "preprocessing": "EXIF transpose; grayscale; tensor bilinear resize; ImageNet normalization",
        "seed": seed,
        "batch_size": batch,
        "requested_epochs": epochs,
        "completed_epochs": len(history),
        "best_epoch": checkpoint["epoch"],
        "parameters": sum(p.numel() for p in model.parameters()),
        "pretrained_weights": "MobileNet_V3_Small_Weights.IMAGENET1K_V1",
        "device": str(device),
        "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
        "torch": str(torch.__version__),
        "torchvision": str(torchvision.__version__),
        "python": platform.python_version(),
        "training_seconds": round(time.monotonic() - started, 2),
        "dataset": audit,
        "checkpoint_sha256": hashlib.sha256(
            (output / "best.pt").read_bytes()
        ).hexdigest(),
        "validation_log_loss_before_calibration": float(
            nn.functional.cross_entropy(val_logits, yv)
        ),
        "validation_log_loss_after_calibration": float(losses.min()),
    }
    write_json(output / "metadata.json", metadata)
    print("Training complete. Test evaluation is a separate command.", flush=True)
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/fer2013-v2"))
    parser.add_argument("--output", type=Path, default=Path("runs/v2"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    train(args.data, args.output, args.epochs, args.batch, args.seed, args.device)
