"""MobileNetV3 transfer learning and the shared tensor preprocessing contract."""

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

IMAGE_SIZE = 128
ARCHITECTURE = "mobilenet_v3_small_128"


def build_model(pretrained=False):
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    model = mobilenet_v3_small(weights=weights)
    model.classifier = nn.Sequential(
        nn.Linear(576, 128), nn.Hardswish(), nn.Dropout(0.25), nn.Linear(128, 2)
    )
    return model


def preprocess(images, augment=False):
    """NCHW uint8 RGB or grayscale -> normalized float RGB at 128x128."""
    if images.ndim != 4 or images.shape[1] not in (1, 3):
        raise ValueError("Expected a batch shaped (N,1,H,W) or (N,3,H,W)")
    images = images.float() / 255
    if images.shape[1] == 1:
        images = images.expand(-1, 3, -1, -1)
    images = F.interpolate(
        images,
        size=(IMAGE_SIZE, IMAGE_SIZE),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    if augment:
        n = images.shape[0]
        angles = (torch.rand(n, device=images.device) - 0.5) * 0.4
        scale = 0.9 + torch.rand(n, device=images.device) * 0.2
        flip = torch.where(torch.rand(n, device=images.device) > 0.5, 1.0, -1.0)
        matrix = torch.zeros(n, 2, 3, device=images.device)
        matrix[:, 0, 0] = angles.cos() * scale * flip
        matrix[:, 0, 1] = -angles.sin() * scale
        matrix[:, 1, 0] = angles.sin() * scale * flip
        matrix[:, 1, 1] = angles.cos() * scale
        matrix[:, :, 2] = (torch.rand(n, 2, device=images.device) - 0.5) * 0.12
        grid = F.affine_grid(matrix, images.shape, align_corners=False)
        images = F.grid_sample(
            images, grid, padding_mode="reflection", align_corners=False
        )
        brightness = 0.85 + torch.rand(n, 1, 1, 1, device=images.device) * 0.3
        images = (images * brightness).clamp(0, 1)
    mean = images.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
    std = images.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
    return (images - mean) / std
