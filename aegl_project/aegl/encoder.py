"""
Vision Encoder phi: x -> z_x  (Algorithm 1, line 1: Zx <- VisionEncoder_phi(X))

A compact convolutional backbone is used so the whole pipeline trains fast
on CPU for demonstration purposes. Swap `VisionEncoder` for a ViT / CLIP /
MAE backbone (as referenced in the literature review, [1][2]) for
production-scale experiments -- the rest of the framework only depends on
the (B, embed_dim) output contract.
"""
import torch
import torch.nn as nn


class VisionEncoder(nn.Module):
    def __init__(self, in_channels: int = 3, embed_dim: int = 128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # /2

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # /4

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(128, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x).flatten(1)
        z_x = self.proj(h)
        return z_x  # (B, embed_dim)
