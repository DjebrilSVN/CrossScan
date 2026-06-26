"""
3D Cross-Modality Volumetric Translation Engine (CT <-> MRI).

Implements a 3D Residual translation network for synthesizing pseudo-MRI from CT volumes
(or pseudo-CT from MRI), establishing the synthetic translation benchmark arm
to compare against native physics domain randomization.
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock3D(nn.Module):
    """3D Residual Block with Instance Normalization and LeakyReLU."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(channels),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv(x)


class CrossModality3DTranslator(nn.Module):
    """
    3D Translation Generator mapping CT <-> MRI volumes.
    Preserves anatomical geometry while transforming radiodensity into magnetic resonance contrast.
    """
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_features: int = 32,
        num_res_blocks: int = 4
    ):
        super().__init__()
        # Initial projection
        self.in_conv = nn.Sequential(
            nn.Conv3d(in_channels, base_features, kernel_size=3, padding=1),
            nn.InstanceNorm3d(base_features),
            nn.LeakyReLU(0.01, inplace=True)
        )
        
        # Downsampling
        self.down1 = nn.Sequential(
            nn.Conv3d(base_features, base_features * 2, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm3d(base_features * 2),
            nn.LeakyReLU(0.01, inplace=True)
        )
        
        # Residual Bottleneck
        res_blocks = [ResidualBlock3D(base_features * 2) for _ in range(num_res_blocks)]
        self.bottleneck = nn.Sequential(*res_blocks)
        
        # Upsampling
        self.up1 = nn.Sequential(
            nn.ConvTranspose3d(base_features * 2, base_features, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm3d(base_features),
            nn.LeakyReLU(0.01, inplace=True)
        )
        
        # Output reconstruction
        self.out_conv = nn.Sequential(
            nn.Conv3d(base_features, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid()  # Normalized radiometry [0, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0 = self.in_conv(x)
        h1 = self.down1(h0)
        bot = self.bottleneck(h1)
        up = self.up1(bot)
        
        # Handle odd spatial dimensions if needed
        if up.shape[2:] != x.shape[2:]:
            up = F.interpolate(up, size=x.shape[2:], mode="trilinear", align_corners=False)
            
        out = self.out_conv(up)
        return out

