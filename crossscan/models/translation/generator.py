"""3D Residual U-Net Architecture for Volumetric CT <-> MRI Translation."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock3D(nn.Module):
    def __init__(self, in_c: int, out_c: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.InstanceNorm3d(out_c, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(out_c, affine=True)
        )
        self.skip = nn.Identity() if in_c == out_c and stride == 1 else nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
            nn.InstanceNorm3d(out_c, affine=True)
        )
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(x) + self.skip(x))


class ResBlock3D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(channels, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(channels, affine=True)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)
