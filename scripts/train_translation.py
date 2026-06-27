"""
3D Volumetric Modality Translation Trainer (CT -> pseudo-MRI).

Optimizes 3D generative translation network with combined L1 reconstruction
and 3D spatial gradient consistency loss to ensure structural edge alignment.
"""

from pathlib import Path
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_3d_gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Measures spatial gradient edge consistency in 3D: ||grad(pred) - grad(target)||_1."""
    dz_p = pred[:, :, 1:, :, :] - pred[:, :, :-1, :, :]
    dz_t = target[:, :, 1:, :, :] - target[:, :, :-1, :, :]
    dy_p = pred[:, :, :, 1:, :] - pred[:, :, :, :-1, :]
    dy_t = target[:, :, :, 1:, :] - target[:, :, :, :-1, :]
    dx_p = pred[:, :, :, :, 1:] - pred[:, :, :, :, :-1]
    dx_t = target[:, :, :, :, 1:] - target[:, :, :, :, :-1]
    
    loss_z = F.l1_loss(dz_p, dz_t)
    loss_y = F.l1_loss(dy_p, dy_t)
    loss_x = F.l1_loss(dx_p, dx_t)
    return (loss_z + loss_y + loss_x) / 3.0
