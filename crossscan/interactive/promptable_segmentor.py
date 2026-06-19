"""
Interactive Promptable 3D Segmentation Architecture for Clinical Human-in-the-Loop Refinement.

Encodes positive and negative radiologist click prompts as 3D Gaussian distance maps,
fusing user interactions with prior segmentations to produce rapid real-time boundary corrections.
"""

from typing import Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from crossscan.core.ontology import NUM_CLASSES
from crossscan.evaluation.metrics import compute_dice_score


def generate_click_heatmap_3d(
    click_coords: List[Tuple[int, int, int]],
    spatial_shape: Tuple[int, int, int],
    sigma: float = 3.0,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Generates a continuous 3D Gaussian distance heatmap from discrete (z, y, x) click coordinates.
    
    Args:
        click_coords: List of (z, y, x) voxel tuples.
        spatial_shape: Target 3D volume dimensions (D, H, W).
        sigma: Standard deviation of the 3D Gaussian kernel.
        device: Target compute device.
        
    Returns:
        torch.Tensor of shape (1, 1, D, H, W) in [0, 1].
    """
    d, h, w = spatial_shape
    heatmap = torch.zeros((1, 1, d, h, w), dtype=torch.float32, device=device)
    
    if not click_coords:
        return heatmap
        
    for z0, y0, x0 in click_coords:
        # Define local bounding box around click to avoid calculating across full volume
        r = int(np.ceil(3.0 * sigma))
        z_min, z_max = max(0, z0 - r), min(d, z0 + r + 1)
        y_min, y_max = max(0, y0 - r), min(h, y0 + r + 1)
        x_min, x_max = max(0, x0 - r), min(w, x0 + r + 1)
        
        zz, yy, xx = torch.meshgrid(
            torch.arange(z_min, z_max, device=device),
            torch.arange(y_min, y_max, device=device),
            torch.arange(x_min, x_max, device=device),
            indexing="ij"
        )
        
        dist_sq = (zz - z0) ** 2 + (yy - y0) ** 2 + (xx - x0) ** 2
        gaussian_patch = torch.exp(-dist_sq / (2.0 * sigma ** 2))
        
        # Max-blend multiple clicks
        heatmap[0, 0, z_min:z_max, y_min:y_max, x_min:x_max] = torch.maximum(
            heatmap[0, 0, z_min:z_max, y_min:y_max, x_min:x_max],
            gaussian_patch
        )
        
    return heatmap


class Promptable3DRefinementNet(nn.Module):
    """
    Interactive 3D Refinement Network.
    
    Inputs:
    - 3D Volumetric Image: (B, 1, D, H, W)
    - Prior Segmentation Mask: (B, 1, D, H, W)
    - Positive Click Heatmap: (B, 1, D, H, W)
    - Negative Click Heatmap: (B, 1, D, H, W)
    Total channels = 4.
    
    Outputs:
    - Refined Organ Logits: (B, 1, D, H, W) or (B, num_classes, D, H, W)
    """
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 1,
        feature_size: int = 16
    ):
        super().__init__()
        self.conv_in = nn.Sequential(
            nn.Conv3d(in_channels, feature_size, kernel_size=3, padding=1),
            nn.InstanceNorm3d(feature_size),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )
        
        self.block1 = nn.Sequential(
            nn.Conv3d(feature_size, feature_size * 2, kernel_size=3, padding=1),
            nn.InstanceNorm3d(feature_size * 2),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )
        
        self.block2 = nn.Sequential(
            nn.Conv3d(feature_size * 2, feature_size, kernel_size=3, padding=1),
            nn.InstanceNorm3d(feature_size),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )
        
        self.conv_out = nn.Conv3d(feature_size, out_channels, kernel_size=1)

    def forward(
        self,
        image: torch.Tensor,
        prior_mask: torch.Tensor,
        pos_clicks: torch.Tensor,
        neg_clicks: torch.Tensor
    ) -> torch.Tensor:
        """
        Concatenates image and interactive prompt channels and computes delta refinement.
        """
        x = torch.cat([image, prior_mask, pos_clicks, neg_clicks], dim=1)
        h = self.conv_in(x)
        h = self.block1(h)
        h = self.block2(h)
        residual = self.conv_out(h)
        # Add residual to prior mask logits
        return prior_mask + residual


class InteractiveRefinementEngine:
    """
    High-level interactive controller simulating radiologist correction workflows.
    """
    def __init__(
        self,
        model: Optional[Promptable3DRefinementNet] = None,
        sigma: float = 3.0,
        device: Optional[torch.device] = None
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model or Promptable3DRefinementNet().to(self.device)
        self.model.eval()
        self.sigma = sigma

    @torch.no_grad()
    def refine_step(
        self,
        image: torch.Tensor,
        current_mask: torch.Tensor,
        pos_coords: List[Tuple[int, int, int]],
        neg_coords: List[Tuple[int, int, int]]
    ) -> torch.Tensor:
        """
        Executes a single interactive refinement step given accumulated positive/negative clicks.
        """
        spatial_shape = image.shape[2:]
        pos_map = generate_click_heatmap_3d(pos_coords, spatial_shape, sigma=self.sigma, device=self.device)
        neg_map = generate_click_heatmap_3d(neg_coords, spatial_shape, sigma=self.sigma, device=self.device)
        
        if image.device != self.device:
            image = image.to(self.device)
        if current_mask.device != self.device:
            current_mask = current_mask.to(self.device)
            
        refined_logits = self.model(image, current_mask, pos_map, neg_map)
        return torch.sigmoid(refined_logits)

