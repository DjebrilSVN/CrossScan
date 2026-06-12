"""
Uncertainty Quantification Engine: Test-Time Augmentation (TTA) and Voxel Entropy.

Enables risk estimation and confidence calibration without ground truth annotations.
"""

from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import scipy.ndimage as ndi
import torch
import torch.nn.functional as F


def compute_voxel_entropy(probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Computes Shannon predictive entropy per voxel:
    H(x) = - sum_{c=1}^C p_c(x) * log(p_c(x) + eps)
    
    Args:
        probs: (B, C, D, H, W) normalized probability tensor (softmax output).
    Returns:
        (B, D, H, W) voxel entropy map.
    """
    entropy = -torch.sum(probs * torch.log(probs + eps), dim=1)
    return entropy


@torch.no_grad()
def predict_with_tta(
    model: torch.nn.Module,
    volume: torch.Tensor,
    flip_axes: Sequence[Tuple[int, ...]] = (
        (),
        (-1,),
        (-2,),
        (-3,),
        (-1, -2),
        (-2, -3),
        (-1, -3),
        (-1, -2, -3),
    )
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Performs 8-pass spatial flip Test-Time Augmentation (TTA).
    
    Args:
        model: PyTorch segmentation model.
        volume: (1, C, D, H, W) input volume tensor.
        flip_axes: Sequence of axis tuples to flip across.
        
    Returns:
        mean_probs: (1, C, D, H, W) ensemble average prediction.
        predictive_entropy: (1, D, H, W) voxel-level epistemic/aleatoric uncertainty.
        epistemic_variance: (1, C, D, H, W) variance across TTA ensemble members.
    """
    model.eval()
    ensemble_probs: List[torch.Tensor] = []
    
    for axes in flip_axes:
        # Apply flip
        x = torch.flip(volume, dims=axes) if axes else volume
        
        # Inference
        out = model(x)
        if isinstance(out, list):
            out = out[0]
        probs = F.softmax(out, dim=1)
        
        # Invert flip
        if axes:
            probs = torch.flip(probs, dims=axes)
            
        ensemble_probs.append(probs)
        
    stacked = torch.stack(ensemble_probs, dim=0)  # (K, B, C, D, H, W)
    mean_probs = torch.mean(stacked, dim=0)
    variance = torch.var(stacked, dim=0)
    entropy = compute_voxel_entropy(mean_probs)
    
    return mean_probs, entropy, variance


def extract_boundary_uncertainty(
    pred_mask: np.ndarray,
    entropy_map: np.ndarray,
    dilation_radius_voxels: int = 2
) -> Dict[int, Dict[str, float]]:
    """
    Extracts uncertainty statistics specifically localized along the anatomical boundary ribbon.
    
    Args:
        pred_mask: 3D integer array of predicted organ labels.
        entropy_map: 3D float array of voxel predictive entropy.
        dilation_radius_voxels: Boundary ribbon thickness in voxels.
        
    Returns:
        Dict mapping class index to boundary uncertainty statistics.
    """
    struct_elem = ndi.generate_binary_structure(3, 1)
    results = {}
    
    for c in np.unique(pred_mask):
        if c == 0:
            continue
            
        mask_c = (pred_mask == c)
        if not np.any(mask_c):
            continue
            
        # Create boundary ribbon: dilated - eroded
        dilated = ndi.binary_dilation(mask_c, structure=struct_elem, iterations=dilation_radius_voxels)
        eroded = ndi.binary_erosion(mask_c, structure=struct_elem, iterations=dilation_radius_voxels)
        boundary = np.logical_xor(dilated, eroded)
        
        if not np.any(boundary):
            continue
            
        boundary_entropy = entropy_map[boundary]
        core_entropy = entropy_map[eroded] if np.any(eroded) else np.array([0.0])
        
        results[int(c)] = {
            "boundary_entropy_mean": float(np.mean(boundary_entropy)),
            "boundary_entropy_p90": float(np.percentile(boundary_entropy, 90)),
            "core_entropy_mean": float(np.mean(core_entropy)),
            "boundary_voxel_count": int(np.sum(boundary))
        }
        
    return results
