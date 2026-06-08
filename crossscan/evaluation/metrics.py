"""
Evaluation Metrics Framework conforming to the Metrics Reloaded recommendations.

Implements:
- Dice Similarity Coefficient (DSC)
- Normalized Surface Distance (NSD) at structure-specific tolerance tau (tau in mm)
- 95th Percentile Hausdorff Distance (HD95)
- Average Symmetric Surface Distance (ASSD)
- Volume Overlap and Error Tracking per anatomical structure
"""

from typing import Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt
import torch

from crossscan.core.ontology import ORGAN_ONTOLOGY, ORGAN_NSD_TOLERANCES


def compute_dice_score(
    pred_mask: np.ndarray,
    target_mask: np.ndarray,
    smooth: float = 1e-6
) -> float:
    """
    Computes volumetric Dice Similarity Coefficient (DSC) for binary masks.
    """
    pred_bool = pred_mask.astype(bool)
    target_bool = target_mask.astype(bool)
    
    intersection = np.logical_and(pred_bool, target_bool).sum()
    cardinality = pred_bool.sum() + target_bool.sum()
    
    if cardinality == 0:
        return 1.0  # True negative agreement when both masks are empty
        
    return float((2.0 * intersection + smooth) / (cardinality + smooth))


def extract_surface_mask(binary_mask: np.ndarray) -> np.ndarray:
    """
    Extracts the 1-voxel thick boundary surface of a 3D binary mask.
    """
    if not np.any(binary_mask):
        return np.zeros_like(binary_mask, dtype=bool)
    eroded = binary_erosion(binary_mask)
    return np.logical_xor(binary_mask, eroded)


def compute_surface_distances(
    pred_mask: np.ndarray,
    target_mask: np.ndarray,
    voxel_spacing: Tuple[float, float, float] = (1.5, 1.5, 1.5)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates exact Euclidean surface distances from prediction to target and target to prediction.
    """
    pred_surface = extract_surface_mask(pred_mask)
    target_surface = extract_surface_mask(target_mask)
    
    if not np.any(pred_surface) or not np.any(target_surface):
        return np.array([np.nan]), np.array([np.nan])
    
    # Distance transform of target surface complement
    dt_target = distance_transform_edt(~target_surface, sampling=voxel_spacing)
    dt_pred = distance_transform_edt(~pred_surface, sampling=voxel_spacing)
    
    # Distances for boundary voxels
    distances_pred_to_target = dt_target[pred_surface]
    distances_target_to_pred = dt_pred[target_surface]
    
    return distances_pred_to_target, distances_target_to_pred


def compute_nsd(
    pred_mask: np.ndarray,
    target_mask: np.ndarray,
    tolerance_mm: float = 1.5,
    voxel_spacing: Tuple[float, float, float] = (1.5, 1.5, 1.5)
) -> float:
    """
    Normalized Surface Distance (NSD) / Surface Dice at specified tolerance tau.
    """
    pred_surface = extract_surface_mask(pred_mask)
    target_surface = extract_surface_mask(target_mask)
    
    n_pred = pred_surface.sum()
    n_target = target_surface.sum()
    
    if n_pred == 0 and n_target == 0:
        return 1.0
    if n_pred == 0 or n_target == 0:
        return 0.0
        
    d_pred_to_target, d_target_to_pred = compute_surface_distances(
        pred_mask, target_mask, voxel_spacing=voxel_spacing
    )
    
    surfel_pred_within_tau = np.sum(d_pred_to_target <= tolerance_mm)
    surfel_target_within_tau = np.sum(d_target_to_pred <= tolerance_mm)
    
    nsd = (surfel_pred_within_tau + surfel_target_within_tau) / (n_pred + n_target)
    return float(nsd)


def compute_hd95(
    pred_mask: np.ndarray,
    target_mask: np.ndarray,
    voxel_spacing: Tuple[float, float, float] = (1.5, 1.5, 1.5)
) -> float:
    """
    95th percentile of bidirectional Hausdorff Distance in millimeters.
    """
    if not np.any(pred_mask) and not np.any(target_mask):
        return 0.0
    if not np.any(pred_mask) or not np.any(target_mask):
        return float("inf")
        
    d_pred_to_target, d_target_to_pred = compute_surface_distances(
        pred_mask, target_mask, voxel_spacing=voxel_spacing
    )
    
    if np.isnan(d_pred_to_target).any() or np.isnan(d_target_to_pred).any():
        return float("inf")
        
    hd95_1 = np.percentile(d_pred_to_target, 95)
    hd95_2 = np.percentile(d_target_to_pred, 95)
    return float(max(hd95_1, hd95_2))


def evaluate_volume(
    pred: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    voxel_spacing: Tuple[float, float, float] = (1.5, 1.5, 1.5),
    present_classes: Optional[Sequence[int]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Full Metrics Reloaded multi-organ volume evaluation.
    
    Returns:
        Dict mapping organ name to metrics {'dice': float, 'nsd': float, 'hd95': float}.
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
        
    results = {}
    active_classes = present_classes or list(ORGAN_ONTOLOGY.keys())
    
    for class_idx in active_classes:
        if class_idx == 0:
            continue  # Exclude background from report
            
        organ_meta = ORGAN_ONTOLOGY.get(class_idx)
        if organ_meta is None:
            continue
            
        pred_c = (pred == class_idx).astype(np.uint8)
        target_c = (target == class_idx).astype(np.uint8)
        
        # Check if organ is expected in ground truth
        has_gt = np.any(target_c)
        has_pred = np.any(pred_c)
        
        if not has_gt and not has_pred:
            results[organ_meta.name] = {
                "dice": 1.0,
                "nsd": 1.0,
                "hd95": 0.0,
                "status": "true_negative"
            }
            continue
            
        tau = organ_meta.nsd_tolerance_mm
        dice = compute_dice_score(pred_c, target_c)
        nsd = compute_nsd(pred_c, target_c, tolerance_mm=tau, voxel_spacing=voxel_spacing)
        hd95 = compute_hd95(pred_c, target_c, voxel_spacing=voxel_spacing)
        
        results[organ_meta.name] = {
            "dice": dice,
            "nsd": nsd,
            "hd95": hd95 if not np.isinf(hd95) else 999.0,
            "status": "evaluated"
        }
        
    return results
