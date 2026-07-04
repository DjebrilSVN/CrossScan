"""
Uncertainty Quantification Engine: Test-Time Augmentation (TTA), Voxel Entropy, and Calibration.

Enables risk estimation, spatial error localization, and confidence calibration
without ground truth annotations.
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
        (B, D, H, W) voxel entropy map in natural units (nats).
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
        if isinstance(out, (list, tuple)):
            out = out[0]
        if torch.is_tensor(out) and out.ndim == 6:
            out = out[:, 0]
            
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
            
        # Create boundary ribbon: dilated XOR eroded
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


def compute_expected_calibration_error(
    probabilities: np.ndarray,
    ground_truth: np.ndarray,
    num_bins: int = 15
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Computes Expected Calibration Error (ECE) and Reliability Diagram data.
    
    ECE = sum_{m=1}^M (|B_m| / N) * |acc(B_m) - conf(B_m)|
    
    Args:
        probabilities: (N,) predicted confidence probabilities in [0, 1].
        ground_truth: (N,) binary correctness labels {0, 1}.
        num_bins: Number of confidence bins.
        
    Returns:
        Dict containing ece, mce (maximum calibration error), bin_confs, bin_accs.
    """
    assert len(probabilities) == len(ground_truth)
    N = len(probabilities)
    if N == 0:
        return {"ece": 0.0, "mce": 0.0, "bin_confs": np.array([]), "bin_accs": np.array([])}
        
    bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)
    bin_confs = []
    bin_accs = []
    bin_weights = []
    
    ece = 0.0
    mce = 0.0
    
    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        if i == num_bins - 1:
            in_bin = (probabilities >= bin_lower) & (probabilities <= bin_upper)
        else:
            in_bin = (probabilities >= bin_lower) & (probabilities < bin_upper)
            
        bin_size = np.sum(in_bin)
        if bin_size > 0:
            avg_conf = np.mean(probabilities[in_bin])
            avg_acc = np.mean(ground_truth[in_bin])
            abs_err = abs(avg_acc - avg_conf)
            
            ece += (bin_size / N) * abs_err
            mce = max(mce, abs_err)
            
            bin_confs.append(float(avg_conf))
            bin_accs.append(float(avg_acc))
            bin_weights.append(int(bin_size))
        else:
            bin_confs.append(float((bin_lower + bin_upper) / 2.0))
            bin_accs.append(float((bin_lower + bin_upper) / 2.0))
            bin_weights.append(0)
            
    return {
        "ece": float(ece),
        "mce": float(mce),
        "bin_confs": np.array(bin_confs),
        "bin_accs": np.array(bin_accs),
        "bin_weights": np.array(bin_weights)
    }
