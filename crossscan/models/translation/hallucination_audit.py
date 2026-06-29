"""
Clinical Anatomical Hallucination Audit Module for Cross-Modality Translation.

Detects:
1. Geometric Volume Dissolution: |Delta V| / V per organ
2. Synthetic Boundary Distortion: Hausdorff 95 surface deviation
3. Pseudo-Structure Fabrication: artificial lesion/mass appearance
"""

from typing import Dict, Any, List, Optional
import numpy as np
from skimage.metrics import structural_similarity as ssim


def compute_normalized_mutual_information(img1: np.ndarray, img2: np.ndarray, bins: int = 64) -> float:
    """Computes Normalized Mutual Information (NMI) between two 3D scalar volumes."""
    hist_2d, _, _ = np.histogram2d(img1.ravel(), img2.ravel(), bins=bins)
    pxy = hist_2d / float(np.sum(hist_2d))
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    
    px = px[px > 0]
    py = py[py > 0]
    pxy = pxy[pxy > 0]
    
    hx = -np.sum(px * np.log2(px))
    hy = -np.sum(py * np.log2(py))
    hxy = -np.sum(pxy * np.log2(pxy))
    
    if (hx + hy) == 0:
        return 0.0
    return float(2.0 * (hx + hy - hxy) / (hx + hy))


def compute_slice_averaged_ssim(vol1: np.ndarray, vol2: np.ndarray) -> float:
    """Computes axial slice-averaged 2D Structural Similarity Index (SSIM)."""
    scores = []
    for s in range(vol1.shape[0]):
        slice1 = vol1[s]
        slice2 = vol2[s]
        data_range = max(float(slice1.max() - slice1.min()), 1e-4)
        score = ssim(slice1, slice2, data_range=data_range)
        scores.append(score)
    return float(np.mean(scores))
