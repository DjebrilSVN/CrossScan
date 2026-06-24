"""
Synthetic Physical Perturbation Battery for Stress-Testing Volumetric Segmentors.

Applies calibrated, physics-derived imaging corruptions to evaluate out-of-distribution
failure modes and quantify graceful degradation:
1. Patient Motion Artifacts (k-space phase perturbation simulating respiratory/cardiac movement).
2. Radiofrequency B1 Inhomogeneity (severe coil sensitivity profiles).
3. Rician & Gaussian Thermal Noise.
4. Slice Thickness Degradation (thick-slice clinical acquisition).
5. Truncated Field of View (FOV) anatomy cutoff.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import scipy.ndimage as ndi
import torch

from crossscan.data.domain_randomization import (
    RandomRicianNoise,
    RandomB1BiasField,
    RandomMRIPhaseGhosting,
    RandomSliceThicknessDegradation
)


def apply_kspace_motion_artifacts(
    image: np.ndarray,
    num_ghosts: int = 4,
    intensity: float = 0.25
) -> np.ndarray:
    """
    Simulates respiratory motion ghosting via k-space phase modulation along phase-encode axis.
    """
    img_t = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
    ghost_op = RandomMRIPhaseGhosting(prob=1.0, num_ghosts=num_ghosts, intensity_range=(intensity, intensity))
    corrupted_t = ghost_op(img_t)
    return corrupted_t.squeeze().numpy().astype(np.float32)


def apply_b1_inhomogeneity(
    image: np.ndarray,
    magnitude: float = 0.6
) -> np.ndarray:
    """
    Simulates severe B1 transmit/receive coil sensitivity drop-off.
    """
    img_t = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
    bias_op = RandomB1BiasField(prob=1.0, magnitude_range=(magnitude, magnitude))
    corrupted_t = bias_op(img_t)
    return corrupted_t.squeeze().numpy().astype(np.float32)


def apply_rician_noise(
    image: np.ndarray,
    sigma: float = 0.08
) -> np.ndarray:
    """
    Simulates Rician noise: sqrt((I + N(0, s))^2 + N(0, s)^2).
    """
    img_t = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
    noise_op = RandomRicianNoise(prob=1.0, sigma_range=(sigma, sigma))
    corrupted_t = noise_op(img_t)
    return corrupted_t.squeeze().numpy().astype(np.float32)


def apply_slice_thickness_degradation(
    image: np.ndarray,
    downsample_factor: int = 3
) -> np.ndarray:
    """
    Simulates thick-slice clinical imaging via slice-direction averaging and subsampling.
    """
    img_t = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
    slice_op = RandomSliceThicknessDegradation(prob=1.0, downsample_factors=(downsample_factor,))
    corrupted_t = slice_op(img_t)
    return corrupted_t.squeeze().numpy().astype(np.float32)


def apply_fov_truncation(
    image: np.ndarray,
    crop_fraction: float = 0.15
) -> np.ndarray:
    """
    Simulates patient truncation when anatomy exceeds scanner bore / FOV limits.
    """
    img = image.copy()
    d, h, w = img.shape
    # Zero out lateral margins (w-axis) and anterior margins (y-axis)
    w_crop = int(w * crop_fraction)
    y_crop = int(h * crop_fraction)
    
    img[:, :y_crop, :] = 0.0
    img[:, :, :w_crop] = 0.0
    img[:, :, w - w_crop:] = 0.0
    return img.astype(np.float32)

