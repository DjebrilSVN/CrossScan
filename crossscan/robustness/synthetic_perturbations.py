"""Calibrated synthetic physical perturbations for stress-testing segmentor tolerance."""
from typing import Tuple
import numpy as np
import scipy.ndimage as ndi


def apply_kspace_motion_artifacts(volume: np.ndarray, num_ghosts: int = 4, intensity: float = 0.25) -> np.ndarray:
    """Simulates periodic discrete k-space phase-encode modulation artifacts (respiratory/pulsatile ghosting)."""
    fft = np.fft.fftn(volume, axes=(-2, -1))
    ghost_vol = np.zeros_like(volume)
    H = volume.shape[-2]
    step = max(1, H // (num_ghosts * 2))
    for shift in range(-num_ghosts, num_ghosts + 1):
        if shift == 0:
            continue
        rolled = np.roll(volume, shift * step, axis=-2)
        ghost_vol += intensity * (1.0 / abs(shift)) * rolled
    return np.clip(volume + ghost_vol, 0.0, 1.0).astype(np.float32)


def apply_b1_inhomogeneity(volume: np.ndarray, magnitude: float = 0.5) -> np.ndarray:
    """Applies low-frequency smooth polynomial B1 transmit/receive bias field inhomogeneity."""
    shape = volume.shape
    z, y, x = np.meshgrid(
        np.linspace(-1, 1, shape[0]),
        np.linspace(-1, 1, shape[1]),
        np.linspace(-1, 1, shape[2]),
        indexing="ij"
    )
    bias = 1.0 + magnitude * (0.35 * x + 0.25 * y**2 - 0.2 * z**2 + 0.15 * x * y)
    bias = bias / np.mean(bias)
    return np.clip(volume * bias, 0.0, 1.0).astype(np.float32)


def apply_rician_noise(volume: np.ndarray, sigma: float = 0.08) -> np.ndarray:
    """Injects MR-physics Rician noise from orthogonal complex Gaussian channels: sqrt((I + N_r)^2 + N_i^2)."""
    n_real = np.random.normal(0, sigma, volume.shape)
    n_imag = np.random.normal(0, sigma, volume.shape)
    rician = np.sqrt((volume + n_real)**2 + n_imag**2)
    return np.clip(rician, 0.0, 1.0).astype(np.float32)


def apply_slice_thickness_degradation(volume: np.ndarray, downsample_factor: int = 3) -> np.ndarray:
    """Simulates thick-slice low-resolution clinical protocols via axial blur and subsampling."""
    blurred = ndi.gaussian_filter1d(volume, sigma=downsample_factor * 0.6, axis=0)
    subsampled = blurred[::downsample_factor, :, :]
    restored = ndi.zoom(subsampled, (volume.shape[0] / subsampled.shape[0], 1.0, 1.0), order=1)
    return np.clip(restored, 0.0, 1.0).astype(np.float32)
