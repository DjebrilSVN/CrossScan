"""
Natural Clinical Shift Simulators for CT & MRI Modalities.

Models real-world clinical fleet variations:
1. MRI Field Strength Shifts (1.5 Tesla vs. 3.0 Tesla).
2. MRI Pulse Sequence Variations (T1 in-phase, T1 out-of-phase with India ink artifact, T2-SPIR fat-suppression).
3. CT Radiation Dose Reductions (Poisson-Gaussian quantum photon noise at 25% and 50% dose).
4. CT Reconstruction Kernels (Sharp bone/edge vs. Soft-tissue smoothing).
"""

from typing import Optional, Tuple, Union
import numpy as np
import scipy.ndimage as ndi
import torch


def simulate_mri_field_strength(
    image: np.ndarray,
    target_field: float = 3.0,
    source_field: float = 1.5
) -> np.ndarray:
    """
    Simulates field strength transition between 1.5T and 3.0T scanners.
    
    At 3.0T:
    - Base SNR increases (~1.7x) but dielectric and B1 transmit inhomogeneities increase.
    - T1 relaxation times prolong, reducing contrast difference in standard spin echoes.
    """
    img = image.copy().astype(np.float32)
    if target_field > source_field:  # 1.5T -> 3.0T
        # Add dielectric center-brightening B1 profile
        d, h, w = img.shape
        zz, yy, xx = np.meshgrid(
            np.linspace(-1, 1, d),
            np.linspace(-1, 1, h),
            np.linspace(-1, 1, w),
            indexing="ij"
        )
        r_sq = (zz ** 2 + yy ** 2 + xx ** 2)
        # Dielectric lens effect: higher signal in center, faster decay towards periphery
        b1_profile = 1.0 + 0.25 * (1.0 - np.clip(r_sq, 0, 1))
        img = img * b1_profile
        # Contrast compression from prolonged T1
        img = np.power(np.clip(img, 0, None), 0.92)
    else:  # 3.0T -> 1.5T
        # Add thermal noise penalty (lower baseline SNR)
        noise = np.random.normal(0, 0.04, img.shape)
        img = np.clip(img + noise, 0, None)
        
    return img.astype(np.float32)


def simulate_mri_sequence_shift(
    image: np.ndarray,
    sequence_type: str = "t1_out_phase"
) -> np.ndarray:
    """
    Simulates contrast shifts across MRI pulse sequences.
    
    Args:
        sequence_type:
            - 't1_out_phase': Out-of-phase chemical shift cancellation (India ink artifact) at water-fat borders.
            - 't2_spir': Spectral presaturation inversion recovery (fat suppression + fluid hyperintensity).
    """
    img = image.copy().astype(np.float32)
    
    if sequence_type == "t1_out_phase":
        # Compute gradient magnitude to identify organ and fat boundaries
        grad = ndi.gaussian_gradient_magnitude(img, sigma=1.0)
        grad_norm = grad / (np.max(grad) + 1e-8)
        # India ink effect: destructive phase cancellation drops boundary voxel intensities to near zero
        cancellation_mask = np.clip(grad_norm * 1.8, 0.0, 0.9)
        img = img * (1.0 - cancellation_mask)
        
    elif sequence_type == "t2_spir":
        # Fat suppression (suppress high-signal adipose tissue) + fluid hyperintensity boost
        # Non-linear mapping: invert lipid range, elevate high-intensity fluid voxels
        p_mid = np.percentile(img[img > 0], 50) if np.any(img > 0) else 0.5
        fluid_mask = img > (p_mid * 1.4)
        img[fluid_mask] = img[fluid_mask] * 1.35
        # Suppress intermediate adipose signal
        adipose_mask = np.logical_and(img > (p_mid * 0.7), img <= (p_mid * 1.2))
        img[adipose_mask] = img[adipose_mask] * 0.65
        
    return np.clip(img, 0, None).astype(np.float32)


def simulate_low_dose_ct(
    image: np.ndarray,
    dose_fraction: float = 0.25,
    photon_flux_i0: float = 1e5
) -> np.ndarray:
    """
    Simulates Low-Dose Computed Tomography (LDCT) via Poisson-Gaussian quantum photon statistics.
    
    Standard CT: full dose photon flux I0.
    Low-Dose CT: I0_eff = dose_fraction * I0.
    Photon transmission: I ~ Poisson(I0_eff * exp(-attenuation)) + N(0, sigma_electronic).
    """
    img = np.clip(image.copy().astype(np.float32), 0.0, 1.0)
    
    # Effective photon counts incident on detector
    i0_eff = max(1e3, photon_flux_i0 * dose_fraction)
    # Transmission intensity (Beer-Lambert attenuation)
    transmission = np.exp(-img * 3.0) * i0_eff
    
    # Poisson quantum noise
    poisson_counts = np.random.poisson(np.clip(transmission, 0, None)).astype(np.float32)
    # Electronic readout noise
    electronic_noise = np.random.normal(0, 10.0, img.shape).astype(np.float32)
    measured_counts = np.clip(poisson_counts + electronic_noise, 1.0, None)
    
    # Invert transmission back to attenuation
    noisy_img = -np.log(measured_counts / i0_eff) / 3.0
    return np.clip(noisy_img, 0.0, 1.0).astype(np.float32)


def simulate_ct_kernel_shift(
    image: np.ndarray,
    kernel_type: str = "sharp"
) -> np.ndarray:
    """
    Simulates CT convolution filter shift (Sharp bone/lung kernel vs. Soft tissue smoothing).
    """
    img = image.copy().astype(np.float32)
    if kernel_type == "sharp":
        # High-pass unsharp mask edge enhancement
        blurred = ndi.gaussian_filter(img, sigma=1.0)
        edges = img - blurred
        sharp = img + 0.75 * edges
        # Add high-frequency noise typical of sharp reconstruction filters
        noise = np.random.normal(0, 0.02, img.shape)
        return np.clip(sharp + noise, 0.0, 1.0).astype(np.float32)
    else:  # "smooth"
        return ndi.gaussian_filter(img, sigma=1.2).astype(np.float32)

