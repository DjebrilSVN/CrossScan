"""Natural clinical shift transforms: scanner field strength, pulse sequences, and CT dose/kernels."""
from typing import Tuple, Optional
import numpy as np
import scipy.ndimage as ndi


def simulate_mri_field_strength(volume: np.ndarray, target_field: float = 3.0, source_field: float = 1.5) -> np.ndarray:
    """Simulates 3.0T dielectric wave-standing intensity variation (shading roll-off)."""
    if target_field == source_field:
        return volume
    shape = volume.shape
    z, y, x = np.meshgrid(
        np.linspace(-1, 1, shape[0]),
        np.linspace(-1, 1, shape[1]),
        np.linspace(-1, 1, shape[2]),
        indexing="ij"
    )
    r2 = (x**2 + y**2 + 0.5 * z**2)
    b1_field = 1.0 - 0.28 * r2 * (target_field / 3.0)
    b1_field = np.clip(b1_field, 0.45, 1.0)
    return (volume * b1_field).astype(np.float32)


def simulate_mri_sequence_shift(volume: np.ndarray, sequence_type: str = "t1_out_phase") -> np.ndarray:
    """Simulates alternate clinical pulse sequences with tissue boundary cancellation or fluid contrast."""
    if sequence_type == "t1_out_phase":
        # Out-of-phase India ink boundary cancellation
        grad = ndi.gaussian_gradient_magnitude(volume, sigma=1.0)
        grad_norm = grad / (np.max(grad) + 1e-6)
        ink_mask = (grad_norm > 0.18).astype(np.float32)
        simulated = volume * (1.0 - 0.65 * ink_mask)
        return np.clip(simulated, 0.0, 1.0).astype(np.float32)
    elif sequence_type == "t2_spir":
        # Hyperintense fluid / suppressed fat contrast
        inverted = 1.0 - volume
        fluid_contrast = np.power(volume, 1.4) * 1.2
        simulated = np.where(volume > 0.6, fluid_contrast, volume * 0.75)
        return np.clip(simulated, 0.0, 1.0).astype(np.float32)
    return volume
