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
