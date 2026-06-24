"""Robustness evaluation battery: natural fleet shifts and synthetic physical perturbations."""
from crossscan.robustness.natural_shifts import (
    simulate_mri_field_strength,
    simulate_mri_sequence_shift,
    simulate_low_dose_ct,
    simulate_ct_kernel_shift,
)
from crossscan.robustness.synthetic_perturbations import (
    apply_kspace_motion_artifacts,
    apply_b1_inhomogeneity,
    apply_rician_noise,
    apply_slice_thickness_degradation,
    apply_fov_truncation,
)

__all__ = [
    "simulate_mri_field_strength",
    "simulate_mri_sequence_shift",
    "simulate_low_dose_ct",
    "simulate_ct_kernel_shift",
    "apply_kspace_motion_artifacts",
    "apply_b1_inhomogeneity",
    "apply_rician_noise",
    "apply_slice_thickness_degradation",
    "apply_fov_truncation",
]

