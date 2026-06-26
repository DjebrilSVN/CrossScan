"""Unit tests for Natural/Synthetic Robustness Battery and Cross-Modality Translation."""
import pytest
import numpy as np
import torch

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


def test_natural_shifts_execution():
    vol = np.ones((16, 32, 32), dtype=np.float32) * 0.5
    vol[4:12, 8:24, 8:24] = 0.8
    
    t3 = simulate_mri_field_strength(vol, target_field=3.0, source_field=1.5)
    assert t3.shape == vol.shape
    assert not np.isnan(t3).any()
    
    out_phase = simulate_mri_sequence_shift(vol, sequence_type="t1_out_phase")
    assert out_phase.shape == vol.shape
    assert not np.isnan(out_phase).any()
    
    t2_spir = simulate_mri_sequence_shift(vol, sequence_type="t2_spir")
    assert t2_spir.shape == vol.shape
    
    ldct = simulate_low_dose_ct(vol, dose_fraction=0.25)
    assert ldct.shape == vol.shape
    assert not np.isnan(ldct).any()
    
    sharp = simulate_ct_kernel_shift(vol, kernel_type="sharp")
    assert sharp.shape == vol.shape


def test_synthetic_perturbations_execution():
    vol = np.ones((16, 32, 32), dtype=np.float32) * 0.5
    vol[4:12, 8:24, 8:24] = 0.9
    
    motion = apply_kspace_motion_artifacts(vol, num_ghosts=2, intensity=0.2)
    assert motion.shape == vol.shape
    
    b1 = apply_b1_inhomogeneity(vol, magnitude=0.4)
    assert b1.shape == vol.shape
    
    rician = apply_rician_noise(vol, sigma=0.05)
    assert rician.shape == vol.shape
    
    thick = apply_slice_thickness_degradation(vol, downsample_factor=2)
    assert thick.shape == vol.shape
    
    fov = apply_fov_truncation(vol, crop_fraction=0.1)
    assert fov.shape == vol.shape
    assert (fov[:, :2, :] == 0).all()

from crossscan.models.translation.generator import CrossModality3DTranslator


def test_cross_modality_3d_translator():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    translator = CrossModality3DTranslator(base_features=8, num_res_blocks=2).to(dev)
    
    x = torch.randn(1, 1, 16, 32, 32, device=dev)
    out = translator(x)
    
    assert out.shape == (1, 1, 16, 32, 32)
    assert (out >= 0.0).all() and (out <= 1.0).all()
