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
from crossscan.models.translation.generator import CrossModality3DTranslator
from crossscan.models.translation.hallucination_audit import (
    AnatomicalHallucinationAuditor,
    compute_normalized_mutual_information,
    compute_slice_averaged_ssim,
)


def test_natural_shifts_execution():
    vol = np.ones((16, 32, 32), dtype=np.float32) * 0.5
    # Organ structure in center
    vol[4:12, 8:24, 8:24] = 0.8
    
    # Field strength
    t3 = simulate_mri_field_strength(vol, target_field=3.0, source_field=1.5)
    assert t3.shape == vol.shape
    assert not np.isnan(t3).any()
    
    # Pulse sequence
    out_phase = simulate_mri_sequence_shift(vol, sequence_type="t1_out_phase")
    assert out_phase.shape == vol.shape
    assert not np.isnan(out_phase).any()
    
    t2_spir = simulate_mri_sequence_shift(vol, sequence_type="t2_spir")
    assert t2_spir.shape == vol.shape
    
    # Low-Dose CT
    ldct = simulate_low_dose_ct(vol, dose_fraction=0.25)
    assert ldct.shape == vol.shape
    assert not np.isnan(ldct).any()
    
    # CT Kernel
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
    # Border voxels must be zeroed
    assert (fov[:, :2, :] == 0).all()


def test_cross_modality_3d_translator():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    translator = CrossModality3DTranslator(base_features=8, num_res_blocks=2).to(dev)
    
    x = torch.randn(1, 1, 16, 32, 32, device=dev)
    out = translator(x)
    
    assert out.shape == (1, 1, 16, 32, 32)
    assert (out >= 0.0).all() and (out <= 1.0).all()


def test_anatomical_hallucination_auditor():
    auditor = AnatomicalHallucinationAuditor()
    
    orig = np.zeros((16, 32, 32), dtype=np.float32)
    orig[4:12, 8:24, 8:24] = 0.8
    
    mask = np.zeros((16, 32, 32), dtype=np.int32)
    mask[4:12, 8:24, 8:24] = 1  # Liver
    
    # 1. Clean synthetic translation (faithful)
    syn_clean = orig.copy() + np.random.normal(0, 0.01, orig.shape)
    rep_clean = auditor.audit_translation(orig, syn_clean, mask_original=mask, mask_translated=mask)
    assert not rep_clean["hallucination_detected"]
    assert rep_clean["ssim"] > 0.80
    assert rep_clean["normalized_mutual_information"] > 0.50
    
    # 2. Hallucinatory translation (organ dissolved)
    syn_hallucinated = np.zeros_like(orig)  # Organ completely vanished
    mask_dissolved = np.zeros_like(mask)
    rep_halluc = auditor.audit_translation(orig, syn_hallucinated, mask_original=mask, mask_translated=mask_dissolved)
    assert rep_halluc["hallucination_detected"]
    assert len(rep_halluc["warnings"]) > 0

