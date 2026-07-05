"""
Unit Tests for Phase 5 Reliability Layer & Conformal Risk Control Triage.
"""

from pathlib import Path
import tempfile
import pytest
import numpy as np
import torch
import torch.nn.functional as F

from crossscan.core.ontology import NUM_CLASSES
from crossscan.reliability.uncertainty import (
    compute_voxel_entropy,
    extract_boundary_uncertainty,
    compute_expected_calibration_error
)
from crossscan.reliability.shift_detector import MahalanobisShiftDetector


def test_voxel_entropy_and_boundary_uncertainty():
    B, C, D, H, W = 1, NUM_CLASSES, 16, 24, 24
    logits = torch.randn(B, C, D, H, W)
    probs = F.softmax(logits, dim=1)
    
    entropy = compute_voxel_entropy(probs)
    assert entropy.shape == (B, D, H, W)
    assert not torch.isnan(entropy).any()
    assert (entropy >= 0.0).all()
    
    pred_mask = torch.argmax(probs, dim=1)[0].cpu().numpy().astype(np.int32)
    pred_mask[4:12, 6:18, 6:18] = 1
    
    entropy_np = entropy[0].cpu().numpy()
    boundary_stats = extract_boundary_uncertainty(pred_mask, entropy_np, dilation_radius_voxels=1)
    
    assert 1 in boundary_stats
    assert "boundary_entropy_mean" in boundary_stats[1]
    assert boundary_stats[1]["boundary_voxel_count"] > 0


def test_expected_calibration_error():
    confs = np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1])
    labels = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    ece_res = compute_expected_calibration_error(confs, labels, num_bins=5)
    assert "ece" in ece_res
    assert ece_res["ece"] < 0.15


def test_mahalanobis_shift_detector():
    detector = MahalanobisShiftDetector(shrinkage_factor=1e-3)
    np.random.seed(42)
    id_latents = np.random.normal(0, 1.0, (20, 32)).astype(np.float32)
    detector.fit(id_latents)
    assert detector.is_fitted
    
    id_sample = np.random.normal(0, 1.0, 32).astype(np.float32)
    dist_id = detector.compute_mahalanobis_distance(id_sample)
    
    ood_sample = np.random.normal(10.0, 1.0, 32).astype(np.float32)
    dist_ood = detector.compute_mahalanobis_distance(ood_sample)
    
    assert dist_ood > dist_id
    assert detector.compute_free_energy(ood_sample) != 0.0
