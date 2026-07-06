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
from crossscan.reliability.labelfree_qc import LabelFreeQualityAssessor
from crossscan.reliability.conformal import (
    ConformalTriageEngine,
    RiskCoverageAnalysis,
    TriageDecision
)


def test_voxel_entropy_and_boundary_uncertainty():
    B, C, D, H, W = 1, NUM_CLASSES, 16, 24, 24
    logits = torch.randn(B, C, D, H, W)
    probs = F.softmax(logits, dim=1)
    
    # 1. Test Shannon entropy calculation
    entropy = compute_voxel_entropy(probs)
    assert entropy.shape == (B, D, H, W)
    assert not torch.isnan(entropy).any()
    assert (entropy >= 0.0).all()
    
    # 2. Test boundary ribbon uncertainty
    pred_mask = torch.argmax(probs, dim=1)[0].cpu().numpy().astype(np.int32)
    # Ensure organ 1 is present in the center
    pred_mask[4:12, 6:18, 6:18] = 1
    
    entropy_np = entropy[0].cpu().numpy()
    boundary_stats = extract_boundary_uncertainty(pred_mask, entropy_np, dilation_radius_voxels=1)
    
    assert 1 in boundary_stats
    assert "boundary_entropy_mean" in boundary_stats[1]
    assert "boundary_entropy_p90" in boundary_stats[1]
    assert boundary_stats[1]["boundary_voxel_count"] > 0


def test_expected_calibration_error():
    # Perfectly calibrated case
    confs = np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1])
    labels = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    ece_res = compute_expected_calibration_error(confs, labels, num_bins=5)
    assert "ece" in ece_res
    assert ece_res["ece"] < 0.15
    
    # Empty case
    empty_res = compute_expected_calibration_error(np.array([]), np.array([]))
    assert empty_res["ece"] == 0.0


def test_mahalanobis_shift_detector():
    detector = MahalanobisShiftDetector(shrinkage_factor=1e-3)
    
    # In-distribution latents centered around 0
    np.random.seed(42)
    id_latents = np.random.normal(0, 1.0, (20, 32)).astype(np.float32)
    detector.fit(id_latents)
    assert detector.is_fitted
    
    # ID test sample
    id_sample = np.random.normal(0, 1.0, 32).astype(np.float32)
    dist_id = detector.compute_mahalanobis_distance(id_sample)
    
    # OOD test sample shifted far away
    ood_sample = np.random.normal(10.0, 1.0, 32).astype(np.float32)
    dist_ood = detector.compute_mahalanobis_distance(ood_sample)
    
    assert dist_ood > dist_id
    assert detector.compute_free_energy(ood_sample) != 0.0
    
    # Test evaluation output
    res = detector.evaluate_scan(ood_sample)
    assert res["is_ood_flagged"] is True
    
    # Test persistence
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / "shift.joblib"
        detector.save(tmp_path)
        loaded = MahalanobisShiftDetector.load(tmp_path)
        assert loaded.is_fitted
        assert loaded.compute_mahalanobis_distance(id_sample) == pytest.approx(dist_id, rel=1e-4)


def test_label_free_qc_regressor():
    assessor = LabelFreeQualityAssessor(regressor_type="ridge")
    
    # Create mock inputs
    vol = np.ones((16, 24, 24), dtype=np.float32) * 0.5
    pred_mask = np.zeros((16, 24, 24), dtype=np.int32)
    pred_mask[4:12, 6:18, 6:18] = 1  # Liver
    probs = np.zeros((NUM_CLASSES, 16, 24, 24), dtype=np.float32)
    probs[0] = 0.1
    probs[1] = 0.9
    
    features = assessor.extract_features(image=vol, pred_mask=pred_mask, probs=probs)
    assert 1 in features
    assert features[1].shape == (12,)
    
    # Mock training
    X = np.random.randn(10, 12).astype(np.float32)
    y = np.clip(np.random.normal(0.88, 0.05, 10), 0.5, 1.0)
    assessor.fit({1: X}, {1: y})
    assert assessor.is_fitted
    
    # Prediction
    pred = assessor.predict({1: features[1]})
    assert "predicted_case_dice" in pred
    assert "predicted_organ_dices" in pred
    assert 0.0 <= pred["predicted_case_dice"] <= 1.0
    
    # Persistence
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / "qc.joblib"
        assessor.save(tmp_path)
        loaded = LabelFreeQualityAssessor.load(tmp_path)
        assert loaded.is_fitted


def test_conformal_risk_control_and_triage():
    engine = ConformalTriageEngine(target_risk_alpha=0.10, reject_dice_floor=0.65)
    
    # Calibrate on mock validation data
    np.random.seed(42)
    cal_true = np.clip(np.random.normal(0.90, 0.04, 25), 0.65, 0.98)
    cal_pred = np.clip(cal_true + np.random.normal(0, 0.02, 25), 0.60, 0.99)
    
    lambda_thresh = engine.calibrate(cal_pred, cal_true)
    assert engine.is_calibrated
    assert 0.60 <= lambda_thresh <= 0.98
    
    # Risk-Coverage analysis
    rc = RiskCoverageAnalysis.compute_curve(cal_pred, cal_true)
    assert "aurc" in rc
    assert "milestones" in rc
    assert rc["milestones"]["cov_70_dice"] > 0.70
    assert 0.0 <= rc["milestones"]["cov_70_risk"] <= 0.30
    
    # Triage decision testing
    # 1. Clean confident case (well above calibrated lambda)
    dec_accept = engine.triage_case(
        case_id="patient_clean",
        predicted_organ_dices={"Liver": 0.97, "Spleen": 0.95, "mean_case_dice": 0.96},
        is_ood_flagged=False
    )
    assert dec_accept.action == "AUTO_ACCEPT"
    assert dec_accept.risk_level == "LOW"
    
    # 2. Flag for review case (one sub-threshold organ)
    dec_review = engine.triage_case(
        case_id="patient_borderline",
        predicted_organ_dices={"Liver": 0.95, "Spleen": 0.72, "mean_case_dice": 0.83},
        is_ood_flagged=False
    )
    assert dec_review.action == "FLAG_FOR_REVIEW"
    assert "Spleen" in dec_review.flagged_organs
    
    # 3. Reject case (OOD or severe failure)
    dec_reject = engine.triage_case(
        case_id="patient_ood",
        predicted_organ_dices={"Liver": 0.55, "Spleen": 0.50, "mean_case_dice": 0.52},
        is_ood_flagged=True
    )
    assert dec_reject.action == "REJECT"
    assert dec_reject.risk_level == "CRITICAL"
