"""Unit tests for Reliability Layer: Uncertainty, Label-Free QC, and Conformal Triage."""
import numpy as np
import torch
from crossscan.reliability.uncertainty import (
    compute_voxel_entropy,
    extract_boundary_uncertainty
)
from crossscan.reliability.labelfree_qc import LabelFreeQualityAssessor
from crossscan.reliability.conformal import (
    RiskCoverageAnalysis,
    ConformalTriageEngine
)


def test_voxel_entropy():
    # 2 classes, confident vs uncertain: shape (B=1, C=2, D=1, H=1, W=1)
    probs_confident = torch.tensor([1.0, 0.0], dtype=torch.float32).reshape(1, 2, 1, 1, 1)
    probs_uncertain = torch.tensor([0.5, 0.5], dtype=torch.float32).reshape(1, 2, 1, 1, 1)
    
    ent_conf = compute_voxel_entropy(probs_confident)
    ent_unc = compute_voxel_entropy(probs_uncertain)
    
    assert ent_conf.item() < 0.01
    assert ent_unc.item() > 0.6  # -2 * 0.5 * ln(0.5) ~ 0.693


def test_label_free_qc_feature_extraction():
    image = np.ones((16, 16, 16), dtype=np.float32)
    pred_mask = np.zeros((16, 16, 16), dtype=np.int32)
    pred_mask[4:12, 4:12, 4:12] = 1  # Liver
    
    # Fake probability map: 9 classes
    probs = np.zeros((9, 16, 16, 16), dtype=np.float32)
    probs[0] = 0.9  # background
    probs[1, 4:12, 4:12, 4:12] = 0.95
    probs[0, 4:12, 4:12, 4:12] = 0.05
    
    assessor = LabelFreeQualityAssessor()
    features = assessor.extract_features(image, pred_mask, probs)
    
    assert 1 in features
    assert features[1].shape in [(9,), (12,)]
    assert features[1][0] > 0.0  # Volume > 0
    assert features[1][2] > 0.9  # Mean foreground prob > 0.9


def test_risk_coverage_analysis():
    # Synthetic cohort of 100 cases
    confidences = np.linspace(0.6, 0.99, 100)
    true_dices = confidences + np.random.normal(0, 0.02, 100)
    true_dices = np.clip(true_dices, 0.0, 1.0)
    
    res = RiskCoverageAnalysis.compute_curve(confidences, true_dices)
    assert len(res["coverages"]) == 100
    assert res["aurc"] > 0.0
    # Top confident cases should have higher Dice than bottom cases
    assert res["dices"][0] >= res["dices"][-1]


def test_conformal_triage_engine():
    cal_pred = np.array([0.95, 0.92, 0.90, 0.85, 0.75, 0.60])
    cal_true = np.array([0.94, 0.91, 0.89, 0.82, 0.70, 0.55])
    
    engine = ConformalTriageEngine(target_risk_alpha=0.10)
    thresh = engine.calibrate(cal_pred, cal_true)
    assert 0.5 <= thresh <= 1.0
    
    decision_good = engine.triage_case(
        "patient_001",
        {"liver": 0.95, "spleen": 0.92, "mean_case_dice": 0.935}
    )
    assert decision_good.action == "AUTO_ACCEPT"
    
    decision_bad = engine.triage_case(
        "patient_002",
        {"liver": 0.50, "spleen": 0.40, "mean_case_dice": 0.45}
    )
    assert decision_bad.action == "REJECT"
