"""Unit tests for Patient-Level Bootstrap and Contamination Auditing."""
import numpy as np
from crossscan.evaluation.bootstrap import compute_patient_bootstrap_ci
from crossscan.evaluation.contamination import audit_test_cohort_contamination


def test_compute_patient_bootstrap_ci():
    # 50 patient Dice scores around 0.88
    scores = np.random.normal(0.88, 0.04, 50)
    scores = np.clip(scores, 0.5, 1.0)
    
    ci = compute_patient_bootstrap_ci(scores, num_bootstrap=500, confidence_level=0.95)
    assert 0.80 <= ci["mean"] <= 0.95
    assert ci["ci_lower"] <= ci["mean"] <= ci["ci_upper"]
    assert ci["std"] > 0.0


def test_audit_test_cohort_contamination():
    # AMOS CT should be detected as contaminated for VISTA3D
    report_amos_ct = audit_test_cohort_contamination("amos", "ct", ["patient_001"])
    assert report_amos_ct["VISTA3D"]["contaminated"] == True
    
    # AMOS MRI is NOT in VISTA3D training sets
    report_amos_mr = audit_test_cohort_contamination("amos", "mr", ["patient_001"])
    assert report_amos_mr["VISTA3D"]["contaminated"] == False

