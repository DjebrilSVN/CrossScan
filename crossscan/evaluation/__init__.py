"""Metrics Reloaded evaluation suite, bootstrap CI, failure taxonomy, and ablation study."""
from crossscan.evaluation.metrics import (
    compute_dice_score,
    compute_nsd,
    compute_hd95,
    evaluate_volume
)
from crossscan.evaluation.bootstrap import compute_patient_bootstrap_ci
from crossscan.evaluation.contamination import audit_test_cohort_contamination
from crossscan.evaluation.failure_taxonomy import classify_organ_failure, audit_kidney_laterality, FailureMode
from crossscan.evaluation.ablation import (
    AblationStudyRunner,
    AblationCondition,
    AblationMetrics,
    ABLATION_CONDITIONS
)

__all__ = [
    "compute_dice_score",
    "compute_nsd",
    "compute_hd95",
    "evaluate_volume",
    "compute_patient_bootstrap_ci",
    "audit_test_cohort_contamination",
    "classify_organ_failure",
    "audit_kidney_laterality",
    "FailureMode",
    "AblationStudyRunner",
    "AblationCondition",
    "AblationMetrics",
    "ABLATION_CONDITIONS",
]
