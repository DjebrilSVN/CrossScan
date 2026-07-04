"""
Reliability Layer: Uncertainty Quantification, OOD Shift Detection, Label-Free QC, and Conformal Triage.
"""

from crossscan.reliability.uncertainty import (
    compute_voxel_entropy,
    predict_with_tta,
    extract_boundary_uncertainty,
    compute_expected_calibration_error
)
from crossscan.reliability.shift_detector import (
    MahalanobisShiftDetector
)
from crossscan.reliability.labelfree_qc import (
    LabelFreeQualityAssessor
)
from crossscan.reliability.conformal import (
    ConformalTriageEngine,
    RiskCoverageAnalysis,
    TriageDecision
)

__all__ = [
    "compute_voxel_entropy",
    "predict_with_tta",
    "extract_boundary_uncertainty",
    "compute_expected_calibration_error",
    "MahalanobisShiftDetector",
    "LabelFreeQualityAssessor",
    "ConformalTriageEngine",
    "RiskCoverageAnalysis",
    "TriageDecision",
]
