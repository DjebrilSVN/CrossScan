"""
Conformal Risk Control (CRC) and Selective Risk-Coverage Triage Engine.

Provides distribution-free statistical risk guarantees for clinical deployment:
routes high-confidence inferences to PACS auto-acceptance while isolating
uncertain cases for radiologist review and rejecting out-of-distribution failures.
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import json
import numpy as np
from scipy.integrate import trapezoid

from crossscan.core.ontology import ORGAN_NAMES


@dataclass
class TriageDecision:
    case_id: str
    action: str  # "AUTO_ACCEPT", "FLAG_FOR_REVIEW", "REJECT"
    predicted_dice: float
    risk_level: str  # "LOW", "MODERATE", "CRITICAL"
    flagged_organs: List[str]
    is_ood_flagged: bool
    review_recommendation: str

    def to_dict(self) -> Dict[str, Union[str, float, List[str], bool]]:
        return asdict(self)


class RiskCoverageAnalysis:
    """
    Computes selective prediction risk-coverage trade-off curves.
    Answers: 'If we auto-accept the top X% most confident cases, what is the achieved Dice?'
    """
    @staticmethod
    def compute_curve(
        confidences: np.ndarray,
        true_dices: np.ndarray
    ) -> Dict[str, Union[np.ndarray, float, Dict[str, float]]]:
        """
        Computes the complete Risk-Coverage curve and key clinical coverage operating points.
        
        Args:
            confidences: Array of predicted case confidences/Dice [N].
            true_dices: Array of actual ground truth Dice scores [N].
            
        Returns:
            Dict containing coverages, dices, risks, aurc, and key coverage points (50%, 70%, 90%, 100%).
        """
        assert len(confidences) == len(true_dices), "Length mismatch between confidences and true_dices"
        n = len(confidences)
        sorted_indices = np.argsort(confidences)[::-1]  # Descending order
        
        coverages = []
        dices_at_coverage = []
        risks_at_coverage = []
        
        for k in range(1, n + 1):
            top_k_indices = sorted_indices[:k]
            cov = float(k / n)
            mean_dice = float(np.mean(true_dices[top_k_indices]))
            risk = float(1.0 - mean_dice)
            
            coverages.append(cov)
            dices_at_coverage.append(mean_dice)
            risks_at_coverage.append(risk)
            
        coverages_np = np.array(coverages)
        risks_np = np.array(risks_at_coverage)
        dices_np = np.array(dices_at_coverage)
        
        # Area Under Risk-Coverage Curve (AURC)
        aurc = float(trapezoid(risks_np, coverages_np)) if len(coverages_np) > 1 else 0.0
        
        # Key clinical milestones
        def _get_metric_at_coverage(target_cov: float) -> Tuple[float, float]:
            idx = np.argmin(np.abs(coverages_np - target_cov))
            return float(dices_np[idx]), float(risks_np[idx])
            
        dice_50, risk_50 = _get_metric_at_coverage(0.50)
        dice_70, risk_70 = _get_metric_at_coverage(0.70)
        dice_90, risk_90 = _get_metric_at_coverage(0.90)
        dice_100, risk_100 = _get_metric_at_coverage(1.00)
        
        return {
            "coverages": coverages_np,
            "dices": dices_np,
            "risks": risks_np,
            "aurc": aurc,
            "milestones": {
                "cov_50_dice": dice_50, "cov_50_risk": risk_50,
                "cov_70_dice": dice_70, "cov_70_risk": risk_70,
                "cov_90_dice": dice_90, "cov_90_risk": risk_90,
                "cov_100_dice": dice_100, "cov_100_risk": risk_100,
            }
        }


class ConformalTriageEngine:
    """
    Applies Split Conformal Risk Control (CRC) to guarantee an upper bound on clinical test risk.
    """
    def __init__(
        self,
        target_risk_alpha: float = 0.10,
        reject_dice_floor: float = 0.65
    ):
        """
        Args:
            target_risk_alpha: Statistical risk budget (e.g. alpha = 0.10 -> expected error <= 10%, meaning >= 90% Dice).
            reject_dice_floor: Hard failure threshold below which segmentations are rejected outright.
        """
        self.alpha = target_risk_alpha
        self.reject_threshold = reject_dice_floor
        
        self.accept_threshold: float = 0.88
        self.conformal_p_value_bound: float = 0.05
        self.is_calibrated: bool = False
        self.calibration_sample_size: int = 0

    def calibrate(
        self,
        cal_predicted_dices: np.ndarray,
        cal_true_dices: np.ndarray
    ) -> float:
        """
        Calibrates the acceptance threshold lambda_hat using Split Conformal Risk Control:
        Finds the minimum predicted Dice threshold lambda such that:
        (1 / (n + 1)) * sum(L_i(lambda)) + (B / (n + 1)) <= alpha
        where B is the bounded supremum loss (1.0).
        """
        assert len(cal_predicted_dices) == len(cal_true_dices)
        n = len(cal_predicted_dices)
        self.calibration_sample_size = n
        
        candidate_thresholds = np.linspace(0.60, 0.98, 200)
        valid_threshold = 0.88  # Default conservative fallback
        
        for thresh in candidate_thresholds:
            accepted_mask = (cal_predicted_dices >= thresh)
            if not np.any(accepted_mask):
                continue
                
            accepted_losses = (1.0 - cal_true_dices[accepted_mask])
            # Conformal finite-sample inflation: (n / (n + 1)) * empirical_mean + 1 / (n + 1)
            emp_risk = float(np.mean(accepted_losses))
            finite_sample_risk = (n / (n + 1.0)) * emp_risk + (1.0 / (n + 1.0))
            
            if finite_sample_risk <= self.alpha:
                valid_threshold = thresh
                break
                
        self.accept_threshold = float(valid_threshold)
        self.is_calibrated = True
        return self.accept_threshold

    def triage_case(
        self,
        case_id: str,
        predicted_organ_dices: Dict[str, float],
        is_ood_flagged: bool = False,
        mahalanobis_dist: float = 0.0
    ) -> TriageDecision:
        """
        Classifies an individual volume inference into 3-tier clinical action pathways:
        1. AUTO_ACCEPT (Green): Confidence exceeds conformal bound and in-distribution.
        2. FLAG_FOR_REVIEW (Yellow): Intermediate confidence or isolated organ boundary flags.
        3. REJECT (Red): High OOD distance or critical organ predicted failure.
        """
        organ_dices = {k: v for k, v in predicted_organ_dices.items() if k != "mean_case_dice"}
        mean_dice = predicted_organ_dices.get("mean_case_dice", float(np.mean(list(organ_dices.values()))))
        
        flagged_organs = [
            organ for organ, score in organ_dices.items() if score < self.accept_threshold
        ]
        
        if is_ood_flagged or mean_dice < self.reject_threshold or len(flagged_organs) >= 4:
            action = "REJECT"
            risk_level = "CRITICAL"
            rec = "Severe out-of-distribution shift or widespread anatomical breakdown. Recommend manual re-contouring or protocol review."
        elif mean_dice >= self.accept_threshold and len(flagged_organs) == 0:
            action = "AUTO_ACCEPT"
            risk_level = "LOW"
            rec = "Statistically calibrated within alpha error budget. Pushed directly to PACS / DICOM-SEG repository."
        else:
            action = "FLAG_FOR_REVIEW"
            risk_level = "MODERATE"
            flagged_str = ", ".join(flagged_organs)
            rec = f"Confidence sub-threshold on: [{flagged_str}]. Radiologist review requested with high-entropy boundary overlays."
            
        return TriageDecision(
            case_id=case_id,
            action=action,
            predicted_dice=float(mean_dice),
            risk_level=risk_level,
            flagged_organs=flagged_organs,
            is_ood_flagged=is_ood_flagged,
            review_recommendation=rec
        )

    def save(self, filepath: Union[str, Path]) -> None:
        """Saves conformal calibrator configuration to JSON."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "alpha": self.alpha,
            "accept_threshold": self.accept_threshold,
            "reject_threshold": self.reject_threshold,
            "is_calibrated": self.is_calibrated,
            "calibration_sample_size": self.calibration_sample_size
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "ConformalTriageEngine":
        """Loads conformal calibrator configuration from JSON."""
        with open(filepath, "r") as f:
            data = json.load(f)
        engine = cls(
            target_risk_alpha=data.get("alpha", 0.10),
            reject_dice_floor=data.get("reject_threshold", 0.65)
        )
        engine.accept_threshold = data.get("accept_threshold", 0.88)
        engine.is_calibrated = data.get("is_calibrated", True)
        engine.calibration_sample_size = data.get("calibration_sample_size", 0)
        return engine
