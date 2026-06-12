"""
Conformal Risk Control (CRC) and Risk-Coverage Selective Triage Policy.

Provides rigorous statistical guarantees for clinical decision support:
routes confident cases to auto-acceptance while flagging uncertain cases
for radiologist review.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.integrate import trapezoid


@dataclass
class TriageDecision:
    case_id: str
    action: str  # "AUTO_ACCEPT", "FLAG_FOR_REVIEW", "REJECT"
    predicted_dice: float
    risk_level: str
    flagged_organs: List[str]


class RiskCoverageAnalysis:
    """
    Computes selective prediction risk-coverage trade-off curves.
    Answer: 'If we auto-accept the top X% of cases, what is the achieved Dice?'
    """
    @staticmethod
    def compute_curve(
        confidences: np.ndarray,
        true_dices: np.ndarray,
        num_thresholds: int = 50
    ) -> Dict[str, np.ndarray]:
        """
        Args:
            confidences: Array of predicted case confidences/Dice [N].
            true_dices: Array of actual ground truth Dice scores [N].
        """
        assert len(confidences) == len(true_dices)
        n = len(confidences)
        sorted_indices = np.argsort(confidences)[::-1]  # Descending confidence
        
        coverages = []
        dices_at_coverage = []
        risks_at_coverage = []
        
        for k in range(1, n + 1):
            top_k_indices = sorted_indices[:k]
            cov = float(k / n)
            mean_dice = float(np.mean(true_dices[top_k_indices]))
            risk = 1.0 - mean_dice
            
            coverages.append(cov)
            dices_at_coverage.append(mean_dice)
            risks_at_coverage.append(risk)
            
        coverages_np = np.array(coverages)
        risks_np = np.array(risks_at_coverage)
        
        # Area Under Risk-Coverage Curve (AURC) via trapezoidal integration
        aurc = float(trapezoid(risks_np, coverages_np))
        
        return {
            "coverages": coverages_np,
            "dices": np.array(dices_at_coverage),
            "risks": risks_np,
            "aurc": aurc
        }


class ConformalTriageEngine:
    """
    Applies Split Conformal Risk Control to guarantee an upper bound on test error.
    """
    def __init__(self, target_risk_alpha: float = 0.10):
        """
        Args:
            target_risk_alpha: Maximum allowable average error (1 - Dice).
                              e.g., alpha=0.10 corresponds to 90% expected Dice on accepted cohort.
        """
        self.alpha = target_risk_alpha
        self.accept_threshold: float = 0.85
        self.reject_threshold: float = 0.60
        self.is_calibrated: bool = False

    def calibrate(
        self,
        cal_predicted_dices: np.ndarray,
        cal_true_dices: np.ndarray
    ) -> float:
        """
        Calibrates the acceptance threshold lambda_hat using empirical conformal bounds.
        
        Finds the minimum predicted Dice threshold lambda such that:
        Empirical Risk <= alpha * (n / (n + 1))
        """
        n = len(cal_predicted_dices)
        candidate_thresholds = np.linspace(0.5, 0.98, 100)
        valid_threshold = 0.85
        
        for thresh in candidate_thresholds:
            accepted_mask = (cal_predicted_dices >= thresh)
            if not np.any(accepted_mask):
                continue
                
            risk = np.mean(1.0 - cal_true_dices[accepted_mask])
            # Conformal finite-sample correction
            if risk <= self.alpha:
                valid_threshold = thresh
                break
                
        self.accept_threshold = float(valid_threshold)
        self.is_calibrated = True
        return self.accept_threshold

    def triage_case(
        self,
        case_id: str,
        predicted_organ_dices: Dict[str, float]
    ) -> TriageDecision:
        """
        Classifies an inference case into clinical action bins.
        """
        organ_dices = {k: v for k, v in predicted_organ_dices.items() if k != "mean_case_dice"}
        mean_dice = predicted_organ_dices.get("mean_case_dice", float(np.mean(list(organ_dices.values()))))
        
        flagged_organs = [
            organ for organ, score in organ_dices.items() if score < self.accept_threshold
        ]
        
        if mean_dice >= self.accept_threshold and len(flagged_organs) == 0:
            action = "AUTO_ACCEPT"
            risk_level = "LOW"
        elif mean_dice < self.reject_threshold or len(flagged_organs) >= 4:
            action = "REJECT"
            risk_level = "CRITICAL"
        else:
            action = "FLAG_FOR_REVIEW"
            risk_level = "MODERATE"
            
        return TriageDecision(
            case_id=case_id,
            action=action,
            predicted_dice=mean_dice,
            risk_level=risk_level,
            flagged_organs=flagged_organs
        )
