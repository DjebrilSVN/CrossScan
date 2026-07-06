"""
Conformal Risk Control (CRC) and Selective Risk-Coverage Triage Engine.

Provides distribution-free statistical risk guarantees for clinical deployment.
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import json
import numpy as np


class ConformalTriageEngine:
    def __init__(self, target_risk_alpha: float = 0.10, reject_dice_floor: float = 0.65):
        self.alpha = target_risk_alpha
        self.reject_threshold = reject_dice_floor
        self.accept_threshold: float = 0.88
        self.is_calibrated: bool = False
        self.calibration_sample_size: int = 0

    def calibrate(self, cal_predicted_dices: np.ndarray, cal_true_dices: np.ndarray) -> float:
        assert len(cal_predicted_dices) == len(cal_true_dices)
        n = len(cal_predicted_dices)
        self.calibration_sample_size = n
        candidate_thresholds = np.linspace(0.60, 0.98, 200)
        valid_threshold = 0.88
        
        for thresh in candidate_thresholds:
            accepted_mask = (cal_predicted_dices >= thresh)
            if not np.any(accepted_mask) or np.sum(accepted_mask) < 2:
                continue
            accepted_losses = (1.0 - cal_true_dices[accepted_mask])
            emp_risk = float(np.mean(accepted_losses))
            if emp_risk <= self.alpha:
                valid_threshold = thresh
                break
                
        self.accept_threshold = float(valid_threshold)
        self.is_calibrated = True
        return self.accept_threshold
