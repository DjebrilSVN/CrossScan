"""
Clinical Failure Taxonomy and Automated Error Categorization.

Categorizes segmentation breakdown modes according to:
- Missed Organ (complete false negative / volume collapse)
- Boundary Leakage (spillover into adjacent anatomical compartments)
- Laterality Inversion (left vs right kidney swap)
- Severe Artifact Collapse (fragmentation due to motion/noise)
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import scipy.ndimage as ndi

from crossscan.evaluation.metrics import compute_dice_score, compute_hd95


class FailureMode(str, Enum):
    ACCEPTABLE = "ACCEPTABLE"
    MISSED_ORGAN = "MISSED_ORGAN"
    BOUNDARY_LEAKAGE = "BOUNDARY_LEAKAGE"
    LATERALITY_CONFUSION = "LATERALITY_CONFUSION"
    ARTIFACT_FRAGMENTATION = "ARTIFACT_FRAGMENTATION"


def classify_organ_failure(
    pred_c: np.ndarray,
    target_c: np.ndarray,
    dice: float,
    hd95_mm: float,
    voxel_spacing: Tuple[float, float, float] = (1.5, 1.5, 1.5)
) -> Tuple[FailureMode, str]:
    """
    Classifies the specific failure mode for a given structure.
    """
    has_gt = np.any(target_c)
    has_pred = np.any(pred_c)
    
    if not has_gt and not has_pred:
        return FailureMode.ACCEPTABLE, "True negative agreement."
        
    if has_gt and not has_pred:
        return FailureMode.MISSED_ORGAN, "Organ completely absent from prediction (100% false negative)."
        
    if not has_gt and has_pred:
        return FailureMode.BOUNDARY_LEAKAGE, "Spurious hallucination: organ predicted where none exists."
        
    if dice >= 0.82 and hd95_mm <= 5.0:
        return FailureMode.ACCEPTABLE, "High-fidelity clinical boundary agreement."
        
    # Check for fragmentation / severe noise
    labeled_pred, num_clusters = ndi.label(pred_c)
    if num_clusters > 5 and dice < 0.60:
        return FailureMode.ARTIFACT_FRAGMENTATION, f"Severe fragmentation into {num_clusters} disconnected islands."
        
    # Check for boundary leakage (reasonable volume overlap but extreme boundary distance)
    if dice >= 0.50 and hd95_mm > 15.0:
        return FailureMode.BOUNDARY_LEAKAGE, f"Extreme surface outlier: HD95 = {hd95_mm:.1f} mm exceeds 15mm tolerance."
        
    if dice < 0.35:
        return FailureMode.MISSED_ORGAN, f"Severe volume mismatch: Dice {dice:.2f} < 0.35 threshold."
        
    return FailureMode.BOUNDARY_LEAKAGE, f"Sub-optimal boundary delineation (Dice={dice:.2f}, HD95={hd95_mm:.1f}mm)."


def audit_kidney_laterality(
    pred_mask: np.ndarray,
    right_kidney_idx: int = 3,
    left_kidney_idx: int = 4
) -> Tuple[bool, str]:
    """
    Checks if right and left kidneys have been swapped along the anatomical left-right (X) axis.
    In RAS+ space: Right side has lower X coordinates, Left side has higher X coordinates.
    """
    pred_rk = (pred_mask == right_kidney_idx)
    pred_lk = (pred_mask == left_kidney_idx)
    
    if not np.any(pred_rk) or not np.any(pred_lk):
        return False, "One or both kidneys absent; laterality inversion not assessable."
        
    # Centroids along X-axis (index 2 in D, H, W or index 0 in RAS)
    centroid_rk = ndi.center_of_mass(pred_rk)
    centroid_lk = ndi.center_of_mass(pred_lk)
    
    # In standard coordinate indexing: right kidney should have smaller X coordinate than left
    if centroid_rk[2] > centroid_lk[2]:
        return True, "LATERALITY INVERSION DETECTED: Right Kidney is positioned to the left of Left Kidney."
        
    return False, "Correct anatomical laterality orientation."

