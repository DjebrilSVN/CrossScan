"""
Interactive Promptable Segmentation and Radiologist Correction Simulator.

Implements automated click placement on false-negative (positive clicks)
and false-positive (negative clicks) error clusters, measuring the key clinical metric:
Clicks-to-Target-Dice (C2D) against manual correction from scratch.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import scipy.ndimage as ndi

from crossscan.evaluation.metrics import compute_dice_score


def find_largest_error_component(error_mask: np.ndarray) -> Optional[Tuple[int, int, int]]:
    """
    Finds the center of mass (centroid) of the largest connected error cluster.
    """
    if not np.any(error_mask):
        return None
        
    labeled, num_features = ndi.label(error_mask)
    if num_features == 0:
        return None
        
    sizes = ndi.sum(error_mask, labeled, range(1, num_features + 1))
    largest_label = int(np.argmax(sizes) + 1)
    
    # Calculate centroid of largest component
    centroid = ndi.center_of_mass(error_mask, labeled, largest_label)
    return (int(round(centroid[0])), int(round(centroid[1])), int(round(centroid[2])))


def simulate_radiologist_clicks(
    pred_mask: np.ndarray,
    target_mask: np.ndarray,
    max_clicks: int = 10,
    target_dice: float = 0.88,
    simulated_brush_radius: int = 3
) -> Dict[str, Union[int, List[float], bool]]:
    """
    Simulates iterative user corrections on a 3D binary mask.
    
    In each step:
    - Finds the largest false-negative (missed organ) or false-positive (leakage) error cluster.
    - Places a click (simulated spherical brush edit).
    - Recomputes Dice.
    - Stops when target Dice is achieved or max_clicks reached.
    
    Returns:
        Dict containing total clicks, Dice progression history, and success flag.
    """
    current_pred = pred_mask.copy().astype(bool)
    true_mask = target_mask.astype(bool)
    
    initial_dice = compute_dice_score(current_pred, true_mask)
    dice_history = [initial_dice]
    
    if initial_dice >= target_dice:
        return {
            "clicks_to_target": 0,
            "dice_history": dice_history,
            "target_achieved": True,
            "initial_dice": initial_dice,
            "final_dice": initial_dice
        }
        
    struct_brush = ndi.generate_binary_structure(3, 1)
    
    for click_idx in range(1, max_clicks + 1):
        # Calculate errors
        fn_mask = np.logical_and(true_mask, ~current_pred)  # Missed organ
        fp_mask = np.logical_and(~true_mask, current_pred)  # Leakage into background
        
        fn_size = np.sum(fn_mask)
        fp_size = np.sum(fp_mask)
        
        if fn_size == 0 and fp_size == 0:
            break
            
        # Target the larger error type
        if fn_size >= fp_size:
            click_coord = find_largest_error_component(fn_mask)
            is_positive = True
        else:
            click_coord = find_largest_error_component(fp_mask)
            is_positive = False
            
        if click_coord is None:
            break
            
        # Apply simulated edit: spherical patch around click
        z, y, x = click_coord
        r = simulated_brush_radius
        d, h, w = current_pred.shape
        
        z_min, z_max = max(0, z - r), min(d, z + r + 1)
        y_min, y_max = max(0, y - r), min(h, y + r + 1)
        x_min, x_max = max(0, x - r), min(w, x + r + 1)
        
        if is_positive:
            # Positive click adds true organ voxels within brush radius
            patch_true = true_mask[z_min:z_max, y_min:y_max, x_min:x_max]
            current_pred[z_min:z_max, y_min:y_max, x_min:x_max] |= patch_true
        else:
            # Negative click removes false-positive voxels within brush radius
            current_pred[z_min:z_max, y_min:y_max, x_min:x_max] &= true_mask[z_min:z_max, y_min:y_max, x_min:x_max]
            
        current_dice = compute_dice_score(current_pred, true_mask)
        dice_history.append(current_dice)
        
        if current_dice >= target_dice:
            return {
                "clicks_to_target": click_idx,
                "dice_history": dice_history,
                "target_achieved": True,
                "initial_dice": initial_dice,
                "final_dice": current_dice
            }
            
    return {
        "clicks_to_target": max_clicks,
        "dice_history": dice_history,
        "target_achieved": False,
        "initial_dice": initial_dice,
        "final_dice": dice_history[-1]
    }


class InteractiveRefinementSession:
    """Manages multi-organ interactive correction session."""
    def __init__(self, initial_pred: np.ndarray, ground_truth: np.ndarray):
        self.initial_pred = initial_pred
        self.ground_truth = ground_truth

    def evaluate_efficiency(self, target_dice: float = 0.88) -> Dict[int, Dict]:
        """
        Runs click simulation for every organ and reports efficiency.
        """
        results = {}
        for c in np.unique(self.ground_truth):
            if c == 0:
                continue
            pred_c = (self.initial_pred == c)
            gt_c = (self.ground_truth == c)
            res = simulate_radiologist_clicks(pred_c, gt_c, target_dice=target_dice)
            results[int(c)] = res
        return results

