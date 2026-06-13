"""Unit tests for Interactive Promptable Segmentation Simulator."""
import numpy as np
from crossscan.interactive.prompt_refinement import (
    find_largest_error_component,
    simulate_radiologist_clicks,
    InteractiveRefinementSession
)


def test_find_largest_error_component():
    mask = np.zeros((20, 20, 20), dtype=bool)
    # Component 1: small (size 8)
    mask[2:4, 2:4, 2:4] = True
    # Component 2: larger (size 64)
    mask[10:14, 10:14, 10:14] = True
    
    centroid = find_largest_error_component(mask)
    assert centroid is not None
    # Centroid should be inside the larger component (around 11.5 -> 11 or 12)
    assert 10 <= centroid[0] <= 13
    assert 10 <= centroid[1] <= 13
    assert 10 <= centroid[2] <= 13


def test_simulate_radiologist_clicks():
    gt = np.zeros((30, 30, 30), dtype=np.uint8)
    gt[10:20, 10:20, 10:20] = 1  # 10x10x10 cube
    
    # Sub-optimal initial prediction: only half the cube
    initial_pred = np.zeros((30, 30, 30), dtype=np.uint8)
    initial_pred[10:20, 10:20, 10:15] = 1
    
    res = simulate_radiologist_clicks(
        initial_pred,
        gt,
        max_clicks=8,
        target_dice=0.85,
        simulated_brush_radius=3
    )
    
    assert res["initial_dice"] < 0.70
    assert res["final_dice"] > res["initial_dice"]
    assert len(res["dice_history"]) > 1
    assert "clicks_to_target" in res

