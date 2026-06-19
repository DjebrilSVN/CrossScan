"""Unit tests for Interactive 3D Refinement and Prompt Simulation."""
import pytest
import numpy as np
import torch

from crossscan.interactive.promptable_segmentor import (
    generate_click_heatmap_3d,
    Promptable3DRefinementNet,
    InteractiveRefinementEngine
)
from crossscan.interactive.prompt_refinement import (
    find_largest_error_component,
    simulate_radiologist_clicks
)


def test_generate_click_heatmap_3d():
    clicks = [(16, 16, 16)]
    shape = (32, 32, 32)
    heatmap = generate_click_heatmap_3d(clicks, shape, sigma=2.0)
    
    assert heatmap.shape == (1, 1, 32, 32, 32)
    assert heatmap[0, 0, 16, 16, 16].item() == pytest.approx(1.0, abs=1e-3)
    # Check that distance drops off
    assert heatmap[0, 0, 0, 0, 0].item() < 1e-4


def test_promptable_refinement_net_forward():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = Promptable3DRefinementNet(in_channels=4, out_channels=1, feature_size=8).to(device)
    
    img = torch.randn(1, 1, 16, 16, 16, device=device)
    prior = torch.zeros(1, 1, 16, 16, 16, device=device)
    pos = torch.zeros(1, 1, 16, 16, 16, device=device)
    neg = torch.zeros(1, 1, 16, 16, 16, device=device)
    
    out = net(img, prior, pos, neg)
    assert out.shape == (1, 1, 16, 16, 16)


def test_interactive_refinement_engine():
    engine = InteractiveRefinementEngine()
    img = torch.randn(1, 1, 16, 16, 16)
    prior = torch.zeros(1, 1, 16, 16, 16)
    
    refined = engine.refine_step(
        img,
        prior,
        pos_coords=[(8, 8, 8)],
        neg_coords=[(2, 2, 2)]
    )
    assert refined.shape == (1, 1, 16, 16, 16)
    assert (refined >= 0.0).all() and (refined <= 1.0).all()


def test_simulate_radiologist_clicks():
    gt = np.zeros((32, 32, 32), dtype=bool)
    gt[10:22, 10:22, 10:22] = True  # Cube
    
    # Empty pred requires clicks to recover target Dice
    pred = np.zeros_like(gt)
    
    res = simulate_radiologist_clicks(
        pred_mask=pred,
        target_mask=gt,
        max_clicks=5,
        target_dice=0.50,
        simulated_brush_radius=4
    )
    assert res["clicks_to_target"] > 0
    assert len(res["dice_history"]) > 1
    assert res["final_dice"] > res["initial_dice"]

