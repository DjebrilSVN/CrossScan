"""Unit tests for Metrics Reloaded compliance (Dice, NSD, HD95)."""
import numpy as np
from crossscan.evaluation.metrics import (
    compute_dice_score,
    compute_nsd,
    compute_hd95,
    evaluate_volume
)


def test_compute_dice_identical():
    mask = np.zeros((20, 20, 20), dtype=np.uint8)
    mask[5:15, 5:15, 5:15] = 1
    assert compute_dice_score(mask, mask) == 1.0


def test_compute_dice_disjoint():
    m1 = np.zeros((20, 20, 20), dtype=np.uint8)
    m2 = np.zeros((20, 20, 20), dtype=np.uint8)
    m1[0:5, 0:5, 0:5] = 1
    m2[10:15, 10:15, 10:15] = 1
    assert compute_dice_score(m1, m2) < 0.001


def test_compute_nsd_within_tolerance():
    m1 = np.zeros((30, 30, 30), dtype=np.uint8)
    m2 = np.zeros((30, 30, 30), dtype=np.uint8)
    m1[10:20, 10:20, 10:20] = 1
    # m2 is 1 voxel dilated along one axis (within 1.5mm tolerance)
    m2[10:20, 10:20, 10:21] = 1
    
    nsd = compute_nsd(m1, m2, tolerance_mm=2.0, voxel_spacing=(1.5, 1.5, 1.5))
    assert nsd > 0.85


def test_evaluate_volume_multi_organ():
    pred = np.zeros((24, 24, 24), dtype=np.int32)
    target = np.zeros((24, 24, 24), dtype=np.int32)
    
    # Class 1: liver
    pred[5:15, 5:15, 5:15] = 1
    target[5:15, 5:15, 5:15] = 1
    
    # Class 2: spleen
    pred[2:5, 2:5, 2:5] = 2
    target[2:5, 2:5, 2:6] = 2
    
    res = evaluate_volume(pred, target, voxel_spacing=(1.5, 1.5, 1.5), present_classes=[1, 2])
    assert "liver" in res
    assert res["liver"]["dice"] == 1.0
    assert "spleen" in res
    assert res["spleen"]["dice"] > 0.7

