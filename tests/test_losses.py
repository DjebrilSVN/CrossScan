"""Unit tests for masked marginal loss on partially annotated datasets."""
import torch
from crossscan.models.losses import MaskedDiceLoss, MaskedCrossEntropyLoss, HarmonizedCrossScanLoss


def test_masked_dice_loss():
    # Batch size 2, 5 classes (0: bg, 1: liver, 2: spleen, 3: rkidney, 4: lkidney), spatial (8, 8, 8)
    b, c, d, h, w = 2, 5, 8, 8, 8
    logits = torch.randn(b, c, d, h, w, requires_grad=True)
    # Ground truth has class 1 (liver)
    targets = torch.zeros(b, d, h, w, dtype=torch.long)
    targets[:, 2:6, 2:6, 2:6] = 1
    
    # Dataset 0 has classes {0, 1, 2}, Dataset 1 has classes {0, 1, 3, 4}
    present_mask = torch.tensor([
        [True, True, True, False, False],
        [True, True, False, True, True]
    ], dtype=torch.bool)
    
    loss_fn = MaskedDiceLoss()
    loss = loss_fn(logits, targets, present_mask=present_mask)
    assert not torch.isnan(loss)
    assert loss.item() >= 0.0
    
    loss.backward()
    assert logits.grad is not None


def test_masked_cross_entropy_loss():
    b, c, d, h, w = 2, 5, 8, 8, 8
    logits = torch.randn(b, c, d, h, w, requires_grad=True)
    targets = torch.zeros(b, d, h, w, dtype=torch.long)
    targets[:, 2:6, 2:6, 2:6] = 1
    
    present_mask = torch.tensor([
        [True, True, False, False, False],
        [True, True, False, False, False]
    ], dtype=torch.bool)
    
    ce_fn = MaskedCrossEntropyLoss()
    loss = ce_fn(logits, targets, present_mask=present_mask)
    assert not torch.isnan(loss)
    loss.backward()
    assert logits.grad is not None


def test_harmonized_crossscan_loss():
    b, c, d, h, w = 2, 5, 8, 8, 8
    logits = torch.randn(b, c, d, h, w, requires_grad=True)
    targets = torch.randint(0, 3, (b, d, h, w))
    present_mask = torch.tensor([
        [True, True, True, False, False],
        [True, True, True, False, False]
    ], dtype=torch.bool)
    
    criterion = HarmonizedCrossScanLoss()
    total_loss = criterion(logits, targets, present_mask=present_mask)
    assert not torch.isnan(total_loss)
    total_loss.backward()
    assert logits.grad is not None

