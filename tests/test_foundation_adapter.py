"""Unit tests for Medical Foundation Model 3D LoRA Adaptation."""
import pytest
import torch
import torch.nn as nn

from crossscan.core.ontology import NUM_CLASSES
from crossscan.models.foundation_adapter import CrossScanFoundationAdapter
from crossscan.models.losses import HarmonizedCrossScanLoss


def test_foundation_adapter_initialization_and_breakdown():
    model = CrossScanFoundationAdapter(
        base_backbone=None,
        out_classes=NUM_CLASSES,
        lora_rank=4,
        lora_alpha=8.0,
        feature_size=16
    )
    breakdown = model.get_parameter_breakdown()
    
    assert breakdown["total_parameters"] > 0
    assert breakdown["trainable_parameters"] > 0
    assert breakdown["frozen_parameters"] > 0
    assert breakdown["trainable_percent"] < 10.0  # Parameter-efficient (< 10%)
    
    # Check that base backbone weights are frozen
    base_weight_grads = [p.requires_grad for name, p in model.named_parameters() if not ("lora_A" in name or "lora_B" in name or "adapter_head" in name)]
    assert all(not req for req in base_weight_grads)
    
    # Check that LoRA and adapter head weights are trainable
    trainable_grads = [p.requires_grad for name, p in model.named_parameters() if ("lora_A" in name or "lora_B" in name or "adapter_head" in name)]
    assert all(req for req in trainable_grads)


def test_foundation_adapter_forward_and_backward():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CrossScanFoundationAdapter(
        base_backbone=None,
        out_classes=NUM_CLASSES,
        lora_rank=4,
        lora_alpha=8.0,
        feature_size=16
    ).to(device)
    
    x = torch.randn(1, 1, 32, 32, 32, device=device)
    target = torch.randint(0, NUM_CLASSES, (1, 1, 32, 32, 32), device=device)
    
    out = model(x)
    assert out.shape == (1, NUM_CLASSES, 32, 32, 32)
    
    criterion = HarmonizedCrossScanLoss().to(device)
    loss = criterion(out, target)
    assert not torch.isnan(loss)
    
    loss.backward()
    
    # Verify that LoRA weights have gradients and frozen weights have None
    for name, p in model.named_parameters():
        if "lora_A" in name or "lora_B" in name or "adapter_head" in name:
            assert p.grad is not None, f"Expected gradient for {name}"
        else:
            assert p.grad is None, f"Expected no gradient for frozen parameter {name}"

