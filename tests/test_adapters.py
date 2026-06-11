"""Unit tests for 3D LoRA Parameter-Efficient Adapters."""
import torch
import torch.nn as nn
from crossscan.models.backbones.adapters import LoRA3DConv, LoRALinear, inject_lora_into_model


def test_lora_3d_conv_zero_init():
    conv = nn.Conv3d(8, 16, kernel_size=3, padding=1)
    x = torch.randn(2, 8, 12, 12, 12)
    
    with torch.no_grad():
        orig_out = conv(x)
        
    lora_conv = LoRA3DConv(conv, rank=4, alpha=8.0)
    
    # At initialization, lora_B is 0, so output must strictly match original conv
    with torch.no_grad():
        lora_out = lora_conv(x)
        
    assert torch.allclose(orig_out, lora_out, atol=1e-5)
    assert not conv.weight.requires_grad
    assert lora_conv.lora_A.weight.requires_grad
    assert lora_conv.lora_B.weight.requires_grad


def test_lora_3d_conv_gradient_flow():
    conv = nn.Conv3d(4, 8, kernel_size=3, padding=1)
    lora_conv = LoRA3DConv(conv, rank=2, alpha=4.0)
    x = torch.randn(1, 4, 8, 8, 8, requires_grad=True)
    
    out = lora_conv(x)
    loss = out.sum()
    loss.backward()
    
    assert conv.weight.grad is None  # Base weight is frozen
    assert lora_conv.lora_A.weight.grad is not None
    assert lora_conv.lora_B.weight.grad is not None


def test_lora_linear():
    lin = nn.Linear(32, 64)
    lora_lin = LoRALinear(lin, rank=4, alpha=8.0)
    x = torch.randn(4, 32)
    
    with torch.no_grad():
        orig_out = lin(x)
        lora_out = lora_lin(x)
        
    assert torch.allclose(orig_out, lora_out, atol=1e-5)
    
    out = lora_lin(x)
    out.sum().backward()
    assert lin.weight.grad is None
    assert lora_lin.lora_A.grad is not None


def test_inject_lora_into_model():
    model = nn.Sequential(
        nn.Conv3d(1, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv3d(16, 32, kernel_size=3, padding=1)
    )
    trainable_before = sum(p.numel() for p in model.parameters() if p.requires_grad)
    adapted_count = inject_lora_into_model(model, target_conv_rank=4)
    trainable_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    assert adapted_count > 0
    # Trainable parameters with LoRA should be significantly less than full fine-tuning
    assert trainable_after < trainable_before

