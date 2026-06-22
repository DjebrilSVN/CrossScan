"""
Parameter-Efficient Fine-Tuning (PEFT) 3D LoRA Adapters for Medical Foundation Models.

Implements Low-Rank Adaptation (LoRA) for 3D Convolutions and Attention Projections:
W' = W_frozen + (alpha / r) * (B @ A)
where B is initialized to 0, ensuring exact identity behavior at initialization.
"""

import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRA3DConv(nn.Module):
    """
    3D Convolutional LoRA Layer.
    Injects rank-r bottleneck matrices alongside a frozen 3D convolutional kernel.
    """
    def __init__(
        self,
        base_conv: nn.Conv3d,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0
    ):
        super().__init__()
        self.base_conv = base_conv
        # Freeze base weights
        self.base_conv.weight.requires_grad = False
        if self.base_conv.bias is not None:
            self.base_conv.bias.requires_grad = False
            
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        
        in_c = base_conv.in_channels
        out_c = base_conv.out_channels
        k_size = base_conv.kernel_size
        stride = base_conv.stride
        padding = base_conv.padding
        dilation = base_conv.dilation
        groups = base_conv.groups
        
        # Down-projection A: 1x1x1 conv from in_c to rank
        self.lora_A = nn.Conv3d(in_c, rank, kernel_size=1, bias=False)
        # Up-projection B: 3x3x3 conv from rank to out_c matching original spatial kernel
        self.lora_B = nn.Conv3d(
            rank, out_c, kernel_size=k_size, stride=stride, padding=padding,
            dilation=dilation, groups=groups, bias=False
        )
        
        # Reset parameters: A uses Kaiming uniform, B initialized to 0
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base frozen pass
        base_out = self.base_conv(x)
        # LoRA delta pass
        lora_out = self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
        return base_out + lora_out


class LoRALinear(nn.Module):
    """
    Linear LoRA Layer for Transformer Attention Projections (Q, K, V).
    """
    def __init__(
        self,
        base_linear: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0
    ):
        super().__init__()
        self.base_linear = base_linear
        self.base_linear.weight.requires_grad = False
        if self.base_linear.bias is not None:
            self.base_linear.bias.requires_grad = False
            
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        
        in_dim = base_linear.in_features
        out_dim = base_linear.out_features
        
        self.lora_A = nn.Parameter(torch.empty(rank, in_dim))
        self.lora_B = nn.Parameter(torch.zeros(out_dim, rank))
        
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_linear(x)
        lora_out = (self.dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return base_out + lora_out


def inject_lora_into_model(
    model: nn.Module,
    target_conv_rank: int = 8,
    target_alpha: float = 16.0,
    inject_conv3d: bool = True
) -> int:
    """
    Traverses model submodules and recursively wraps target layers with LoRA adapters.
    Returns the total count of adapted parameters.
    """
    adapted_count = 0
    for name, module in list(model.named_children()):
        if isinstance(module, (LoRA3DConv, LoRALinear)):
            continue
        if inject_conv3d and isinstance(module, nn.Conv3d) and module.kernel_size[0] > 1:
            lora_conv = LoRA3DConv(module, rank=target_conv_rank, alpha=target_alpha)
            setattr(model, name, lora_conv)
            adapted_count += sum(p.numel() for p in lora_conv.parameters() if p.requires_grad)
        else:
            adapted_count += inject_lora_into_model(
                module, target_conv_rank, target_alpha, inject_conv3d
            )
            
    return adapted_count

