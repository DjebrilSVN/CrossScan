"""
Medical Foundation Model Adaptation Engine with Parameter-Efficient 3D LoRA.

Enables adapting large 3D medical backbones (VISTA3D, SAM-Med3D, MedSAM2 style encoders)
with parameter-efficient low-rank adapters (< 2% trainable weights) while keeping
base foundation features frozen to prevent catastrophic forgetting.
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from crossscan.core.ontology import NUM_CLASSES
from crossscan.models.joint_segmentor import CrossScanJointSegmentor
from crossscan.models.backbones.adapters import LoRA3DConv, LoRALinear, inject_lora_into_model


class CrossScanFoundationAdapter(nn.Module):
    """
    Wraps a 3D medical foundation model with frozen backbone weights,
    3D Low-Rank Adaptation (LoRA) layers, and a task-specific segmentation head.
    """
    def __init__(
        self,
        base_backbone: Optional[nn.Module] = None,
        out_classes: int = NUM_CLASSES,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        inject_conv3d: bool = True,
        feature_size: int = 32
    ):
        super().__init__()
        self.out_classes = out_classes
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        
        # 1. Base Backbone (if not supplied, initialize 3D DynUNet encoder)
        if base_backbone is None:
            self.backbone = CrossScanJointSegmentor(
                in_channels=1,
                out_classes=out_classes,
                feature_size=feature_size,
                num_scales=4,
                deep_supervision=False,
                use_domain_randomization=False
            )
        else:
            self.backbone = base_backbone
            
        # 2. Freeze all base backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # 3. Inject parameter-efficient 3D LoRA adapters
        self.adapted_param_count = inject_lora_into_model(
            self.backbone,
            target_conv_rank=lora_rank,
            target_alpha=lora_alpha,
            inject_conv3d=inject_conv3d
        )
        
        # 4. Trainable domain-adaptation refinement projection head
        self.adapter_head = nn.Sequential(
            nn.Conv3d(out_classes, feature_size, kernel_size=3, padding=1),
            nn.InstanceNorm3d(feature_size),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv3d(feature_size, out_classes, kernel_size=1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: frozen foundation backbone + LoRA delta + lightweight adaptation head.
        """
        base_logits = self.backbone(x)
        if isinstance(base_logits, (list, tuple)):
            base_logits = base_logits[0]
            
        residual_logits = self.adapter_head(base_logits)
        return base_logits + residual_logits

    def get_parameter_breakdown(self) -> Dict[str, Union[int, float]]:
        """
        Computes the parameter efficiency profile: total, trainable, and percentage.
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        pct_trainable = (trainable / max(1, total)) * 100.0
        
        return {
            "total_parameters": total,
            "trainable_parameters": trainable,
            "frozen_parameters": frozen,
            "trainable_percent": pct_trainable,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha
        }

