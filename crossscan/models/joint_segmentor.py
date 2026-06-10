"""
Joint Modality-Agnostic 3D Segmentor with Bottleneck Latent Hooks.

Architecture:
- 3D Residual Encoder UNet (nnU-Net v2 residual preset design)
- Deep supervision heads at intermediate decoder scales
- Bottleneck latent embedding extraction for OOD detection and Quality Control
- Integrated physics-informed domain randomization for CT and MRI generalization
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.nets import DynUNet
from monai.inferers import sliding_window_inference
from crossscan.core.ontology import NUM_CLASSES
from crossscan.data.domain_randomization import DomainRandomization3D


class CrossScanJointSegmentor(nn.Module):
    """
    Unified 3D CT/MRI Segmentor with Latent Feature Extraction Hooks.
    """
    def __init__(
        self,
        in_channels: int = 1,
        out_classes: int = NUM_CLASSES,
        feature_size: int = 32,
        num_scales: int = 4,
        deep_supervision: bool = True,
        use_domain_randomization: bool = True
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_classes = out_classes
        self.deep_supervision = deep_supervision
        self.use_domain_randomization = use_domain_randomization
        
        # Domain Randomizer applied conditionally during training
        self.domain_randomizer = DomainRandomization3D() if use_domain_randomization else nn.Identity()
        
        # Encoder channels progression: e.g. [32, 64, 128, 256]
        filters = [feature_size * (2 ** i) for i in range(num_scales)]
        strides = [1] + [2] * (num_scales - 1)
        
        num_up = num_scales - 1
        computed_deep_supr = max(0, min(2, num_up - 1)) if deep_supervision and num_up > 1 else 0
        
        # Build DynUNet backbone (residual encoder preset)
        self.net = DynUNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_classes,
            kernel_size=[[3, 3, 3]] * num_scales,
            strides=[[s, s, s] for s in strides],
            upsample_kernel_size=[[2, 2, 2]] * (num_scales - 1),
            filters=filters,
            dropout=0.1,
            norm_name=("instance", {"affine": True}),
            act_name=("leakyrelu", {"negative_slope": 0.01, "inplace": True}),
            deep_supervision=deep_supervision and computed_deep_supr > 0,
            deep_supr_num=computed_deep_supr,
            res_block=True
        )
        
        # Bottleneck latent feature dimension
        self.bottleneck_dim = filters[-1]
        self._latest_bottleneck_features: Optional[torch.Tensor] = None

    def forward(
        self,
        x: torch.Tensor,
        apply_randomization: bool = False
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass with optional on-the-fly domain randomization.
        """
        if self.training and apply_randomization:
            x = self.domain_randomizer(x)
            
        outputs = self.net(x)
        return outputs

    def extract_bottleneck_representation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts global average pooled bottleneck embeddings [B, bottleneck_dim]
        for Out-Of-Distribution (OOD) detection and Label-Free Quality Control.
        """
        # Pass through input block and downsampling stages
        features = self.net.input_block(x)
        for down_block in self.net.downsamples:
            features = down_block(features)
        bottleneck = self.net.bottleneck(features)
        
        # Global spatial average pooling over 3D dimensions (D, H, W)
        pooled = F.adaptive_avg_pool3d(bottleneck, (1, 1, 1)).flatten(1)
        return pooled

    @torch.no_grad()
    def predict_volume_sliding_window(
        self,
        volume: torch.Tensor,
        roi_size: Tuple[int, int, int] = (96, 96, 96),
        sw_batch_size: int = 1,
        overlap: float = 0.5,
        mode: str = "gaussian"
    ) -> torch.Tensor:
        """
        Memory-efficient sliding-window inference with Gaussian patch blending.
        Enables seamless execution on 8GB local RTX 3080 GPUs.
        """
        self.eval()
        logits = sliding_window_inference(
            inputs=volume,
            roi_size=roi_size,
            sw_batch_size=sw_batch_size,
            predictor=lambda patch: self.net(patch),
            overlap=overlap,
            mode=mode,
            progress=False
        )
        if isinstance(logits, list):
            logits = logits[0]
        return logits
