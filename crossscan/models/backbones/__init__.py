"""Backbone architectures and parameter-efficient adapters."""
from crossscan.models.backbones.adapters import LoRA3DConv, LoRALinear, inject_lora_into_model

__all__ = ["LoRA3DConv", "LoRALinear", "inject_lora_into_model"]
