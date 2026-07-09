"""
Hardware Profiler, Latency Benchmarking, and ONNX Model Exporter.

Measures inference latency (seconds per volume), GPU peak VRAM memory (GB),
and voxel processing throughput across FP32, AMP FP16, and ONNX execution graphs.
"""

from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast

from crossscan.models.joint_segmentor import CrossScanJointSegmentor
from crossscan.core.ontology import NUM_CLASSES


class InferenceProfiler:
    """
    Profiles 3D segmentation inference latency, GPU memory, and exports ONNX graphs.
    """
    def __init__(self, model: CrossScanJointSegmentor, device: torch.device):
        self.model = model.to(device)
        self.device = device
