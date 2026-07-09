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
import onnx

from crossscan.models.joint_segmentor import CrossScanJointSegmentor
from crossscan.core.ontology import NUM_CLASSES


class InferenceProfiler:
    """
    Profiles 3D segmentation inference latency, GPU memory, and exports ONNX graphs.
    """
    def __init__(self, model: CrossScanJointSegmentor, device: torch.device):
        self.model = model.to(device)
        self.device = device

    def benchmark_latency_and_vram(
        self,
        volume_shape: Tuple[int, int, int] = (64, 128, 128),
        num_warmup: int = 2,
        num_runs: int = 5,
        use_amp: bool = True
    ) -> Dict[str, Union[float, int, str]]:
        """
        Profiles inference speed and peak allocated CUDA memory.
        """
        self.model.eval()
        x = torch.randn(1, 1, *volume_shape, device=self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(num_warmup):
                if use_amp and self.device.type == "cuda":
                    with autocast(device_type="cuda", dtype=torch.float16):
                        _ = self.model(x)
                else:
                    _ = self.model(x)
                    
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.synchronize(self.device)
            
        latencies = []
        with torch.no_grad():
            for _ in range(num_runs):
                t0 = time.perf_counter()
                if use_amp and self.device.type == "cuda":
                    with autocast(device_type="cuda", dtype=torch.float16):
                        _ = self.model(x)
                else:
                    _ = self.model(x)
                    
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                latencies.append(time.perf_counter() - t0)
                
        mean_latency = float(np.mean(latencies))
        p95_latency = float(np.percentile(latencies, 95))
        
        # Total voxels processed
        total_voxels = float(np.prod(volume_shape))
        throughput_mvox_sec = (total_voxels / mean_latency) / 1e6
        
        peak_vram_gb = 0.0
        if self.device.type == "cuda":
            peak_vram_gb = float(torch.cuda.max_memory_allocated(self.device) / (1024 ** 3))
            
        return {
            "mean_latency_sec": mean_latency,
            "p95_latency_sec": p95_latency,
            "throughput_mvox_sec": throughput_mvox_sec,
            "peak_vram_gb": peak_vram_gb,
            "precision": "FP16 (AMP)" if use_amp else "FP32",
            "volume_shape": f"{volume_shape[0]}x{volume_shape[1]}x{volume_shape[2]}"
        }

    def export_onnx(
        self,
        output_path: Union[str, Path],
        input_shape: Tuple[int, int, int] = (32, 64, 64),
        opset_version: int = 17
    ) -> Path:
        """
        Exports the 3D model architecture to standard ONNX with dynamic spatial axes.
        """
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        
        self.model.eval()
        dummy_input = torch.randn(1, 1, *input_shape, device=self.device)
        
        # Wrapper to strip deep supervision list in eval mode
        class ExportWrapper(nn.Module):
            def __init__(self, base_model):
                super().__init__()
                self.base = base_model
            def forward(self, x):
                out = self.base(x)
                if isinstance(out, (list, tuple)):
                    return out[0]
                if torch.is_tensor(out) and out.ndim == 6:
                    return out[:, 0]
                return out

        wrapper = ExportWrapper(self.model).to(self.device)
        
        torch.onnx.export(
            wrapper,
            dummy_input,
            str(out_p),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["input_volume"],
            output_names=["segmentation_logits"],
            dynamic_axes={
                "input_volume": {0: "batch_size", 2: "depth", 3: "height", 4: "width"},
                "segmentation_logits": {0: "batch_size", 2: "depth", 3: "height", 4: "width"}
            }
        )
        
        # Verify ONNX model validity
        onnx_model = onnx.load(str(out_p))
        onnx.checker.check_model(onnx_model)
        
        return out_p

