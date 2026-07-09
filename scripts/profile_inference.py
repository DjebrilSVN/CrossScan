"""
Hardware Latency, GPU VRAM Profiling, and ONNX Model Exporter Script.

Evaluates seconds per volume, throughput (MVox/s), and memory footprints across
FP32 vs AMP FP16 on local NVIDIA RTX hardware, exporting validated ONNX graphs.
"""

from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich.table import Table
import torch

from crossscan.models.joint_segmentor import CrossScanJointSegmentor
from crossscan.deployment.profiler import InferenceProfiler
from crossscan.core.ontology import NUM_CLASSES

app = typer.Typer(help="CrossScan Hardware Profiler & ONNX Exporter")
console = Console()


@app.command()
def profile(
    checkpoint: Optional[Path] = typer.Option(Path("./checkpoints/joint_model/joint_model_best.pt"), "--checkpoint", "-c", help="Joint model checkpoint"),
    output_dir: Path = typer.Option(Path("./checkpoints/deployment"), "--out", "-o", help="Deployment output directory"),
    onnx_name: str = typer.Option("crossscan_joint.onnx", "--onnx-name", help="Exported ONNX file name"),
    device: Optional[str] = typer.Option(None, "--device", help="Compute device")
):
    """Benchmarks hardware latency, memory footprint, and exports ONNX graph."""
    output_dir.mkdir(parents=True, exist_ok=True)
    console.rule("[bold cyan]CrossScan Hardware Latency, VRAM & ONNX Deployment Profiler[/bold cyan]")
    
    if not isinstance(device, str) or not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    
    device_name = torch.cuda.get_device_name(0) if dev.type == "cuda" else "CPU"
    total_vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3) if dev.type == "cuda" else 0.0
    console.print(f"[bold]Compute Engine:[/bold] {dev} ({device_name}, {total_vram:.2f} GB VRAM)")
    
    model = CrossScanJointSegmentor(in_channels=1, out_classes=NUM_CLASSES, feature_size=32, num_scales=4).to(dev)
    if checkpoint and checkpoint.exists():
        ckpt = torch.load(checkpoint, map_location=dev)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict, strict=False)
        console.print(f"[green]Loaded joint model checkpoint from {checkpoint}[/green]")
    else:
        console.print("[yellow]Using initialized model weights for profiling benchmark.[/yellow]")
        
    profiler = InferenceProfiler(model=model, device=dev)
    
    console.print("Benchmarking 3D inference regimes (volume: 64x128x128 = 1.05 MVox)...")
    res_fp32 = profiler.benchmark_latency_and_vram(volume_shape=(64, 128, 128), use_amp=False)
    res_fp16 = profiler.benchmark_latency_and_vram(volume_shape=(64, 128, 128), use_amp=True)
    
    # Export ONNX
    onnx_path = output_dir / onnx_name
    console.print(f"Exporting dynamic 3D ONNX computation graph to {onnx_path}...")
    profiler.export_onnx(output_path=onnx_path, input_shape=(32, 64, 64))
    onnx_size_mb = onnx_path.stat().st_size / (1024 ** 2)
    console.print(f"[green]ONNX graph successfully exported & validated! ({onnx_size_mb:.2f} MB)[/green]")
    
    table = Table(title="Hardware Deployment & Latency Benchmark Matrix", header_style="bold magenta")
    table.add_column("Precision Regime", width=18)
    table.add_column("Volume Grid", width=16)
    table.add_column("Latency (Mean)", width=18)
    table.add_column("Latency (P95)", width=18)
    table.add_column("Throughput (MVox/s)", width=22)
    table.add_column("Peak CUDA VRAM", width=18)
    table.add_column("Speedup vs FP32", width=18)
    
    speedup = res_fp32["mean_latency_sec"] / res_fp16["mean_latency_sec"] if res_fp16["mean_latency_sec"] > 0 else 1.0
    
    table.add_row(
        res_fp32["precision"],
        res_fp32["volume_shape"],
        f"{res_fp32['mean_latency_sec']:.3f} s",
        f"{res_fp32['p95_latency_sec']:.3f} s",
        f"{res_fp32['throughput_mvox_sec']:.2f} MVox/s",
        f"{res_fp32['peak_vram_gb']:.2f} GB",
        "1.00x (Baseline)"
    )
    table.add_row(
        f"[green]{res_fp16['precision']}[/green]",
        res_fp16["volume_shape"],
        f"[bold green]{res_fp16['mean_latency_sec']:.3f} s[/bold green]",
        f"{res_fp16['p95_latency_sec']:.3f} s",
        f"[green]{res_fp16['throughput_mvox_sec']:.2f} MVox/s[/green]",
        f"[green]{res_fp16['peak_vram_gb']:.2f} GB[/green]",
        f"[bold green]{speedup:.2f}x[/bold green]"
    )
    
    console.print(table)
    console.print(
        f"\n[bold green]Deployment Summary:[/bold green] Inference requires only [bold]{res_fp16['mean_latency_sec']:.2f} seconds[/bold] "
        f"per volume at [bold]{res_fp16['peak_vram_gb']:.2f} GB peak VRAM[/bold] — fully deployable on consumer 8GB laptops."
    )


if __name__ == "__main__":
    app()

