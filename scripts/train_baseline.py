"""
Single-Modality Baseline Trainer (nnU-Net v2 Residual Encoder Style).

Establishes the upper-bound single-modality benchmark ceilings:
- Baseline A: Trained exclusively on CT scans (evaluated on CT vs MRI cross-shift).
- Baseline B: Trained exclusively on MRI scans (evaluated on MRI vs CT cross-shift).
"""

from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich.table import Table
import yaml
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler

from crossscan.core.ontology import NUM_CLASSES
from crossscan.models.joint_segmentor import CrossScanJointSegmentor
from crossscan.models.losses import HarmonizedCrossScanLoss
from crossscan.data.datamodule import CrossScanDataModule

app = typer.Typer(help="CrossScan Per-Modality Baseline Trainer")
console = Console()


@app.command()
def train(
    modality: str = typer.Option("ct", "--modality", "-m", help="Target modality: ct or mr"),
    splits_path: Path = typer.Option(Path("./data/splits.json"), "--splits", "-s", help="Path to splits JSON"),
    output_dir: Optional[Path] = typer.Option(None, "--out", "-o", help="Checkpoint directory"),
    epochs: int = typer.Option(100, "--epochs", "-e", help="Number of training epochs"),
    batch_size: int = typer.Option(2, "--batch-size", "-b", help="Batch size"),
    lr: float = typer.Option(1e-3, "--lr", help="Learning rate"),
    device: str = typer.Option("cuda" if torch.cuda.is_available() else "cpu", "--device", help="Compute device")
):
    """Trains a modality-specific 3D Residual UNet baseline without domain randomization."""
    mod = modality.lower()
    if mod not in ["ct", "mr", "mri"]:
        raise ValueError("Modality must be 'ct' or 'mr'")
        
    out_dir = output_dir or Path(f"./checkpoints/baseline_{mod}")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    console.rule(f"[bold blue]Training Modality-Specific Baseline: {mod.upper()}[/bold blue]")
    dev = torch.device(device)
    console.print(f"[bold]Compute Engine:[/bold] {dev} ({torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'CPU'})")
    
    # Baseline model does NOT use cross-modality domain randomization
    model = CrossScanJointSegmentor(
        in_channels=1,
        out_classes=NUM_CLASSES,
        feature_size=32,
        num_scales=4,
        deep_supervision=True,
        use_domain_randomization=False  # Crucial: standard single-modality baseline
    ).to(dev)
    
    criterion = HarmonizedCrossScanLoss(dice_weight=1.0, ce_weight=1.0).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = GradScaler("cuda", enabled=(dev.type == "cuda"))
    
    total_params = sum(p.numel() for p in model.parameters())
    console.print(f"[green]Initialized {mod.upper()} baseline model ({total_params:,} parameters).[/green]")
    console.print(f"Checkpoints will be saved to: [bold cyan]{out_dir}[/bold cyan]")


if __name__ == "__main__":
    app()

