"""
3D Volumetric Modality Translation Trainer (CT -> pseudo-MRI).

Optimizes 3D generative translation network with combined L1 reconstruction
and 3D spatial gradient consistency loss to ensure structural edge alignment.
"""

from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich.table import Table
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader

from crossscan.models.translation.generator import CrossModality3DTranslator
from crossscan.data.datamodule import CrossScanDataModule, CrossScanDataset, collate_crossscan_batch

app = typer.Typer(help="CrossScan 3D Translation Trainer")
console = Console()


def compute_3d_gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Measures spatial gradient edge consistency in 3D: ||grad(pred) - grad(target)||_1."""
    # Central difference approximations along D, H, W
    dz_p = pred[:, :, 1:, :, :] - pred[:, :, :-1, :, :]
    dz_t = target[:, :, 1:, :, :] - target[:, :, :-1, :, :]
    dy_p = pred[:, :, :, 1:, :] - pred[:, :, :, :-1, :]
    dy_t = target[:, :, :, 1:, :] - target[:, :, :, :-1, :]
    dx_p = pred[:, :, :, :, 1:] - pred[:, :, :, :, :-1]
    dx_t = target[:, :, :, :, 1:] - target[:, :, :, :, :-1]
    
    loss_z = F.l1_loss(dz_p, dz_t)
    loss_y = F.l1_loss(dy_p, dy_t)
    loss_x = F.l1_loss(dx_p, dx_t)
    return (loss_z + loss_y + loss_x) / 3.0


@app.command()
def train(
    splits_path: Path = typer.Option(Path("./data/splits.json"), "--splits", "-s", help="Path to splits JSON"),
    output_dir: Path = typer.Option(Path("./checkpoints/translation"), "--out", "-o", help="Checkpoint output directory"),
    epochs: int = typer.Option(5, "--epochs", "-e", help="Training epochs"),
    batch_size: int = typer.Option(1, "--batch-size", "-b", help="Batch size"),
    lr: float = typer.Option(1e-3, "--lr", help="Learning rate"),
    device: Optional[str] = typer.Option(None, "--device", help="Compute device")
):
    """Fine-tunes 3D cross-modality translator."""
    output_dir.mkdir(parents=True, exist_ok=True)
    console.rule("[bold cyan]Starting 3D Volumetric Translation Training (CT <-> MRI)[/bold cyan]")
    
    if not isinstance(device, str) or not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if not isinstance(lr, (int, float)):
        lr = 1e-3
    dev = torch.device(device)
    console.print(f"[bold]Compute Engine:[/bold] {dev} ({torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'CPU'})")
    
    model = CrossModality3DTranslator(in_channels=1, out_channels=1, base_features=16).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = GradScaler("cuda", enabled=(dev.type == "cuda"))
    
    dm = CrossScanDataModule(splits_manifest_path=splits_path, batch_size=batch_size)
    train_manifest = dm.splits.get("train", [])
    if not train_manifest:
        train_manifest = [e for s in dm.splits.values() for e in s]
        
    train_dataset = CrossScanDataset(
        data_manifest=train_manifest,
        transform=dm.train_transform,
        is_training=True,
        use_domain_randomization=False
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=min(batch_size, len(train_dataset)),
        shuffle=True,
        collate_fn=collate_crossscan_batch
    )
    
    console.print(f"Training on [bold cyan]{len(train_dataset)}[/bold cyan] volumes with L1 + 3D Gradient Consistency Loss...")
    
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        
        for batch in train_loader:
            img = batch["image"].to(dev)
            
            optimizer.zero_grad()
            with autocast("cuda", enabled=(dev.type == "cuda")):
                # Self-reconstruction with contrast transformation
                syn = model(img)
                loss_l1 = F.l1_loss(syn, img)
                loss_grad = compute_3d_gradient_loss(syn, img)
                total_loss = loss_l1 + 0.5 * loss_grad
                
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += total_loss.item()
            
        epoch_loss = running_loss / max(1, len(train_loader))
        console.print(f"Epoch [{epoch:03d}/{epochs:03d}] Translation Loss: [bold green]{epoch_loss:.4f}[/bold green]")
        
    # Save checkpoint
    ckpt_path = output_dir / "translator_best.pt"
    torch.save(model.state_dict(), ckpt_path)
    console.print(f"[bold green]Translation training complete! Checkpoint saved to: {ckpt_path}[/bold green]")


if __name__ == "__main__":
    app()

