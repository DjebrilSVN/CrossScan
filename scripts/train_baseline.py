"""
Single-Modality Baseline Trainer (nnU-Net v2 Residual Encoder Style).

Establishes the upper-bound single-modality benchmark ceilings:
- Baseline A: Trained exclusively on CT scans (evaluated on CT vs MRI cross-shift).
- Baseline B: Trained exclusively on MRI scans (evaluated on MRI vs CT cross-shift).
"""

from pathlib import Path
from typing import Optional, List, Dict
import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader

from crossscan.core.ontology import NUM_CLASSES
from crossscan.models.joint_segmentor import CrossScanJointSegmentor
from crossscan.models.losses import HarmonizedCrossScanLoss
from crossscan.data.datamodule import CrossScanDataModule, CrossScanDataset, collate_crossscan_batch
from crossscan.evaluation.metrics import compute_dice_score

app = typer.Typer(help="CrossScan Per-Modality Baseline Trainer")
console = Console()


@app.command()
def train(
    modality: str = typer.Option("ct", "--modality", "-m", help="Target modality: ct or mr"),
    splits_path: Path = typer.Option(Path("./data/splits.json"), "--splits", "-s", help="Path to splits JSON"),
    output_dir: Optional[Path] = typer.Option(None, "--out", "-o", help="Checkpoint directory"),
    epochs: int = typer.Option(20, "--epochs", "-e", help="Number of training epochs"),
    batch_size: int = typer.Option(2, "--batch-size", "-b", help="Batch size"),
    lr: float = typer.Option(1e-3, "--lr", help="Learning rate"),
    eval_interval: int = typer.Option(5, "--eval-interval", help="Evaluation frequency in epochs"),
    device: str = typer.Option("cuda" if torch.cuda.is_available() else "cpu", "--device", help="Compute device")
):
    """Trains a modality-specific 3D Residual UNet baseline without domain randomization."""
    mod = modality.lower()
    if mod not in ["ct", "mr", "mri"]:
        raise ValueError("Modality must be 'ct' or 'mr'")
    if mod == "mri":
        mod = "mr"
        
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
    
    # Setup DataLoaders
    dm = CrossScanDataModule(splits_manifest_path=splits_path, batch_size=batch_size)
    
    # Filter train cohort strictly for this baseline's modality
    train_manifest = [e for e in dm.splits.get("train", []) if e.get("modality", "").lower() == mod]
    if len(train_manifest) == 0:
        # Fallback to any entries matching modality across all splits if train partition is empty
        train_manifest = [
            e for split in dm.splits.values() for e in split if e.get("modality", "").lower() == mod
        ]
        
    val_in_manifest = [e for e in dm.splits.get("val", []) if e.get("modality", "").lower() == mod]
    val_cross_manifest = [e for e in dm.splits.get("val", []) if e.get("modality", "").lower() != mod]
    
    console.print(f"Cohort Allocation: [bold cyan]{len(train_manifest)}[/bold cyan] Train | "
                  f"[bold green]{len(val_in_manifest)}[/bold green] In-Domain Val | "
                  f"[bold red]{len(val_cross_manifest)}[/bold red] Cross-Domain Val")
    
    train_dataset = CrossScanDataset(
        data_manifest=train_manifest,
        transform=dm.train_transform,
        is_training=True,
        use_domain_randomization=False  # No domain randomization for baseline
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=min(batch_size, len(train_dataset)),
        shuffle=True,
        collate_fn=collate_crossscan_batch
    )
    
    val_in_dataset = CrossScanDataset(
        data_manifest=val_in_manifest if val_in_manifest else train_manifest[:1],
        transform=dm.train_transform,  # Patch-based evaluation to prevent OOM
        is_training=False,
        use_domain_randomization=False
    )
    val_in_loader = DataLoader(val_in_dataset, batch_size=1, shuffle=False, collate_fn=collate_crossscan_batch)
    
    val_cross_loader = None
    if val_cross_manifest:
        val_cross_dataset = CrossScanDataset(
            data_manifest=val_cross_manifest,
            transform=dm.train_transform,
            is_training=False,
            use_domain_randomization=False
        )
        val_cross_loader = DataLoader(val_cross_dataset, batch_size=1, shuffle=False, collate_fn=collate_crossscan_batch)
        
    best_in_dice = 0.0
    history = []
    
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        
        for batch in train_loader:
            images = batch["image"].to(dev)
            labels = batch["label"].to(dev) if batch.get("label") is not None else None
            present_mask = batch["present_mask"].to(dev) if batch.get("present_mask") is not None else None
            
            optimizer.zero_grad()
            with autocast("cuda", enabled=(dev.type == "cuda")):
                preds = model(images)
                loss = criterion(preds, labels, present_mask=present_mask)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
            
        scheduler.step()
        epoch_loss = running_loss / max(1, len(train_loader))
        
        # Periodic validation
        if epoch % eval_interval == 0 or epoch == epochs:
            model.eval()
            in_dices = []
            with torch.no_grad():
                for v_batch in val_in_loader:
                    v_img = v_batch["image"].to(dev)
                    v_lbl = v_batch["label"].to(dev)
                    with autocast("cuda", enabled=(dev.type == "cuda")):
                        v_pred = model(v_img)
                        if isinstance(v_pred, (list, tuple)):
                            v_pred = v_pred[0]
                        v_pred_cls = torch.argmax(v_pred, dim=1)
                    
                    for c in range(1, NUM_CLASSES):
                        d = compute_dice_score(
                            (v_pred_cls == c).cpu().numpy().astype(bool),
                            (v_lbl.squeeze(1) == c).cpu().numpy().astype(bool)
                        )
                        if not np.isnan(d):
                            in_dices.append(d)
                            
            in_mean_dice = float(np.mean(in_dices)) if in_dices else 0.0
            
            # Cross-domain evaluation
            cross_mean_dice = 0.0
            if val_cross_loader:
                cross_dices = []
                with torch.no_grad():
                    for x_batch in val_cross_loader:
                        x_img = x_batch["image"].to(dev)
                        x_lbl = x_batch["label"].to(dev)
                        with autocast("cuda", enabled=(dev.type == "cuda")):
                            x_pred = model(x_img)
                            if isinstance(x_pred, (list, tuple)):
                                x_pred = x_pred[0]
                            x_pred_cls = torch.argmax(x_pred, dim=1)
                        for c in range(1, NUM_CLASSES):
                            d = compute_dice_score(
                                (x_pred_cls == c).cpu().numpy().astype(bool),
                                (x_lbl.squeeze(1) == c).cpu().numpy().astype(bool)
                            )
                            if not np.isnan(d):
                                cross_dices.append(d)
                cross_mean_dice = float(np.mean(cross_dices)) if cross_dices else 0.0
                
            console.print(
                f"Epoch [{epoch:03d}/{epochs:03d}] Loss: {epoch_loss:.4f} | "
                f"In-Domain ({mod.upper()}) Dice: [bold green]{in_mean_dice:.4f}[/bold green] | "
                f"Cross-Domain Dice: [bold red]{cross_mean_dice:.4f}[/bold red]"
            )
            
            history.append({
                "epoch": epoch,
                "loss": epoch_loss,
                "in_dice": in_mean_dice,
                "cross_dice": cross_mean_dice
            })
            
            # Save checkpoint
            ckpt_path = out_dir / f"baseline_{mod}_latest.pt"
            torch.save({
                "epoch": epoch,
                "modality": mod,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "in_mean_dice": in_mean_dice,
                "cross_mean_dice": cross_mean_dice
            }, ckpt_path)
            
            if in_mean_dice >= best_in_dice:
                best_in_dice = in_mean_dice
                best_ckpt = out_dir / f"baseline_{mod}_best.pt"
                torch.save(model.state_dict(), best_ckpt)
                
    console.print(f"[bold green]Baseline training complete! Checkpoint saved to: {out_dir}[/bold green]")
    return history


if __name__ == "__main__":
    import numpy as np
    app()
