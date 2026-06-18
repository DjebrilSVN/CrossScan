"""
Training Engine for the Unified Joint Modality-Agnostic 3D Segmentor.

Features:
- Physics-informed domain randomization (SynthSeg style: Rician noise, B1 bias, motion, downsampling)
- Masked Marginal Loss for partially labeled multi-center datasets
- Automatic Mixed Precision (AMP FP16) for high throughput on NVIDIA GPUs
- Patient-level validation tracking across CT and MRI simultaneously
"""

from pathlib import Path
from typing import Optional, List, Dict
import typer
from rich.console import Console
from rich.table import Table
import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader

from crossscan.core.ontology import NUM_CLASSES
from crossscan.models.joint_segmentor import CrossScanJointSegmentor
from crossscan.models.losses import HarmonizedCrossScanLoss
from crossscan.data.datamodule import CrossScanDataModule, CrossScanDataset, collate_crossscan_batch
from crossscan.evaluation.metrics import compute_dice_score, compute_nsd

app = typer.Typer(help="CrossScan Joint Model Trainer")
console = Console()


@app.command()
def train(
    config_path: Path = typer.Option(Path("./configs/joint_model.yaml"), "--config", "-c", help="Path to config YAML"),
    splits_path: Path = typer.Option(Path("./data/splits.json"), "--splits", "-s", help="Path to data splits JSON"),
    output_dir: Path = typer.Option(Path("./checkpoints/joint_model"), "--out", "-o", help="Checkpoint output directory"),
    epochs: int = typer.Option(20, "--epochs", "-e", help="Number of training epochs"),
    batch_size: int = typer.Option(2, "--batch-size", "-b", help="Batch size per step"),
    lr: float = typer.Option(1e-3, "--lr", help="Initial learning rate"),
    eval_interval: int = typer.Option(5, "--eval-interval", help="Evaluation frequency in epochs"),
    device: str = typer.Option("cuda" if torch.cuda.is_available() else "cpu", "--device", help="Compute device")
):
    """Executes joint multi-modality CT/MRI training with domain randomization."""
    output_dir.mkdir(parents=True, exist_ok=True)
    console.rule("[bold green]Starting CrossScan Joint 3D Model Training[/bold green]")
    
    dev = torch.device(device)
    console.print(f"[bold]Target Device:[/bold] {dev} ({torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'CPU'})")
    
    # 1. Initialize Joint Modality-Agnostic Model
    model = CrossScanJointSegmentor(
        in_channels=1,
        out_classes=NUM_CLASSES,
        feature_size=32,
        num_scales=4,
        deep_supervision=True,
        use_domain_randomization=True
    ).to(dev)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    console.print(f"[bold]Parameters:[/bold] {trainable_params:,} trainable / {total_params:,} total")
    
    # 2. Criterion, Optimizer, Scaler
    criterion = HarmonizedCrossScanLoss(dice_weight=1.0, ce_weight=1.0).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = GradScaler("cuda", enabled=(dev.type == "cuda"))
    
    # 3. Setup DataLoaders (combining CT and MRI)
    dm = CrossScanDataModule(splits_manifest_path=splits_path, batch_size=batch_size)
    
    train_manifest = dm.splits.get("train", [])
    if len(train_manifest) == 0:
        train_manifest = [e for split in dm.splits.values() for e in split]
        
    val_ct_manifest = [e for e in dm.splits.get("val", []) if e.get("modality", "").lower() == "ct"]
    val_mr_manifest = [e for e in dm.splits.get("val", []) if e.get("modality", "").lower() in ["mr", "mri"]]
    
    console.print(f"Joint Training Cohort: [bold cyan]{len(train_manifest)}[/bold cyan] Mixed CT/MRI Cases")
    console.print(f"Validation Cohorts:    [bold blue]{len(val_ct_manifest)}[/bold blue] CT | [bold magenta]{len(val_mr_manifest)}[/bold magenta] MRI")
    
    train_dataset = CrossScanDataset(
        data_manifest=train_manifest,
        transform=dm.train_transform,
        is_training=True,
        use_domain_randomization=True
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=min(batch_size, len(train_dataset)),
        shuffle=True,
        collate_fn=collate_crossscan_batch
    )
    
    val_ct_loader = DataLoader(
        CrossScanDataset(val_ct_manifest if val_ct_manifest else train_manifest[:1], transform=dm.train_transform, is_training=False),
        batch_size=1, shuffle=False, collate_fn=collate_crossscan_batch
    )
    val_mr_loader = DataLoader(
        CrossScanDataset(val_mr_manifest if val_mr_manifest else train_manifest[-1:], transform=dm.train_transform, is_training=False),
        batch_size=1, shuffle=False, collate_fn=collate_crossscan_batch
    )
    
    best_joint_dice = 0.0
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
            ct_dices = []
            mr_dices = []
            
            with torch.no_grad():
                for v_batch in val_ct_loader:
                    v_img = v_batch["image"].to(dev)
                    v_lbl = v_batch["label"].to(dev)
                    with autocast("cuda", enabled=(dev.type == "cuda")):
                        v_pred = model(v_img)
                        if isinstance(v_pred, (list, tuple)):
                            v_pred = v_pred[0]
                        v_cls = torch.argmax(v_pred, dim=1)
                    for c in range(1, NUM_CLASSES):
                        d = compute_dice_score(
                            (v_cls == c).cpu().numpy().astype(bool),
                            (v_lbl.squeeze(1) == c).cpu().numpy().astype(bool)
                        )
                        if not np.isnan(d):
                            ct_dices.append(d)
                            
                for v_batch in val_mr_loader:
                    v_img = v_batch["image"].to(dev)
                    v_lbl = v_batch["label"].to(dev)
                    with autocast("cuda", enabled=(dev.type == "cuda")):
                        v_pred = model(v_img)
                        if isinstance(v_pred, (list, tuple)):
                            v_pred = v_pred[0]
                        v_cls = torch.argmax(v_pred, dim=1)
                    for c in range(1, NUM_CLASSES):
                        d = compute_dice_score(
                            (v_cls == c).cpu().numpy().astype(bool),
                            (v_lbl.squeeze(1) == c).cpu().numpy().astype(bool)
                        )
                        if not np.isnan(d):
                            mr_dices.append(d)
                            
            ct_mean = float(np.mean(ct_dices)) if ct_dices else 0.0
            mr_mean = float(np.mean(mr_dices)) if mr_dices else 0.0
            joint_mean = (ct_mean + mr_mean) / 2.0
            
            console.print(
                f"Epoch [{epoch:03d}/{epochs:03d}] Loss: {epoch_loss:.4f} | "
                f"CT Dice: [bold cyan]{ct_mean:.4f}[/bold cyan] | "
                f"MRI Dice: [bold magenta]{mr_mean:.4f}[/bold magenta] | "
                f"Joint: [bold green]{joint_mean:.4f}[/bold green]"
            )
            
            history.append({
                "epoch": epoch,
                "loss": epoch_loss,
                "ct_dice": ct_mean,
                "mr_dice": mr_mean,
                "joint_dice": joint_mean
            })
            
            # Save checkpoints
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "ct_dice": ct_mean,
                "mr_dice": mr_mean,
                "joint_dice": joint_mean
            }, output_dir / "joint_model_latest.pt")
            
            if joint_mean >= best_joint_dice:
                best_joint_dice = joint_mean
                torch.save(model.state_dict(), output_dir / "joint_model_best.pt")
                
    console.print(f"[bold green]Joint training completed! Checkpoints saved to: {output_dir}[/bold green]")
    return history


if __name__ == "__main__":
    app()
