"""
Lightweight Foundation Model Adaptation Trainer using 3D Parameter-Efficient LoRA.

Fine-tunes rank-r convolutional LoRA adapters (< 2% trainable parameters)
on mixed CT/MRI clinical cohorts while freezing the underlying foundation backbone.
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
from crossscan.models.foundation_adapter import CrossScanFoundationAdapter
from crossscan.models.losses import HarmonizedCrossScanLoss
from crossscan.data.datamodule import CrossScanDataModule, CrossScanDataset, collate_crossscan_batch
from crossscan.evaluation.metrics import compute_dice_score

app = typer.Typer(help="CrossScan Foundation Model LoRA Trainer")
console = Console()


@app.command()
def train(
    splits_path: Path = typer.Option(Path("./data/splits.json"), "--splits", "-s", help="Path to splits JSON"),
    output_dir: Path = typer.Option(Path("./checkpoints/foundation_adapter"), "--out", "-o", help="Checkpoint output directory"),
    lora_rank: int = typer.Option(8, "--rank", "-r", help="LoRA rank bottleneck dimension"),
    epochs: int = typer.Option(10, "--epochs", "-e", help="Number of fine-tuning epochs"),
    batch_size: int = typer.Option(2, "--batch-size", "-b", help="Batch size"),
    lr: float = typer.Option(5e-4, "--lr", help="Learning rate for adapter weights"),
    eval_interval: int = typer.Option(2, "--eval-interval", help="Evaluation frequency in epochs"),
    device: str = typer.Option("cuda" if torch.cuda.is_available() else "cpu", "--device", help="Compute device")
):
    """Executes parameter-efficient 3D LoRA fine-tuning on clinical multi-modality data."""
    output_dir.mkdir(parents=True, exist_ok=True)
    console.rule("[bold cyan]Starting Foundation Model 3D LoRA Adaptation[/bold cyan]")
    
    dev = torch.device(device)
    console.print(f"[bold]Compute Engine:[/bold] {dev} ({torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'CPU'})")
    
    # 1. Initialize Foundation Model with Frozen Backbone + 3D LoRA
    model = CrossScanFoundationAdapter(
        base_backbone=None,
        out_classes=NUM_CLASSES,
        lora_rank=lora_rank,
        lora_alpha=float(lora_rank * 2),
        feature_size=32
    ).to(dev)
    
    breakdown = model.get_parameter_breakdown()
    console.print(f"[bold]Total Parameters:[/bold]     {breakdown['total_parameters']:,}")
    console.print(f"[bold green]Trainable Parameters:[/bold green] {breakdown['trainable_parameters']:,} ([bold green]{breakdown['trainable_percent']:.2f}%[/bold green])")
    console.print(f"[bold dim]Frozen Parameters:[/bold dim]    {breakdown['frozen_parameters']:,}")
    
    # 2. Criterion and Optimizer (only optimising trainable adapter parameters)
    criterion = HarmonizedCrossScanLoss(dice_weight=1.0, ce_weight=1.0).to(dev)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=1e-4
    )
    scaler = GradScaler("cuda", enabled=(dev.type == "cuda"))
    
    # 3. DataModule
    dm = CrossScanDataModule(splits_manifest_path=splits_path, batch_size=batch_size)
    train_manifest = dm.splits.get("train", [])
    if len(train_manifest) == 0:
        train_manifest = [e for s in dm.splits.values() for e in s]
        
    val_manifest = dm.splits.get("val", [])
    if len(val_manifest) == 0:
        val_manifest = train_manifest[:1]
        
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
    
    val_loader = DataLoader(
        CrossScanDataset(val_manifest, transform=dm.train_transform, is_training=False),
        batch_size=1,
        shuffle=False,
        collate_fn=collate_crossscan_batch
    )
    
    best_dice = 0.0
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
            
        epoch_loss = running_loss / max(1, len(train_loader))
        
        if epoch % eval_interval == 0 or epoch == epochs:
            model.eval()
            dices = []
            with torch.no_grad():
                for v_batch in val_loader:
                    v_img = v_batch["image"].to(dev)
                    v_lbl = v_batch["label"].to(dev)
                    with autocast("cuda", enabled=(dev.type == "cuda")):
                        v_pred = model(v_img)
                        v_cls = torch.argmax(v_pred, dim=1)
                    for c in range(1, NUM_CLASSES):
                        d = compute_dice_score(
                            (v_cls == c).cpu().numpy().astype(bool),
                            (v_lbl.squeeze(1) == c).cpu().numpy().astype(bool)
                        )
                        if not np.isnan(d):
                            dices.append(d)
                            
            mean_val_dice = float(np.mean(dices)) if dices else 0.0
            console.print(
                f"Epoch [{epoch:03d}/{epochs:03d}] Loss: {epoch_loss:.4f} | "
                f"Validation Mean Dice: [bold green]{mean_val_dice:.4f}[/bold green]"
            )
            
            history.append({"epoch": epoch, "loss": epoch_loss, "val_dice": mean_val_dice})
            
            # Save checkpoint
            if mean_val_dice >= best_dice:
                best_dice = mean_val_dice
                # Save only trainable adapter parameters for maximum storage efficiency
                trainable_state = {k: v for k, v in model.state_dict().items() if "lora_" in k or "adapter_head" in k}
                torch.save(trainable_state, output_dir / "adapter_best.pt")
                
    console.print(f"[bold green]LoRA Adaptation completed! Lightweight adapter saved to: {output_dir / 'adapter_best.pt'}[/bold green]")
    return history


if __name__ == "__main__":
    app()

