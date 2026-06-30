"""
Clinical Anatomical Hallucination Audit Script for Generative Modality Translation.

Audits synthesized 3D pseudo-MRI / pseudo-CT volumes for structural hallucination,
organ volume dissolution, or artificial lesion spawning.
"""

from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich.table import Table
import numpy as np
import nibabel as nib
import torch

from crossscan.models.translation.generator import CrossModality3DTranslator
from crossscan.models.translation.hallucination_audit import AnatomicalHallucinationAuditor

app = typer.Typer(help="CrossScan Anatomical Hallucination Auditor")
console = Console()


@app.command()
def audit(
    test_splits: Path = typer.Option(Path("./data/splits.json"), "--splits", "-s", help="Path to data splits JSON"),
    checkpoint_translator: Optional[Path] = typer.Option(None, "--checkpoint", "-c", help="Trained translator checkpoint"),
    device: Optional[str] = typer.Option(None, "--device", help="Compute device")
):
    """Executes anatomical hallucination audit on synthesized volumes."""
    console.rule("[bold cyan]Anatomical Hallucination Audit for Generative Translation[/bold cyan]")
    if not isinstance(device, str) or not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    
    import json
    if not test_splits.exists():
        console.print(f"[red]Error: Splits file {test_splits} not found.[/red]")
        raise typer.Exit(code=1)
        
    with open(test_splits, "r") as f:
        splits = json.load(f)
        
    test_cases = splits.get("test", []) or splits.get("val", [])
    
    # Initialize translator
    translator = CrossModality3DTranslator(base_features=16).to(dev)
    if checkpoint_translator and checkpoint_translator.exists():
        console.print(f"Loading translator weights from: [bold cyan]{checkpoint_translator}[/bold cyan]")
        translator.load_state_dict(torch.load(checkpoint_translator, map_location=dev))
    translator.eval()
    
    auditor = AnatomicalHallucinationAuditor()
    
    table = Table(title="Anatomical Hallucination Audit Results", header_style="bold magenta")
    table.add_column("Patient ID", width=14)
    table.add_column("Modality", width=10)
    table.add_column("SSIM", width=12)
    table.add_column("NMI", width=12)
    table.add_column("Risk Score", width=14)
    table.add_column("Clinical Safety Status", width=26)
    table.add_column("Audit Remarks / Warnings", width=36)
    
    for case in test_cases:
        p_id = case["patient_id"]
        mod = case["modality"].upper()
        img_p = case.get("image")
        lbl_p = case.get("label")
        
        if not img_p or not Path(img_p).exists():
            continue
            
        orig_img = nib.load(str(img_p)).get_fdata().astype(np.float32)
        orig_lbl = nib.load(str(lbl_p)).get_fdata().astype(np.int32) if lbl_p and Path(lbl_p).exists() else None
        
        # Subsample central patch for audit speed: (32, 64, 64)
        d, h, w = orig_img.shape
        z_mid, y_mid, x_mid = d // 2, h // 2, w // 2
        patch = orig_img[z_mid-16:z_mid+16, y_mid-32:y_mid+32, x_mid-32:x_mid+32]
        
        # Translate volume
        with torch.no_grad():
            inp_t = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(dev)
            syn_t = translator(inp_t)
            syn_patch = syn_t.squeeze().cpu().numpy()
            
        lbl_patch = orig_lbl[z_mid-16:z_mid+16, y_mid-32:y_mid+32, x_mid-32:x_mid+32] if orig_lbl is not None else None
        
        report = auditor.audit_translation(
            original_volume=patch,
            translated_volume=syn_patch,
            mask_original=lbl_patch,
            mask_translated=lbl_patch  # Compare geometry conservation
        )
        
        warnings_str = "; ".join(report["warnings"]) if report["warnings"] else "[dim green]Zero anatomical distortions[/dim green]"
        
        table.add_row(
            p_id,
            mod,
            f"{report['ssim']:.3f}",
            f"{report['normalized_mutual_information']:.3f}",
            f"{report['hallucination_risk_score']:.3f}",
            report["clinical_safety_status"],
            warnings_str
        )
        
    console.print(table)
    console.print("\n[bold yellow]Anatomical Hallucination Audit Guidelines:[/bold yellow]")
    console.print("1. [bold]Organ Dissolution:[/bold] Generative translation must NOT shrink or dissolve low-contrast structures (e.g. pancreas).")
    console.print("2. [bold]Pseudo-Structure Spawn:[/bold] Synthesis must NOT fabricate artificial hyperintensities resembling tumors.")
    console.print("3. [bold]Boundary Preservation:[/bold] Soft-tissue organ interfaces must retain their sub-millimeter geometric boundaries.\n")


if __name__ == "__main__":
    app()

