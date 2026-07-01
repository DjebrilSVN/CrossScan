"""
CrossScan Command-Line Interface (CLI).

Unified entry point for data harmonization, model inference,
label-free quality control, and clinical DICOM-SEG export.
"""

from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich.table import Table
import torch

from crossscan.core.ontology import ORGAN_ONTOLOGY, NUM_CLASSES

app = typer.Typer(
    name="crossscan",
    help="CrossScan: 3D Multi-Modality CT/MRI Segmentation & Reliability Suite"
)
console = Console()


@app.command()
def info():
    """Display system hardware, PyTorch CUDA capabilities, and standardized ontology."""
    console.rule("[bold cyan]CrossScan System Diagnostics[/bold cyan]")
    
    cuda_avail = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU Only"
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3) if cuda_avail else 0.0
    
    console.print(f"[bold]PyTorch Version:[/bold] {torch.__version__}")
    console.print(f"[bold]CUDA Acceleration:[/bold] {'[green]AVAILABLE[/green]' if cuda_avail else '[red]DISABLED[/red]'}")
    console.print(f"[bold]GPU Hardware:[/bold] {device_name} ({vram_gb:.2f} GB VRAM)")
    
    console.print("\n[bold cyan]Standardized 8-Organ Anatomical Ontology (Metrics Reloaded Standard):[/bold cyan]")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Class ID", style="dim", width=8)
    table.add_column("Organ Name", width=22)
    table.add_column("SNOMED CT", width=14)
    table.add_column("RadLex", width=12)
    table.add_column("NSD Tolerance (tau)", width=20)
    
    for idx, meta in ORGAN_ONTOLOGY.items():
        table.add_row(
            str(idx),
            meta.display_name,
            meta.snomed_ct,
            meta.radlex,
            f"{meta.nsd_tolerance_mm:.1f} mm" if meta.nsd_tolerance_mm > 0 else "N/A"
        )
        
    console.print(table)


@app.command()
def fetch_data(
    output_dir: Path = typer.Option(Path("./data/raw"), "--out", "-o", help="Target output directory for raw data"),
    ct_cases: int = typer.Option(5, "--ct-cases", "-c", help="Number of real CT volumes to download"),
    mr_cases: int = typer.Option(5, "--mr-cases", "-m", help="Number of real MRI volumes to download"),
):
    """Acquire real multi-modality CT & MRI patient volumes from AMOS22 benchmark."""
    from crossscan.data.fetcher import fetch_amos_cohort
    fetch_amos_cohort(output_base_dir=output_dir, num_ct_cases=ct_cases, num_mr_cases=mr_cases)


@app.command()
def download_manifest(
    dataset: str = typer.Option("amos_ct", help="Dataset key: amos_ct, amos_mr, totalsegmentator_ct, totalsegmentator_mr, chaos"),
    output_dir: str = typer.Option("./data", help="Target directory for downloads")
):
    """Inspect official peer-reviewed datasets and generate manifest."""
    from crossscan.data.download_manifest import DATASET_CATALOG
    
    if dataset not in DATASET_CATALOG:
        console.print(f"[red]Error: Dataset '{dataset}' not in catalog. Choose from: {list(DATASET_CATALOG.keys())}[/red]")
        raise typer.Exit(code=1)
        
    src = DATASET_CATALOG[dataset]
    console.rule(f"[bold green]Dataset: {src.name}[/bold green]")
    console.print(f"[bold]Modality:[/bold] {src.modality}")
    console.print(f"[bold]Volumes:[/bold] {src.num_volumes}")
    console.print(f"[bold]Citation:[/bold] {src.citation}")
    console.print(f"[bold]Zenodo URL:[/bold] {src.zenodo_url}")
    console.print(f"[bold]License:[/bold] {src.license}")


@app.command()
def prepare_cohort(
    raw_dir: Path = typer.Option(Path("./data/raw"), "--raw-dir", "-r", help="Directory containing raw scans"),
    output_dir: Path = typer.Option(Path("./data/processed"), "--out", "-o", help="Target processed directory"),
    splits_out: Path = typer.Option(Path("./data/splits.json"), "--splits-out", "-s", help="Path for splits JSON"),
    n4_bias_correct: bool = typer.Option(False, "--n4", help="Apply N4ITK bias field correction for MRI")
):
    """Harmonize multi-modality raw scans into canonical RAS+, normalized radiometry, and 8-organ space."""
    import subprocess
    import sys
    cmd = [
        sys.executable, "scripts/prepare_data.py", "prepare-cohort",
        "--raw-dir", str(raw_dir),
        "--out", str(output_dir),
        "--splits-out", str(splits_out)
    ]
    if n4_bias_correct:
        cmd.append("--n4")
    subprocess.run(cmd, check=True)


@app.command()
def benchmark(
    ct_baseline: Optional[Path] = typer.Option(None, "--ct-baseline", help="CT baseline checkpoint"),
    mr_baseline: Optional[Path] = typer.Option(None, "--mr-baseline", help="MRI baseline checkpoint"),
    joint_model: Optional[Path] = typer.Option(None, "--joint", help="Joint model checkpoint"),
    num_eval_cases: int = typer.Option(15, "--num-cases", "-n", help="Number of evaluation cases")
):
    """Run the 3-arm benchmark matrix and evaluate cross-modality domain collapse."""
    from scripts.benchmark_models import run_benchmark
    run_benchmark(checkpoint_ct_baseline=ct_baseline, checkpoint_mr_baseline=mr_baseline, checkpoint_joint=joint_model, num_eval_cases=num_eval_cases)


@app.command()
def audit_contamination():
    """Audit evaluation test cohorts against published foundation model pretraining rosters."""
    from scripts.audit_contamination import audit
    audit()


@app.command()
def train_adapter(
    splits_path: Path = typer.Option(Path("./data/splits.json"), "--splits", "-s", help="Path to splits JSON"),
    output_dir: Path = typer.Option(Path("./checkpoints/foundation_adapter"), "--out", "-o", help="Checkpoint directory"),
    rank: int = typer.Option(8, "--rank", "-r", help="LoRA rank"),
    epochs: int = typer.Option(5, "--epochs", "-e", help="Training epochs"),
    batch_size: int = typer.Option(2, "--batch-size", "-b", help="Batch size"),
    lr: float = typer.Option(5e-4, "--lr", help="Learning rate")
):
    """Fine-tune 3D parameter-efficient LoRA adapter on clinical data."""
    from scripts.train_adapter import train as train_lora
    train_lora(splits_path=splits_path, output_dir=output_dir, lora_rank=rank, epochs=epochs, batch_size=batch_size, lr=lr)


@app.command()
def evaluate_interactive(
    splits_path: Path = typer.Option(Path("./data/splits.json"), "--splits", "-s", help="Path to splits JSON"),
    target_dice: float = typer.Option(0.85, "--target-dice", "-t", help="Target Dice score"),
    max_clicks: int = typer.Option(10, "--max-clicks", "-m", help="Max allowable clicks")
):
    """Simulate radiologist interactive correction and measure Clicks-to-Target-Dice (C2D)."""
    from scripts.evaluate_interactive import evaluate as eval_interact
    eval_interact(test_splits=splits_path, target_dice=target_dice, max_clicks=max_clicks)


@app.command()
def evaluate_robustness(
    mode: str = typer.Option("both", "--mode", "-m", help="Shift mode: natural, synthetic, or both"),
    num_cases: int = typer.Option(15, "--num-cases", "-n", help="Number of bootstrap evaluation samples")
):
    """Benchmark performance degradation across natural fleet shifts and synthetic perturbations."""
    from scripts.evaluate_robustness import benchmark_natural_shifts, benchmark_synthetic_shifts
    if mode in ["natural", "both"]:
        benchmark_natural_shifts(num_cases=num_cases)
    if mode in ["synthetic", "both"]:
        benchmark_synthetic_shifts(num_synthetic_cases=num_cases)


@app.command()
def audit_hallucination(
    splits_path: Path = typer.Option(Path("./data/splits.json"), "--splits", "-s", help="Path to splits JSON"),
    checkpoint: Optional[Path] = typer.Option(None, "--checkpoint", "-c", help="Path to translator weights")
):
    """Audit generative cross-modality translation for anatomical hallucinations and volume dissolution."""
    from scripts.audit_hallucination import audit as audit_halluc
    audit_halluc(test_splits=splits_path, checkpoint_translator=checkpoint)


@app.command()
def train_translation(
    splits_path: Path = typer.Option(Path("./data/splits.json"), "--splits", "-s", help="Path to data splits JSON"),
    output_dir: Path = typer.Option(Path("./checkpoints/translation"), "--out", "-o", help="Checkpoint output directory"),
    epochs: int = typer.Option(5, "--epochs", "-e", help="Training epochs"),
    batch_size: int = typer.Option(1, "--batch-size", "-b", help="Batch size")
):
    """Train 3D volumetric translation generator (CT <-> MRI) with gradient edge consistency."""
    from scripts.train_translation import train as train_transl
    train_transl(splits_path=splits_path, output_dir=output_dir, epochs=epochs, batch_size=batch_size)


if __name__ == "__main__":
    app()

