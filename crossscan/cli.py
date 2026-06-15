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


if __name__ == "__main__":
    app()

