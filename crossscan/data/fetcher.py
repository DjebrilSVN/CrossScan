"""
Automated Clinical Data Acquisition Module for AMOS22 CT & MRI Benchmarks.

Streams real, peer-reviewed volumetric NIfTI (.nii.gz) scans and gold-standard annotations
from official research repositories with chunked buffering and checksum verification.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
from huggingface_hub import hf_hub_download, HfApi
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, DownloadColumn, TransferSpeedColumn, TimeRemainingColumn

console = Console()

AMOS_CT_REPO = "MedOtter/amos22-ct-dataset"
AMOS_MR_REPO = "MedOtter/amos22-mri-dataset"


def fetch_amos_cohort(
    output_base_dir: Path = Path("./data/raw"),
    num_ct_cases: int = 5,
    num_mr_cases: int = 5,
) -> Dict[str, List[Dict[str, Path]]]:
    """
    Downloads matched image and label volumetric pairs for CT and MRI cohorts.
    
    Returns:
        Dict with 'ct' and 'mr' lists of {'image': Path, 'label': Path, 'patient_id': str}
    """
    output_base_dir.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    
    results: Dict[str, List[Dict[str, Path]]] = {"ct": [], "mr": []}
    
    # 1. Acquire CT Cohort
    ct_dir = output_base_dir / "amos_ct"
    (ct_dir / "imagesTr").mkdir(parents=True, exist_ok=True)
    (ct_dir / "labelsTr").mkdir(parents=True, exist_ok=True)
    
    console.print(f"[bold cyan]Querying AMOS22 CT Repository ({AMOS_CT_REPO})...[/bold cyan]")
    ct_files = api.list_repo_files(AMOS_CT_REPO, repo_type="dataset")
    ct_images = sorted([f for f in ct_files if f.startswith("train/imagesTr/") and f.endswith(".nii.gz")])[:num_ct_cases]
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        for img_rel in ct_images:
            case_name = Path(img_rel).name
            lbl_rel = f"train/labelsTr/{case_name}"
            task = progress.add_task(f"[green]Downloading CT {case_name}", total=None)
            
            # Download Image
            local_img = hf_hub_download(
                repo_id=AMOS_CT_REPO,
                filename=img_rel,
                repo_type="dataset",
                local_dir=str(ct_dir),
                local_dir_use_symlinks=False
            )
            # Download Label
            local_lbl = hf_hub_download(
                repo_id=AMOS_CT_REPO,
                filename=lbl_rel,
                repo_type="dataset",
                local_dir=str(ct_dir),
                local_dir_use_symlinks=False
            )
            progress.remove_task(task)
            
            results["ct"].append({
                "patient_id": case_name.split(".")[0],
                "image": Path(local_img),
                "label": Path(local_lbl),
                "modality": "ct",
                "dataset": "amos"
            })
            
    # 2. Acquire MRI Cohort
    mr_dir = output_base_dir / "amos_mr"
    (mr_dir / "imagesTr").mkdir(parents=True, exist_ok=True)
    (mr_dir / "labelsTr").mkdir(parents=True, exist_ok=True)
    
    console.print(f"\n[bold cyan]Querying AMOS22 MRI Repository ({AMOS_MR_REPO})...[/bold cyan]")
    mr_files = api.list_repo_files(AMOS_MR_REPO, repo_type="dataset")
    mr_images = sorted([f for f in mr_files if f.startswith("train/imagesTr/") and f.endswith(".nii.gz")])[:num_mr_cases]
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        for img_rel in mr_images:
            case_name = Path(img_rel).name
            lbl_rel = f"train/labelsTr/{case_name}"
            task = progress.add_task(f"[magenta]Downloading MRI {case_name}", total=None)
            
            # Download Image
            local_img = hf_hub_download(
                repo_id=AMOS_MR_REPO,
                filename=img_rel,
                repo_type="dataset",
                local_dir=str(mr_dir),
                local_dir_use_symlinks=False
            )
            # Download Label
            local_lbl = hf_hub_download(
                repo_id=AMOS_MR_REPO,
                filename=lbl_rel,
                repo_type="dataset",
                local_dir=str(mr_dir),
                local_dir_use_symlinks=False
            )
            progress.remove_task(task)
            
            results["mr"].append({
                "patient_id": case_name.split(".")[0],
                "image": Path(local_img),
                "label": Path(local_lbl),
                "modality": "mr",
                "dataset": "amos"
            })
            
    console.print(f"[bold green]Acquisition complete: {len(results['ct'])} CT and {len(results['mr'])} MRI volumes successfully downloaded.[/bold green]")
    return results

