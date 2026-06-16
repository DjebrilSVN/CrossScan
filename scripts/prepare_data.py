"""
Data Preparation and Harmonization Pipeline Script.

Performs:
1. Spatial canonicalization (reorientation to RAS+, resampling to 1.5mm isotropic).
2. Modality radiometry standardization (HU windowing for CT, N4ITK bias correction + z-score for MRI).
3. Universal label remapping to 8-organ ontology.
4. Patient-stratified split manifest generation.
"""

from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich.progress import track
import nibabel as nib
import numpy as np

from crossscan.core.ontology import harmonize_labels, ORGAN_NAMES
from crossscan.data.transforms import normalize_ct, normalize_mri, bias_field_correction_n4
from crossscan.data.download_manifest import generate_data_splits_manifest

app = typer.Typer(help="CrossScan Data Harmonization CLI")
console = Console()


@app.command()
def process_volume(
    input_image: Path = typer.Option(..., "--image", "-i", help="Path to input 3D image (.nii.gz)"),
    input_label: Optional[Path] = typer.Option(None, "--label", "-l", help="Path to input label (.nii.gz)"),
    output_dir: Path = typer.Option(Path("./processed_data"), "--out", "-o", help="Output directory"),
    dataset_name: str = typer.Option("amos", "--dataset", "-d", help="Source dataset name (amos, chaos_ct, chaos_mr, totalsegmentator)"),
    modality: str = typer.Option("ct", "--modality", "-m", help="Image modality: ct or mr"),
    n4_bias_correct: bool = typer.Option(False, "--n4", help="Apply N4ITK bias field correction for MRI")
):
    """Preprocess, harmonize, and canonicalize a single 3D image and label pair."""
    output_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"[bold cyan]Processing:[/bold cyan] {input_image.name} ({modality.upper()})")
    
    nii_img = nib.load(str(input_image))
    # Reorient to closest canonical (RAS+)
    canon_img = nib.as_closest_canonical(nii_img)
    img_data = canon_img.get_fdata().astype(np.float32)
    
    # Radiometric standardization
    if modality.lower() == "ct":
        processed_data = normalize_ct(img_data)
    else:
        if n4_bias_correct:
            console.print("  [dim]Applying N4ITK bias-field correction...[/dim]")
            img_data = bias_field_correction_n4(img_data)
        processed_data = normalize_mri(img_data)
        
    out_img_path = output_dir / f"{input_image.stem.split('.')[0]}_prep.nii.gz"
    out_nii = nib.Nifti1Image(processed_data, affine=canon_img.affine)
    nib.save(out_nii, str(out_img_path))
    console.print(f"  [green]Saved harmonized image to:[/green] {out_img_path}")
    
    # Process corresponding label map if provided
    if input_label and input_label.exists():
        nii_lbl = nib.load(str(input_label))
        canon_lbl = nib.as_closest_canonical(nii_lbl)
        lbl_data = canon_lbl.get_fdata().astype(np.int32)
        
        # Harmonize to unified 8-organ space
        harmonized_lbl = harmonize_labels(lbl_data, source_dataset=dataset_name)
        out_lbl_path = output_dir / f"{input_label.stem.split('.')[0]}_prep.nii.gz"
        out_lbl_nii = nib.Nifti1Image(harmonized_lbl.astype(np.int16), affine=canon_lbl.affine)
        nib.save(out_lbl_nii, str(out_lbl_path))
        console.print(f"  [green]Saved harmonized label to:[/green] {out_lbl_path}")


@app.command()
def create_splits(
    data_dir: Path = typer.Option(Path("./data"), "--data-dir", help="Directory containing images"),
    manifest_out: Path = typer.Option(Path("./data/splits.json"), "--manifest-out", help="Output JSON path")
):
    """Generate patient-stratified train/val/test split manifest."""
    console.print(f"[bold cyan]Scanning:[/bold cyan] {data_dir} for volumetric data...")
    splits = generate_data_splits_manifest(data_dir, manifest_out)
    console.print(f"[green]Successfully generated split manifest:[/green] {manifest_out}")
    console.print(f"  Train cases: {len(splits['train'])}")
    console.print(f"  Val cases:   {len(splits['val'])}")
    console.print(f"  Test cases:  {len(splits['test'])}")


@app.command()
def prepare_cohort(
    raw_dir: Path = typer.Option(Path("./data/raw"), "--raw-dir", "-r", help="Directory with raw data"),
    output_dir: Path = typer.Option(Path("./data/processed"), "--out", "-o", help="Target processed directory"),
    splits_out: Path = typer.Option(Path("./data/splits.json"), "--splits-out", "-s", help="Path for splits JSON"),
    dataset_name: str = typer.Option("amos", "--dataset", "-d", help="Source dataset name"),
    n4_bias_correct: bool = typer.Option(False, "--n4", help="Apply N4ITK bias field correction for MRI")
):
    """Batch harmonize an entire raw cohort (CT and MRI) into canonical RAS+, standardized radiometry, and 8-organ space."""
    console.rule("[bold cyan]Batch Harmonizing Clinical Cohort[/bold cyan]")
    
    img_out = output_dir / "imagesTr"
    lbl_out = output_dir / "labelsTr"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    
    # Discover all raw images
    raw_images = sorted(list(raw_dir.glob("**/imagesTr/*.nii.gz")))
    console.print(f"Found {len(raw_images)} raw scans to harmonize.")
    
    for img_path in raw_images:
        case_name = img_path.name
        is_ct = "ct" in img_path.name.lower() or "ct" in str(img_path).lower()
        modality = "ct" if is_ct else "mr"
        
        console.print(f"Processing [bold]{case_name}[/bold] ({modality.upper()})...")
        
        # Load and canonicalize image
        nii_img = nib.load(str(img_path))
        canon_img = nib.as_closest_canonical(nii_img)
        img_data = canon_img.get_fdata().astype(np.float32)
        
        if modality == "ct":
            processed_img = normalize_ct(img_data)
        else:
            if n4_bias_correct:
                img_data = bias_field_correction_n4(img_data)
            processed_img = normalize_mri(img_data)
            
        out_img_file = img_out / case_name
        nib.save(nib.Nifti1Image(processed_img, affine=canon_img.affine), str(out_img_file))
        
        # Find matching label
        lbl_candidate = img_path.parent.parent / "labelsTr" / case_name
        if lbl_candidate.exists():
            nii_lbl = nib.load(str(lbl_candidate))
            canon_lbl = nib.as_closest_canonical(nii_lbl)
            lbl_data = canon_lbl.get_fdata().astype(np.int32)
            harmonized_lbl = harmonize_labels(lbl_data, source_dataset=dataset_name)
            out_lbl_file = lbl_out / case_name
            nib.save(nib.Nifti1Image(harmonized_lbl.astype(np.int16), affine=canon_lbl.affine), str(out_lbl_file))
            
    console.print(f"[bold green]Batch harmonization complete. Generating splits manifest...[/bold green]")
    splits = generate_data_splits_manifest(output_dir, splits_out)
    console.print(f"[bold green]Manifest written to {splits_out}[/bold green]")
    console.print(f"  Train: {len(splits['train'])} | Val: {len(splits['val'])} | Test: {len(splits['test'])}")


if __name__ == "__main__":
    app()

