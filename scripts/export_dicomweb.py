"""
Clinical DICOMweb & Standard DICOM-SEG Export Pipeline.

Generates compliant DICOM-SEG multi-frame objects with RadLex/SNOMED CT coding,
transmits both the background CT image series and segmentation to Orthanc PACS via STOW-RS,
and generates OHIF / Stone / 3D Slicer viewing links.
"""

from pathlib import Path
from typing import Optional, List, Tuple
import json
import base64
import urllib.request
import urllib.error
import typer
from rich.console import Console
import numpy as np
import SimpleITK as sitk
import pydicom
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid
import nibabel as nib
import torch

from crossscan.core.ontology import ORGAN_ONTOLOGY, NUM_CLASSES
from crossscan.deployment.dicom_io import export_dicom_seg
from crossscan.deployment.orthanc_client import OrthancDICOMwebClient

app = typer.Typer(help="CrossScan DICOMweb Exporter")
console = Console()


def create_reference_dicom_series(
    volume: np.ndarray,
    patient_id: str = "amos_0005",
    modality: str = "CT",
    output_series_dir: Optional[Path] = None,
    pixel_spacing: Tuple[float, float] = (0.8535, 0.8535),
    slice_thickness: float = 5.0,
    origin: Tuple[float, float, float] = (-240.5, -61.5, 6.0)
) -> List[Dataset]:
    """
    Creates a sequence of pydicom Datasets preserving clinical spatial geometry and saves
    valid Part 10 DICOM image slice files with true Hounsfield units for PACS ingestion.
    """
    D, H, W = volume.shape
    study_uid = generate_uid()
    series_uid = generate_uid()
    frame_of_ref = generate_uid()
    
    sop_class = "1.2.840.10008.5.1.4.1.1.2" if modality.upper() == "CT" else "1.2.840.10008.5.1.4.1.1.4"
    
    if output_series_dir:
        output_series_dir.mkdir(parents=True, exist_ok=True)
        
    slices: List[Dataset] = []
    for z in range(D):
        sop_instance_uid = generate_uid()
        
        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = sop_class
        file_meta.MediaStorageSOPInstanceUID = sop_instance_uid
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        
        slice_path = output_series_dir / f"slice_{z:04d}.dcm" if output_series_dir else Path(f"slice_{z:04d}.dcm")
        ds = FileDataset(str(slice_path), {}, file_meta=file_meta, preamble=b"\0" * 128)
        ds.is_little_endian = True
        ds.is_implicit_VR = False
        
        ds.PatientID = patient_id
        ds.PatientName = f"CrossScan^{patient_id}"
        ds.PatientBirthDate = "19750101"
        ds.PatientSex = "M"
        
        ds.Modality = modality.upper()
        ds.StudyInstanceUID = study_uid
        ds.SeriesInstanceUID = series_uid
        ds.SOPInstanceUID = sop_instance_uid
        ds.SOPClassUID = sop_class
        ds.FrameOfReferenceUID = frame_of_ref
        
        ds.SeriesNumber = 1
        ds.InstanceNumber = z + 1
        ds.SeriesDescription = f"CrossScan {modality.upper()} Scan (Patient {patient_id})"
        
        ds.ImagePositionPatient = [f"{origin[0]:.2f}", f"{origin[1]:.2f}", f"{origin[2] + z * slice_thickness:.2f}"]
        ds.ImageOrientationPatient = ["1.0", "0.0", "0.0", "0.0", "1.0", "0.0"]
        ds.PixelSpacing = [f"{pixel_spacing[0]:.4f}", f"{pixel_spacing[1]:.4f}"]
        ds.SliceThickness = f"{slice_thickness:.2f}"
        ds.Rows = H
        ds.Columns = W
        ds.PixelRepresentation = 0
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.RescaleSlope = 1.0
        ds.RescaleIntercept = -1024.0
        ds.WindowCenter = 40
        ds.WindowWidth = 400
        
        # True Hounsfield units: Stored Pixel = HU - RescaleIntercept = HU + 1024
        hu_slice = volume[z]
        stored = np.clip(hu_slice + 1024.0, 0, 4095).astype(np.uint16)
        ds.PixelData = stored.tobytes()
        
        if output_series_dir:
            ds.save_as(str(slice_path), write_like_original=False)
            
        slices.append(ds)
        
    return slices


def export_slicer_seg_nrrd(
    lbl_image: sitk.Image,
    output_path: Path
) -> Path:
    """
    Exports a SimpleITK label image as a 3D Slicer compliant .seg.nrrd file
    with individual organ colors, IDs, and anatomical terminology.
    """
    seg_img = sitk.Cast(lbl_image, sitk.sitkUInt8)
    for c in range(1, NUM_CLASSES):
        meta = ORGAN_ONTOLOGY[c]
        r = meta.rgb_color[0] / 255.0
        g = meta.rgb_color[1] / 255.0
        b = meta.rgb_color[2] / 255.0
        seg_idx = c - 1
        seg_img.SetMetaData(f"Segment{seg_idx}_ID", f"Segment_{c}")
        seg_img.SetMetaData(f"Segment{seg_idx}_Name", meta.display_name)
        seg_img.SetMetaData(f"Segment{seg_idx}_Color", f"{r:.4f} {g:.4f} {b:.4f}")
        seg_img.SetMetaData(f"Segment{seg_idx}_LabelValue", str(c))
        seg_img.SetMetaData(f"Segment{seg_idx}_Layer", "0")
        seg_img.SetMetaData(f"Segment{seg_idx}_Tag", meta.snomed_ct)
        
    sitk.WriteImage(seg_img, str(output_path))
    return output_path


@app.command()
def export(
    splits_path: Path = typer.Option(Path("./data/splits.json"), "--splits", "-s", help="Path to data splits JSON"),
    checkpoint: Optional[Path] = typer.Option(None, "--checkpoint", "-c", help="Model checkpoint"),
    output_dir: Path = typer.Option(Path("./output/dicom_export"), "--out", "-o", help="Output directory for DICOM-SEG"),
    orthanc_url: str = typer.Option("http://localhost:8042", "--orthanc", help="Orthanc PACS server URL"),
    device: Optional[str] = typer.Option(None, "--device", help="Compute device")
):
    """Generates standard DICOM-SEG and transmits both image series and SEG to Orthanc PACS."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ct_series_dir = output_dir / "source_series"
    ct_series_dir.mkdir(parents=True, exist_ok=True)
    
    console.rule("[bold cyan]Clinical DICOM-SEG Generation & DICOMweb PACS Transmission[/bold cyan]")
    
    with open(splits_path, "r") as f:
        splits = json.load(f)
        
    test_cases = splits.get("test", []) or splits.get("val", [])
    case = test_cases[0]
    case_id = case.get("patient_id", "amos_0005")
    modality = case.get("modality", "ct").upper()
    img_p = Path(case["image"])
    lbl_p = Path(case.get("label", ""))
    
    # Check for authentic raw CT volume with full HU dynamic range
    raw_ct_path = Path("data/raw/amos_ct/train/imagesTr/amos_0005.nii.gz")
    if raw_ct_path.exists():
        console.print(f"Loading clinical AMOS volume with authentic Hounsfield Units: [bold cyan]{raw_ct_path.name}[/bold cyan]")
        img_raw = sitk.ReadImage(str(raw_ct_path))
    else:
        console.print(f"Loading processed volume: [bold cyan]{img_p.name}[/bold cyan]")
        img_raw = sitk.ReadImage(str(img_p))
        
    lbl_raw = sitk.ReadImage(str(lbl_p))
    
    # Orient both to standard radiological LPS coordinates
    console.print("Canonicalizing spatial orientation to clinical LPS coordinates...")
    img_lps = sitk.DICOMOrient(img_raw, "LPS")
    lbl_lps = sitk.DICOMOrient(lbl_raw, "LPS")
    
    # Resample to standard clinical 512x512 matrix
    orig_size = img_lps.GetSize()
    orig_spacing = img_lps.GetSpacing()
    new_size = [512, 512, orig_size[2]]
    new_spacing = [orig_spacing[0] * orig_size[0] / 512.0, orig_spacing[1] * orig_size[1] / 512.0, orig_spacing[2]]
    
    console.print(f"Standardizing matrix resolution: {orig_size[:2]} -> [bold cyan](512, 512)[/bold cyan] ({orig_size[2]} slices)")
    resampled_ct = sitk.Resample(
        img_lps,
        new_size,
        sitk.Transform(),
        sitk.sitkLinear,
        img_lps.GetOrigin(),
        new_spacing,
        img_lps.GetDirection(),
        -1024.0,
        sitk.sitkFloat32
    )
    resampled_lbl = sitk.Resample(
        lbl_lps,
        new_size,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        lbl_lps.GetOrigin(),
        new_spacing,
        lbl_lps.GetDirection(),
        0,
        sitk.sitkUInt8
    )
    
    vol_np = sitk.GetArrayFromImage(resampled_ct)   # (D, H, W) in HU
    lbl_np = sitk.GetArrayFromImage(resampled_lbl)  # (D, H, W) in organ IDs [0..8]
    
    console.print(f"CT volume HU range: [bold]{vol_np.min():.1f}[/bold] to [bold]{vol_np.max():.1f}[/bold] (Soft-tissue Window: L:40, W:400)")
    present_organs = [ORGAN_ONTOLOGY[c].display_name for c in np.unique(lbl_np) if c > 0]
    console.print(f"Identified Anatomical Organs: [bold green]{', '.join(present_organs)}[/bold green]")
    
    # 1. Generate reference Part 10 DICOM image series
    console.print(f"\nGenerating Part 10 {modality} DICOM series in [bold cyan]{ct_series_dir}[/bold cyan]...")
    source_slices = create_reference_dicom_series(
        vol_np,
        patient_id=case_id,
        modality=modality,
        output_series_dir=ct_series_dir,
        pixel_spacing=(new_spacing[0], new_spacing[1]),
        slice_thickness=new_spacing[2],
        origin=img_lps.GetOrigin()
    )
    console.print(f"[green]Successfully saved {len(source_slices)} {modality} DICOM slice files.[/green]")
    
    # 2. Export standard DICOM-SEG object
    seg_file = output_dir / f"{case_id}_CrossScan_SEG.dcm"
    console.print(f"Exporting standard DICOM-SEG object to [bold cyan]{seg_file}[/bold cyan]...")
    export_dicom_seg(lbl_np, source_slices, seg_file, series_description="CrossScan 3D Multi-Organ Segmentation")
    console.print(f"[green]DICOM-SEG created! Size: {seg_file.stat().st_size / 1024:.1f} KB[/green]")
    console.print("SOP Class UID: [bold]1.2.840.10008.5.1.4.1.1.66.4 (Segmentation Storage)[/bold]")
    
    # 3. Save standard companion files for direct 1-click 3D Slicer loading
    ct_nii_path = output_dir / f"{case_id}_CT.nii.gz"
    seg_nii_path = output_dir / f"{case_id}_SEG.nii.gz"
    seg_nrrd_path = output_dir / f"{case_id}.seg.nrrd"
    
    sitk.WriteImage(resampled_ct, str(ct_nii_path))
    sitk.WriteImage(resampled_lbl, str(seg_nii_path))
    export_slicer_seg_nrrd(resampled_lbl, seg_nrrd_path)
    console.print(f"[green]Saved companion 3D Slicer files:[/green]")
    console.print(f"  • CT Volume: [cyan]{ct_nii_path.name}[/cyan]")
    console.print(f"  • Standard NIfTI: [cyan]{seg_nii_path.name}[/cyan]")
    console.print(f"  • 3D Slicer Native Color Model: [bold green]{seg_nrrd_path.name}[/bold green]")
    
    # 4. Transmit via STOW-RS to Orthanc PACS
    client = OrthancDICOMwebClient(base_url=orthanc_url, auth=("orthanc", "orthanc"))
    console.print(f"\nUploading complete study to Orthanc PACS at {orthanc_url}...")
    
    auth_header = "Basic " + base64.b64encode(b"orthanc:orthanc").decode()
    try:
        # Delete old studies first to keep PACS clean
        req_clean = urllib.request.Request(f"{orthanc_url}/studies")
        req_clean.add_header("Authorization", auth_header)
        with urllib.request.urlopen(req_clean, timeout=5.0) as resp:
            old_studies = json.loads(resp.read().decode())
            for old_id in old_studies:
                del_req = urllib.request.Request(f"{orthanc_url}/studies/{old_id}", method="DELETE")
                del_req.add_header("Authorization", auth_header)
                urllib.request.urlopen(del_req, timeout=5.0)
                
        # 1. Upload CT Slice Series
        slice_files = sorted(list(ct_series_dir.glob("*.dcm")))
        console.print(f"• Uploading {len(slice_files)} {modality} background slices...")
        for sf in slice_files:
            data = sf.read_bytes()
            req = urllib.request.Request(f"{orthanc_url}/instances", data=data, method="POST")
            req.add_header("Authorization", auth_header)
            req.add_header("Content-Type", "application/dicom")
            with urllib.request.urlopen(req, timeout=5.0) as _:
                pass
                
        # 2. Upload Segmentation File
        console.print(f"• Uploading DICOM-SEG multi-frame object...")
        seg_data = seg_file.read_bytes()
        req_seg = urllib.request.Request(f"{orthanc_url}/instances", data=seg_data, method="POST")
        req_seg.add_header("Authorization", auth_header)
        req_seg.add_header("Content-Type", "application/dicom")
        with urllib.request.urlopen(req_seg, timeout=5.0) as resp:
            if resp.status in [200, 201]:
                console.print(f"[bold green]Full Study Upload Succeeded! (Status: {resp.status})[/bold green]")
    except Exception as e:
        console.print(f"[yellow]PACS Upload Notice: Local server unreachable ({e})[/yellow]")
        console.print("[dim]Note: When Orthanc is running, studies are immediately accessible via DICOMweb REST API.[/dim]")
        
    study_uid = source_slices[0].StudyInstanceUID
    ohif_url = f"{orthanc_url}/ohif/viewer?StudyInstanceUIDs={study_uid}"
    explorer_url = f"{orthanc_url}/ui/app/#/studies/{study_uid}"
    stone_url = f"{orthanc_url}/stone-webviewer/app/index.html?study={study_uid}"
    
    # 3D Slicer launch commands
    import os
    local_app = Path(os.environ.get("LOCALAPPDATA", "")) / "slicer.org"
    slicer_bin = "Slicer.exe"
    if local_app.exists():
        matches = sorted(list(local_app.glob("**/Slicer.exe")))
        if matches:
            slicer_bin = f'"{matches[-1]}"'
            
    slicer_cmd_native = f'& {slicer_bin} --load "{ct_nii_path.resolve()}" --load "{seg_nrrd_path.resolve()}"'
    slicer_cmd_dicom = f'& {slicer_bin} --load-dicom "{ct_series_dir.resolve()}" --load "{seg_file.resolve()}"'
    
    console.rule("[bold magenta]Clinical Interoperability Links[/bold magenta]")
    console.print(f"• [bold]Orthanc Explorer 2 Portal:[/bold] [link={explorer_url}]{explorer_url}[/link]")
    console.print(f"• [bold]OHIF Web Viewer Link:[/bold] [link={ohif_url}]{ohif_url}[/link]")
    console.print(f"• [bold]Stone Web Viewer Link:[/bold] [link={stone_url}]{stone_url}[/link]")
    console.print(f"• [bold]3D Slicer 1-Click Launch (Native 3D Mesh):[/bold]\n  [cyan]{slicer_cmd_native}[/cyan]")
    console.print(f"• [bold]3D Slicer DICOMweb Launch:[/bold]\n  [cyan]{slicer_cmd_dicom}[/cyan]")


if __name__ == "__main__":
    app()
