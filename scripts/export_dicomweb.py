"""
Clinical DICOMweb & Standard DICOM-SEG Export Pipeline.

Generates compliant DICOM-SEG multi-frame objects with RadLex/SNOMED CT coding,
transmits them to Orthanc PACS via STOW-RS, and generates OHIF / 3D Slicer viewing links.
"""

from pathlib import Path
from typing import Optional, List
import json
import typer
from rich.console import Console
import numpy as np
import pydicom
from pydicom.dataset import Dataset
from pydicom.uid import generate_uid
import torch

from crossscan.core.ontology import ORGAN_ONTOLOGY, NUM_CLASSES
from crossscan.deployment.dicom_io import export_dicom_seg
from crossscan.deployment.orthanc_client import OrthancDICOMwebClient
from crossscan.data.transforms import load_and_preprocess_volume
from crossscan.models.joint_segmentor import CrossScanJointSegmentor

app = typer.Typer(help="CrossScan DICOMweb Exporter")
console = Console()


def create_mock_reference_dicom_series(
    volume: np.ndarray,
    patient_id: str = "AMOS_TEST_001",
    modality: str = "CT"
) -> List[Dataset]:
    """Creates a sequence of pydicom Datasets preserving spatial geometry for DICOM-SEG export."""
    D, H, W = volume.shape
    study_uid = generate_uid()
    series_uid = generate_uid()
    frame_of_ref = generate_uid()
    
    slices: List[Dataset] = []
    for z in range(D):
        ds = Dataset()
        ds.PatientID = patient_id
        ds.PatientName = f"CrossScan^{patient_id}"
        ds.Modality = modality.upper()
        ds.StudyInstanceUID = study_uid
        ds.SeriesInstanceUID = series_uid
        ds.SOPInstanceUID = generate_uid()
        ds.FrameOfReferenceUID = frame_of_ref
        ds.ImagePositionPatient = ["-180.0", "-180.0", f"{z * 1.5:.2f}"]
        ds.ImageOrientationPatient = ["1.0", "0.0", "0.0", "0.0", "1.0", "0.0"]
        ds.PixelSpacing = ["1.5", "1.5"]
        ds.SliceThickness = "1.5"
        ds.Rows = H
        ds.Columns = W
        ds.PixelRepresentation = 0
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.RescaleSlope = 1.0
        ds.RescaleIntercept = 0.0
        
        # Scale to 16-bit integer
        slice_arr = np.clip(volume[z] * 1000.0, 0, 4095).astype(np.uint16)
        ds.PixelData = slice_arr.tobytes()
        slices.append(ds)
        
    return slices


@app.command()
def export(
    splits_path: Path = typer.Option(Path("./data/splits.json"), "--splits", "-s", help="Path to data splits JSON"),
    checkpoint: Optional[Path] = typer.Option(Path("./checkpoints/joint_model/joint_model_best.pt"), "--checkpoint", "-c", help="Joint model checkpoint"),
    output_dir: Path = typer.Option(Path("./output/dicom_export"), "--out", "-o", help="Output directory for DICOM-SEG"),
    orthanc_url: str = typer.Option("http://localhost:8042", "--orthanc", help="Orthanc PACS server URL"),
    device: Optional[str] = typer.Option(None, "--device", help="Compute device")
):
    """Generates standard DICOM-SEG and transmits to Orthanc PACS via STOW-RS."""
    output_dir.mkdir(parents=True, exist_ok=True)
    console.rule("[bold cyan]Clinical DICOM-SEG Generation & DICOMweb PACS Transmission[/bold cyan]")
    
    if not isinstance(device, str) or not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    
    with open(splits_path, "r") as f:
        splits = json.load(f)
        
    test_cases = splits.get("test", []) or splits.get("val", [])
    case = test_cases[0]
    case_id = case.get("patient_id", case.get("id", "Patient_001"))
    modality = case.get("modality", "ct").upper()
    img_p = Path(case["image"])
    
    console.print(f"Ingesting test volume: [bold]{case_id}[/bold] ({modality}) from {img_p}")
    vol_tensor = load_and_preprocess_volume(img_p, modality=modality.lower()).unsqueeze(0).to(dev)
    
    # Run segmentation inference
    model = CrossScanJointSegmentor(in_channels=1, out_classes=NUM_CLASSES, feature_size=32, num_scales=4).to(dev)
    if checkpoint and checkpoint.exists():
        ckpt = torch.load(checkpoint, map_location=dev)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    with torch.no_grad():
        logits = model(vol_tensor)
        if torch.is_tensor(logits) and logits.ndim == 6:
            logits = logits[:, 0]
        probs = torch.softmax(logits, dim=1)
        pred_mask = torch.argmax(probs, dim=1)[0].cpu().numpy().astype(np.int32)
        
    vol_np = vol_tensor[0, 0].cpu().numpy()
    
    console.print(f"Creating reference DICOM metadata and spatial headers...")
    source_slices = create_mock_reference_dicom_series(vol_np, patient_id=case_id, modality=modality)
    
    # Export DICOM-SEG
    seg_file = output_dir / f"{case_id}_CrossScan_SEG.dcm"
    console.print(f"Exporting standard DICOM-SEG object to [bold cyan]{seg_file}[/bold cyan]...")
    export_dicom_seg(pred_mask, source_slices, seg_file, series_description="CrossScan 3D Multi-Organ Segmentation")
    console.print(f"[green]DICOM-SEG successfully created! Size: {seg_file.stat().st_size / 1024:.1f} KB[/green]")
    console.print(f"SOP Class UID: [bold]1.2.840.10008.5.1.4.1.1.66.4 (Segmentation Storage)[/bold]")
    
    # Transmit via STOW-RS to Orthanc PACS
    client = OrthancDICOMwebClient(base_url=orthanc_url)
    console.print(f"\nAttempting STOW-RS upload to Orthanc PACS: {orthanc_url}/dicom-web/studies...")
    stow_res = client.stow_store_instance(seg_file)
    
    if stow_res.get("success"):
        console.print(f"[bold green]STOW-RS Upload Succeeded! (Status: {stow_res['status_code']})[/bold green]")
    else:
        console.print(f"[yellow]STOW-RS Status: {stow_res.get('error', 'Simulated offline PACS environment')}[/yellow]")
        console.print("[dim]Note: In production with running Orthanc, this instance is immediately accessible via DICOMweb REST API.[/dim]")
        
    study_uid = source_slices[0].StudyInstanceUID
    ohif_url = client.get_ohif_viewer_url(study_uid)
    slicer_cmd = client.get_slicer_command(output_dir, seg_file)
    
    console.rule("[bold magenta]Clinical Interoperability Links[/bold magenta]")
    console.print(f"• [bold]OHIF Web Viewer Link:[/bold] [link={ohif_url}]{ohif_url}[/link]")
    console.print(f"• [bold]3D Slicer Direct Launch:[/bold] [cyan]{slicer_cmd}[/cyan]")


if __name__ == "__main__":
    app()

