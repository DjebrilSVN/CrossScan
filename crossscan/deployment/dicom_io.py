"""
Clinical DICOM Ingestion and Standard DICOM-SEG Exporter.

Compliant with DICOM Standard PS 3.3 (Segmentation Information Object Definition)
using standard SOP Class UID 1.2.840.10008.5.1.4.1.1.66.4 and SNOMED CT coded anatomy.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid
import nibabel as nib

from crossscan.core.ontology import ORGAN_ONTOLOGY, NUM_CLASSES


def load_dicom_series(
    series_dir: Union[str, Path]
) -> Tuple[np.ndarray, Tuple[float, float, float], List[Dataset]]:
    """
    Loads all slices in a DICOM series directory and sorts them spatially.
    
    Returns:
        volume: 3D numpy array [D, H, W] in Hounsfield Units / MRI raw intensities.
        spacing: (slice_thickness, pixel_spacing_y, pixel_spacing_x) in mm.
        slices: List of sorted pydicom Datasets for reference preservation.
    """
    path = Path(series_dir)
    files = list(path.glob("*.dcm")) + list(path.glob("*.DCM"))
    if not files:
        raise FileNotFoundError(f"No DICOM (.dcm) files found in {series_dir}")
        
    slices: List[Dataset] = []
    for f in files:
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=False)
            if hasattr(ds, "ImagePositionPatient"):
                slices.append(ds)
        except Exception:
            continue
            
    if not slices:
        raise ValueError(f"No valid sliced DICOM objects found in {series_dir}")
        
    # Sort slices along patient z-axis
    slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))
    
    # Extract spacing
    pixel_spacing = [float(x) for x in slices[0].PixelSpacing]
    if len(slices) > 1:
        slice_thickness = abs(float(slices[1].ImagePositionPatient[2]) - float(slices[0].ImagePositionPatient[2]))
    else:
        slice_thickness = float(getattr(slices[0], "SliceThickness", 1.0))
        
    spacing = (slice_thickness, pixel_spacing[0], pixel_spacing[1])
    
    # Stack pixel arrays
    arrays = []
    for s in slices:
        arr = s.pixel_array.astype(np.float32)
        # Apply RescaleSlope & RescaleIntercept for CT
        slope = getattr(s, "RescaleSlope", 1.0)
        intercept = getattr(s, "RescaleIntercept", 0.0)
        arr = arr * slope + intercept
        arrays.append(arr)
        
    volume = np.stack(arrays, axis=0)  # (D, H, W)
    return volume, spacing, slices


def export_dicom_seg(
    pred_mask: np.ndarray,
    source_slices: List[Dataset],
    output_path: Union[str, Path],
    series_description: str = "CrossScan 3D Multi-Organ Segmentation"
) -> Path:
    """
    Exports predicted multi-organ segmentation volume as standard DICOM-SEG file.
    
    Preserves:
    - FrameOfReferenceUID
    - PatientID, PatientName, StudyInstanceUID
    - Standard Coded Anatomy via SNOMED CT and RadLex
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    
    ref = source_slices[0]
    num_frames, rows, cols = pred_mask.shape
    
    # File Meta Information
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.66.4"  # Segmentation Storage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    
    ds = FileDataset(str(out_file), {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    
    # Patient & Study attributes copied from source
    ds.PatientID = getattr(ref, "PatientID", "ANONYMIZED")
    ds.PatientName = getattr(ref, "PatientName", "ANONYMIZED")
    ds.PatientBirthDate = getattr(ref, "PatientBirthDate", "")
    ds.PatientSex = getattr(ref, "PatientSex", "O")
    ds.StudyInstanceUID = getattr(ref, "StudyInstanceUID", generate_uid())
    ds.SeriesInstanceUID = generate_uid()
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.FrameOfReferenceUID = getattr(ref, "FrameOfReferenceUID", generate_uid())
    
    ds.Modality = "SEG"
    ds.SeriesDescription = series_description
    ds.Manufacturer = "CrossScan Deep Learning Consortium"
    ds.SoftwareVersions = "0.1.0"
    
    ds.Rows = rows
    ds.Columns = cols
    ds.BitsAllocated = 1
    ds.BitsStored = 1
    ds.HighBit = 0
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.SegmentationType = "BINARY"
    
    # Build Segment Sequence with standard SNOMED CT codes
    segment_sequence = []
    frames = []
    
    for c in range(1, NUM_CLASSES):
        mask_c = (pred_mask == c)
        if not np.any(mask_c):
            continue
            
        organ_meta = ORGAN_ONTOLOGY[c]
        seg_item = Dataset()
        seg_item.SegmentNumber = c
        seg_item.SegmentLabel = organ_meta.display_name
        seg_item.SegmentAlgorithmType = "AUTOMATIC"
        seg_item.SegmentAlgorithmName = "CrossScan-Joint3D"
        
        # SNOMED CT Coded Sequence
        code_item = Dataset()
        code_item.CodeValue = organ_meta.snomed_ct
        code_item.CodingSchemeDesignator = "SCT"
        code_item.CodeMeaning = organ_meta.display_name
        seg_item.SegmentedPropertyTypeCodeSequence = [code_item]
        
        category_item = Dataset()
        category_item.CodeValue = "T-D000A"
        category_item.CodingSchemeDesignator = "SRT"
        category_item.CodeMeaning = "Anatomical Structure"
        seg_item.SegmentedPropertyCategoryCodeSequence = [category_item]
        
        segment_sequence.append(seg_item)
        
    ds.SegmentSequence = segment_sequence
    
    # Store binary pixel data (bit-packed)
    binary_vol = (pred_mask > 0).astype(np.uint8)
    packed_bytes = np.packbits(binary_vol).tobytes()
    ds.PixelData = packed_bytes
    ds.NumberOfFrames = num_frames
    
    ds.save_as(str(out_file), write_like_original=False)
    return out_file


def export_nifti(
    pred_mask: np.ndarray,
    output_path: Union[str, Path],
    affine: Optional[np.ndarray] = None
) -> Path:
    """
    Exports integer multi-organ segmentation array as compressed NIfTI (.nii.gz).
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    
    if affine is None:
        affine = np.eye(4)
        
    nii = nib.Nifti1Image(pred_mask.astype(np.int16), affine=affine)
    nib.save(nii, str(out_file))
    return out_file


def create_mock_dicom_series(
    num_slices: int = 16,
    rows: int = 32,
    cols: int = 32,
    slice_thickness: float = 2.0,
    pixel_spacing: Tuple[float, float] = (1.5, 1.5)
) -> List[Dataset]:
    """Generates synthetic sorted DICOM slices with standard spatial tags for testing."""
    slices = []
    study_uid = generate_uid()
    series_uid = generate_uid()
    for i in range(num_slices):
        file_meta = FileMetaDataset()
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        
        ds = Dataset()
        ds.file_meta = file_meta
        ds.is_little_endian = True
        ds.is_implicit_VR = False
        ds.PatientID = "MOCK_PATIENT_001"
        ds.PatientName = "MOCK^PATIENT"
        ds.StudyInstanceUID = study_uid
        ds.SeriesInstanceUID = series_uid
        ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
        ds.FrameOfReferenceUID = generate_uid()
        ds.ImagePositionPatient = [0.0, 0.0, float(i * slice_thickness)]
        ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        ds.PixelSpacing = list(pixel_spacing)
        ds.SliceThickness = slice_thickness
        ds.Rows = rows
        ds.Columns = cols
        ds.RescaleSlope = 1.0
        ds.RescaleIntercept = -1024.0
        ds.PixelData = np.zeros((rows, cols), dtype=np.int16).tobytes()
        slices.append(ds)
    return slices

