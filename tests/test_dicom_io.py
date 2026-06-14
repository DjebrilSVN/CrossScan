"""Unit tests for DICOM & NIfTI deployment I/O."""
import tempfile
from pathlib import Path
import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid
import nibabel as nib

from crossscan.deployment.dicom_io import export_dicom_seg, export_nifti


def create_mock_dicom_slice(z_pos: float = 0.0) -> Dataset:
    file_meta = FileMetaDataset()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"  # CT Image Storage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    
    ds = Dataset()
    ds.file_meta = file_meta
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.PatientID = "TEST_PATIENT_001"
    ds.PatientName = "MOCK^PATIENT"
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.FrameOfReferenceUID = generate_uid()
    ds.ImagePositionPatient = [0.0, 0.0, z_pos]
    ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    ds.PixelSpacing = [1.5, 1.5]
    ds.SliceThickness = 2.0
    ds.Rows = 32
    ds.Columns = 32
    ds.RescaleSlope = 1.0
    ds.RescaleIntercept = -1024.0
    ds.PixelData = np.zeros((32, 32), dtype=np.int16).tobytes()
    return ds


def test_export_nifti():
    with tempfile.TemporaryDirectory() as tmpdir:
        out_nii = Path(tmpdir) / "pred.nii.gz"
        mask = np.zeros((16, 16, 16), dtype=np.int32)
        mask[4:8, 4:8, 4:8] = 1
        
        saved = export_nifti(mask, out_nii)
        assert saved.exists()
        
        loaded = nib.load(str(saved)).get_fdata()
        assert loaded.shape == (16, 16, 16)
        assert np.sum(loaded == 1) == 64


def test_export_dicom_seg():
    with tempfile.TemporaryDirectory() as tmpdir:
        out_seg = Path(tmpdir) / "output_seg.dcm"
        slices = [create_mock_dicom_slice(z) for z in [0.0, 2.0, 4.0, 6.0]]
        
        mask = np.zeros((4, 32, 32), dtype=np.int32)
        mask[1:3, 10:20, 10:20] = 1  # Liver
        mask[2:4, 5:10, 5:10] = 2    # Spleen
        
        saved_file = export_dicom_seg(mask, slices, out_seg)
        assert saved_file.exists()
        
        ds = pydicom.dcmread(str(saved_file))
        assert ds.SOPClassUID == "1.2.840.10008.5.1.4.1.1.66.4"
        assert ds.PatientID == "TEST_PATIENT_001"
        assert len(ds.SegmentSequence) == 2
        assert ds.SegmentSequence[0].SegmentLabel == "Liver"
        assert ds.SegmentSequence[0].SegmentedPropertyTypeCodeSequence[0].CodeValue == "10200004"

