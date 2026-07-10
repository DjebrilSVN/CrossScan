"""
MONAI Deploy Application Package (MAP) Architecture for CrossScan.

Packages the complete end-to-end pipeline into standardized clinical operators:
1. DICOMDataLoaderOperator: Ingests 3D patient slices.
2. HarmonizationOperator: Applies canonical RAS+ and radiometry standardization.
3. CrossScanInferenceOperator: Modality-agnostic joint 3D inference.
4. ConformalTriageOperator: Evaluates self-auditing failure prediction & CRC risk.
5. DICOMSEGWriterOperator: Encapsulates standard DICOM-SEG (SOP 1.2.840...66.4).
"""

from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from crossscan.core.ontology import NUM_CLASSES, ORGAN_ONTOLOGY
from crossscan.models.joint_segmentor import CrossScanJointSegmentor
from crossscan.reliability.conformal import ConformalTriageEngine, TriageDecision
from crossscan.reliability.shift_detector import MahalanobisShiftDetector
from crossscan.reliability.labelfree_qc import LabelFreeQualityAssessor
from crossscan.deployment.dicom_io import export_dicom_seg


class DICOMDataLoaderOperator:
    """Ingests multi-frame or multi-slice DICOM series."""
    def compute(self, series_path: Union[str, Path]) -> Dict:
        from crossscan.deployment.dicom_io import load_dicom_series
        volume, spacing, slices = load_dicom_series(series_path)
        return {"volume": volume, "spacing": spacing, "source_slices": slices}


class HarmonizationOperator:
    """Preprocesses input arrays into canonical radiometry and multiple-of-16 spatial grids."""
    def compute(self, data: Dict, modality: str = "ct") -> Dict:
        from crossscan.data.transforms import normalize_ct, normalize_mri
        vol = data["volume"]
        if modality.lower() == "ct":
            norm = normalize_ct(vol)
        else:
            norm = normalize_mri(vol)
        tensor = torch.from_numpy(norm).float().unsqueeze(0).unsqueeze(0)
        # Pad to multiple of 16 for DynUNet
        d, h, w = tensor.shape[2:]
        pad_d = (16 - d % 16) % 16
        pad_h = (16 - h % 16) % 16
        pad_w = (16 - w % 16) % 16
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            tensor = F.pad(tensor, (0, pad_w, 0, pad_h, 0, pad_d), mode="constant", value=0.0)
        data["tensor"] = tensor
        data["original_shape"] = (d, h, w)
        return data
