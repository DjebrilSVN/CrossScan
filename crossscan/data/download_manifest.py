"""
Dataset Acquisition, Integrity Verification, and Manifest Registry.

Curates official, peer-reviewed public benchmarks:
- AMOS22 (Zenodo: 7265705) - CT & MRI abdominal multi-organ
- TotalSegmentator CT v2 (Zenodo: 6802614) & TotalSegmentator MRI (Zenodo: 11494582)
- CHAOS ISBI 2019 (Zenodo: 3431873) - Multi-sequence MRI (T1-DUAL, T2-SPIR) & CT
"""

import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import urllib.request
from tqdm import tqdm


@dataclass
class DatasetSource:
    name: str
    modality: str
    num_volumes: int
    citation: str
    zenodo_url: str
    zenodo_record_id: str
    expected_files: List[str]
    license: str


DATASET_CATALOG: Dict[str, DatasetSource] = {
    "amos_ct": DatasetSource(
        name="AMOS22 CT",
        modality="CT",
        num_volumes=500,
        citation="Ji et al., AMOS: A Large-Scale Abdominal Multi-Organ Benchmark, IEEE TMI 2022.",
        zenodo_url="https://zenodo.org/record/7265705",
        zenodo_record_id="7265705",
        expected_files=["amos22.zip"],
        license="CC BY-NC 4.0"
    ),
    "amos_mr": DatasetSource(
        name="AMOS22 MRI",
        modality="MRI",
        num_volumes=100,
        citation="Ji et al., AMOS: A Large-Scale Abdominal Multi-Organ Benchmark, IEEE TMI 2022.",
        zenodo_url="https://zenodo.org/record/7265705",
        zenodo_record_id="7265705",
        expected_files=["amos22.zip"],
        license="CC BY-NC 4.0"
    ),
    "totalsegmentator_ct": DatasetSource(
        name="TotalSegmentator CT v2",
        modality="CT",
        num_volumes=1204,
        citation="Wasserthal et al., TotalSegmentator: Robust Segmentation of 104 Anatomical Structures, Radiology: AI 2023.",
        zenodo_url="https://zenodo.org/record/6802614",
        zenodo_record_id="6802614",
        expected_files=["Totalsegmentator_dataset_v201.zip"],
        license="CC BY 4.0"
    ),
    "totalsegmentator_mr": DatasetSource(
        name="TotalSegmentator MRI",
        modality="MRI",
        num_volumes=1050,
        citation="Wasserthal et al., TotalSegmentator MRI, Zenodo 2024.",
        zenodo_url="https://zenodo.org/records/11494582",
        zenodo_record_id="11494582",
        expected_files=["Totalsegmentator_mri_dataset.zip"],
        license="CC BY 4.0"
    ),
    "chaos": DatasetSource(
        name="CHAOS ISBI 2019",
        modality="CT_and_MRI",
        num_volumes=80,
        citation="Kavur et al., CHAOS Challenge - Combined (CT-MR) Healthy Abdominal Organ Segmentation, Medical Image Analysis 2021.",
        zenodo_url="https://zenodo.org/records/3431873",
        zenodo_record_id="3431873",
        expected_files=["CHAOS_Train_Sets.zip"],
        license="CC BY-SA 4.0"
    )
}


def verify_file_sha256(filepath: Union[str, Path], expected_sha256: str) -> bool:
    """Computes SHA256 hash of a file with chunked buffering."""
    p = Path(filepath)
    if not p.exists():
        return False
        
    hasher = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(1024 * 1024):
            hasher.update(chunk)
            
    return hasher.hexdigest().lower() == expected_sha256.lower()


def generate_data_splits_manifest(
    data_dir: Union[str, Path],
    output_manifest_path: Union[str, Path],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42
) -> Dict[str, List[Dict[str, str]]]:
    """
    Creates patient-stratified split manifest preventing patient data leakage across splits.
    """
    import numpy as np
    path = Path(data_dir)
    image_files = sorted(list(path.glob("**/*_image.nii.gz")) + list(path.glob("**/imagesTr/*.nii.gz")))
    
    entries = []
    for img_p in image_files:
        # Determine matching label
        lbl_p = img_p.parent.parent / "labelsTr" / img_p.name
        if not lbl_p.exists():
            lbl_p = Path(str(img_p).replace("image", "label"))
            
        # Determine modality
        modality = "ct" if "ct" in img_p.name.lower() or "ct" in str(img_p.parent).lower() else "mr"
        
        patient_id = img_p.name.split(".")[0].split("_")[0]
        entries.append({
            "patient_id": patient_id,
            "image": str(img_p.resolve()),
            "label": str(lbl_p.resolve()) if lbl_p.exists() else "",
            "modality": modality
        })
        
    # Group by patient
    rng = np.random.RandomState(random_seed)
    patients = sorted(list(set(e["patient_id"] for e in entries)))
    rng.shuffle(patients)
    
    n_p = len(patients)
    n_train = int(n_p * train_ratio)
    n_val = int(n_p * val_ratio)
    
    train_patients = set(patients[:n_train])
    val_patients = set(patients[n_train:n_train + n_val])
    test_patients = set(patients[n_train + n_val:])
    
    splits = {
        "train": [e for e in entries if e["patient_id"] in train_patients],
        "val": [e for e in entries if e["patient_id"] in val_patients],
        "test": [e for e in entries if e["patient_id"] in test_patients],
    }
    
    out_p = Path(output_manifest_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w") as f:
        json.dump(splits, f, indent=2)
        
    return splits

