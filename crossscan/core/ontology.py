"""
Standardized Anatomical Label Ontology and Cross-Dataset Harmonization Mappings.

Reconciles divergent multi-center annotation protocols (AMOS, TotalSegmentator, CHAOS)
into a unified 8-organ semantic space with official SNOMED CT & RadLex medical terminologies
and Metrics Reloaded surface tolerance thresholds.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Union
import numpy as np
import torch


@dataclass(frozen=True)
class OrganMetadata:
    """Standardized anatomical metadata for an organ structure."""
    index: int
    name: str
    display_name: str
    snomed_ct: str
    radlex: str
    nsd_tolerance_mm: float  # Metrics Reloaded boundary tolerance (tau)
    rgb_color: Tuple[int, int, int]


ORGAN_ONTOLOGY: Dict[int, OrganMetadata] = {
    0: OrganMetadata(
        index=0,
        name="background",
        display_name="Background",
        snomed_ct="N/A",
        radlex="N/A",
        nsd_tolerance_mm=0.0,
        rgb_color=(0, 0, 0)
    ),
    1: OrganMetadata(
        index=1,
        name="liver",
        display_name="Liver",
        snomed_ct="10200004",
        radlex="RID58",
        nsd_tolerance_mm=2.0,
        rgb_color=(255, 0, 0)
    ),
    2: OrganMetadata(
        index=2,
        name="spleen",
        display_name="Spleen",
        snomed_ct="78961009",
        radlex="RID86",
        nsd_tolerance_mm=1.5,
        rgb_color=(0, 255, 0)
    ),
    3: OrganMetadata(
        index=3,
        name="kidney_right",
        display_name="Right Kidney",
        snomed_ct="24964005",
        radlex="RID205",
        nsd_tolerance_mm=1.5,
        rgb_color=(0, 0, 255)
    ),
    4: OrganMetadata(
        index=4,
        name="kidney_left",
        display_name="Left Kidney",
        snomed_ct="18639004",
        radlex="RID206",
        nsd_tolerance_mm=1.5,
        rgb_color=(255, 255, 0)
    ),
    5: OrganMetadata(
        index=5,
        name="pancreas",
        display_name="Pancreas",
        snomed_ct="15776009",
        radlex="RID170",
        nsd_tolerance_mm=1.5,
        rgb_color=(255, 0, 255)
    ),
    6: OrganMetadata(
        index=6,
        name="stomach",
        display_name="Stomach",
        snomed_ct="34516001",
        radlex="RID80",
        nsd_tolerance_mm=2.0,
        rgb_color=(0, 255, 255)
    ),
    7: OrganMetadata(
        index=7,
        name="aorta",
        display_name="Abdominal Aorta",
        snomed_ct="281130003",
        radlex="RID1001",
        nsd_tolerance_mm=1.0,
        rgb_color=(255, 128, 0)
    ),
    8: OrganMetadata(
        index=8,
        name="inferior_vena_cava",
        display_name="Inferior Vena Cava",
        snomed_ct="64131007",
        radlex="RID1220",
        nsd_tolerance_mm=1.0,
        rgb_color=(0, 128, 255)
    )
}

NUM_CLASSES = len(ORGAN_ONTOLOGY)
NUM_FOREGROUND_CLASSES = NUM_CLASSES - 1

ORGAN_NAMES: List[str] = [m.name for m in ORGAN_ONTOLOGY.values()]
ORGAN_INDEX_TO_NAME: Dict[int, str] = {idx: m.name for idx, m in ORGAN_ONTOLOGY.items()}
ORGAN_NAME_TO_INDEX: Dict[str, int] = {m.name: idx for idx, m in ORGAN_ONTOLOGY.items()}
ORGAN_NSD_TOLERANCES: Dict[int, float] = {idx: m.nsd_tolerance_mm for idx, m in ORGAN_ONTOLOGY.items()}

# ==============================================================================
# Mapping from source datasets to the standardized 8-organ ontology
# ==============================================================================

# AMOS22 (15 organs):
# 1: spleen, 2: right kidney, 3: left kidney, 4: gallbladder, 5: esophagus,
# 6: liver, 7: stomach, 8: aorta, 9: inferior vena cava, 10: portal vein & splenic vein,
# 11: pancreas, 12: right adrenal gland, 13: left adrenal gland, 14: duodenum, 15: bladder
AMOS_TO_UNIFIED: Dict[int, int] = {
    0: 0,   # Background
    1: 2,   # spleen -> spleen
    2: 3,   # right kidney -> kidney_right
    3: 4,   # left kidney -> kidney_left
    6: 1,   # liver -> liver
    7: 6,   # stomach -> stomach
    8: 7,   # aorta -> aorta
    9: 8,   # inferior vena cava -> inferior_vena_cava
    11: 5,  # pancreas -> pancreas
    # All other organs (4, 5, 10, 12, 13, 14, 15) are ignored or mapped to 0 (or masked)
}

# CHAOS ISBI 2019 (CT & MRI):
# Task 1/2: 1: Liver
# Task 3/4/5 (MRI): 1: Liver, 2: Right Kidney, 3: Left Kidney, 4: Spleen
CHAOS_CT_TO_UNIFIED: Dict[int, int] = {
    0: 0,
    1: 1,   # liver
}

CHAOS_MR_TO_UNIFIED: Dict[int, int] = {
    0: 0,
    1: 1,   # liver
    2: 3,   # right kidney
    3: 4,   # left kidney
    4: 2,   # spleen
}

# TotalSegmentator (Subset of 117 classes):
# 1: spleen, 2: kidney_right, 3: kidney_left, 5: liver, 6: stomach, 7: aorta, 8: inferior_vena_cava, 10: pancreas
TOTALSEGMENTATOR_TO_UNIFIED: Dict[int, int] = {
    0: 0,
    1: 2,   # spleen
    2: 3,   # kidney_right
    3: 4,   # kidney_left
    5: 1,   # liver
    6: 6,   # stomach
    7: 7,   # aorta
    8: 8,   # inferior_vena_cava
    10: 5,  # pancreas
}

DATASET_LABEL_MAPS: Dict[str, Dict[int, int]] = {
    "amos": AMOS_TO_UNIFIED,
    "chaos_ct": CHAOS_CT_TO_UNIFIED,
    "chaos_mr": CHAOS_MR_TO_UNIFIED,
    "totalsegmentator": TOTALSEGMENTATOR_TO_UNIFIED,
}

DATASET_PRESENT_CLASSES: Dict[str, Set[int]] = {
    "amos": {1, 2, 3, 4, 5, 6, 7, 8},
    "chaos_ct": {1},
    "chaos_mr": {1, 2, 3, 4},
    "totalsegmentator": {1, 2, 3, 4, 5, 6, 7, 8},
}


def harmonize_labels(
    labels: Union[np.ndarray, torch.Tensor],
    source_dataset: str,
    ignore_unmapped: bool = True
) -> Union[np.ndarray, torch.Tensor]:
    """
    Map label indices from a source clinical dataset into the CrossScan unified ontology.
    
    Args:
        labels: 3D or 4D label tensor/array with original integer values.
        source_dataset: Dataset identifier ('amos', 'chaos_ct', 'chaos_mr', 'totalsegmentator').
        ignore_unmapped: If True, values not explicitly mapped become 0 (background).
        
    Returns:
        Harmonized label array/tensor with identical shape and dtype.
    """
    key = source_dataset.lower()
    if key not in DATASET_LABEL_MAPS:
        raise ValueError(f"Unknown dataset '{source_dataset}'. Supported: {list(DATASET_LABEL_MAPS.keys())}")
    
    label_map = DATASET_LABEL_MAPS[key]
    is_torch = isinstance(labels, torch.Tensor)
    device = labels.device if is_torch else None
    
    if is_torch:
        np_labels = labels.detach().cpu().numpy()
    else:
        np_labels = labels
        
    out = np.zeros_like(np_labels)
    for src_val, target_val in label_map.items():
        if src_val != 0:
            out[np_labels == src_val] = target_val
            
    if is_torch:
        return torch.from_numpy(out).to(device=device, dtype=labels.dtype)
    return out


def get_present_classes_mask(source_dataset: str, num_classes: int = NUM_CLASSES) -> torch.Tensor:
    """
    Returns a binary 1D tensor [num_classes] indicating which classes are annotated
    in the given dataset. Used by marginal/masked loss functions to prevent false penalties.
    """
    key = source_dataset.lower()
    present_set = DATASET_PRESENT_CLASSES.get(key, set(range(num_classes)))
    mask = torch.zeros(num_classes, dtype=torch.bool)
    mask[0] = True  # Background is always considered known
    for c in present_set:
        if c < num_classes:
            mask[c] = True
    return mask

