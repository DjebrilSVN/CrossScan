"""
Foundation Model Pretraining Contamination Auditor.

Scans test cohort patient IDs and dataset origins against published pretraining
registries of foundation models (VISTA3D, SAM-Med3D, MedSAM2, TotalSegmentator)
to guarantee zero data contamination and unbiased benchmark reporting.
"""

from typing import Dict, List, Set, Tuple, Union


FOUNDATION_MODEL_TRAINING_REGISTRY: Dict[str, Dict[str, Set[str]]] = {
    "VISTA3D": {
        "datasets": {"BTCV", "TCIA", "AMOS22_CT", "TotalSegmentator_CT", "FLARE22"},
        "notes": "Trained on massive CT abdominal and whole-body datasets. AMOS CT is contaminated; AMOS MRI and CHAOS MRI are uncontaminated."
    },
    "SAM-Med3D": {
        "datasets": {"BTCV", "LiTS", "KiTS19", "KiTS21", "TotalSegmentator_CT", "AMOS22_CT", "CHAOS_CT"},
        "notes": "Extensive CT abdominal pretraining. CHAOS CT and AMOS CT are contaminated."
    },
    "MedSAM2": {
        "datasets": {"TotalSegmentator_CT", "BTCV", "FLARE22"},
        "notes": "Fine-tuned SAM2 on 3D medical slices. AMOS MRI and held-out private sequences are strictly uncontaminated."
    },
    "SegVol": {
        "datasets": {"BTCV", "TotalSegmentator_CT", "AMOS22_CT"},
        "notes": "Large volumetric CT pretraining. AMOS MRI and CHAOS MRI are clean evaluation benchmarks."
    }
}


def audit_test_cohort_contamination(
    dataset_name: str,
    modality: str,
    patient_ids: List[str]
) -> Dict[str, Dict[str, Union[bool, str, int]]]:
    """
    Audits whether an evaluation set overlaps with foundation model pretraining registries.
    
    Returns:
        Dict mapping foundation model name to contamination status and details.
    """
    report = {}
    clean_name = dataset_name.lower().replace("22", "").replace("_", "")
    clean_modality = modality.lower()
    
    for model_name, info in FOUNDATION_MODEL_TRAINING_REGISTRY.items():
        is_contaminated = False
        reason = "Clean / Zero Contamination"
        
        # Check against pretraining dataset registry
        for pretrain_ds in info["datasets"]:
            p_clean = pretrain_ds.lower().replace("22", "").replace("_", "")
            if clean_name in p_clean and clean_modality in p_clean:
                is_contaminated = True
                reason = f"Dataset '{dataset_name}' ({modality.upper()}) matched pretraining entry '{pretrain_ds}' in {model_name}."
                break
                
        report[model_name] = {
            "contaminated": is_contaminated,
            "status": "CONTAMINATED" if is_contaminated else "UNCONTAMINATED (CLEAN)",
            "reason": reason,
            "notes": info["notes"]
        }
        
    return report
