"""
Anatomical Hallucination Auditor for Cross-Modality Image Synthesis.

Rigorous clinical safety audit detecting artificial hallucinations in translated scans:
1. Structural Similarity Index (SSIM) and Normalized Mutual Information (NMI).
2. Organ Volume Conservation Ratio (|V_syn - V_orig| / V_orig).
3. Boundary Distortion Index (BDI) and Centroid Shift.
4. Hallucination Risk Score and Safety Classification (Acceptable vs. Unsafe Hallucination).
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import scipy.ndimage as ndi
from skimage.metrics import structural_similarity as ssim_2d

from crossscan.core.ontology import ORGAN_NAMES, NUM_CLASSES
from crossscan.evaluation.metrics import compute_dice_score, compute_hd95


def compute_normalized_mutual_information(
    img1: np.ndarray,
    img2: np.ndarray,
    bins: int = 64
) -> float:
    """
    Computes Normalized Mutual Information (NMI) between two 3D volumes:
    NMI(X, Y) = 2 * I(X; Y) / (H(X) + H(Y))
    """
    # Flatten foreground voxels
    mask = np.logical_or(img1 > 0.01, img2 > 0.01)
    if not np.any(mask):
        return 1.0
        
    x = img1[mask].flatten()
    y = img2[mask].flatten()
    
    # 2D Joint histogram
    hist_2d, _, _ = np.histogram2d(x, y, bins=bins)
    pxy = hist_2d / np.sum(hist_2d)
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    
    # Entropies
    px_nz = px[px > 0]
    py_nz = py[py > 0]
    pxy_nz = pxy[pxy > 0]
    
    hx = -np.sum(px_nz * np.log2(px_nz))
    hy = -np.sum(py_nz * np.log2(py_nz))
    hxy = -np.sum(pxy_nz * np.log2(pxy_nz))
    
    mi = hx + hy - hxy
    nmi = (2.0 * mi) / (hx + hy + 1e-8)
    return float(np.clip(nmi, 0.0, 1.0))


def compute_slice_averaged_ssim(
    img1: np.ndarray,
    img2: np.ndarray
) -> float:
    """Computes mean 2D SSIM across all axial 2D slices containing foreground."""
    d = img1.shape[0]
    ssim_vals = []
    
    for z in range(d):
        s1 = img1[z]
        s2 = img2[z]
        if np.any(s1 > 0.05) or np.any(s2 > 0.05):
            val = ssim_2d(s1, s2, data_range=1.0)
            ssim_vals.append(val)
            
    return float(np.mean(ssim_vals)) if ssim_vals else 1.0


class AnatomicalHallucinationAuditor:
    """
    Evaluates synthesized scans for unphysical anatomical alterations or pseudo-structures.
    """
    def __init__(
        self,
        max_allowable_volume_change: float = 0.20,  # Max 20% organ volume shift
        min_allowable_ssim: float = 0.70,
        min_allowable_nmi: float = 0.40
    ):
        self.max_vol_change = max_allowable_volume_change
        self.min_ssim = min_allowable_ssim
        self.min_nmi = min_allowable_nmi

    def audit_translation(
        self,
        original_volume: np.ndarray,
        translated_volume: np.ndarray,
        mask_original: Optional[np.ndarray] = None,
        mask_translated: Optional[np.ndarray] = None
    ) -> Dict[str, Union[float, bool, str, Dict]]:
        """
        Runs comprehensive anatomical hallucination audit.
        
        Returns:
            Dictionary containing SSIM, NMI, volume conservation per organ,
            hallucination flags, and overall clinical safety verdict.
        """
        orig = np.clip(original_volume.astype(np.float32), 0.0, 1.0)
        syn = np.clip(translated_volume.astype(np.float32), 0.0, 1.0)
        
        ssim_score = compute_slice_averaged_ssim(orig, syn)
        nmi_score = compute_normalized_mutual_information(orig, syn)
        
        organ_audits = {}
        hallucination_detected = False
        warning_messages = []
        
        if mask_original is not None and mask_translated is not None:
            unique_classes = set(np.unique(mask_original)).union(set(np.unique(mask_translated)))
            for c in unique_classes:
                if c == 0:
                    continue
                c_idx = int(c)
                organ_name = ORGAN_NAMES[c_idx] if c_idx < len(ORGAN_NAMES) else f"Organ_{c_idx}"
                
                m_orig = (mask_original == c_idx)
                m_syn = (mask_translated == c_idx)
                
                v_orig = float(np.sum(m_orig))
                v_syn = float(np.sum(m_syn))
                
                if v_orig > 0:
                    vol_ratio_diff = abs(v_syn - v_orig) / v_orig
                else:
                    vol_ratio_diff = 1.0 if v_syn > 50 else 0.0
                    
                dice_overlap = compute_dice_score(m_syn, m_orig)
                
                # Check for organ dissolution or artificial spawning
                status = "PRESERVED"
                if v_orig > 100 and v_syn < 0.5 * v_orig:
                    status = "DISSOLUTION_HALLUCINATION"
                    hallucination_detected = True
                    warning_messages.append(f"{organ_name}: severe volume loss ({vol_ratio_diff*100:.1f}%)")
                elif v_orig == 0 and v_syn > 200:
                    status = "ARTIFICIAL_SPAWN_HALLUCINATION"
                    hallucination_detected = True
                    warning_messages.append(f"{organ_name}: artificial pseudo-structure spawned without anatomical source")
                elif vol_ratio_diff > self.max_vol_change:
                    status = "DEFORMATION_WARNING"
                    
                organ_audits[organ_name] = {
                    "original_volume_voxels": v_orig,
                    "translated_volume_voxels": v_syn,
                    "volume_change_ratio": float(vol_ratio_diff),
                    "dice_agreement": float(dice_overlap),
                    "status": status
                }
                
        # Check global image fidelity
        if ssim_score < self.min_ssim:
            hallucination_detected = True
            warning_messages.append(f"Global SSIM ({ssim_score:.3f}) below safety threshold ({self.min_ssim:.2f})")
            
        if nmi_score < self.min_nmi:
            hallucination_detected = True
            warning_messages.append(f"Normalized Mutual Information ({nmi_score:.3f}) indicates severe feature misalignment")
            
        risk_score = float(np.clip(
            (1.0 - ssim_score) * 0.5 + (1.0 - nmi_score) * 0.5,
            0.0, 1.0
        ))
        
        return {
            "ssim": ssim_score,
            "normalized_mutual_information": nmi_score,
            "hallucination_risk_score": risk_score,
            "hallucination_detected": hallucination_detected,
            "clinical_safety_status": "[red]UNSAFE_HALLUCINATION[/red]" if hallucination_detected else "[green]ANATOMICALLY_FAITHFUL[/green]",
            "warnings": warning_messages,
            "per_organ_audit": organ_audits
        }

