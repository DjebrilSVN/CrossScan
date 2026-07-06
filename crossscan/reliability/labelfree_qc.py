"""
Label-Free Quality Control (QC) Regressor.

Predicts per-case and per-organ Dice without ground truth annotations
using uncertainty ribbons, probability margins, organ morphology, and latent OOD metrics.
"""

from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import numpy as np
import scipy.ndimage as ndi
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
import joblib

from crossscan.core.ontology import ORGAN_ONTOLOGY, NUM_CLASSES
from crossscan.reliability.uncertainty import extract_boundary_uncertainty, compute_voxel_entropy
from crossscan.reliability.shift_detector import MahalanobisShiftDetector


class LabelFreeQualityAssessor:
    def __init__(self, regressor_type: str = "gbr"):
        self.regressor_type = regressor_type
        self.models: Dict[int, Union[GradientBoostingRegressor, Ridge]] = {}
        self.feature_names = [
            "vol_mm3", "surface_to_vol", "mean_prob", "p10_prob", "prob_margin",
            "mean_boundary_entropy", "p90_boundary_entropy", "mean_core_entropy",
            "organ_snr", "mahalanobis_dist", "free_energy", "mean_case_entropy"
        ]
        self.is_fitted = False

    def extract_features(
        self,
        image: np.ndarray,
        pred_mask: np.ndarray,
        probs: np.ndarray,
        entropy_map: Optional[np.ndarray] = None,
        latent_features: Optional[np.ndarray] = None,
        shift_detector: Optional[MahalanobisShiftDetector] = None,
        voxel_spacing: Tuple[float, float, float] = (1.5, 1.5, 1.5)
    ) -> Dict[int, np.ndarray]:
        if entropy_map is None:
            eps = 1e-8
            entropy_map = -np.sum(probs * np.log(probs + eps), axis=0)
            
        mean_case_entropy = float(np.mean(entropy_map))
        voxel_vol_mm3 = float(np.prod(voxel_spacing))
        boundary_stats = extract_boundary_uncertainty(pred_mask, entropy_map)
        
        m_dist = 0.0
        free_energy = 0.0
        if latent_features is not None and shift_detector is not None and shift_detector.is_fitted:
            m_dist = shift_detector.compute_mahalanobis_distance(latent_features)
            free_energy = shift_detector.compute_free_energy(latent_features)
            
        top2_probs = np.partition(probs, -2, axis=0)[-2:]
        prob_margin = top2_probs[1] - top2_probs[0]
        
        feature_dict = {}
        for c in range(1, NUM_CLASSES):
            mask_c = (pred_mask == c)
            vol_voxels = int(np.sum(mask_c))
            if vol_voxels == 0:
                feature_dict[c] = np.zeros(12, dtype=np.float32)
                feature_dict[c][11] = mean_case_entropy
                continue
                
            vol_mm3 = vol_voxels * voxel_vol_mm3
            eroded = ndi.binary_erosion(mask_c)
            surface_voxels = np.sum(np.logical_xor(mask_c, eroded))
            surface_to_vol = float(surface_voxels / (vol_voxels + 1e-6))
            
            fg_probs = probs[c][mask_c]
            mean_prob = float(np.mean(fg_probs))
            p10_prob = float(np.percentile(fg_probs, 10))
            margin_c = float(np.mean(prob_margin[mask_c]))
            
            b_info = boundary_stats.get(c, {
                "boundary_entropy_mean": 0.0,
                "boundary_entropy_p90": 0.0,
                "core_entropy_mean": 0.0
            })
            
            img_c = image[mask_c]
            mean_intensity = float(np.mean(img_c))
            std_intensity = float(np.std(img_c)) + 1e-6
            organ_snr = mean_intensity / std_intensity
            
            feature_dict[c] = np.array([
                vol_mm3, surface_to_vol, mean_prob, p10_prob, margin_c,
                b_info["boundary_entropy_mean"], b_info["boundary_entropy_p90"],
                b_info["core_entropy_mean"], organ_snr, m_dist, free_energy, mean_case_entropy
            ], dtype=np.float32)
            
        return feature_dict
