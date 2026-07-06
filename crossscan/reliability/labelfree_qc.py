"""
Label-Free Quality Control (QC) Regressor.

Predicts per-case and per-organ Dice without ground truth annotations
using uncertainty ribbons, probability margins, organ morphology, and latent OOD metrics.
Enables automated triage routing so radiologists review only high-risk or sub-threshold cases.
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
    """
    Predicts segmentation performance (Dice) using radiomic, morphological,
    uncertainty, and bottleneck latent features extracted at inference time without ground truth.
    """
    def __init__(self, regressor_type: str = "gbr"):
        self.regressor_type = regressor_type
        # One regressor per organ class + one case-level aggregate regressor (key: 0)
        self.models: Dict[int, Union[GradientBoostingRegressor, Ridge]] = {}
        self.feature_names = [
            "vol_mm3",
            "surface_to_vol",
            "mean_prob",
            "p10_prob",
            "prob_margin",
            "mean_boundary_entropy",
            "p90_boundary_entropy",
            "mean_core_entropy",
            "organ_snr",
            "mahalanobis_dist",
            "free_energy",
            "mean_case_entropy"
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
        """
        Extracts 12-dimensional failure-predictive feature vector for each predicted organ.
        
        Args:
            image: (D, H, W) normalized intensity volume.
            pred_mask: (D, H, W) integer predicted class indices.
            probs: (C, D, H, W) softmax class probabilities.
            entropy_map: Optional precomputed voxel entropy map.
            latent_features: Optional bottleneck pooled latent representation.
            shift_detector: Optional fitted MahalanobisShiftDetector.
            voxel_spacing: Physical voxel dimensions in mm.
            
        Returns:
            Dict mapping organ class id (1..NUM_CLASSES-1) to (12,) feature vector.
        """
        if entropy_map is None:
            eps = 1e-8
            entropy_map = -np.sum(probs * np.log(probs + eps), axis=0)
            
        mean_case_entropy = float(np.mean(entropy_map))
        voxel_vol_mm3 = float(np.prod(voxel_spacing))
        boundary_stats = extract_boundary_uncertainty(pred_mask, entropy_map)
        
        # OOD metrics from bottleneck
        m_dist = 0.0
        free_energy = 0.0
        if latent_features is not None and shift_detector is not None and shift_detector.is_fitted:
            m_dist = shift_detector.compute_mahalanobis_distance(latent_features)
            free_energy = shift_detector.compute_free_energy(latent_features)
            
        top2_probs = np.partition(probs, -2, axis=0)[-2:]  # (2, D, H, W)
        prob_margin = top2_probs[1] - top2_probs[0]         # Margin between top-2 classes
        
        feature_dict = {}
        
        for c in range(1, NUM_CLASSES):
            mask_c = (pred_mask == c)
            vol_voxels = int(np.sum(mask_c))
            
            if vol_voxels == 0:
                # Organ missing in prediction: provide default zeroed/penalized feature vector
                feature_dict[c] = np.zeros(12, dtype=np.float32)
                feature_dict[c][11] = mean_case_entropy
                continue
                
            vol_mm3 = vol_voxels * voxel_vol_mm3
            
            # Morphological boundary surface area via erosion
            eroded = ndi.binary_erosion(mask_c)
            surface_voxels = np.sum(np.logical_xor(mask_c, eroded))
            surface_to_vol = float(surface_voxels / (vol_voxels + 1e-6))
            
            # Foreground probability statistics
            fg_probs = probs[c][mask_c]
            mean_prob = float(np.mean(fg_probs))
            p10_prob = float(np.percentile(fg_probs, 10))
            
            # Margin in predicted structure
            margin_c = float(np.mean(prob_margin[mask_c]))
            
            # Boundary ribbon entropy
            b_info = boundary_stats.get(c, {
                "boundary_entropy_mean": 0.0,
                "boundary_entropy_p90": 0.0,
                "core_entropy_mean": 0.0
            })
            
            # Radiometric Signal-to-Noise Ratio (SNR) within predicted structure
            img_c = image[mask_c]
            mean_intensity = float(np.mean(img_c))
            std_intensity = float(np.std(img_c)) + 1e-6
            organ_snr = mean_intensity / std_intensity
            
            vec = np.array([
                vol_mm3,
                surface_to_vol,
                mean_prob,
                p10_prob,
                margin_c,
                b_info["boundary_entropy_mean"],
                b_info["boundary_entropy_p90"],
                b_info["core_entropy_mean"],
                organ_snr,
                m_dist,
                free_energy,
                mean_case_entropy
            ], dtype=np.float32)
            
            feature_dict[c] = vec
            
        return feature_dict

    def _create_regressor(self) -> Union[GradientBoostingRegressor, Ridge]:
        if self.regressor_type == "ridge":
            return Ridge(alpha=1.0)
        return GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.8,
            random_state=42
        )

    def fit(
        self,
        features_by_class: Dict[int, np.ndarray],
        dices_by_class: Dict[int, np.ndarray]
    ) -> "LabelFreeQualityAssessor":
        """
        Trains organ-specific regressors and a case-level aggregate regressor.
        
        Args:
            features_by_class: Dict mapping class_id -> (N, 12) matrix.
            dices_by_class: Dict mapping class_id -> (N,) ground truth Dice scores.
        """
        all_case_features = []
        all_case_dices = []
        
        for c, X in features_by_class.items():
            if c not in dices_by_class:
                continue
            y = dices_by_class[c]
            if len(X) < 2:
                continue
                
            model = self._create_regressor()
            model.fit(X, y)
            self.models[c] = model
            
            all_case_features.append(X)
            all_case_dices.append(y)
            
        # Fit global case-level aggregate regressor under key 0
        if len(all_case_features) > 0:
            X_global = np.concatenate(all_case_features, axis=0)
            y_global = np.concatenate(all_case_dices, axis=0)
            global_model = self._create_regressor()
            global_model.fit(X_global, y_global)
            self.models[0] = global_model
            
        self.is_fitted = True
        return self

    def predict(
        self,
        features_dict: Dict[int, np.ndarray]
    ) -> Dict[str, Union[float, Dict[int, float]]]:
        """
        Predicts per-organ Dice and case-level overall Dice without ground truth.
        
        Returns:
            Dict containing:
                - "predicted_organ_dices": Dict[int, float]
                - "predicted_case_dice": float
                - "predicted_failure_risk": float (1.0 - predicted_case_dice)
        """
        if not self.is_fitted:
            raise RuntimeError("LabelFreeQualityAssessor must be fitted before predict() can be called.")
            
        predicted_organs: Dict[int, float] = {}
        pred_list = []
        
        for c, vec in features_dict.items():
            if np.all(vec == 0):
                # Organ not predicted
                predicted_organs[c] = 0.0
                continue
                
            vec_2d = vec.reshape(1, -1)
            model = self.models.get(c, self.models.get(0))
            if model is not None:
                pred_dice = float(np.clip(model.predict(vec_2d)[0], 0.0, 1.0))
            else:
                pred_dice = 0.5
                
            predicted_organs[c] = pred_dice
            pred_list.append(pred_dice)
            
        case_dice = float(np.mean(pred_list)) if len(pred_list) > 0 else 0.0
        
        return {
            "predicted_organ_dices": predicted_organs,
            "predicted_case_dice": case_dice,
            "predicted_failure_risk": float(1.0 - case_dice)
        }

    def save(self, filepath: Union[str, Path]) -> None:
        """Saves fitted assessor model to disk."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "LabelFreeQualityAssessor":
        """Loads fitted assessor model from disk."""
        return joblib.load(filepath)
