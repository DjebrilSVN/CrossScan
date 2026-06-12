"""
Label-Free Quality Control (QC) Regressor.

Predicts per-case and per-organ Dice without ground truth annotations.
Enables automated triage routing: radiologist reviews only sub-threshold cases.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import scipy.ndimage as ndi
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
import joblib

from crossscan.core.ontology import ORGAN_ONTOLOGY, NUM_CLASSES
from crossscan.reliability.uncertainty import extract_boundary_uncertainty, compute_voxel_entropy


class LabelFreeQualityAssessor:
    """
    Predicts segmentation performance (Dice) using radiomic, morphological,
    and uncertainty features extracted at test-time without ground truth.
    """
    def __init__(self, regressor_type: str = "gbr"):
        self.regressor_type = regressor_type
        # One regressor per organ class + one case-level aggregate regressor
        self.models: Dict[int, Union[GradientBoostingRegressor, Ridge]] = {}
        self.is_fitted = False

    def extract_features(
        self,
        image: np.ndarray,
        pred_mask: np.ndarray,
        probs: np.ndarray,
        entropy_map: Optional[np.ndarray] = None,
        voxel_spacing: Tuple[float, float, float] = (1.5, 1.5, 1.5)
    ) -> Dict[int, np.ndarray]:
        """
        Extracts 9-dimensional failure-predictive feature vector for each predicted organ:
        [volume_mm3, surface_volume_ratio, mean_prob, p10_prob, margin_top2,
         mean_boundary_entropy, p90_boundary_entropy, mean_core_entropy, image_organ_snr]
        """
        if entropy_map is None:
            # probs shape: (C, D, H, W)
            eps = 1e-8
            entropy_map = -np.sum(probs * np.log(probs + eps), axis=0)
            
        voxel_vol_mm3 = float(np.prod(voxel_spacing))
        boundary_stats = extract_boundary_uncertainty(pred_mask, entropy_map)
        
        feature_dict = {}
        top2_probs = np.partition(probs, -2, axis=0)[-2:]  # (2, D, H, W)
        prob_margin = top2_probs[1] - top2_probs[0]         # Difference between top 2 classes
        
        for c in range(1, NUM_CLASSES):
            mask_c = (pred_mask == c)
            vol_voxels = int(np.sum(mask_c))
            
            if vol_voxels == 0:
                # Organ not predicted (predicted volume 0)
                feature_dict[c] = np.zeros(9, dtype=np.float32)
                continue
                
            vol_mm3 = vol_voxels * voxel_vol_mm3
            
            # Surface area via erosion
            eroded = ndi.binary_erosion(mask_c)
            surface_voxels = np.sum(np.logical_xor(mask_c, eroded))
            surface_to_vol = float(surface_voxels / (vol_voxels + 1e-6))
            
            # Probability stats
            fg_probs = probs[c][mask_c]
            mean_prob = float(np.mean(fg_probs))
            p10_prob = float(np.percentile(fg_probs, 10))
            
            # Margin in predicted region
            margin_c = float(np.mean(prob_margin[mask_c]))
            
            # Boundary uncertainty
            b_info = boundary_stats.get(c, {
                "boundary_entropy_mean": 0.0,
                "boundary_entropy_p90": 0.0,
                "core_entropy_mean": 0.0
            })
            
            # Image SNR within predicted structure
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
                organ_snr
            ], dtype=np.float32)
            
            feature_dict[c] = vec
            
        return feature_dict

    def fit(
        self,
        training_features: Dict[int, np.ndarray],
        true_dices: Dict[int, np.ndarray]
    ) -> None:
        """
        Fits regressor on validation cohort using extracted features and known ground truth Dice scores.
        
        Args:
            training_features: Mapping organ_index -> Feature matrix [N_samples, 9]
            true_dices: Mapping organ_index -> True Dice vector [N_samples]
        """
        for c in range(1, NUM_CLASSES):
            if c not in training_features or len(training_features[c]) == 0:
                continue
                
            X = training_features[c]
            y = true_dices[c]
            
            if self.regressor_type == "gbr":
                model = GradientBoostingRegressor(
                    n_estimators=100,
                    max_depth=3,
                    learning_rate=0.05,
                    random_state=42
                )
            else:
                model = Ridge(alpha=1.0)
                
            model.fit(X, y)
            self.models[c] = model
            
        self.is_fitted = True

    def predict_dice(
        self,
        features: Dict[int, np.ndarray]
    ) -> Dict[str, float]:
        """
        Predicts Dice score for each organ without ground truth.
        """
        predictions = {}
        scores = []
        
        for c in range(1, NUM_CLASSES):
            organ_name = ORGAN_ONTOLOGY[c].name
            if c not in features:
                predictions[organ_name] = 0.0
                continue
                
            x_vec = features[c].reshape(1, -1)
            
            # If organ was not predicted at all (zero vector)
            if np.all(x_vec == 0):
                pred_dice = 0.0
            elif c in self.models:
                pred_dice = float(np.clip(self.models[c].predict(x_vec)[0], 0.0, 1.0))
            else:
                # Heuristic fallback if model not fitted: use mean foreground probability
                pred_dice = float(np.clip(x_vec[0, 2], 0.0, 1.0))
                
            predictions[organ_name] = pred_dice
            scores.append(pred_dice)
            
        predictions["mean_case_dice"] = float(np.mean(scores)) if scores else 0.0
        return predictions

    def save(self, filepath: str) -> None:
        joblib.dump({"models": self.models, "type": self.regressor_type}, filepath)

    def load(self, filepath: str) -> None:
        data = joblib.load(filepath)
        self.models = data["models"]
        self.regressor_type = data["type"]
        self.is_fitted = True

