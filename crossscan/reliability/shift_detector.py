"""
Out-Of-Distribution (OOD) & Scanner Shift Detector.

Detects scanner fleet shifts, sequence discrepancies, and anomalous anatomies
using bottleneck latent Mahalanobis distance and shrinkage covariance.
"""

from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import json
import numpy as np
import joblib


class MahalanobisShiftDetector:
    """
    Detects clinical acquisition shifts and OOD scans by modeling
    the in-distribution (ID) latent representation in the segmentor bottleneck.
    """
    def __init__(self, energy_temperature: float = 1.0, shrinkage_factor: float = 1e-4):
        self.energy_temperature = energy_temperature
        self.shrinkage_factor = shrinkage_factor
        
        self.mean_vector: Optional[np.ndarray] = None
        self.inv_cov_matrix: Optional[np.ndarray] = None
        self.mahalanobis_threshold_95: float = 15.0
        self.is_fitted: bool = False

    def fit(self, id_latent_features: np.ndarray) -> "MahalanobisShiftDetector":
        assert id_latent_features.ndim == 2
        N, D = id_latent_features.shape
        self.mean_vector = np.mean(id_latent_features, axis=0)
        cov = np.cov(id_latent_features, rowvar=False)
        reg_cov = cov + self.shrinkage_factor * np.eye(D)
        self.inv_cov_matrix = np.linalg.pinv(reg_cov)
        
        dists = [self.compute_mahalanobis_distance(id_latent_features[i]) for i in range(N)]
        self.mahalanobis_threshold_95 = float(np.percentile(dists, 95)) if len(dists) > 0 else 10.0
        self.is_fitted = True
        return self

    def compute_mahalanobis_distance(self, latent_vector: np.ndarray) -> float:
        if self.mean_vector is None or self.inv_cov_matrix is None:
            return 0.0
        diff = latent_vector - self.mean_vector
        dist_sq = float(np.dot(np.dot(diff, self.inv_cov_matrix), diff))
        return float(np.sqrt(max(0.0, dist_sq)))
