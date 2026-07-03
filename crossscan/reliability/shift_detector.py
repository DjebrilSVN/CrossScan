"""
Out-Of-Distribution (OOD) & Scanner Shift Detector.

Detects scanner fleet shifts, sequence discrepancies, and anomalous anatomies
using bottleneck latent Mahalanobis distance, free energy scoring, and 
radiometric distribution statistics.
"""

from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import json
import numpy as np
from scipy.spatial.distance import mahalanobis
from scipy.stats import wasserstein_distance
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
        self.ref_histogram: Optional[np.ndarray] = None
        self.ref_bin_edges: Optional[np.ndarray] = None
        
        # Calibrated threshold for 95% ID retention
        self.mahalanobis_threshold_95: float = 15.0
        self.energy_threshold_95: float = 0.0
        self.is_fitted: bool = False

    def fit(
        self,
        id_latent_features: np.ndarray,
        id_intensity_volumes: Optional[List[np.ndarray]] = None
    ) -> "MahalanobisShiftDetector":
        """
        Fits empirical Gaussian distribution on in-distribution latent vectors.
        
        Args:
            id_latent_features: (N, D) matrix of pooled bottleneck feature vectors.
            id_intensity_volumes: Optional list of raw 3D volumes for histogram shift tracking.
        """
        assert id_latent_features.ndim == 2, f"Expected 2D array (N, D), got {id_latent_features.shape}"
        N, D = id_latent_features.shape
        
        # 1. Compute empirical mean
        self.mean_vector = np.mean(id_latent_features, axis=0)
        
        # 2. Compute covariance with Ledoit-Wolf / Tikhonov shrinkage
        cov = np.cov(id_latent_features, rowvar=False)
        if cov.ndim == 0:
            cov = np.array([[cov]])
            
        # Regularization / shrinkage
        reg_cov = cov + self.shrinkage_factor * np.eye(D)
        
        # Robust pseudo-inverse
        try:
            self.inv_cov_matrix = np.linalg.pinv(reg_cov)
        except np.linalg.LinAlgError:
            self.inv_cov_matrix = np.eye(D)
            
        # 3. Compute baseline Mahalanobis distances & energies to calibrate 95th percentile
        dists = [
            self.compute_mahalanobis_distance(id_latent_features[i])
            for i in range(N)
        ]
        energies = [
            self.compute_free_energy(id_latent_features[i])
            for i in range(N)
        ]
        
        self.mahalanobis_threshold_95 = float(np.percentile(dists, 95)) if len(dists) > 0 else 10.0
        self.energy_threshold_95 = float(np.percentile(energies, 95)) if len(energies) > 0 else 0.0
        
        # 4. Reference intensity distribution
        if id_intensity_volumes and len(id_intensity_volumes) > 0:
            all_voxels = np.concatenate([v.ravel()[::100] for v in id_intensity_volumes])
            counts, bin_edges = np.histogram(all_voxels, bins=64, range=(0.0, 1.0), density=True)
            self.ref_histogram = counts
            self.ref_bin_edges = bin_edges
            
        self.is_fitted = True
        return self

    def compute_mahalanobis_distance(self, latent_vector: np.ndarray) -> float:
        """
        Computes D_M(z) = sqrt((z - mu)^T Sigma^-1 (z - mu)).
        """
        if self.mean_vector is None or self.inv_cov_matrix is None:
            return 0.0
        diff = latent_vector - self.mean_vector
        dist_sq = float(np.dot(np.dot(diff, self.inv_cov_matrix), diff))
        return float(np.sqrt(max(0.0, dist_sq)))

    def compute_free_energy(self, latent_vector: np.ndarray) -> float:
        """
        Computes negative Helmholtz free energy: E(z; T) = -T * logsumexp(z / T).
        Higher energy indicates outlier / out-of-distribution state.
        """
        z_scaled = latent_vector / self.energy_temperature
        # Numerically stable logsumexp
        max_val = np.max(z_scaled)
        energy = -self.energy_temperature * (max_val + np.log(np.sum(np.exp(z_scaled - max_val)) + 1e-8))
        return float(energy)

    def compute_histogram_wasserstein(self, volume: np.ndarray) -> float:
        """
        Measures statistical radiometric distribution divergence against training reference.
        """
        if self.ref_histogram is None:
            return 0.0
        sub_voxels = volume.ravel()[::100]
        test_counts, _ = np.histogram(sub_voxels, bins=len(self.ref_histogram), range=(0.0, 1.0), density=True)
        return float(wasserstein_distance(self.ref_histogram, test_counts))

    def evaluate_scan(
        self,
        latent_features: np.ndarray,
        volume: Optional[np.ndarray] = None
    ) -> Dict[str, Union[float, bool]]:
        """
        Comprehensive OOD evaluation of a single volume inference.
        
        Returns:
            Dict containing mahalanobis_distance, free_energy, wasserstein_shift,
            and is_ood_flagged.
        """
        m_dist = self.compute_mahalanobis_distance(latent_features)
        energy = self.compute_free_energy(latent_features)
        w_dist = self.compute_histogram_wasserstein(volume) if volume is not None else 0.0
        
        # Flagged as OOD if Mahalanobis exceeds 95th percentile ID threshold
        is_ood = (m_dist > self.mahalanobis_threshold_95 * 1.5) or (w_dist > 0.35)
        
        return {
            "mahalanobis_distance": m_dist,
            "mahalanobis_threshold_95": self.mahalanobis_threshold_95,
            "free_energy": energy,
            "wasserstein_shift": w_dist,
            "is_ood_flagged": bool(is_ood)
        }

    def save(self, filepath: Union[str, Path]) -> None:
        """Saves fitted shift detector state to disk."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "MahalanobisShiftDetector":
        """Loads fitted shift detector state from disk."""
        return joblib.load(filepath)

