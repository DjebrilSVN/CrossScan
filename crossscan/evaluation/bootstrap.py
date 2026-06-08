"""
Patient-Level Non-Parametric Bootstrap Confidence Interval Estimator.

Compliant with Metrics Reloaded: Resamples strictly at the patient level
(never across individual 2D slices or voxels) to preserve within-patient spatial correlation.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np


def compute_patient_bootstrap_ci(
    patient_metric_values: Union[List[float], np.ndarray],
    num_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_seed: int = 42
) -> Dict[str, float]:
    """
    Computes non-parametric bootstrap confidence interval over patient cohort.
    
    Args:
        patient_metric_values: Array of per-patient metric scores (e.g. Dice).
        num_bootstrap: Number of bootstrap resamples (default: 1000).
        confidence_level: Confidence level (default: 0.95 for 95% CI).
        
    Returns:
        Dict with keys: 'mean', 'std', 'ci_lower', 'ci_upper', 'median'
    """
    values = np.asarray(patient_metric_values, dtype=np.float64)
    # Remove NaN values from cases where organ was not present
    values = values[~np.isnan(values)]
    n = len(values)
    
    if n == 0:
        return {"mean": 0.0, "std": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "median": 0.0}
    if n == 1:
        v = float(values[0])
        return {"mean": v, "std": 0.0, "ci_lower": v, "ci_upper": v, "median": v}
        
    rng = np.random.RandomState(random_seed)
    bootstrap_means = np.empty(num_bootstrap, dtype=np.float64)
    
    for b in range(num_bootstrap):
        sample = rng.choice(values, size=n, replace=True)
        bootstrap_means[b] = np.mean(sample)
        
    alpha = (1.0 - confidence_level) / 2.0
    ci_lower = float(np.percentile(bootstrap_means, 100 * alpha))
    ci_upper = float(np.percentile(bootstrap_means, 100 * (1.0 - alpha)))
    
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "median": float(np.median(values))
    }

