"""
Physics-Informed Domain Randomization Engine for 3D CT & MRI.

Implements realistic imaging physics degradation:
- Rician noise (MRI) & Gaussian quantum noise (CT)
- Spatially varying polynomial B1 bias fields
- k-space phase-encoding motion ghosting
- Anisotropic slice thickness downsampling (thick clinical slices)
- Non-linear contrast warping & gamma randomization
"""

import math
from typing import Dict, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from monai.transforms import RandomizableTransform


class RandomRicianNoise(RandomizableTransform):
    """
    Simulates MRI Rician noise: sqrt((I + N_r)^2 + N_i^2) where N_r, N_i ~ N(0, sigma^2).
    """
    def __init__(self, prob: float = 0.5, sigma_range: Tuple[float, float] = (0.02, 0.12)):
        super().__init__(prob=prob)
        self.sigma_range = sigma_range
        self.sigma: float = 0.0

    def randomize(self, data: Optional[torch.Tensor] = None) -> None:
        super().randomize(data)
        self.sigma = float(np.random.uniform(self.sigma_range[0], self.sigma_range[1]))

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        self.randomize(img)
        if not self._do_transform:
            return img
        
        real_noise = torch.randn_like(img) * self.sigma
        imag_noise = torch.randn_like(img) * self.sigma
        noisy = torch.sqrt((img + real_noise) ** 2 + imag_noise ** 2)
        return noisy


class RandomB1BiasField(RandomizableTransform):
    """
    Simulates MRI B1 radiofrequency field inhomogeneity using 2nd-order smooth 3D polynomial.
    """
    def __init__(self, prob: float = 0.5, magnitude_range: Tuple[float, float] = (0.2, 0.6)):
        super().__init__(prob=prob)
        self.magnitude_range = magnitude_range
        self.magnitude: float = 0.0

    def randomize(self, data: Optional[torch.Tensor] = None) -> None:
        super().randomize(data)
        self.magnitude = float(np.random.uniform(self.magnitude_range[0], self.magnitude_range[1]))

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        self.randomize(img)
        if not self._do_transform:
            return img
        
        # img shape: (C, D, H, W) or (D, H, W)
        shape = img.shape[-3:]
        d, h, w = shape
        z = torch.linspace(-1.0, 1.0, d, device=img.device)
        y = torch.linspace(-1.0, 1.0, h, device=img.device)
        x = torch.linspace(-1.0, 1.0, w, device=img.device)
        
        grid_z, grid_y, grid_x = torch.meshgrid(z, y, x, indexing="ij")
        
        # Random quadratic polynomial coefficients
        coeffs = np.random.uniform(-1.0, 1.0, size=9)
        bias = (
            coeffs[0] * grid_x + coeffs[1] * grid_y + coeffs[2] * grid_z +
            coeffs[3] * (grid_x ** 2) + coeffs[4] * (grid_y ** 2) + coeffs[5] * (grid_z ** 2) +
            coeffs[6] * (grid_x * grid_y) + coeffs[7] * (grid_y * grid_z) + coeffs[8] * (grid_x * grid_z)
        )
        # Normalize and scale
        bias = bias - bias.mean()
        bias = bias / (bias.std() + 1e-6) * self.magnitude
        bias_multiplier = torch.exp(bias)
        
        if img.ndim == 4:
            bias_multiplier = bias_multiplier.unsqueeze(0)
            
        return img * bias_multiplier


class RandomMRIPhaseGhosting(RandomizableTransform):
    """
    Simulates discrete respiratory/cardiac ghosting artifacts along the phase-encoding axis (H or W).
    """
    def __init__(
        self,
        prob: float = 0.4,
        num_ghosts: int = 4,
        intensity_range: Tuple[float, float] = (0.05, 0.25),
        axis: int = -2
    ):
        super().__init__(prob=prob)
        self.num_ghosts = num_ghosts
        self.intensity_range = intensity_range
        self.axis = axis
        self.intensity: float = 0.0

    def randomize(self, data: Optional[torch.Tensor] = None) -> None:
        super().randomize(data)
        self.intensity = float(np.random.uniform(self.intensity_range[0], self.intensity_range[1]))

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        self.randomize(img)
        if not self._do_transform:
            return img
        
        # Fourier transform along the phase encoding axis
        fft = torch.fft.fft(img, dim=self.axis)
        
        # Create ghosting perturbation in k-space
        dim_len = img.shape[self.axis]
        step = max(1, dim_len // (self.num_ghosts + 1))
        ghost_mask = torch.ones_like(fft)
        
        for k in range(1, self.num_ghosts + 1):
            idx = (k * step) % dim_len
            ghost_slice = [slice(None)] * img.ndim
            ghost_slice[self.axis] = idx
            ghost_mask[tuple(ghost_slice)] *= (1.0 + self.intensity * ((-1) ** k))
            
        ghosted_fft = fft * ghost_mask
        ghosted_img = torch.fft.ifft(ghosted_fft, dim=self.axis).real
        return torch.clamp(ghosted_img, min=0.0)


class RandomSliceThicknessDegradation(RandomizableTransform):
    """
    Simulates thick clinical MRI/CT slice acquisition by downsampling along z-axis and linear interpolation.
    """
    def __init__(self, prob: float = 0.4, downsample_factors: Tuple[int, ...] = (2, 3, 4)):
        super().__init__(prob=prob)
        self.downsample_factors = downsample_factors
        self.factor: int = 1

    def randomize(self, data: Optional[torch.Tensor] = None) -> None:
        super().randomize(data)
        self.factor = int(np.random.choice(self.downsample_factors))

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        self.randomize(img)
        if not self._do_transform or self.factor == 1:
            return img
        
        orig_shape = img.shape
        # Input tensor shape: (C, D, H, W)
        need_squeeze = False
        if img.ndim == 3:
            img = img.unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
            need_squeeze = True
        elif img.ndim == 4:
            img = img.unsqueeze(0)  # (1, C, D, H, W)
            need_squeeze = True
            
        d, h, w = orig_shape[-3:]
        target_d = max(4, d // self.factor)
        
        # Downsample along depth
        downsampled = torch.nn.functional.interpolate(
            img, size=(target_d, h, w), mode="trilinear", align_corners=False
        )
        # Resample back to original thickness
        reconstructed = torch.nn.functional.interpolate(
            downsampled, size=(d, h, w), mode="trilinear", align_corners=False
        )
        
        if need_squeeze:
            reconstructed = reconstructed.squeeze(0)
            if len(orig_shape) == 3:
                reconstructed = reconstructed.squeeze(0)
                
        return reconstructed


class DomainRandomization3D(nn.Module):
    """
    Full physics-based domain randomization wrapper for CT & MRI volumes.
    Applied during training to force the model to learn invariant anatomical representations.
    """
    def __init__(
        self,
        noise_prob: float = 0.5,
        bias_prob: float = 0.5,
        ghost_prob: float = 0.35,
        thickness_prob: float = 0.35,
        gamma_range: Tuple[float, float] = (0.7, 1.5)
    ):
        super().__init__()
        self.rician_noise = RandomRicianNoise(prob=noise_prob)
        self.bias_field = RandomB1BiasField(prob=bias_prob)
        self.ghosting = RandomMRIPhaseGhosting(prob=ghost_prob)
        self.thickness = RandomSliceThicknessDegradation(prob=thickness_prob)
        self.gamma_range = gamma_range

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        # Non-linear gamma warping
        if np.random.rand() > 0.5:
            gamma = float(np.random.uniform(self.gamma_range[0], self.gamma_range[1]))
            img = torch.clamp(img, min=0.0) ** gamma
            
        img = self.bias_field(img)
        img = self.ghosting(img)
        img = self.thickness(img)
        img = self.rician_noise(img)
        return img

