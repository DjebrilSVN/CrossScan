"""Unit tests for physics-informed domain randomization module."""
import torch
import numpy as np
from crossscan.data.domain_randomization import (
    RandomRicianNoise,
    RandomB1BiasField,
    RandomMRIPhaseGhosting,
    RandomSliceThicknessDegradation,
    DomainRandomization3D
)


def test_rician_noise():
    img = torch.ones((1, 16, 16, 16), dtype=torch.float32)
    noise_op = RandomRicianNoise(prob=1.0, sigma_range=(0.05, 0.1))
    noisy = noise_op(img)
    assert noisy.shape == img.shape
    assert not torch.allclose(noisy, img)
    assert torch.all(noisy >= 0.0)


def test_b1_bias_field():
    img = torch.ones((1, 16, 16, 16), dtype=torch.float32)
    bias_op = RandomB1BiasField(prob=1.0, magnitude_range=(0.3, 0.5))
    biased = bias_op(img)
    assert biased.shape == img.shape
    assert not torch.allclose(biased, img)
    # Bias field should be smooth and strictly positive
    assert torch.all(biased > 0.0)


def test_slice_thickness_degradation():
    img = torch.randn((1, 16, 16, 16), dtype=torch.float32)
    thick_op = RandomSliceThicknessDegradation(prob=1.0, downsample_factors=(2,))
    degraded = thick_op(img)
    assert degraded.shape == img.shape


def test_domain_randomization_pipeline():
    img = torch.ones((1, 1, 24, 24, 24), dtype=torch.float32) * 0.5
    pipeline = DomainRandomization3D()
    out = pipeline(img)
    assert out.shape == img.shape
    assert not torch.isnan(out).any()

