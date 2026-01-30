# test_diffusion_schedules.py

import pytest
import torch
from synthetic_generation.diffusion.schedules import linear_beta_schedule, cosine_beta_schedule

# ------------------------------
# linear_beta_schedule tests
# ------------------------------

def test_linear_beta_schedule_shape():
    """Returns tensor with correct length"""
    betas = linear_beta_schedule(num_timesteps=1000)
    assert betas.shape == (1000,)

def test_linear_beta_schedule_range():
    """Betas are monotonically increasing from beta_start to beta_end"""
    beta_start = 1e-4
    beta_end = 0.02
    betas = linear_beta_schedule(beta_start=beta_start, beta_end=beta_end, num_timesteps=1000)
    
    assert torch.allclose(betas[0], torch.tensor(beta_start), atol=1e-6)
    assert torch.allclose(betas[-1], torch.tensor(beta_end), atol=1e-6)
    assert torch.all(betas[1:] >= betas[:-1])  # Monotonically increasing

def test_linear_beta_schedule_custom_params():
    """Works with custom parameters"""
    betas = linear_beta_schedule(beta_start=0.001, beta_end=0.1, num_timesteps=500)
    assert betas.shape == (500,)
    assert betas[0] >= 0.001
    assert betas[-1] <= 0.1

@pytest.mark.parametrize("num_timesteps", [0, -1, -100])
def test_linear_beta_schedule_invalid_timesteps(num_timesteps):
    """Raises for non-positive timesteps"""
    with pytest.raises(ValueError, match="positive"):
        linear_beta_schedule(num_timesteps=num_timesteps)

def test_linear_beta_schedule_beta_end_less_than_start():
    """Raises if beta_end <= beta_start"""
    with pytest.raises(ValueError, match="beta_end must be greater"):
        linear_beta_schedule(beta_start=0.02, beta_end=0.01, num_timesteps=1000)

def test_linear_beta_schedule_zero_beta_start():
    """Raises if beta_start <= 0"""
    with pytest.raises(ValueError, match="beta_start should be > 0"):
        linear_beta_schedule(beta_start=0.0, beta_end=0.02, num_timesteps=1000)

def test_linear_beta_schedule_negative_beta_start():
    """Raises if beta_start < 0"""
    with pytest.raises(ValueError, match="beta_start should be > 0"):
        linear_beta_schedule(beta_start=-0.001, beta_end=0.02, num_timesteps=1000)

# ------------------------------
# cosine_beta_schedule tests
# ------------------------------

def test_cosine_beta_schedule_shape():
    """Returns tensor with correct length"""
    betas = cosine_beta_schedule(num_timesteps=1000)
    assert betas.shape == (1000,)

def test_cosine_beta_schedule_clipped():
    """All betas are clipped to [0.0001, 0.9999]"""
    betas = cosine_beta_schedule(num_timesteps=1000)
    assert torch.all(betas >= 0.0001)
    assert torch.all(betas <= 0.9999)

def test_cosine_beta_schedule_custom_s():
    """Works with custom s parameter"""
    betas1 = cosine_beta_schedule(num_timesteps=1000, s=0.008)
    betas2 = cosine_beta_schedule(num_timesteps=1000, s=0.02)
    
    # Different s should produce different schedules
    assert not torch.allclose(betas1, betas2)

@pytest.mark.parametrize("num_timesteps", [0, -1, -100])
def test_cosine_beta_schedule_invalid_timesteps(num_timesteps):
    """Raises for non-positive timesteps"""
    with pytest.raises(ValueError, match="positive"):
        cosine_beta_schedule(num_timesteps=num_timesteps)

@pytest.mark.parametrize("s", [-0.1, 1.0, 1.5])
def test_cosine_beta_schedule_invalid_s(s):
    """Raises if s not in [0, 1)"""
    with pytest.raises(ValueError, match="interval"):
        cosine_beta_schedule(num_timesteps=1000, s=s)

def test_cosine_beta_schedule_produces_valid_alphas():
    """Betas should produce valid alphas (1 - beta > 0)"""
    betas = cosine_beta_schedule(num_timesteps=1000)
    alphas = 1 - betas
    assert torch.all(alphas > 0)
    assert torch.all(alphas < 1)

# ------------------------------
# Comparison tests
# ------------------------------

def test_linear_vs_cosine_different():
    """Linear and cosine schedules produce different betas"""
    linear_betas = linear_beta_schedule(num_timesteps=1000)
    cosine_betas = cosine_beta_schedule(num_timesteps=1000)
    
    assert not torch.allclose(linear_betas, cosine_betas)