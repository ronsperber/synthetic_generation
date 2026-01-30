# test_diffusion_sampling.py

import pytest
import torch
from synthetic_generation.diffusion.sampling import q_sample, p_sample
from synthetic_generation.diffusion.models import DiffusionNet
from synthetic_generation.diffusion.schedules import linear_beta_schedule

@pytest.fixture
def diffusion_components():
    """Create common components for testing"""
    num_timesteps = 100
    betas = linear_beta_schedule(num_timesteps=num_timesteps)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)
    
    return {
        'num_timesteps': num_timesteps,
        'betas': betas,
        'alphas': alphas,
        'alphas_cumprod': alphas_cumprod,
        'alphas_cumprod_prev': alphas_cumprod_prev,
        'sqrt_alphas_cumprod': sqrt_alphas_cumprod,
        'sqrt_one_minus_alphas_cumprod': sqrt_one_minus_alphas_cumprod,
    }

# ------------------------------
# q_sample tests (forward diffusion)
# ------------------------------

def test_q_sample_shape(diffusion_components):
    """q_sample returns correct shape"""
    x0 = torch.randn(10, 2)
    t = torch.randint(0, diffusion_components['num_timesteps'], (10,))
    
    x_t, noise = q_sample(
        x0=x0,
        t=t,
        sqrt_alphas_cumprod=diffusion_components['sqrt_alphas_cumprod'],
        sqrt_one_minus_alphas_cumprod=diffusion_components['sqrt_one_minus_alphas_cumprod']
    )
    
    assert x_t.shape == x0.shape
    assert noise.shape == x0.shape

def test_q_sample_with_custom_noise(diffusion_components):
    """q_sample uses provided noise"""
    x0 = torch.randn(5, 3)
    t = torch.randint(0, diffusion_components['num_timesteps'], (5,))
    custom_noise = torch.randn(5, 3)
    
    x_t, returned_noise = q_sample(
        x0=x0,
        t=t,
        sqrt_alphas_cumprod=diffusion_components['sqrt_alphas_cumprod'],
        sqrt_one_minus_alphas_cumprod=diffusion_components['sqrt_one_minus_alphas_cumprod'],
        noise=custom_noise
    )
    
    assert torch.allclose(returned_noise, custom_noise)

def test_q_sample_deterministic_with_same_noise(diffusion_components):
    """Same x0, t, noise produces same x_t"""
    x0 = torch.randn(5, 2)
    t = torch.randint(0, diffusion_components['num_timesteps'], (5,))
    noise = torch.randn(5, 2)
    
    x_t1, _ = q_sample(
        x0=x0, t=t,
        sqrt_alphas_cumprod=diffusion_components['sqrt_alphas_cumprod'],
        sqrt_one_minus_alphas_cumprod=diffusion_components['sqrt_one_minus_alphas_cumprod'],
        noise=noise
    )
    
    x_t2, _ = q_sample(
        x0=x0, t=t,
        sqrt_alphas_cumprod=diffusion_components['sqrt_alphas_cumprod'],
        sqrt_one_minus_alphas_cumprod=diffusion_components['sqrt_one_minus_alphas_cumprod'],
        noise=noise
    )
    
    assert torch.allclose(x_t1, x_t2)

def test_q_sample_more_noise_at_later_timesteps(diffusion_components):
    """x_t should be noisier at later timesteps"""
    x0 = torch.ones(1, 2)
    noise = torch.randn(1, 2)
    
    # Early timestep
    t_early = torch.tensor([10])
    x_early, _ = q_sample(
        x0=x0, t=t_early,
        sqrt_alphas_cumprod=diffusion_components['sqrt_alphas_cumprod'],
        sqrt_one_minus_alphas_cumprod=diffusion_components['sqrt_one_minus_alphas_cumprod'],
        noise=noise
    )
    
    # Late timestep
    t_late = torch.tensor([90])
    x_late, _ = q_sample(
        x0=x0, t=t_late,
        sqrt_alphas_cumprod=diffusion_components['sqrt_alphas_cumprod'],
        sqrt_one_minus_alphas_cumprod=diffusion_components['sqrt_one_minus_alphas_cumprod'],
        noise=noise
    )
    
    # Later timestep should be further from x0
    dist_early = torch.dist(x_early, x0)
    dist_late = torch.dist(x_late, x0)
    assert dist_late > dist_early

# ------------------------------
# p_sample tests (reverse diffusion)
# ------------------------------

def test_p_sample_shape(diffusion_components):
    """p_sample returns correct shape"""
    model = DiffusionNet(data_dim=2)
    x_t = torch.randn(5, 2)
    t = 50
    
    x_prev = p_sample(
        model=model,
        x_t=x_t,
        t=t,
        betas=diffusion_components['betas'],
        alphas=diffusion_components['alphas'],
        alphas_cumprod=diffusion_components['alphas_cumprod'],
        alphas_cumprod_prev=diffusion_components['alphas_cumprod_prev']
    )
    
    assert x_prev.shape == x_t.shape

def test_p_sample_at_t0_deterministic(diffusion_components):
    """At t=0, p_sample should not add noise"""
    model = DiffusionNet(data_dim=2)
    x_t = torch.randn(5, 2)
    t = 0
    
    # Run twice, should get same result
    x_prev1 = p_sample(
        model=model, x_t=x_t, t=t,
        betas=diffusion_components['betas'],
        alphas=diffusion_components['alphas'],
        alphas_cumprod=diffusion_components['alphas_cumprod'],
        alphas_cumprod_prev=diffusion_components['alphas_cumprod_prev']
    )
    
    x_prev2 = p_sample(
        model=model, x_t=x_t, t=t,
        betas=diffusion_components['betas'],
        alphas=diffusion_components['alphas'],
        alphas_cumprod=diffusion_components['alphas_cumprod'],
        alphas_cumprod_prev=diffusion_components['alphas_cumprod_prev']
    )
    
    assert torch.allclose(x_prev1, x_prev2)

def test_p_sample_at_t_greater_than_0_stochastic(diffusion_components):
    """At t>0, p_sample should add noise (stochastic)"""
    model = DiffusionNet(data_dim=2)
    x_t = torch.randn(5, 2)
    t = 50
    
    # Run twice, should get different results due to noise
    x_prev1 = p_sample(
        model=model, x_t=x_t, t=t,
        betas=diffusion_components['betas'],
        alphas=diffusion_components['alphas'],
        alphas_cumprod=diffusion_components['alphas_cumprod'],
        alphas_cumprod_prev=diffusion_components['alphas_cumprod_prev']
    )
    
    x_prev2 = p_sample(
        model=model, x_t=x_t, t=t,
        betas=diffusion_components['betas'],
        alphas=diffusion_components['alphas'],
        alphas_cumprod=diffusion_components['alphas_cumprod'],
        alphas_cumprod_prev=diffusion_components['alphas_cumprod_prev']
    )
    
    # Should be different due to random noise
    assert not torch.allclose(x_prev1, x_prev2)

def test_p_sample_with_conditioning(diffusion_components):
    """p_sample works with conditioning"""
    model = DiffusionNet(data_dim=2, conditional_dim=3)
    x_t = torch.randn(5, 2)
    c = torch.randn(5, 3)
    t = 50
    
    x_prev = p_sample(
        model=model, x_t=x_t, t=t,
        betas=diffusion_components['betas'],
        alphas=diffusion_components['alphas'],
        alphas_cumprod=diffusion_components['alphas_cumprod'],
        alphas_cumprod_prev=diffusion_components['alphas_cumprod_prev'],
        c=c
    )
    
    assert x_prev.shape == x_t.shape

def test_p_sample_no_grad():
    """p_sample doesn't compute gradients"""
    model = DiffusionNet(data_dim=2)
    x_t = torch.randn(5, 2, requires_grad=True)
    
    betas = linear_beta_schedule(num_timesteps=100)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])
    
    x_prev = p_sample(
        model=model, x_t=x_t, t=50,
        betas=betas, alphas=alphas,
        alphas_cumprod=alphas_cumprod,
        alphas_cumprod_prev=alphas_cumprod_prev
    )
    
    # p_sample is decorated with @torch.no_grad(), so output shouldn't require grad
    assert not x_prev.requires_grad