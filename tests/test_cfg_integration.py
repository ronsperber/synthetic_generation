import torch
import pytest
import numpy as np
from synthetic_generation.diffusion.process import DiffusionProcess, make_null_conditional

# --- Dummy model for DiffusionProcess ---
class DummyDiffusionModel(torch.nn.Module):
    def forward(self, x, t, c=None):
        # deterministic function for testing: just adds mean of conditional
        if c is None:
            return torch.zeros_like(x)
        return c.mean(dim=1, keepdim=True).expand_as(x)

# --- Fixtures ---
@pytest.fixture
def diffusion_process():
    data_dim = 4
    num_timesteps = 6  # very small for testing
    model = DummyDiffusionModel()
    process = DiffusionProcess(data_dim=data_dim, num_timesteps=num_timesteps, model=model)
    
    # fake schedules
    betas = torch.linspace(0.01, 0.1, num_timesteps)
    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])
    
    process.betas = betas
    process.alphas = alphas
    process.alphas_cumprod = alphas_cumprod
    process.alphas_cumprod_prev = alphas_cumprod_prev
    
    return process

@pytest.fixture
def conditional_tensor():
    return torch.randn(3, 2)  # batch_size=3, cond_dim=2

# --- Tests ---

def test_generate_samples_unconditional(diffusion_process):
    num_samples = 3
    samples = diffusion_process.generate_samples(num_samples=num_samples)
    assert samples.shape == (num_samples, diffusion_process.data_dim)
    # outputs should be finite numbers
    assert torch.isfinite(samples).all()

def test_generate_samples_conditional(diffusion_process, conditional_tensor):
    num_samples = conditional_tensor.shape[0]
    # simple conditional test
    samples = diffusion_process.generate_samples(
        num_samples=num_samples,
        c=conditional_tensor,
        guidance_scale=1.0
    )
    assert samples.shape[0] == num_samples
    assert samples.shape[1] == diffusion_process.data_dim
    assert torch.isfinite(samples).all()

def test_generate_samples_cfg(diffusion_process, conditional_tensor):
    num_samples = conditional_tensor.shape[0]
    samples = diffusion_process.generate_samples(
        num_samples=num_samples,
        c=conditional_tensor,
        guidance_scale=1.5
    )
    assert samples.shape == (num_samples, diffusion_process.data_dim)
    assert torch.isfinite(samples).all()

def test_generate_samples_ddim_unconditional(diffusion_process):
    num_samples = 2
    samples = diffusion_process.generate_samples_ddim(
        num_samples=num_samples,
        num_inference_steps=3,
        eta=0.0
    )
    assert samples.shape == (num_samples, diffusion_process.data_dim)
    assert torch.isfinite(samples).all()

def test_generate_samples_ddim_cfg(diffusion_process, conditional_tensor):
    num_samples = conditional_tensor.shape[0]
    samples = diffusion_process.generate_samples_ddim(
        num_samples=num_samples,
        num_inference_steps=3,
        eta=0.0,
        c=conditional_tensor,
        guidance_scale=1.2
    )
    assert samples.shape == (num_samples, diffusion_process.data_dim)
    assert torch.isfinite(samples).all()

def test_ddim_deterministic_consistency(diffusion_process):
    num_samples = 2
    torch.manual_seed(42)
    x1 = diffusion_process.generate_samples_ddim(
        num_samples=num_samples,
        num_inference_steps=3,
        eta=0.0
    )
    torch.manual_seed(42)
    x2 = diffusion_process.generate_samples_ddim(
        num_samples=num_samples,
        num_inference_steps=3,
        eta=0.0
    )
    # With eta=0 (deterministic), outputs should be identical
    assert torch.allclose(x1, x2), "DDIM deterministic outputs differ!"

def test_cfg_changes_output(diffusion_process, conditional_tensor):
    num_samples = conditional_tensor.shape[0]
    # Unconditional
    x_uncond = diffusion_process.generate_samples(
        num_samples=num_samples,
        c=conditional_tensor,
        guidance_scale=1.0
    )
    # With guidance > 1
    x_cfg = diffusion_process.generate_samples(
        num_samples=num_samples,
        c=conditional_tensor,
        guidance_scale=1.5
    )
    # Outputs should differ
    assert not torch.allclose(x_uncond, x_cfg), "CFG did not change the output!"

def test_ddim_cfg_changes_output(diffusion_process, conditional_tensor):
    num_samples = conditional_tensor.shape[0]
    x_uncond = diffusion_process.generate_samples_ddim(
        num_samples=num_samples,
        num_inference_steps=3,
        eta=0.0,
        c=conditional_tensor,
        guidance_scale=1.0
    )
    x_cfg = diffusion_process.generate_samples_ddim(
        num_samples=num_samples,
        num_inference_steps=3,
        eta=0.0,
        c=conditional_tensor,
        guidance_scale=1.5
    )
    assert not torch.allclose(x_uncond, x_cfg), "DDIM CFG did not change the output!"
