import torch
import pytest
from synthetic_generation.diffusion.process import make_null_conditional
from synthetic_generation.diffusion.sampling import p_sample_cfg, ddim_sample_cfg

# --- Dummy model for testing ---
class DummyModel(torch.nn.Module):
    def forward(self, x, t, c=None):
        # Return something deterministic based on c if given
        if c is None:
            return torch.zeros_like(x)
        return c.mean(dim=1, keepdim=True).expand_as(x)

# --- Fixtures ---
@pytest.fixture
def dummy_data():
    x_t = torch.randn(3, 4)  # batch_size=3, data_dim=4
    c = torch.randn(3, 2)    # conditional dim = 2
    null_c = make_null_conditional(c)  # should be zeros
    betas = torch.linspace(0.01, 0.1, 5)
    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])
    return x_t, c, null_c, betas, alphas, alphas_cumprod, alphas_cumprod_prev

@pytest.fixture
def model():
    return DummyModel()

# --- Tests ---
def test_p_sample_cfg_shape(model, dummy_data):
    x_t, c, null_c, betas, alphas, alphas_cumprod, alphas_cumprod_prev = dummy_data
    t = 2
    guidance = 1.5
    x_next = p_sample_cfg(
        model=model,
        x_t=x_t,
        t=t,
        betas=betas,
        alphas=alphas,
        alphas_cumprod=alphas_cumprod,
        alphas_cumprod_prev=alphas_cumprod_prev,
        c=c,
        c_null=null_c,
        guidance_scale=guidance
    )
    assert x_next.shape == x_t.shape
    # Should not be identical to unconditional if guidance > 1
    x_uncond = p_sample_cfg(
        model=model,
        x_t=x_t,
        t=t,
        betas=betas,
        alphas=alphas,
        alphas_cumprod=alphas_cumprod,
        alphas_cumprod_prev=alphas_cumprod_prev,
        c=c,
        c_null=null_c,
        guidance_scale=1.0
    )
    assert not torch.allclose(x_next, x_uncond)

def test_ddim_sample_cfg_deterministic(model, dummy_data):
    x_t, c, null_c, _, _, alphas_cumprod, _ = dummy_data
    t, t_prev = 4, 3
    guidance = 1.2
    # eta=0 means deterministic
    x_next = ddim_sample_cfg(
        model=model,
        x_t=x_t,
        t=t,
        t_prev=t_prev,
        alphas_cumprod=alphas_cumprod,
        eta=0.0,
        c=c,
        c_null=null_c,
        guidance_scale=guidance
    )
    assert x_next.shape == x_t.shape
    # Re-run and check determinism
    x_next2 = ddim_sample_cfg(
        model=model,
        x_t=x_t,
        t=t,
        t_prev=t_prev,
        alphas_cumprod=alphas_cumprod,
        eta=0.0,
        c=c,
        c_null=null_c,
        guidance_scale=guidance
    )
    assert torch.allclose(x_next, x_next2)
