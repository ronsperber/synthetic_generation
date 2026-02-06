import torch
import pytest
from synthetic_generation.diffusion.process import make_null_conditional
from synthetic_generation.diffusion.sampling import p_sample_cfg, ddim_sample_cfg

# --- Dummy model for testing ---
class DummyModel(torch.nn.Module):
    def forward(self, x, t, c=None):
        # deterministic output based on conditional
        if c is None:
            return torch.zeros_like(x)
        return c.mean(dim=1, keepdim=True).expand_as(x)

# --- Fixtures ---
@pytest.fixture
def dummy_data():
    x_t = torch.randn(4, 5)  # batch_size=4, data_dim=5
    c = torch.randn(4, 3)    # conditional dim = 3
    null_c = make_null_conditional(c)
    betas = torch.linspace(0.01, 0.1, 6)
    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])
    return x_t, c, null_c, betas, alphas, alphas_cumprod, alphas_cumprod_prev

@pytest.fixture
def model():
    return DummyModel()

# --- Advanced Tests ---

def test_make_null_conditional_shapes():
    c = torch.randn(3, 2)
    null = make_null_conditional(c)
    assert null.shape == c.shape
    assert torch.all(null == 0)

    # test passing a non-zero null_token
    null_token = torch.tensor([1.0, 2.0])
    null2 = make_null_conditional(c, null_token=null_token)
    assert null2.shape == c.shape
    for i in range(c.shape[0]):
        assert torch.allclose(null2[i], null_token)

def test_p_sample_cfg_interpolation(model, dummy_data):
    x_t, c, null_c, betas, alphas, alphas_cumprod, alphas_cumprod_prev = dummy_data
    t = 3

    # guidance_scale = 1 should produce output same as conditional
    x_guided_1 = p_sample_cfg(
        model=model, x_t=x_t, t=t,
        betas=betas, alphas=alphas,
        alphas_cumprod=alphas_cumprod,
        alphas_cumprod_prev=alphas_cumprod_prev,
        c=c, c_null=null_c, guidance_scale=1.0
    )

    # guidance_scale > 1 should move output away from unconditional
    x_guided_2 = p_sample_cfg(
        model=model, x_t=x_t, t=t,
        betas=betas, alphas=alphas,
        alphas_cumprod=alphas_cumprod,
        alphas_cumprod_prev=alphas_cumprod_prev,
        c=c, c_null=null_c, guidance_scale=1.5
    )
    assert not torch.allclose(x_guided_1, x_guided_2)

def test_ddim_sample_cfg_deterministic_and_stochastic(model, dummy_data):
    x_t, c, null_c, _, _, alphas_cumprod, _ = dummy_data
    t, t_prev = 5, 4

    # deterministic DDIM with eta=0
    x1 = ddim_sample_cfg(model=model, x_t=x_t, t=t, t_prev=t_prev,
                         alphas_cumprod=alphas_cumprod, eta=0.0,
                         c=c, c_null=null_c, guidance_scale=1.2)
    x2 = ddim_sample_cfg(model=model, x_t=x_t, t=t, t_prev=t_prev,
                         alphas_cumprod=alphas_cumprod, eta=0.0,
                         c=c, c_null=null_c, guidance_scale=1.2)
    assert torch.allclose(x1, x2)

    # stochastic DDIM with eta > 0 should produce different outputs
    x3 = ddim_sample_cfg(model=model, x_t=x_t, t=t, t_prev=t_prev,
                         alphas_cumprod=alphas_cumprod, eta=1.0,
                         c=c, c_null=null_c, guidance_scale=1.2)
    x4 = ddim_sample_cfg(model=model, x_t=x_t, t=t, t_prev=t_prev,
                         alphas_cumprod=alphas_cumprod, eta=1.0,
                         c=c, c_null=null_c, guidance_scale=1.2)
    # Very small chance outputs are identical, but extremely unlikely
    assert not torch.allclose(x3, x4)

def test_p_sample_cfg_with_none_conditional(model, dummy_data):
    x_t, _, _, betas, alphas, alphas_cumprod, alphas_cumprod_prev = dummy_data
    t = 2
    # None conditional should work and produce zeros
    x_out = p_sample_cfg(
        model=model, x_t=x_t, t=t,
        betas=betas, alphas=alphas,
        alphas_cumprod=alphas_cumprod,
        alphas_cumprod_prev=alphas_cumprod_prev,
        c=None, c_null=None, guidance_scale=1.0
    )
    assert x_out.shape == x_t.shape
