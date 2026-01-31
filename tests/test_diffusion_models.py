# test_diffusion_modules.py

import pytest
import torch
from synthetic_generation.diffusion.models import SinusoidalTimeEmbedding, MLPTimeEmbedding, DiffusionNet

# ------------------------------
# SinusoidalTimeEmbedding tests
# ------------------------------
@pytest.mark.parametrize("embedding_dim", [8, 16, 32])
def test_sinusoidal_even_embedding_dim(embedding_dim):
    t = torch.arange(5)
    emb = SinusoidalTimeEmbedding(embedding_dim)
    out = emb(t)
    assert out.shape == (5, embedding_dim)
    assert torch.all(torch.isfinite(out))

@pytest.mark.parametrize("embedding_dim", [7, 15, 31])
def test_sinusoidal_odd_embedding_dim_raises(embedding_dim):
    with pytest.raises(ValueError):
        SinusoidalTimeEmbedding(embedding_dim)

# ------------------------------
# MLPTimeEmbedding tests
# ------------------------------
@pytest.mark.parametrize("num_hidden_layers, hidden_dims", [
    (0, (64,64)),
    (2, (128,128)),
    (3, [(10,20),(20,30),(30,40)]),
])
def test_mlp_time_embedding_various(num_hidden_layers, hidden_dims):
    t = torch.tensor([0, 5, 10])
    emb = MLPTimeEmbedding(num_hidden_layers=num_hidden_layers, hidden_dims=hidden_dims, embedding_dim=16)
    out = emb(t)
    assert out.shape == (3, 16)
    assert torch.all(torch.isfinite(out))

# ------------------------------
# DiffusionNet tests
# ------------------------------
@pytest.mark.parametrize("embedding", [
    None,
    MLPTimeEmbedding(num_hidden_layers=0, hidden_dims=(64,64), embedding_dim=16)
])
def test_diffusion_net_various_embeddings(embedding):
    batch_size = 3
    data_dim = 2
    t = torch.tensor([0, 10, 20])
    x_t = torch.randn(batch_size, data_dim)
    net = DiffusionNet(data_dim, time_embedding=embedding)
    out = net(x_t, t)
    assert out.shape == (batch_size, data_dim)
    assert torch.all(torch.isfinite(out))

# ------------------------------
# DiffusionNet edge cases
# ------------------------------

def test_diffusion_net_invalid_embedding():
    """Raises if embedding has no embedding_dim attribute"""
    class BadEmbedding(torch.nn.Module):
        def forward(self, t):
            return t
    with pytest.raises(AttributeError):
        DiffusionNet(data_dim=3, time_embedding=BadEmbedding())

def test_diffusion_net_forward_time_dim_check():
    """Raises if t is not 1D"""
    batch_size = 2
    data_dim = 3
    x_t = torch.randn(batch_size, data_dim)
    t = torch.randn(batch_size, 2)  # invalid shape
    
    net = DiffusionNet(data_dim)
    with pytest.raises(ValueError):
        net(x_t, t)

def test_diffusion_net_zero_hidden_custom_embedding():
    """Forward works with zero hidden layers in embedding"""
    batch_size = 4
    data_dim = 2
    t = torch.tensor([0, 10, 20, 30])
    x_t = torch.randn(batch_size, data_dim)
    
    embedding = MLPTimeEmbedding(num_hidden_layers=0, hidden_dims=(64,64), embedding_dim=16)
    net = DiffusionNet(data_dim, time_embedding=embedding)
    
    out = net(x_t, t)
    assert out.shape == (batch_size, data_dim)
    assert torch.all(torch.isfinite(out))

def test_diffusion_net_large_batch_and_timesteps():
    """Forward works with large batch and timestep indices"""
    batch_size = 128
    data_dim = 10
    T = 1000
    x_t = torch.randn(batch_size, data_dim)
    t = torch.randint(0, T, (batch_size,))
    
    net = DiffusionNet(data_dim)
    out = net(x_t, t)
    assert out.shape == (batch_size, data_dim)
    assert torch.all(torch.isfinite(out))

def test_diffusion_net_with_conditioning():
    """Test that conditioning works correctly"""
    batch_size = 4
    data_dim = 2
    conditional_dim = 3
    
    x_t = torch.randn(batch_size, data_dim)
    t = torch.tensor([0, 10, 20, 30])
    c = torch.randn(batch_size, conditional_dim)
    
    net = DiffusionNet(data_dim, conditional_dim=conditional_dim)
    out = net(x_t, t, c)
    assert out.shape == (batch_size, data_dim)
    assert torch.all(torch.isfinite(out))

def test_diffusion_net_conditional_missing_c_raises():
    """Should raise if conditional_dim > 0 but c is None"""
    net = DiffusionNet(data_dim=2, conditional_dim=3)
    x_t = torch.randn(4, 2)
    t = torch.tensor([0, 10, 20, 30])
    
    with pytest.raises(ValueError, match="Conditional dimension is > 0"):
        net(x_t, t, c=None)

def test_diffusion_net_conditional_batch_mismatch_raises():
    """Should raise if c batch size doesn't match x_t"""
    net = DiffusionNet(data_dim=2, conditional_dim=3)
    x_t = torch.randn(4, 2)
    t = torch.tensor([0, 10, 20, 30])
    c = torch.randn(5, 3)  # Wrong batch size
    
    with pytest.raises(ValueError, match="same batch size"):
        net(x_t, t, c)

def test_diffusion_net_embedding_dim_from_attribute():
    """Uses embedding.embedding_dim if present"""
    embedding = SinusoidalTimeEmbedding(embedding_dim=64)
    net = DiffusionNet(data_dim=2, time_embedding=embedding)
    assert net.time_embedding_dim == 64

def test_diffusion_net_embedding_dim_explicit():
    """Uses explicit embedding_dim parameter if embedding lacks attribute"""
    class CustomEmbedding(torch.nn.Module):
        def forward(self, t):
            return torch.randn(t.shape[0], 32)
    
    embedding = CustomEmbedding()
    net = DiffusionNet(data_dim=2, time_embedding=embedding, time_embedding_dim=32)
    assert net.time_embedding_dim == 32

def test_sinusoidal_deterministic():
    """Same input produces same output"""
    emb = SinusoidalTimeEmbedding(32)
    t = torch.tensor([0.0, 5.0, 10.0])
    out1 = emb(t)
    out2 = emb(t)
    assert torch.allclose(out1, out2)

def test_sinusoidal_distinct_timesteps():
    """Different timesteps produce different embeddings"""
    emb = SinusoidalTimeEmbedding(32)
    t1 = torch.tensor([0.0])
    t2 = torch.tensor([1.0])
    out1 = emb(t1)
    out2 = emb(t2)
    assert not torch.allclose(out1, out2)

def test_mlp_embedding_normalizes_timesteps():
    """Timesteps are normalized to [0, 1]"""
    num_timesteps = 1000
    emb = MLPTimeEmbedding(num_timesteps=num_timesteps, embedding_dim=16)
    
    # t=0 should map to 0, t=999 should map to 1
    # We can't directly test normalization without white-box access,
    # but we can verify it accepts full range
    t = torch.tensor([0, 500, 999])
    out = emb(t)
    assert out.shape == (3, 16)
    assert torch.all(torch.isfinite(out))

def test_mlp_embedding_has_embedding_dim_attribute():
    """MLPTimeEmbedding exposes embedding_dim"""
    emb = MLPTimeEmbedding(embedding_dim=48)
    assert hasattr(emb, 'embedding_dim')
    assert emb.embedding_dim == 48

def test_diffusion_net_gradients_flow():
    """Gradients flow through the network"""
    net = DiffusionNet(data_dim=2)
    x_t = torch.randn(4, 2, requires_grad=True)
    t = torch.tensor([0, 10, 20, 30])
    
    out = net(x_t, t)
    loss = out.sum()
    loss.backward()
    
    assert x_t.grad is not None
    assert not torch.all(x_t.grad == 0)