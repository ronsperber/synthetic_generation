# test_diffusion_modules.py

import pytest
import torch
from synthetic_generation.diffusion.models import SinusoidalTimeEmbedding, MLPTimeEmbedding, DiffusionNet

# ------------------------------
# SinusoidalTimeEmbedding tests
# ------------------------------
@pytest.mark.parametrize("embed_dim", [8, 16, 32])
def test_sinusoidal_even_embed_dim(embed_dim):
    t = torch.arange(5)
    emb = SinusoidalTimeEmbedding(embed_dim)
    out = emb(t)
    assert out.shape == (5, embed_dim)
    assert torch.all(torch.isfinite(out))

@pytest.mark.parametrize("embed_dim", [7, 15, 31])
def test_sinusoidal_odd_embed_dim_raises(embed_dim):
    with pytest.raises(ValueError):
        SinusoidalTimeEmbedding(embed_dim)

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
    emb = MLPTimeEmbedding(num_hidden_layers=num_hidden_layers, hidden_dims=hidden_dims, embed_dim=16)
    out = emb(t)
    assert out.shape == (3, 16)
    assert torch.all(torch.isfinite(out))

# ------------------------------
# DiffusionNet tests
# ------------------------------
@pytest.mark.parametrize("embedding", [
    None,
    MLPTimeEmbedding(num_hidden_layers=0, hidden_dims=(64,64), embed_dim=16)
])
def test_diffusion_net_various_embeddings(embedding):
    batch_size = 3
    data_dim = 2
    t = torch.tensor([0, 10, 20])
    x_t = torch.randn(batch_size, data_dim)
    net = DiffusionNet(data_dim, embedding=embedding)
    out = net(x_t, t)
    assert out.shape == (batch_size, data_dim)
    assert torch.all(torch.isfinite(out))
