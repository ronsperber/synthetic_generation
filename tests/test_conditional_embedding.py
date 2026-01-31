
import pytest
import torch
import torch.nn as nn
from synthetic_generation.diffusion.models import DiffusionNet


@pytest.fixture
def basic_components():
    """Basic setup for tests"""
    data_dim = 4
    batch_size = 6
    x_t = torch.randn(batch_size, data_dim)
    t = torch.randint(0, 100, (batch_size,))
    return {
        'data_dim': data_dim,
        'batch_size': batch_size,
        'x_t': x_t,
        't': t,
    }


def test_forward_identity_conditional(basic_components):
    """Conditional embedding is Identity by default"""
    batch_size = basic_components['batch_size']
    x_t = basic_components['x_t']
    t = basic_components['t']
    conditional_dim = 3

    c = torch.randn(batch_size, conditional_dim)

    net = DiffusionNet(
        data_dim=basic_components['data_dim'],
        conditional_dim=conditional_dim
    )

    out = net(x_t, t, c)
    assert out.shape == x_t.shape


def test_forward_nn_embedding_conditional(basic_components):
    """Conditional embedding is an nn.Embedding"""
    batch_size = basic_components['batch_size']
    x_t = basic_components['x_t']
    t = basic_components['t']

    num_categories = 5
    embed_dim = 8
    conditional_dim = embed_dim

    labels = torch.randint(0, num_categories, (batch_size,))
    embedding = nn.Embedding(num_categories, embed_dim)

    net = DiffusionNet(
        data_dim=basic_components['data_dim'],
        conditional_dim=conditional_dim,
        conditional_embedding=embedding
    )

    out = net(x_t, t, labels)
    assert out.shape == x_t.shape


def test_missing_conditional_raises(basic_components):
    """Raises ValueError if conditional_dim > 0 but no c passed"""
    net = DiffusionNet(
        data_dim=basic_components['data_dim'],
        conditional_dim=2
    )
    with pytest.raises(ValueError):
        net(basic_components['x_t'], basic_components['t'])


def test_mismatched_batch_conditional_raises(basic_components):
    """Raises ValueError if conditional batch size doesn't match x_t"""
    batch_size = basic_components['batch_size']
    x_t = basic_components['x_t']
    t = basic_components['t']

    c = torch.randn(batch_size + 1, 2)
    net = DiffusionNet(
        data_dim=basic_components['data_dim'],
        conditional_dim=2
    )
    with pytest.raises(ValueError):
        net(x_t, t, c)


def test_embedding_dim_mismatch_raises(basic_components):
    """Raises ValueError if conditional_embedding.embedding_dim != conditional_dim"""
    batch_size = basic_components['batch_size']
    x_t = basic_components['x_t']
    t = basic_components['t']

    c = torch.randint(0, 4, (batch_size,))
    embedding = nn.Embedding(4, 3)  # embedding_dim=3

    # conditional_dim != embedding_dim
    with pytest.raises(ValueError):
        DiffusionNet(
            data_dim=basic_components['data_dim'],
            conditional_dim=5,
            conditional_embedding=embedding
        )
