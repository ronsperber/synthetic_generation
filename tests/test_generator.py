import pytest
import torch
import torch.nn as nn
from synthetic_generation.gan.models import Generator, OutputHead

def test_generator_single_output():
    torch.manual_seed(42)
    G = Generator(noise_dim=3, num_hidden_layers=2, hidden_dims=[(3, 8), (8, 8)], out_dim=5)
    z = torch.randn(10, 3)
    out = G(z)
    assert out.shape == (10, 5)
    assert torch.isfinite(out).all()

def test_generator_multiple_heads():
    torch.manual_seed(42)
    heads = [
        OutputHead(dim=2, activation=nn.Identity, name="head1"),
        OutputHead(dim=3, activation=nn.ReLU, name="head2"),
        OutputHead(dim=1, activation=nn.Sigmoid, name="head3")
    ]
    G = Generator(noise_dim=4, num_hidden_layers=2, hidden_dims=[(4, 8), (8, 8)], output_heads=heads)
    z = torch.randn(7, 4)
    out = G(z)
    expected_dim = sum(head.dim for head in heads)
    assert out.shape == (7, expected_dim)
    assert (out[:, 2:5] >= 0).all()
    assert ((0 <= out[:, 5:6]) & (out[:, 5:6] <= 1)).all()

def test_generator_multiple_heads_with_conditional():
    torch.manual_seed(0)
    heads = [
        OutputHead(dim=2, activation=nn.Identity, name="head1"),
        OutputHead(dim=3, activation=nn.ReLU, name="head2"),
    ]
    G = Generator(
        noise_dim=4,
        num_hidden_layers=2,
        hidden_dims=[(6, 8), (8, 8)],
        output_heads=heads,
        use_conditional=True,
        conditional_dim=2,
    )
    z = torch.randn(5, 4)
    c = torch.randn(5, 2)
    out = G(z, c)
    assert out.shape == (5, 5)

def test_generator_generate_with_multiple_heads():
    torch.manual_seed(0)
    heads = [
        OutputHead(dim=3, activation=nn.Identity, name="h1"),
        OutputHead(dim=2, activation=nn.Tanh, name="h2"),
    ]
    G = Generator(noise_dim=5, num_hidden_layers=2, hidden_dims=[(5, 10), (10, 10)], output_heads=heads)
    out = G.generate(9)
    assert out.shape == (9, 5)
    assert torch.isfinite(out).all()

def test_generator_head_order_preserved():
    torch.manual_seed(0)
    heads = [
        OutputHead(dim=1, activation=nn.Identity, name="a"),
        OutputHead(dim=1, activation=nn.Identity, name="b"),
        OutputHead(dim=1, activation=nn.Identity, name="c"),
    ]
    G = Generator(noise_dim=3, num_hidden_layers=1, hidden_dims=[(3, 4)], output_heads=heads)
    z = torch.randn(4, 3)
    out = G(z)
    assert out[:, 0:1].shape == (4, 1)
    assert out[:, 1:2].shape == (4, 1)
    assert out[:, 2:3].shape == (4, 1)

def test_generator_missing_out_dim_and_heads_raises():
    with pytest.raises(ValueError):
        Generator(noise_dim=3, num_hidden_layers=1, hidden_dims=[(3, 4)], out_dim=None, output_heads=None)

def test_generator_multiple_heads_backward():
    torch.manual_seed(0)
    heads = [
        OutputHead(dim=2, activation=nn.Identity, name="h1"),
        OutputHead(dim=2, activation=nn.ReLU, name="h2"),
    ]
    G = Generator(noise_dim=4, num_hidden_layers=2, hidden_dims=[(4, 8), (8, 8)], output_heads=heads)
    z = torch.randn(6, 4)
    out = G(z)
    loss = out.mean()
    loss.backward()
    grads = [p.grad for p in G.parameters() if p.grad is not None]
    assert len(grads) > 0
