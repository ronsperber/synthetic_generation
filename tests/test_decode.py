import torch
import torch.nn as nn
import pytest
from synthetic_generation.gan.models import Generator, OutputHead  
from synthetic_generation.gan.activations import GumbelSoftmax
def test_decode_forward():
    # Simple generator with one head
    head = OutputHead(dim=1, activation=nn.Identity, decode=nn.Sigmoid, name="test_head")
    G = Generator(
        noise_dim=2,
        num_hidden_layers=1,
        hidden_dims=(4,4),
        output_heads=[head],
        hidden_activation=nn.ReLU()
    )

    # Fixed input
    z = torch.tensor([[0.0, 1.0], [1.0, -1.0]])

    # Training mode -> decode should NOT be applied
    G.train()
    out_train = G(z)
    # output should match linear layer output (after activation)
    assert out_train.shape == (2,1)
    
    # Eval mode -> decode should be applied
    G.eval()
    out_eval = G(z)
    assert out_eval.shape == (2,1)
    
    # Sigmoid should change the values
    assert torch.all(out_eval >= 0) and torch.all(out_eval <= 1)
    # Should differ from training output if linear output goes outside [0,1]
    assert not torch.allclose(out_eval, out_train)

def test_decode_callable_vs_module():
    # Head with decode as callable class
    head_class = OutputHead(dim=1, activation=nn.Identity, decode=nn.Sigmoid, name="head_class")
    # Head with decode as instantiated module
    head_instance = OutputHead(dim=1, activation=nn.Identity, decode=nn.Sigmoid(), name="head_instance")

    for head in [head_class, head_instance]:
        G = Generator(
            noise_dim=2,
            num_hidden_layers=1,
            hidden_dims=(4,4),
            output_heads=[head],
            hidden_activation=nn.ReLU()
        )
        z = torch.randn(3,2)
        G.eval()
        out = G(z)
        assert torch.all(out >= 0) and torch.all(out <= 1)

def test_no_decode_is_identity():
    # Head with no decode
    head = OutputHead(dim=1, activation=nn.Identity, name="no_decode")
    G = Generator(
        noise_dim=2,
        num_hidden_layers=1,
        hidden_dims=(4,4),
        output_heads=[head],
        hidden_activation=nn.ReLU()
    )
    z = torch.randn(3,2)
    G.eval()
    out_eval = G(z)
    G.train()
    out_train = G(z)
    # Should be identical if decode is not defined
    assert torch.allclose(out_eval, out_train)


def test_multihead_decode():
    # Define multiple heads
    heads = [
        OutputHead(dim=1, activation=nn.Identity, decode=nn.Sigmoid, name="sigmoid_head"),
        OutputHead(dim=2, activation=nn.Identity, decode=nn.Identity, name="identity_head"),
        OutputHead(dim=3, activation=nn.Identity, decode=GumbelSoftmax(), name="gumbel_head")
    ]

    G = Generator(
        noise_dim=5,
        num_hidden_layers=2,
        hidden_dims=[(8,8), (8,8)],
        output_heads=heads,
        hidden_activation=nn.ReLU()
    )

    z = torch.randn(4,5)

    # Training mode -> decode should NOT be applied
    G.train()
    out_train = G(z)
    assert out_train.shape == (4, 1+2+3)
    
    # Eval mode -> decode should be applied
    G.eval()
    out_eval = G(z)
    assert out_eval.shape == (4, 1+2+3)
    
    # Sigmoid head should be in [0,1]
    assert torch.all((0 <= out_eval[:,0]) & (out_eval[:,0] <= 1))
    # Identity head should be unchanged (compared to training output slice)
    assert torch.allclose(out_eval[:,1:3], out_train[:,1:3])
    # check gumbel head
    gumbel_slice = out_eval[:, 3:6]  # slice for the 3-dim Gumbel head
    # Check that every element is either 0 or 1
    assert torch.all((gumbel_slice == 0) | (gumbel_slice == 1))
    # Check that each row sums to exactly 1
    assert torch.all(gumbel_slice.sum(dim=1) == 1)

