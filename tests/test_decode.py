import torch
import torch.nn as nn
import pytest
from gan.models import Generator, OutputHead  # adjust imports

def test_decode_forward():
    # Simple generator with one head
    head = OutputHead(dim=1, activation=nn.Identity, decode=nn.Sigmoid, name="test_head")
    G = Generator(
        noise_dim=2,
        num_hidden_layers=1,
        hidden_dims=(4,4),
        output_heads=[head],
        hidden_activation=nn.ReLU(),
        use_conditional=False
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
            hidden_activation=nn.ReLU(),
            use_conditional=False
        )
        z = torch.randn(3,2)
        G.eval()
        out = G(z)
        assert torch.all(out >= 0) and torch.all(out <= 1)

def test_decode_none_is_identity():
    # Head with no decode
    head = OutputHead(dim=1, activation=nn.Identity, decode=None, name="no_decode")
    G = Generator(
        noise_dim=2,
        num_hidden_layers=1,
        hidden_dims=(4,4),
        output_heads=[head],
        hidden_activation=nn.ReLU(),
        use_conditional=False
    )
    z = torch.randn(3,2)
    G.eval()
    out_eval = G(z)
    G.train()
    out_train = G(z)
    # Should be identical if decode is None
    assert torch.allclose(out_eval, out_train)
