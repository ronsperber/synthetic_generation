import torch
import torch.nn as nn
import pytest
from gan.models import Generator, Discriminator, OutputHead

import torch
import torch.nn as nn

def test_generator_forward_runs_with_activation():
    G = Generator(
        noise_dim=3,
        num_hidden_layers=1,
        hidden_dims=[(3, 3)],
        hidden_activation=nn.ReLU(),
        output_heads=[OutputHead(dim=2,activation=nn.ReLU())]
    )

    z = torch.randn(4, 3)  # batch of 4 samples

    output = G.forward(z)

    # Basic sanity checks
    assert output.shape == (4, 2), "Output should match the sum of head dimensions"
    assert isinstance(G.activation, nn.ReLU), "Activation attribute should store the instance"
    assert torch.all(output >= 0), "ReLU should zero out negative values"
