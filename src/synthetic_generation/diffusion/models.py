"""
module with model classes to use for diffusion
"""
from typing import Sequence, TypeAlias, List
from collections.abc import Callable
import torch
import torch.nn as nn
import math
# types used for the classes
LayerDims = tuple[int, int]
HiddenDims: TypeAlias = LayerDims | Sequence[LayerDims]
ActivationFactory: TypeAlias = Callable[[], nn.Module] | nn.Module


class SinusoidalTimeEmbedding(nn.Module):
    """
    simple sinusoidal embedding
    """
    def __init__(self, embed_dim: int):
        super().__init__()
        embed_dim = int(embed_dim)
        if embed_dim % 2 != 0:
            raise ValueError("Embedding dimension for sinusoidal embedding must be even")
        self.embed_dim = embed_dim
    
    def forward(self, t: torch.Tensor):
        """
        Args:
            t: (batch_size,) tensor of timesteps
        Returns:
            embeddings: (batch_size, embed_dim) tensor
        """
        half_dim = self.embed_dim // 2
        # Create frequencies: 1, 1/10, 1/100, 1/1000, ...
        t = t.float()
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=t.device) * -embeddings)
        
        # Scale timesteps by frequencies
        embeddings = t[:, None] * embeddings[None, :]
        
        # Apply sin to first half, cos to second half
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        
        return embeddings
    
class BaseMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: HiddenDims,
        num_hidden_layers: int,
        activation: ActivationFactory = nn.ReLU
    ):
        super().__init__()
        if callable(activation) and not isinstance(activation, nn.Module):
            self.activation = activation()
        else:
            self.activation = activation

        if isinstance(hidden_dims, tuple):
            hidden_dims = [hidden_dims] * max(1, num_hidden_layers)
        self.hidden_dims = hidden_dims
        self.num_hidden_layers = num_hidden_layers

        # Input/output dims for the first/last layers
        self.input_outdim = hidden_dims[0][0] 
        self.output_indim = hidden_dims[-1][1] 

        self.input_layer = nn.Linear(input_dim, self.input_outdim)
        self.output_layer = nn.Linear(self.output_indim, output_dim)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(in_dim, out_dim) for in_dim, out_dim in hidden_dims
        ])

    def forward_layers(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.input_layer(x))
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        return self.output_layer(x)


class MLPTimeEmbedding(BaseMLP):
    def __init__(self, embed_dim=32, num_hidden_layers=2, hidden_dims=(128,128), activation=nn.ReLU, num_time_steps=1000):
        super().__init__(input_dim=1, output_dim=embed_dim,
                         hidden_dims=hidden_dims, num_hidden_layers=num_hidden_layers, activation=activation)
        self.num_time_steps = num_time_steps
        self.embed_dim = embed_dim
    def forward(self, t: torch.Tensor):
        t = t.unsqueeze(-1).float() / (self.num_time_steps - 1)
        return self.forward_layers(t)

class DiffusionNet(BaseMLP):
    def __init__(self, data_dim: int, embedding: nn.Module | None = None,
                 num_hidden_layers: int = 2, hidden_dims: HiddenDims = (128,128),
                 activation: ActivationFactory = nn.ReLU):
        if embedding is None:
            embedding = MLPTimeEmbedding()
        if  not hasattr(embedding, 'embed_dim'):
            raise AttributeError(
                f"Time embedding {type(embedding).__name__} must have 'embed_dim' attribute"
            )       
        self.embed_dim = embedding.embed_dim
        super().__init__(input_dim=data_dim + self.embed_dim,
                         output_dim=data_dim,
                         hidden_dims=hidden_dims,
                         num_hidden_layers=num_hidden_layers,
                         activation=activation)
        self.embedding = embedding
        

    def forward(self, x_t: torch.Tensor, t: torch.Tensor):
        if t.dim() > 1:
            raise ValueError("time input should be 1 dimensional")
        t_embed = self.embedding(t)
        x = torch.cat([x_t, t_embed], dim=1)
        return self.forward_layers(x)


        
        
