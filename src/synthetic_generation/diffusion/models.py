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
    def __init__(self, embed_dim):
        super().__init__()
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
    
class MLPTimeEmbedding(nn.Module):
    """
    MLP embedding for time
    """
    def __init__(
            self,
            num_hidden_layers: int = 2,
            activation: ActivationFactory = nn.ReLU,
            hidden_dims: HiddenDims = (128,128),
            embed_dim: int = 32,
            num_time_steps: int = 1000
    ):
        super().__init__()
        self.num_hidden_layers = num_hidden_layers
        if isinstance(activation, nn.Module):
            self.activation = activation()
        else:
            self.activation = activation
        if isinstance(hidden_dims, tuple):
            hidden_dims = [hidden_dims] * num_hidden_layers
        if num_hidden_layers == 0:
            self.input_outdim = 128
            self.output_indum = 128
        else:
            self.input_outdim = hidden_dims[0][0]
            self.output_indim = hidden_dims[-1][1]
        self.hidden_dims = hidden_dims
        self.embed_dim = embed_dim
        self.num_time_steps = num_time_steps
        self.input_layer = nn.Linear(1, self.input_outdim)
        self.output_layer = nn.Linear(self.output_indim, self.embed_dim)
        self.hidden_layers = nn.ModuleList()
        for hidden_dim in self.hidden_dims:
            in_dim, out_dim = hidden_dim
            self.hidden_layers.append(nn.Linear(in_dim, out_dim))
            
    def forward(self, t:torch.Tensor):
        # add extra dimension to make t (batch_size, 1)
        # and normalize
        t = t.unsqueeze(-1).float() / self.num_time_steps
        t = self.activation(self.input_layer(t))
        for layer in self.hidden_layers:
            t = self.activation(layer(t))
        return self.output_layer(t)


class DiffusionNet(nn.Module):
    def __init__(
            self,
            data_dim : int, 
            embedding: nn.Module | None = None,
            num_hidden_layers : int = 2,
            hidden_dims: HiddenDims = (128, 128),
            activation : ActivationFactory = nn.ReLU,
    ):  
        super().__init__()
        self.data_dim = data_dim
        self.num_hidden_layers = num_hidden_layers
        if isinstance(hidden_dims, tuple):
            hidden_dims = [hidden_dims] * num_hidden_layers
        self.hidden_dims = hidden_dims
        self.hidden_layers=nn.ModuleList()
        if isinstance(activation, nn.Module):
            self.activation = activation()
        else:
            self.activation = activation
        if embedding is None:
            embedding = MLPTimeEmbedding()
        if  not hasattr(embedding, 'embed_dim'):
            raise AttributeError(
                f"Time embedding {type(embedding).__name__} must have 'embed_dim' attribute"
            )
        self.embedding = embedding
        self.embed_dim = embedding.embed_dim
        if self.num_hidden_layers == 0:
            self.input_outdim = 128
            self.output_indim = 128
        else:
            self.input_outdim = hidden_dims[0][0]
            self.output_indim = hidden_dims[-1][1]
        self.input_layer = nn.Linear(self.data_dim + self.embed_dim, self.input_outdim)
        self.output_layer = nn.Linear(self.output_indim, self.data_dim)
        self.hidden_layers = nn.ModuleList()
        for hidden_dim in self.hidden_dims:
            in_dim, out_dim = hidden_dim
            self.hidden_layers.append(nn.Linear(in_dim, out_dim))

    def forward(
            self,
            x_t: torch.Tensor,
            t: torch.Tensor
    ):
        if t.dim() > 1:
            raise ValueError("time input should be 1 dimensional")
        t_embed = self.embedding(t)
        xt_cat = torch.cat([x_t, t_embed], dim=1)
        xt_cat = self.activation(self.input_layer(xt_cat))
        for layer in self.hidden_layers:
            xt_cat = self.activation(layer(xt_cat))
        return self.output_layer(xt_cat)

    
        
        
