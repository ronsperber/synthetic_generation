"""
module with model classes to use for diffusion
"""
import torch
import torch.nn as nn
import math

class SinusoidalTimeEmbedding(nn.Module):
    """
    simple sinusoidal embedding
    """
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
    
    def forward(self, t):
        """
        Args:
            t: (batch_size,) tensor of timesteps
        Returns:
            embeddings: (batch_size, embed_dim) tensor
        """
        half_dim = self.embed_dim // 2
        # Create frequencies: 1, 1/10, 1/100, 1/1000, ...
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
            num_layers: int = 2,
            activation: nn.Module = nn.ReLU,
            layer_width: int = 128,
            embed_dim: int = 32
    ):
        super().__init__()
        self.num_layers = num_layers
        self.activation = activation
        self.layer_width = layer_width
        self.embed_dim = embed_dim
        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.layers.append(nn.Linear(1, self.layer_width))
        for _ in range(self.num_layers - 2):
            self.layers.append(nn.Linear(self.layer_width, self.layer_width))
        for _ in range(self.num_layers - 1):
                self.activations.append(activation())
        self.layers.append(nn.Linear(self.layer_width, self.embed_dim))
            
    def forward(self, t:torch.Tensor):
        # add extra dimension to make t (batch_size, 1)
        t = t.unsqueeze(-1)
        for i,layer in enumerate(self.layers[:-1]):
            t = self.activations[i](layer(t))
        return self.layers[-1](t)


class DiffusionNet(nn.Module):
    def __init__(
            self,
            data_dim : int, 
            embedding: nn.Module | None = None,
            num_hidden_layers : int = 2,
            layer_width : int = 128,
            activation : nn.Module = nn.ReLU,
            time_steps : int = 1000
    ):
        self.embedding = embedding
        self.data_dim = data_dim
        self.num_hidden_layers = num_hidden_layers
        self.layer_width = layer_width
        self.T = time_steps
        self.out_layers=nn.ModuleList()
        self.out_activations=nn.ModuleList()
        if  not hasattr(embedding, 'embed_dim'):
            raise AttributeError(
                f"Time embedding {type(embedding).__name__} must have 'embed_dim' attribute"
            )
        self.embed_dim = embedding.embed_dim
        self.out_layers.append(nn.Linear(self.embed_dim + self.data_dim, self.layer_width))
        for _ in range(self.out_layers - 2):
            self.out_layers.append(nn.Linear(layer_width, layer_width))
        self.out_layers.append(nn.Linear(self.layer_width, self.data_dim))
        for _ in range(self.out_layers - 1):
            self.out_activations.append(activation())
        if embedding is not None:
            self.embedding = embedding
        else:
            self.embedding = MLPTimeEmbedding()
    def forward(
            self,
            x_t: torch.Tensor,
            t: torch.Tensor
    ):
        if t.dim() > 1:
            raise ValueError("time input should be 1 dimensional")
        t_embed = self.embedding(t.float() / self.T)
        xt_cat = torch.cat([x_t, t_embed], dim=1)
        for i, layer in enumerate(self.out_layers[:-1]):
            xt_cat = self.out_activations[i](layer(xt_cat))
        return self.out_layers[-1](xt_cat)


    
        
        
