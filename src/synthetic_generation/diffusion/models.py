"""
module with model classes to use for diffusion
"""
from typing import Sequence, TypeAlias
from collections.abc import Callable
import math
import torch
import torch.nn as nn

# types used for the classes
LayerDims = tuple[int, int]
HiddenDims: TypeAlias = LayerDims | Sequence[LayerDims]
ActivationFactory: TypeAlias = Callable[[], nn.Module] | nn.Module


class SinusoidalTimeEmbedding(nn.Module):
    """
    simple sinusoidal embedding
    """
    def __init__(self, embedding_dim: int):
        super().__init__()
        embedding_dim = int(embedding_dim)
        if embedding_dim % 2 != 0:
            raise ValueError("Embedding dimension for sinusoidal embedding must be even")
        self.init_args = {
            "embedding_dim" : embedding_dim
        }
        self.embedding_dim = embedding_dim
    
    def forward(self, t: torch.Tensor):
        """
        Parameters
        ----------
        t: torch.Tensor
            (batch_size,) tensor of timesteps
        Returns
        --------
        embeddings: torch.Tensor
            (batch_size, embedding_dim) tensor
        """
        half_dim = self.embedding_dim // 2
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
    """
    Class for a basic MLP 
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: HiddenDims,
        num_hidden_layers: int,
        activation: ActivationFactory | None = None
    ):
        """
        Parameters
        ----------
        input_dim : int
            input dimension for the input layer
        output_dim : int
            output dimension for the output layer
        hidden_dims : HiddenDims
            single tuple (n,n) or list of tuples for the sizes of hidden layers
        num_hidden_layers: int
            number of hidden layers
        activation: ActivationFactory | None
            activation function to be used for all layers except output layer
            when None, nn.ReLU is used
        """
        super().__init__()
        self.init_args = {
            "input_dim": input_dim,
            "output_dim": output_dim,
            "hidden_dims": hidden_dims,
            "num_hidden_layers": num_hidden_layers,
            "activation" : activation
        }
        # make sure the activation is a callable function
        if activation is None:
            activation=nn.ReLU
        if callable(activation) and not isinstance(activation, nn.Module):
            self.activation = activation()
        else:
            self.activation = activation
        if isinstance(hidden_dims, tuple):
            hidden_dims = [hidden_dims] * max(1, num_hidden_layers)
        self.hidden_dims = hidden_dims
        self.num_hidden_layers = num_hidden_layers
        # validate hidden layer dimensions when there are hidden layers
        if self.num_hidden_layers > 0:
            # validate that number of pairs is number of layers
            if len(self.hidden_dims) != self.num_hidden_layers:
                raise ValueError(
                    "Number of hidden layers and length of hidden_dims must be equal"
                )
            # validate that each hidden dim is a tuple (in_dim, out_dim)
            for i, h in enumerate(self.hidden_dims):
                if not (isinstance(h, tuple) and len(h) == 2):
                    raise ValueError(f"hidden_dims[{i} must be a tuple (in_dim, out_dim), got {h}]")
            # validate that output dimension of a layer matches input dimension of next
            for i in range(len(self.hidden_dims) - 1):
                if self.hidden_dims[i][1] != self.hidden_dims[i + 1][0]:
                    raise ValueError(
                        f"hidden_dims[{i}][1] ({self.hidden_dims[i][1]}) "
                        f"!= hidden_dims[{i + 1}][0] ({self.hidden_dims[i + 1][0]})"
                        )

        # Input/output dims for the first/last layers
        self.input_outdim = hidden_dims[0][0] 
        self.output_indim = hidden_dims[-1][1] 

        self.input_layer = nn.Linear(input_dim, self.input_outdim)
        self.output_layer = nn.Linear(self.output_indim, output_dim)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(in_dim, out_dim) for in_dim, out_dim in hidden_dims
        ])

    def forward_layers(self, x: torch.Tensor) -> torch.Tensor:
        """
        passing input through all layers
        Parameters
        ----------
        x : torch.Tensor
            input tensor
        Returns
        -------
        torch.Tensor
            tensor output from the MLP
        """
        x = self.activation(self.input_layer(x))
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        return self.output_layer(x)


class MLPTimeEmbedding(BaseMLP):
    """
    MLP Time embedding class
    """
    def __init__(
            self,
            embedding_dim: int = 32,
            num_hidden_layers:int = 2, 
            hidden_dims:HiddenDims = (128,128),
            activation:ActivationFactory | None = None,
            num_timesteps:int = 1000):
        """
        Parameters
        embedding_dim : int
            the dimension of the space to embed t
        num_hidden_layers: int
            number of hidden layers
        hidden_dims: HiddenDims
            dimensions to use for the hidden layers
        activation : ActivationFactory | None
            activation function for layers other than output layer
            When None, it will default to nn.ReLU
        num_timesteps: int
            number of time steps possible for t
        """
        super().__init__(
            input_dim=1,
            output_dim=embedding_dim,
            hidden_dims=hidden_dims,
            num_hidden_layers=num_hidden_layers,
            activation=activation
            )
        self.init_args = {
            "embedding_dim": embedding_dim,
            "num_hidden_layers": num_hidden_layers,
            "hidden_dims": hidden_dims,
            "activation": activation,
            "num_timesteps": num_timesteps
        }
        self.num_timesteps = num_timesteps
        self.embedding_dim = embedding_dim
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        forward method
        Parameters
        ----------
        t: torch.Tensor
            time steps
        Returns
        -------
        torch.Tensor
            embedded time step tensor
        """
        t = t.unsqueeze(-1).float() / (self.num_timesteps - 1)
        return self.forward_layers(t)

class DiffusionNet(BaseMLP):
    """
    Class for Diffusion model
    """
    def __init__(
            self,
            data_dim: int,
            conditional_dim: int = 0,
            conditional_embedding: nn.Module | None = None,
            time_embedding_dim : int | None = None,
            time_embedding: nn.Module | None = None,
            num_hidden_layers: int = 2,
            hidden_dims: HiddenDims = (128,128),
            activation: ActivationFactory | None = None
            ):
        """
        Parameters
        ----------
        data_dim : int
            dimension of data that is being generated
        conditional_dim : int
            dimension of conditioning tensor or embedded conditioning tensor
        conditional_embedding : nn.Module | None
            when not None, embedding module to use for conditioning tensor
        time_embedding_dim: int | None
            when the time_embedding module has no embedding_dim attribute, used for dimension of embedding
            ignored if embedding has the attribute
        time_embedding : nn.Module | None
            when not None, the embedding module for the time input
        num_hidden_layers: int
            number of hidden layers to use post embedding
        hidden_dims: int
            dimensions for hidden layers post embedding
        activation: ActivationFactory | None
            activation to be used post embedding. When None, nn.ReLU is the default
        """
        self.init_args = {
            "data_dim": data_dim,
            "conditional_dim": conditional_dim,
            "conditional_embedding": conditional_embedding,
            "time_embedding_dim": time_embedding_dim,
            "time_embedding": time_embedding,
            "num_hidden_layers": num_hidden_layers,
            "hidden_dims": hidden_dims,
            "activation": activation
        }
        # if no embedding is specified, use default MLP Time embedding
        time_embedding = time_embedding or MLPTimeEmbedding()
        # make sure the embedding has an embedding_dim attribute needed to know dimension of time embedding
        if  hasattr(time_embedding, 'embedding_dim'):
            self.time_embedding_dim = time_embedding.embedding_dim
        elif time_embedding_dim is not None:
            self.time_embedding_dim = time_embedding_dim
        else:
            raise AttributeError(
                f"Time embedding {type(time_embedding).__name__} must have 'embedding_dim' attribute "
                " or embedding_dim must be specified."
            )       
        self.conditional_dim = conditional_dim
        self.data_dim = data_dim
        super().__init__(input_dim=data_dim + self.time_embedding_dim + self.conditional_dim,
                         output_dim=data_dim,
                         hidden_dims=hidden_dims,
                         num_hidden_layers=num_hidden_layers,
                         activation=activation)
        if conditional_dim > 0:
            # validate set conditional embedding when conditional_dim > 0
            conditional_embedding = conditional_embedding or nn.Identity()
            self.conditional_embedding = conditional_embedding
            if hasattr(conditional_embedding,'embedding_dim'):
                if conditional_embedding.embedding_dim != conditional_dim:
                    raise ValueError(
                        f"conditional_dim {conditional_dim} does not match "
                        f"conditional_embedding.embedding_dim {conditional_embedding.embedding_dim}"
                    )
        else: 
            self.conditional_embedding = None
        self.time_embedding = time_embedding
        
        

    def forward(
            self,
            x_t: torch.Tensor,
            t: torch.Tensor,
            c: torch.Tensor | None = None
             ) -> torch.Tensor:
        """
        Parameters
        x_t : torch.Tensor
            x input with noise coming from timestep t
        t: torch.Tensor
            tensor of time steps
        c : torch.Tensor | None
            optional conditional tensor
        Returns
        -------
        torch.Tensor
            output of passing x_t, t, c to model
        """
        # verify that t is a 1-d vector
        if t.dim() > 1:
            raise ValueError("time input should be 1 dimensional")
        # embed t using the embedding module
        t_embed = self.time_embedding(t)
        # verify that c is correct when the conditional dimension is > 0 
        if self.conditional_dim > 0:
            if c is None:
                raise ValueError("Conditional dimension is > 0 , but no conditional was passed")
            if c.shape[0] != x_t.shape[0]:
                raise ValueError("Conditional tensor must have same batch size as x_t")
            c = self.conditional_embedding(c)
            x = torch.cat([x_t, t_embed, c], dim=1)
        else:
            x = torch.cat([x_t, t_embed], dim=1)
        return self.forward_layers(x)

