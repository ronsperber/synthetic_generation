"""
module with model classes to use for diffusion
"""
from typing import Sequence, TypeAlias
import warnings
from collections.abc import Callable
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import math
from .sampling import p_sample, q_sample
from synthetic_generation.data_utils import make_dataloader
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
        Parameters
        ----------
        t: torch.Tensor
            (batch_size,) tensor of timesteps
        Returns
        --------
        embeddings: torch.Tensor
            (batch_size, embed_dim) tensor
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
    """
    Class for a basic MLP 
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: HiddenDims,
        num_hidden_layers: int,
        activation: ActivationFactory = nn.ReLU
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
        activation: ActivationFactory
            activation function to be used for all layers except output layer
        """
        super().__init__()
        # make sure the activation is a callable function
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
            embed_dim: int = 32,
            num_hidden_layers:int = 2, 
            hidden_dims:HiddenDims = (128,128),
            activation:ActivationFactory = nn.ReLU,
            num_time_steps:int = 1000):
        """
        Parameters
        embed_dim : int
            the dimension of the space to embed t
        num_hidden_layers: int
            number of hidden layers
        hidden_dims: HiddenDims
            dimensions to use for the hidden layers
        activation : ActivationFactory
            activation function for layers other than output layer
        num_time_steps: int
            number of time steps possible for t
        """
        super().__init__(
            input_dim=1,
            output_dim=embed_dim,
            hidden_dims=hidden_dims,
            num_hidden_layers=num_hidden_layers,
            activation=activation
            )
        self.num_time_steps = num_time_steps
        self.embed_dim = embed_dim
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
        t = t.unsqueeze(-1).float() / (self.num_time_steps - 1)
        return self.forward_layers(t)

class DiffusionNet(BaseMLP):
    """
    Class for Diffusion model
    """
    def __init__(
            self,
            data_dim: int,
            conditional_dim: int = 0,
            embedding: nn.Module | None = None,
            num_hidden_layers: int = 2,
            hidden_dims: HiddenDims = (128,128),
            activation: ActivationFactory = nn.ReLU
            ):
        """
        Parameters
        ----------
        data_dim : int
            dimension of data that is being generated
        conditional dim : int
            dimension of conditioning tensor
        embedding : nn.Module | None
            when not None, the embedding module
        num_hidden_layers: int
            number of hidden layers to use post embedding
        hidden_dims: int
            dimensions for hidden layers post embedding
        activation: ActivationFactory
            activation to be used post embedding
        """
        # if no embedding is specified, use default MLP Time embedding
        if embedding is None:
            embedding = MLPTimeEmbedding()
        # make sure the embedding has an embed_dim attribute needed to know dimension of time embedding
        if  not hasattr(embedding, 'embed_dim'):
            raise AttributeError(
                f"Time embedding {type(embedding).__name__} must have 'embed_dim' attribute"
            )       
        self.embed_dim = embedding.embed_dim
        self.conditional_dim = conditional_dim
        self.data_dim = data_dim
        # create the network with data_dim + embed_dim + conditional_dim inputs
        super().__init__(input_dim=data_dim + self.embed_dim + self.conditional_dim,
                         output_dim=data_dim,
                         hidden_dims=hidden_dims,
                         num_hidden_layers=num_hidden_layers,
                         activation=activation)
        self.embedding = embedding
        

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
        t_embed = self.embedding(t)
        # verify that c is correct when the conditional dimension is > 0 
        if self.conditional_dim > 0:
            if c is None:
                raise ValueError("Condtional dimension is > 0 , but no condtional was passed")
            if c.shape[0] != x_t.shape[0]:
                raise ValueError("Conditional tensor must have same batch size as x_t")
            x = torch.cat([x_t, t_embed, c], dim=1)
        else:
            x = torch.cat([x_t, t_embed], dim=1)
        return self.forward_layers(x)

class DiffusionProcess:
    """
    class to store Diffusion model, train model, and generate a sample
    """
    def __init__(
            self,
            model : nn.Module,
            num_timesteps: int = 1000,
            beta_schedule : str= "linear",
            beta_start: float = 0.0,
            beta_end: float = 0.2,
            data_dim: int | None = None
    ):
        """
        Parameters
        ----------
        model : nn.Module
            model that will learn the diffusion process
        num_timesteps: int
            number of time steps to be used
        beta_schedule: str
            description of schedule ("linear" or "cosine")
        beta_start : float
            start of betas
        beta_end : float
            end of betas
        data_dim: int | None
            dimension of the data if the model doesn't have it
        """
        self.model = model
        # if the model has a data_dim we use that and ignore data_dim passed (if any)
        if hasattr(model, "data_dim"):
            self.data_dim = model.data_dim
        else:
            if data_dim is not None:
                self.data_dim = data_dim
            else:
                raise ValueError(
                    "Either the model must have a data_dim or you must set one yourself"
                )
        self.device = next(model.parameters()).device
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        beta_schedule_lower=beta_schedule.strip().lower()
        if beta_schedule_lower == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        elif beta_schedule_lower == "cosine":
            self.betas = self._cosine_beta_schedule(num_timesteps)
        else:
            raise ValueError(
                f"Unknown beta_schedule: '{beta_schedule}'. "
                f"Expected 'linear' or 'cosine'."
                )
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)
        
        # Move to device
        self.betas = self.betas.to(self.device)
        self.alphas = self.alphas.to(self.device)
        self.alphas_cumprod = self.alphas_cumprod.to(self.device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(self.device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(self.device)
    # TO DO : Document below, add _cosine_beta_schedule
    def train(
            self,
            X : torch.Tensor | DataLoader,
            c : torch.Tensor | None = None,
            epochs: int = 100,
            batch_size: int = 512,
            lr: float = 1e-4
    ):
        if epochs <= 0:
            raise ValueError("Number of epochs must be positive")
        if isinstance(X, torch.Tensor):
            dataloader = make_dataloader(X=X, c=c, batch_size=batch_size, shuffle=True)
        elif isinstance(X, DataLoader):
            if c is not None:
                warnings.warn(
                    "The c parameter is ignored when X is a DataLoader. "
                    "Include the c parameter in X instead"
                )
            dataloader = X
        else:
            raise TypeError("X must be a torch.Tensor or DataLoader")
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        pbar = tqdm(range(1, epochs+1), desc = "Diffusion training")
        for _ in pbar:
            epoch_loss = 0
            for batch in dataloader:
                if len(batch) == 1:
                    X_batch = batch[0].to(self.device)
                    c_batch = None
                else:
                    X_batch, c_batch = batch
                    X_batch = X_batch.to(self.device)
                    c_batch = c_batch.to(self.device)
                t = torch.randint(0, self.num_timesteps, (X_batch.size(0),), device=self.device)
                x_t, noise = q_sample(
                    x0=X_batch,
                    sqrt_alphas_cumprod=self.sqrt_one_minus_alphas_cumprod,
                    sqrt_one_minus_alphas_cumprod=self.sqrt_one_minus_alphas_cumprod,
                    t=t
                )
                noise_pred = self.model(x_t, t, c_batch)
                loss = F.mse_loss(noise_pred, noise)
                opt.zero_grad()
                loss.backward()
                opt.step()
                epoch_loss += loss.item() * X_batch.size(0)
            epoch_loss /= len(dataloader.dataset)
            pbar.set_postfix({"Loss" : f"{epoch_loss:.4f}"})
        pbar.close()

    @torch.no_grad
    def generate_samples(
            self,
            num_samples : int,
            c : torch.Tensor | None = None
    ):
        if c is not None:
            if c.shape[0] != num_samples:
                raise ValueError(
                    "Conditional c must have length equal to the number of samples to be generated"
                    )
        # generate noise
        x_t= torch.randn(num_samples, self.data_dim, device=self.device)
        for t in reversed(range(self.num_timesteps)):
            x_t = p_sample(
                x_t = x_t,
                t=t,
                betas=self.betas,
                alphas=self.alphas,
                alphas_cumprod=self.alphas_cumprod,
                c=c
            )
        return x_t.cpu()

