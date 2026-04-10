import warnings
from typing import Callable
from tqdm.auto import tqdm
import numpy as np
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from .sampling import p_sample, q_sample, ddim_sample, p_sample_cfg, ddim_sample_cfg
from .schedules import linear_beta_schedule
from .model_saving import save_diffusion_checkpoint, load_diffusion_checkpoint
from synthetic_generation.data_utils import make_dataloader

def make_null_conditional(c_batch: torch.Tensor, null_token: torch.Tensor | None = None):
    """
    Create a 'null' conditional embedding for classifier-free guidance.
    
    Parameters
    ----------
    c_batch : torch.Tensor
        The batch of conditional tensors (batch_size, conditional_dim)
    null_token : torch.Tensor | None
        If None, use zeros_like(c_batch)
        If a tensor, expand it along batch dimension
    """
    batch_size = c_batch.shape[0]
    if null_token is None:
        return torch.zeros_like(c_batch)
    else:
        # expand to batch size
        return null_token.unsqueeze(0).expand(batch_size, -1)

class DiffusionProcess:
    """
    Class to hold Diffusion Model along with methods to
    train, generate sample data, save model, and load a model from a checkpoint
    
    DiffusionProcess expects `model` to have the following attributes:

    - data_dim : int | tuple [int,...]
    - conditional_dim : int
    - time_embedding : nn.Module or None, optionally with `init_args` dict
    - conditional_embedding : nn.Module or None, optionally with `init_args` dict 
    -  num_hidden_layers : int
    - hidden_dims : tuple[int, ...]
    - activation : nn.Module

    Custom models must implement these attributes to be compatible with
    `save_diffusion_checkpoint` and `load_diffusion_checkpoint`.
    """

    def __init__(
            self,
            model : nn.Module,
            betas : torch.Tensor | None = None,
            num_timesteps: int = 1000,
            data_dim: int | tuple[int,...] | None = None
    ):
        """
        Parameters
        ----------
        model : nn.Module
            model that will learn the diffusion process
        num_timesteps: int
            number of time steps to be used
        betas: torch.Tensor (optional)
            tensor of beta values for scheduler
        data_dim: tuple[int,...] | None
            dimensions of the data if the model doesn't have it
        """
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.train_config = None # set in case we try to save the model before training
        self.null_token = None # set for CFG 
        # if the model has a data_dim we use that and ignore data_dim passed (if any)
        if hasattr(model, "data_dim"):
            self.data_dim = (model.data_dim,) if isinstance(model.data_dim,int) else model.data_dim
        else:
            if data_dim is not None:
                self.data_dim = (data_dim,) if isinstance(data_dim,int) else data_dim
            else:
                raise AttributeError(
                    f"Model {type(model).__name__} must have 'data_dim' attribute "
                    " or data_dim must be specified."
            )
        self.num_timesteps = num_timesteps
        if betas is None:
            betas = linear_beta_schedule(num_timesteps=num_timesteps)
        self.betas = betas
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=self.device), self.alphas_cumprod[:-1]])
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)
        
        # Move to device
        self.betas = self.betas.to(self.device)
        self.alphas = self.alphas.to(self.device)
        self.alphas_cumprod = self.alphas_cumprod.to(self.device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(self.device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(self.device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(self.device)

    def train(
            self,
            X : torch.Tensor | DataLoader,
            c : torch.Tensor | None = None,
            epochs: int = 100,
            batch_size: int = 512,
            lr: float = 1e-4,
            p_null: float = 0.0,
            null_token : torch.Tensor | None = None,
            return_history: bool = False,
            epoch_callback: Callable[[dict], bool] | None = None,
            callback_every: int = 1,


    ):
        """
        method to train self.model on dataset X
        Parameters
        ----------
        X : torch.Tensor | DataLoader
            dataset to learn to synthesize
        c : torch.Tensor | None
            optional conditional Tensor
        epochs: int
            number of epochs to train
        batch_size : int
            batch size to use for DataLoader. 
            if DataLoader is passed, this is ignored
        lr: float
            learning rate to use while training
        p_null : float
            probability of using null token for classifier free guidance
        null_token: torch.Tensor | None
            optional null conditional tensor to use for classifier free guidance
        return_history: bool
            whether or not to return the loss history
        epoch_callback : Callable[[dict], bool] | None
            optional callback called every callback_every epochs.
            receives a state dict with keys: epoch, loss, model, history
            return True to stop training early, False to continue.
        callback_every : int
            how often to call epoch_callback (default 1 = every epoch)
        Returns
        -------
            (Optional) epoch_losses : list
                list of tuples (epoch, epoch_loss)
        """
        self.train_config = {
            "epochs": epochs,
            "batch_size": batch_size,
            "lr" : lr,
            "p_null": p_null,
            "null_conditional": null_token
        }
        self.null_token = null_token
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
        epoch_losses = []
        for epoch in pbar:
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
                    sqrt_alphas_cumprod=self.sqrt_alphas_cumprod,
                    sqrt_one_minus_alphas_cumprod=self.sqrt_one_minus_alphas_cumprod,
                    t=t
                )
                # when a conditional exists we check for CFG
                if p_null > 0 and c_batch is not None: 
                    c_uncond = make_null_conditional(c_batch=c_batch, null_token=self.null_token)
                    mask = (torch.rand(c_batch.shape[0], device=c_batch.device) < p_null).float()
                    c_batch = mask[:, None] * c_uncond + (1 - mask[:, None]) * c_batch
                noise_pred = self.model(x_t, t, c_batch)
                loss = F.mse_loss(noise_pred, noise)
                opt.zero_grad()
                loss.backward()
                opt.step()
                epoch_loss += loss.item() * X_batch.size(0)
            epoch_loss /= len(dataloader.dataset)
            epoch_losses.append((epoch,epoch_loss))
            if epoch_callback is not None and epoch % callback_every == 0:
                callback_state = {
                    "epoch": epoch,
                    "model": self.model,
                    "loss": epoch_loss,
                    "history": epoch_losses,
                }
                should_stop = epoch_callback(callback_state)
                if should_stop:
                    break
            pbar.set_postfix({"Loss" : f"{epoch_loss:.4f}"})
        pbar.close()
        if return_history:
            return epoch_losses

    @torch.no_grad()
    def generate_samples(
            self,
            num_samples : int,
            c : torch.Tensor | None = None,
            guidance_scale : float = 1.0
    ) -> torch.Tensor:
        """
        method to use on trained model to generate new samples
        Parameters
        ----------
        num_samples : int
            number of samples to generate
        c: torch.Tensor | None
            optional conditional tensor
        guidance_scale : float
            for CFG scale to use in generating
        Returns
        -------
        x_t : torch.Tensor
            generated sample after going backwards from t = num_timesteps -> 0
        """
        if c is not None:
            if c.shape[0] != num_samples:
                raise ValueError(
                    "Conditional c must have length equal to the number of samples to be generated"
                    )
        # generate noise
        x_t= torch.randn(num_samples, *self.data_dim, device=self.device)
        if c is not None and guidance_scale != 1.0:
            c_null=make_null_conditional(c_batch=c, null_token=self.null_token)
        # go backwards one timestep at a time to denoise
        for t in tqdm(
            reversed(range(self.num_timesteps)),
            desc="Sampling",
            total=self.num_timesteps
            ):
            if c is None or guidance_scale == 1.0:
                x_t = p_sample(
                    model=self.model,
                    x_t = x_t,
                    t=t,
                    betas=self.betas,
                    alphas=self.alphas,
                    alphas_cumprod=self.alphas_cumprod,
                    alphas_cumprod_prev=self.alphas_cumprod_prev,
                    c=c
                )
            else:
                x_t = p_sample_cfg(
                    model=self.model,
                    x_t = x_t,
                    t=t,
                    betas=self.betas,
                    alphas=self.alphas,
                    alphas_cumprod=self.alphas_cumprod,
                    alphas_cumprod_prev=self.alphas_cumprod_prev,
                    c=c,
                    c_null=c_null,
                    guidance_scale=guidance_scale
                )
        
        return x_t.cpu()
 

    @torch.no_grad()
    def generate_samples_ddim(
        self,
        num_samples: int,
        num_inference_steps: int = 50,
        eta: float = 0.0,
        c: torch.Tensor | None = None,
        guidance_scale: float = 1.0
    ) -> torch.Tensor:
        """
        Generate samples using DDIM sampling (faster than DDPM)
    
        DDIM allows skipping timesteps during sampling, enabling much faster
        generation with minimal quality loss. With num_inference_steps=50,
        this is ~20x faster than standard DDPM sampling.
    
        Parameters
        ----------
        num_samples : int
            Number of samples to generate
        num_inference_steps : int, default=50
            Number of denoising steps. Fewer steps = faster sampling.
            Typical values: 20-100 (vs 1000 for DDPM)
        eta : float, default=0.0
            Stochasticity parameter:
            - eta=0.0: Deterministic DDIM (recommended)
            - eta=1.0: Stochastic (similar to DDPM)
        c : torch.Tensor, optional
            Conditioning information
        guidance_scale : float
            guidance_scale to be used for CFG
    
        Returns
        -------
        samples : torch.Tensor
            Generated samples (on CPU)
        """
        if c is not None:
            if c.shape[0] != num_samples:
                raise ValueError(
                    "Conditional c must have length equal to the number of samples"
                )
    
        # Create evenly-spaced timestep schedule
        # E.g., if num_timesteps=1000 and num_inference_steps=50,
        # we get [0, 20, 40, 60, ..., 980, 1000]
        timesteps = np.linspace(0, self.num_timesteps - 1, num_inference_steps, dtype=int)
        if c is not None and guidance_scale != 1.0:
            c_null=make_null_conditional(c_batch=c, null_token=self.null_token)
        # Start from pure noise
        x = torch.randn(num_samples, *self.data_dim, device=self.device)
    
        # Iteratively denoise using DDIM
        for i in tqdm(reversed(range(len(timesteps))),
            desc=f"DDIM Sampling ({num_inference_steps} steps)",
            total=len(timesteps)
        ):
            t = timesteps[i]
            t_prev = timesteps[i - 1] if i > 0 else -1
            if c is None or guidance_scale == 1.0:
                x = ddim_sample(
                    model=self.model,
                    x_t=x,
                    t=t,
                    t_prev=t_prev,
                    alphas_cumprod=self.alphas_cumprod,
                    eta=eta,
                    c=c
                )
            else:
                x = ddim_sample_cfg(
                    model=self.model,
                    x_t=x,
                    t=t,
                    t_prev=t_prev,
                    alphas_cumprod=self.alphas_cumprod,
                    eta=eta,
                    c=c,
                    c_null=c_null,
                    guidance_scale=guidance_scale
                )
    
        return x.cpu()
    
    def save(
            self,
            path: str,
    ):
        """
        method to save process
        Parameters
        ----------
            path : str
                location to save the process
        """
        save_diffusion_checkpoint(process=self, path=path, train_config=self.train_config)

    def train_save(
            self,
            path: str,
            X: torch.Tensor | DataLoader,
            c: torch.Tensor | None = None,
            **kwargs
    ):
        """
        method to train the model on data and save to a checkpoint
        Parameters
        path : str
            path to save checkpoint
        X : torch.Tensor | DataLoader
            data to train on
        c : torch.Tensor | None
            conditional tensor when not None
        **kwargs:
            additional arguments to use in train
        """

        history = self.train(X=X, c=c, **kwargs) 
        self.save(path=path)
        return history
    
    @classmethod
    def load_process(
        cls,
        path: str,
        model_classes: dict | None = None,
        activation_dict: dict | None = None):
        """
        method to create a new process from a checkpoint
        Parameters
        ----------
        path: str
            path where the checkpoint was saved to
        model_classes : dict | None
            optional dict of custom model classes with "name": class
        activation_dict : dict | None
            optional dict of activations with "name": activation
        Returns
        -------
        DiffusionProcess
            the process built from the checkpoint stored
        """
        process_args, _, _  = load_diffusion_checkpoint(
            path=path,
            model_classes=model_classes,
            activation_dict=activation_dict
            )
        return cls(**process_args)