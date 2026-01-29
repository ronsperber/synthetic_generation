"""
module with schedules for diffusion
"""
import torch

def cosine_beta_schedule(
            timesteps: int,
            s: float=0.008
            )->torch.Tensor:
        """
        Cosine beta schedule as proposed in Improved DDPM.
    
        Creates a smoother noise schedule than linear, which can help
        preserve variance and improve sample quality.
    
        Parameters
        ----------
        timesteps : int
            Number of diffusion timesteps
        s : float, optional
            Small offset to prevent beta from being too small near t=0.
            Controls the steepness of the schedule. Default: 0.008 (from paper)
    
        Returns
        -------
        betas : torch.Tensor
            Beta values for each timestep, clipped to [0.0001, 0.9999]
    
        References
        ----------
        Nichol & Dhariwal (2021): Improved Denoising Diffusion Probabilistic Models
        https://arxiv.org/abs/2102.09672
        """
        if timesteps <=0:
                raise ValueError("Number of timesteps must be positive")
        if (s < 0.0) or (s >= 1.0):
                raise ValueError("s should be in the interval [0,1) ")
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

def linear_beta_schedule(
                beta_start: float = 1e-4,
                beta_end: float = 0.02,
                num_timesteps: int = 1000
) -> torch.Tensor:
        """
        compute linear schedule betas
        Parameters
        ---------
        beta_start : float
            start value for the betas
        beta_end: float
            end value for the betas
        num_timesteps: int
            number of timesteps for the betas
        Returns
        betas : torch.Tensor
            betas for this schedule
        """
        if num_timesteps <= 0:
                raise ValueError("Number of timesteps must be a positive integer")
        if beta_end <= beta_start:
                raise ValueError("beta_end must be greater than beta_start")
        if beta_start <= 0.0:
                raise ValueError("beta_start should be > 0")
        return torch.linspace(beta_start, beta_end, num_timesteps)