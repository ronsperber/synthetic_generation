import torch
import torch.nn as nn
def q_sample(
        x0: torch.Tensor, 
        t: torch.Tensor,
        sqrt_alphas_cumprod: torch.Tensor,
        sqrt_one_minus_alphas_cumprod: torch.Tensor,
        noise=None
        ):
    """
    Sample x_t from x_0 at timestep t
    """
    # verify correct number of dimensions
    if noise is None:
        noise = torch.randn_like(x0)
    return (
        sqrt_alphas_cumprod[t][:, None] * x0 +
        sqrt_one_minus_alphas_cumprod[t][:, None] * noise
    ), noise

@torch.no_grad()
def p_sample(
    model:nn.Module,
    x_t: torch.Tensor,
    t: int,
    betas: torch.Tensor,
    alphas: torch.Tensor,
    alphas_cumprod: torch.Tensor,
    alphas_cumprod_prev: torch.Tensor,
    c: torch.Tensor | None = None
):
    """
    Sample x_{t-1} from x_t
    """
    beta_t = betas[t]
    alpha_t = alphas[t]
    alpha_bar_t = alphas_cumprod[t]
    alpha_bar_prev = alphas_cumprod_prev[t]

    eps_theta = model(x_t, torch.full((x_t.size(0),), t, device=x_t.device), c)
    mean = (1 / torch.sqrt(alpha_t)) * (
        x_t - beta_t / torch.sqrt(1 - alpha_bar_t) * eps_theta
    )

    if t > 0:
        noise = torch.randn_like(x_t)
        posterior_variance = beta_t * (1 - alpha_bar_prev) / (1 - alpha_bar_t)
        sigma = torch.sqrt(posterior_variance)
        return mean + sigma * noise
    else:
        return mean
    
@torch.no_grad()
def ddim_sample(
    model: nn.Module,
    x_t: torch.Tensor,
    t: int,
    t_prev: int,
    alphas_cumprod: torch.Tensor,
    eta: float = 0.0,
    c: torch.Tensor | None = None
) -> torch.Tensor:
    """
    Single DDIM denoising step
    
    Parameters
    ----------
    model : nn.Module
        Denoising model
    x_t : torch.Tensor
        Noised data at timestep t
    t : int
        Current timestep
    t_prev : int
        Target previous timestep (can skip steps, e.g., t=100, t_prev=50)
    alphas_cumprod : torch.Tensor
        Cumulative product of alphas for all timesteps
    eta : float
        Stochasticity parameter:
        - eta=0: Deterministic DDIM
        - eta=1: Stochastic (similar to DDPM)
    c : torch.Tensor, optional
        Conditioning information
    
    Returns
    -------
    x_prev : torch.Tensor
        Denoised sample at timestep t_prev
    """
    # Get alpha values for current and previous timesteps
    alpha_bar_t = alphas_cumprod[t]
    alpha_bar_prev = alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=alphas_cumprod.device)
    
    # Predict noise at current timestep
    eps_theta = model(x_t, torch.full((x_t.size(0),), t, device=x_t.device), c)
    
    # Step 1: Predict x0 (the original clean data)
    pred_x0 = (x_t - torch.sqrt(1 - alpha_bar_t) * eps_theta) / torch.sqrt(alpha_bar_t)
    
    # Step 2: Compute "direction" component (deterministic part)
    # This is the noise component when jumping to t_prev
    sigma_t = eta * torch.sqrt(
        (1 - alpha_bar_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_prev)
    )
    dir_xt = torch.sqrt(1 - alpha_bar_prev - sigma_t**2) * eps_theta
    
    # Step 3: Compute x_prev = sqrt(α_bar_prev) * x0 + "direction"
    x_prev = torch.sqrt(alpha_bar_prev) * pred_x0 + dir_xt
    
    # Step 4: Add stochastic noise if eta > 0 and not final step
    if eta > 0 and t_prev >= 0:
        noise = torch.randn_like(x_t)
        x_prev = x_prev + sigma_t * noise
    
    return x_prev