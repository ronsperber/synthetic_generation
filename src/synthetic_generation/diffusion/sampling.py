import torch
from .models import BaseMLP
def q_sample(
        x0: torch.Tensor, 
        sqrt_alphas_cumprod: torch.Tensor,
        sqrt_one_minus_alphas_cumprod: torch.Tensor,
        t: torch.Tensor,
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
    model:BaseMLP,
    x_t: torch.Tensor,
    t: torch.Tensor,
    betas: torch.Tensor,
    alphas: torch.Tensor,
    alphas_cumprod: torch.Tensor):
    """
    Sample x_{t-1} from x_t
    """
    beta_t = betas[t]
    alpha_t = alphas[t]
    alpha_bar_t = alphas_cumprod[t]

    eps_theta = model(x_t, torch.full((x_t.size(0),), t, device=x_t.device))
    mean = (1 / torch.sqrt(alpha_t)) * (
        x_t - beta_t / torch.sqrt(1 - alpha_bar_t) * eps_theta
    )

    if t > 0:
        noise = torch.randn_like(x_t)
        sigma = torch.sqrt(beta_t)
        return mean + sigma * noise
    else:
        return mean