import torch

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