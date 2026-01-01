"""
module of helper functions for this project
"""

import torch
from torch.utils.data import TensorDataset, DataLoader
from .models import Discriminator


def make_dataloader(
    X: torch.Tensor,
    c: torch.Tensor | None = None,
    batch_size: int = 64,
    shuffle: bool = True,
):
    """
    function to turn data into a DataLoader
    Parameters
    X : torch.Tensor
        input data
    c : Optional torch.Tensor
        For a conditional GAN/WGAN, the condtional tensor
    batch_size: int
        batch size for the DataLoader
    shuffle : bool
        shuffle parameter for DataLoader
    Returns
    -------
    DataLoader
        DataLoader from X and optionally c
    """
    if c is not None:
        dataset = TensorDataset(X, c)
    else:
        dataset = TensorDataset(X)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def gradient_penalty(
    D: Discriminator,
    x_real: torch.Tensor,
    x_fake: torch.Tensor,
    c: torch.Tensor | None = None,
    lambda_gp: float = 10.0,
):
    """
    function to compute gradient penalty for WGAN-GP
    Parameters
    ----------
    D: Discriminator
        the critic for the model
    x_real : torch.Tensor
        a tensor of real data
    x_fake : torch.Tensor
        a tensor of data from the Generator
    c: Optional torch.Tensor
        for a condtional WGAN-GP, the conditional tensor
    lambda_gp : float
        multiplier used to compute the penalty
    """
    batch_size = x_real.size(0)
    # interpolate a random point between x_real and x_fake
    eps = torch.rand(batch_size, 1, device=x_real.device)
    eps = eps.expand_as(x_real)
    x_hat = eps * x_real + (1 - eps) * x_fake
    x_hat.requires_grad_(True)
    # get the critic score for x_hat
    d_hat = D(x_hat, c)
    # get partial derivative of d_hat with respect to x_hat
    grads = torch.autograd.grad(
        outputs=d_hat,
        inputs=x_hat,
        grad_outputs=torch.ones_like(d_hat),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    # compute how badly the critic violates the Lipshcitz constraint along x_hat
    grads = grads.view(batch_size, -1)
    gp = ((grads.norm(2, dim=1) - 1) ** 2).mean()
    # multiply by lambda_gp
    return lambda_gp * gp
