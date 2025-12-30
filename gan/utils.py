import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from .models import Generator, Discriminator

def make_dataloader(X, c=None, batch_size=64, shuffle=True):
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
        lambda_gp: float=10.0
        ):
    batch_size = x_real.size(0)
    eps = torch.rand(batch_size, 1, device=x_real.device)
    eps = eps.expand_as(x_real)
    x_hat = eps * x_real + (1 - eps) * x_fake
    x_hat.requires_grad_(True)
    d_hat = D(x_hat, c)
    grads = torch.autograd.grad(
        outputs=d_hat,
        inputs=x_hat,
        grad_outputs=torch.ones_like(d_hat),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    grads = grads.view(batch_size, -1)
    gp = ((grads.norm(2, dim=1) - 1) ** 2).mean()
    return lambda_gp * gp
