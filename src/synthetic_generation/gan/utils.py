"""
module of helper functions for this project
"""
from typing import Type, Callable
import torch
import torch.nn as nn
from .models import Discriminator, Generator

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


def cov_matrix(data : torch.Tensor, unbiased: bool = False):
    """
    Returns the covariance matrix of a batch of data
    Parameters
    ----------
    data : torch.Tensor
        tensor with dimensions (batch_size, num_features) that we 
        want to get covariance matrix for
    unbiased : bool
        whether we want unbiased estimator (divide by batch_size - 1)
    Returns
    -------
    torch.Tensor
        (num_features, num_features) matrix with the (i,j)th entry E((X_i-E(X_i))(X_j - E(X_j)))
    """
    if data.dim() != 2:
        raise ValueError("Data should be two dimensional (batch_size, num_features)")
    centered_data = data - data.mean(0, keepdim=True)
    batch_size = data.shape[0]
    if unbiased and batch_size < 2:
        raise ValueError("Can not use unbiased covariance with batch size < 2")
    divisor = batch_size - 1 if unbiased else batch_size
    return centered_data.T @ centered_data / divisor # has shape (num_features, num_features)

def cov_penalty(f_fake: torch.Tensor, f_real: torch.Tensor, unbiased: bool = False):
    """
    Compute the covariance/variance penalty term for feature matching
    Parameters
    ----------
    f_fake : torch.Tensor
        features from the fake data
    f_real : torch.Tensor
        features from the real data
    unbiased : bool
        whether or not to use unbiased statistics
    Returns
    torch.Tensor
        mean of sum of all variance differences and all covariance differences
    """
    cov_real = cov_matrix(f_real, unbiased)
    cov_fake = cov_matrix(f_fake, unbiased)
    # compute the square of the differences
    cov_diff = cov_real - cov_fake
    cov_diff_2 = cov_diff ** 2
    # get the upper triangular version so each covariance appears once
    cov_diff_2_upper = torch.triu(cov_diff_2)
    # get the correct number of entries
    num_features = cov_diff_2_upper.shape[0]
    divisor = num_features * (num_features + 1) // 2
    return cov_diff_2_upper.sum() / divisor

def feature_entropy(f_fake:torch.Tensor, unbiased=False, eps=1e-8):
    """
    computes the entropy of a sample in the feature space
    Parameters
    ----------
    f_fake : torch.Tensor
        sample in the feature space
    unbiased : bool
        whether to use biased or unbiased calculation for the covariance matrix
    eps : float
        size of term to be added for numerical stability
    Returns
    -------
    torch.Tensor
        -log (det Covariance matrix) (negative to maximize entropy)
    """
    feature_dim = f_fake.shape[1]
    cov = cov_matrix(f_fake, unbiased)
    s, logdet = torch.slogdet(cov + eps * torch.eye(feature_dim, device=f_fake.device))
    penalty = torch.tensor(1e6, device=f_fake.device)
    return torch.where(s > 0, -logdet / feature_dim, penalty)