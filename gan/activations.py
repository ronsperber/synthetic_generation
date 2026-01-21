"""
custom activations to use in output heads for a GAN
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftplusShift(nn.Module):
    """
    Softplus activation shifted to enforce x >= min_value.
    """
    def __init__(self, min_value: float = 0.0, beta: float = 1.0):
        super().__init__()
        self.min_value = min_value
        self.beta = beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softplus(x, beta=self.beta) + self.min_value


class TanhShiftScale(nn.Module):
    """
    Tanh activation scaled and shifted to [min_value, max_value].
    """
    def __init__(self, min_value: float, max_value: float):
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (torch.tanh(x) + 1) / 2 * (self.max_value - self.min_value) + self.min_value


class ClampedIdentity(nn.Module):
    """
    Identity activation with optional clamping.
    """
    def __init__(self, min_value: float | None = None, max_value: float | None = None, eps: float = 1e-6):
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.min_value is not None or self.max_value is not None:
            if self.min_value is None:
                return torch.clamp(x, max = self.max_value + self.eps)
            elif self.max_value is None:
                return torch.clamp(x , min = self.min_value - self.eps)
            else:
                return torch.clamp(x, min=self.min_value  - self.eps, max=self.max_value + self.eps)
        return x

class BoundedSigmoid(nn.Module):
    """
    Sigmoid to go between minimum and maximum values
    """
    def __init__(self, min_value : float, max_value: float):
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x):
        return torch.sigmoid(x) * (self.max_value - self.min_value) + self.min_value
    
class GumbelSoftmax(nn.Module):
    def __init__(self, tau: float = 1.0, hard: bool = False, dim: int = -1):
        super().__init__()
        self.tau = tau
        self.hard = hard
        self.dim = dim

    def forward(self, x):
        return F.gumbel_softmax(x, tau=self.tau, hard=self.hard, dim=self.dim)
