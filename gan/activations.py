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
        if self.training:
            min_val = self.min_value - self.eps if self.min_value is not None else None
            max_val = self.max_value + self.eps if self.max_value is not None else None
        else:
            min_val = self.min_value
            max_val = self.max_value
        if min_val is not None or max_val is not None:
            return torch.clamp(x, min=min_val, max=max_val)
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
    """
    Gumbel Softmax defaulting to hard on inference
    """
    def __init__(self, tau: float = 1.0, hard_inference: bool = True, dim: int = -1):
        super().__init__()
        self.tau = tau
        self.hard_inference = hard_inference
        self.dim = dim

    def forward(self, x):
        if self.training:
            hard = False
        else:
            hard = self.hard_inference
        return F.gumbel_softmax(x, tau=self.tau, hard=hard, dim=self.dim)
    
class RoundedClamp(nn.Module):
    """
    class to round to nearest whole integer and then clamp if desired at extremes
    this is only to be used for decode, not during training because of gradient issues
    """
    def __init__(self, min_val: float | None = None, max_val: float | None = None):
        """
        Parameters
        min_val float | None
            when not None, lower bound for clamp
        max_val float | None
            when not None, upper bound for clamp
        """
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val
    def forward(self, x):
        rounded = torch.round(x)
        if self.min_val is not None or self.max_val is not None:
            return torch.clamp(rounded, min=self.min_val, max=self.max_val)
        return rounded