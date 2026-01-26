"""
data processing utilities
"""
import torch
from torch.utils.data import TensorDataset, DataLoader

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