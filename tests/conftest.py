import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from synthetic_generation.gan.models import Generator, Discriminator

@pytest.fixture
def sample_data():
    torch.manual_seed(42)
    return torch.randn(100, 10)

@pytest.fixture
def sample_conditional_data():
    torch.manual_seed(42)
    X = torch.randn(100, 10)
    class_indices = torch.randint(0, 3, (100,))
    c = torch.nn.functional.one_hot(class_indices, num_classes=3).float()
    return X, c

@pytest.fixture
def models():
    torch.manual_seed(42)
    G = Generator(noise_dim=20, num_hidden_layers=2, out_dim=10, hidden_dims=(32, 32), conditional_dim=0)
    D = Discriminator(feature_dim=10, num_hidden_layers=2, hidden_dims=(32, 32), conditional_dim=0)
    return G, D

@pytest.fixture
def conditional_models():
    torch.manual_seed(42)
    G = Generator(noise_dim=20, num_hidden_layers=2, out_dim=10, hidden_dims=(32, 32),
                  use_conditional=True, conditional_dim=3)
    D = Discriminator(feature_dim=10, num_hidden_layers=2, hidden_dims=(32, 32),
                      use_conditional=True, conditional_dim=3)
    return G, D

@pytest.fixture
def dataloader(sample_data):
    return DataLoader(TensorDataset(sample_data), batch_size=16)
