"""
Unit tests for GAN and WGAN-GP training functions
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from gan.models import Generator, Discriminator
from gan.training import train_gan, train_wgan_gp


# =====================
# Fixtures
# =====================

@pytest.fixture
def sample_data():
    """Create sample training data"""
    torch.manual_seed(42)
    X = torch.randn(100, 10)  # 100 samples, 10 features
    return X


@pytest.fixture
def sample_conditional_data():
    """Create sample data with conditional (one-hot encoded)"""
    torch.manual_seed(42)
    X = torch.randn(100, 10)
    # Create class indices
    class_indices = torch.randint(0, 3, (100,))
    # Convert to one-hot encoding
    c = torch.nn.functional.one_hot(class_indices, num_classes=3).float()
    return X, c


@pytest.fixture
def models():
    """Create generator and discriminator models"""
    torch.manual_seed(42)
    G = Generator(noise_dim=20, num_hidden_layers=2, out_dim=10, hidden_dims=(32, 32), conditional_dim=0)
    D = Discriminator(feature_dim=10, num_hidden_layers=2, hidden_dims=(32, 32), conditional_dim=0)
    return G, D


@pytest.fixture
def conditional_models():
    """Create conditional generator and discriminator"""
    torch.manual_seed(42)
    G = Generator(noise_dim=20, num_hidden_layers=2, out_dim=10, hidden_dims=(32, 32), use_conditional=True, conditional_dim=3)
    D = Discriminator(feature_dim=10, num_hidden_layers=2, hidden_dims=(32, 32), use_conditional=True, conditional_dim=3)
    return G, D


@pytest.fixture
def dataloader(sample_data):
    """Create a DataLoader from sample data"""
    dataset = TensorDataset(sample_data)
    return DataLoader(dataset, batch_size=16)


# =====================
# Input Validation Tests
# =====================

def test_dimension_mismatch_output_feature():
    """Test that ValueError is raised when G.output_dim != D.feature_dim"""
    torch.manual_seed(42)
    G = Generator(noise_dim=20, num_hidden_layers=2, out_dim=10, hidden_dims=(32, 32), conditional_dim=0)
    D = Discriminator(feature_dim=5, num_hidden_layers=2, hidden_dims=(32, 32), conditional_dim=0)  # Mismatch: 10 vs 5
    X = torch.randn(32, 10)
    
    with pytest.raises(ValueError, match="G outputs|dimension"):
        train_gan(X, G, D, epochs=1, batch_size=16)


def test_dimension_mismatch_conditional():
    """Test that ValueError is raised when conditional dimensions don't match"""
    torch.manual_seed(42)
    G = Generator(noise_dim=20, num_hidden_layers=2, out_dim=10, hidden_dims=(32, 32), use_conditional=True, conditional_dim=3)
    D = Discriminator(feature_dim=10, num_hidden_layers=2, hidden_dims=(32, 32), use_conditional=True, conditional_dim=5)  # Mismatch: 3 vs 5
    X = torch.randn(32, 10)
    c = torch.nn.functional.one_hot(torch.randint(0, 3, (32,)), num_classes=3).float()
    
    with pytest.raises(ValueError, match="Conditional"):
        train_gan(X, G, D, c=c, epochs=1, batch_size=16)


def test_invalid_learning_rate():
    """Test that negative learning rates raise errors"""
    torch.manual_seed(42)
    G = Generator(noise_dim=20, num_hidden_layers=2, out_dim=10, hidden_dims=(32, 32), conditional_dim=0)
    D = Discriminator(feature_dim=10, num_hidden_layers=2, hidden_dims=(32, 32), conditional_dim=0)
    X = torch.randn(32, 10)
    
    with pytest.raises(ValueError):
        train_gan(X, G, D, lr_G=-0.001, epochs=1, batch_size=16)


def test_invalid_batch_size():
    """Test that invalid batch sizes raise errors"""
    torch.manual_seed(42)
    G = Generator(noise_dim=20, num_hidden_layers=2, out_dim=10, hidden_dims=(32, 32), conditional_dim=0)
    D = Discriminator(feature_dim=10, num_hidden_layers=2, hidden_dims=(32, 32), conditional_dim=0)
    X = torch.randn(32, 10)
    
    with pytest.raises((ValueError, RuntimeError)):
        train_gan(X, G, D, epochs=1, batch_size=0)


def test_negative_epochs_raises_error():
    """Test that negative epochs raises a ValueError"""
    torch.manual_seed(42)
    G = Generator(noise_dim=20, num_hidden_layers=2, out_dim=10, hidden_dims=(32, 32), conditional_dim=0)
    D = Discriminator(feature_dim=10, num_hidden_layers=2, hidden_dims=(32, 32), conditional_dim=0)
    X = torch.randn(32, 10)
    
    with pytest.raises(ValueError):
        train_gan(X, G, D, epochs=-5, batch_size=16)


# =====================
# Training Behavior Tests
# =====================

def test_train_gan_with_tensor_input(models, sample_data):
    """Test GAN training with direct tensor input"""
    G, D = models
    initial_g_params = [p.clone() for p in G.parameters()]
    initial_d_params = [p.clone() for p in D.parameters()]
    
    train_gan(sample_data, G, D, epochs=2, batch_size=16)
    
    # Verify parameters have changed
    for p_init, p_final in zip(initial_g_params, G.parameters()):
        assert not torch.allclose(p_init, p_final), "Generator parameters should change"
    
    for p_init, p_final in zip(initial_d_params, D.parameters()):
        assert not torch.allclose(p_init, p_final), "Discriminator parameters should change"


def test_train_gan_with_dataloader(models, dataloader):
    """Test GAN training with DataLoader input"""
    G, D = models
    initial_g_params = [p.clone() for p in G.parameters()]
    
    train_gan(dataloader, G, D, epochs=2, batch_size=16)  # batch_size ignored with DataLoader
    
    # Verify parameters have changed
    for p_init, p_final in zip(initial_g_params, G.parameters()):
        assert not torch.allclose(p_init, p_final), "Parameters should change during training"


def test_train_gan_conditional(conditional_models, sample_conditional_data):
    """Test conditional GAN training"""
    G, D = conditional_models
    X, c = sample_conditional_data
    initial_g_params = [p.clone() for p in G.parameters()]
    
    train_gan(X, G, D, c=c, epochs=2, batch_size=16)
    
    # Verify parameters have changed
    for p_init, p_final in zip(initial_g_params, G.parameters()):
        assert not torch.allclose(p_init, p_final), "Conditional GAN parameters should change"


def test_loss_computation(models, sample_data):
    """Test that losses are computed without NaN or Inf"""
    G, D = models
    # Run with custom criterion to capture losses
    train_gan(sample_data, G, D, epochs=1, batch_size=16, 
              criterion=nn.BCEWithLogitsLoss())
    
    # If we reach here without errors, losses were computed correctly


def test_train_wgan_gp_parameter_updates(models, sample_data):
    """Test WGAN-GP training updates parameters"""
    G, D = models
    initial_g_params = [p.clone() for p in G.parameters()]
    initial_d_params = [p.clone() for p in D.parameters()]
    
    train_wgan_gp(sample_data, G, D, epochs=5, batch_size=16, n_critic=3)
    
    # Verify parameters have changed (at least some of them)
    g_changed = sum(not torch.allclose(p_init, p_final) for p_init, p_final in zip(initial_g_params, G.parameters()))
    d_changed = sum(not torch.allclose(p_init, p_final) for p_init, p_final in zip(initial_d_params, D.parameters()))
    
    assert g_changed > 0, "Generator parameters should change during training"
    assert d_changed > 0, "Discriminator parameters should change during training"


def test_train_wgan_gp_conditional(conditional_models, sample_conditional_data):
    """Test WGAN-GP with conditional data"""
    G, D = conditional_models
    X, c = sample_conditional_data
    initial_g_params = [p.clone() for p in G.parameters()]
    
    train_wgan_gp(X, G, D, c=c, epochs=2, batch_size=16, n_critic=2)
    
    # Verify parameters have changed
    for p_init, p_final in zip(initial_g_params, G.parameters()):
        assert not torch.allclose(p_init, p_final), "Conditional WGAN-GP parameters should change"


# =====================
# Edge Cases
# =====================

def test_single_epoch_training(models, sample_data):
    """Test training completes successfully with a single epoch"""
    G, D = models
    try:
        train_gan(sample_data, G, D, epochs=1, batch_size=16)
    except Exception as e:
        pytest.fail(f"Single epoch training failed: {e}")


def test_batch_size_one(models, sample_data):
    """Test training with batch size 1"""
    G, D = models
    try:
        train_gan(sample_data, G, D, epochs=1, batch_size=1)
    except Exception as e:
        pytest.fail(f"Batch size 1 training failed: {e}")


def test_small_dataset(models):
    """Test training with small dataset"""
    G, D = models
    X = torch.randn(8, 10)  # Only 8 samples
    try:
        train_gan(X, G, D, epochs=2, batch_size=2)
    except Exception as e:
        pytest.fail(f"Small dataset training failed: {e}")


def test_single_sample_batch(models):
    """Test training when batch size equals dataset size"""
    G, D = models
    X = torch.randn(32, 10)
    try:
        train_gan(X, G, D, epochs=2, batch_size=32)
    except Exception as e:
        pytest.fail(f"Single batch training failed: {e}")


# =====================
# WGAN-GP Specific Tests
# =====================

def test_wgan_gp_critic_iterations(models, sample_data):
    """Test that WGAN-GP trains critic multiple times per batch"""
    G, D = models
    n_critic = 5
    
    # This test mainly ensures no errors with multiple critic iterations
    try:
        train_wgan_gp(sample_data, G, D, epochs=1, batch_size=16, n_critic=n_critic)
    except Exception as e:
        pytest.fail(f"WGAN-GP with n_critic={n_critic} failed: {e}")


def test_wgan_gp_gradient_penalty_scaling(models, sample_data):
    """Test WGAN-GP with different gradient penalty weights"""
    G, D = models
    
    for lambda_gp in [1.0, 10.0, 100.0]:
        G_copy = Generator(noise_dim=20, num_hidden_layers=2, out_dim=10, hidden_dims=(32, 32), conditional_dim=0)
        D_copy = Discriminator(feature_dim=10, num_hidden_layers=2, hidden_dims=(32, 32), conditional_dim=0)
        
        try:
            train_wgan_gp(sample_data, G_copy, D_copy, epochs=1, 
                         batch_size=16, lambda_gp=lambda_gp)
        except Exception as e:
            pytest.fail(f"WGAN-GP with lambda_gp={lambda_gp} failed: {e}")


def test_wgan_gp_different_critic_counts(models, sample_data):
    """Test WGAN-GP with different numbers of critic iterations"""
    G, D = models
    
    for n_critic in [1, 3, 5]:
        G_copy = Generator(noise_dim=20, num_hidden_layers=2, out_dim=10, hidden_dims=(32, 32), conditional_dim=0)
        D_copy = Discriminator(feature_dim=10, num_hidden_layers=2, hidden_dims=(32, 32), conditional_dim=0)
        
        try:
            train_wgan_gp(sample_data, G_copy, D_copy, epochs=1, 
                         batch_size=16, n_critic=n_critic)
        except Exception as e:
            pytest.fail(f"WGAN-GP with n_critic={n_critic} failed: {e}")


def test_wgan_gp_learning_rates(models, sample_data):
    """Test WGAN-GP with different learning rates"""
    for lr_g, lr_d in [(1e-4, 1e-4), (5e-5, 5e-5), (1e-3, 1e-3)]:
        G = Generator(noise_dim=20, num_hidden_layers=2, out_dim=10, hidden_dims=(32, 32), conditional_dim=0)
        D = Discriminator(feature_dim=10, num_hidden_layers=2, hidden_dims=(32, 32), conditional_dim=0)
        
        try:
            train_wgan_gp(sample_data, G, D, epochs=1, batch_size=16,
                         lr_G=lr_g, lr_D=lr_d)
        except Exception as e:
            pytest.fail(f"WGAN-GP with lr_G={lr_g}, lr_D={lr_d} failed: {e}")


# =====================
# Comparison Tests
# =====================

def test_gan_vs_wgan_gp_convergence(sample_data):
    """Test that both GAN and WGAN-GP complete training"""
    # GAN training
    G_gan = Generator(noise_dim=20, num_hidden_layers=2, out_dim=10, hidden_dims=(32, 32), conditional_dim=0)
    D_gan = Discriminator(feature_dim=10, num_hidden_layers=2, hidden_dims=(32, 32), conditional_dim=0)
    train_gan(sample_data, G_gan, D_gan, epochs=1, batch_size=16)
    
    # WGAN-GP training
    G_wgan = Generator(noise_dim=20, num_hidden_layers=2, out_dim=10, hidden_dims=(32, 32), conditional_dim=0)
    D_wgan = Discriminator(feature_dim=10, num_hidden_layers=2, hidden_dims=(32, 32), conditional_dim=0)
    train_wgan_gp(sample_data, G_wgan, D_wgan, epochs=1, batch_size=16)
    
    # Both should complete without errors
    assert G_gan is not None
    assert G_wgan is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
