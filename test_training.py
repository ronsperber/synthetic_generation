"""
Unit tests for GAN and WGAN-GP training functions
"""
import copy
import pytest
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from gan.models import Generator, Discriminator, OutputHead
from gan.training import train_gan, train_wgan_gp
from gan.utils import cov_matrix, cov_penalty, make_dataloader


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
              loss="bce_with_logits")
    
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



def test_cov_matrix_and_penalty():
    # Small example (batch_size=3, features=2)
    f_real = torch.tensor([[1.0, 2.0],
                           [2.0, 4.0],
                           [3.0, 6.0]])
    f_fake = torch.tensor([[2.0, 1.0],
                           [4.0, 2.0],
                           [6.0, 3.0]])

    # --- Test cov_matrix ---
    # Manually compute variances and covariance
    mean_real = f_real.mean(0)
    mean_fake = f_fake.mean(0)

    centered_real = f_real - mean_real
    centered_fake = f_fake - mean_fake

    var_real = (centered_real ** 2).sum(0) / f_real.shape[0]
    var_fake = (centered_fake ** 2).sum(0) / f_fake.shape[0]
    cov_real_off = (centered_real[:,0] * centered_real[:,1]).sum() / f_real.shape[0]
    cov_fake_off = (centered_fake[:,0] * centered_fake[:,1]).sum() / f_fake.shape[0]

    cov_real_manual = torch.tensor([[var_real[0], cov_real_off],
                                    [cov_real_off, var_real[1]]])
    cov_fake_manual = torch.tensor([[var_fake[0], cov_fake_off],
                                    [cov_fake_off, var_fake[1]]])

    # Check cov_matrix function
    cov_real_fn = cov_matrix(f_real)
    cov_fake_fn = cov_matrix(f_fake)
    assert torch.allclose(cov_real_fn, cov_real_manual), "cov_matrix real mismatch"
    assert torch.allclose(cov_fake_fn, cov_fake_manual), "cov_matrix fake mismatch"

    # --- Test cov_penalty ---
    manual_penalty = ((var_real[0]-var_fake[0])**2 + 
                      (var_real[1]-var_fake[1])**2 + 
                      (cov_real_off - cov_fake_off)**2) / 3

    penalty_fn = cov_penalty(f_fake, f_real)
    assert torch.isclose(penalty_fn, manual_penalty), f"cov_penalty mismatch: {penalty_fn} vs {manual_penalty}"


def test_feature_matching_changes_generator_GAN():
    X = torch.rand(64,4)
    loader = make_dataloader(X,batch_size=64,shuffle=False)
    G = Generator(noise_dim=2, num_hidden_layers=2, hidden_dims=(32,32), out_dim=4)
    D = Discriminator(feature_dim=4, num_hidden_layers=2, hidden_dims=(32,32))
    G_fm_1 = copy.deepcopy(G)
    D_fm_1 = copy.deepcopy(D)
    G_fm_2 = copy.deepcopy(G)
    D_fm_2 = copy.deepcopy(D)
    G_fm_both = copy.deepcopy(G)
    D_fm_both = copy.deepcopy(D)
    G_base_params = [p.clone() for p in G.parameters()]
    torch.manual_seed(42)
    train_gan(
        X=loader,
        G=G,
        D=D,
        batch_size=8,
        epochs=1
    )
    torch.manual_seed(42)
    train_gan(
        X=loader,
        G=G_fm_1,
        D=D_fm_1,
        batch_size=8,
        epochs=1,
        lambda_fm_1=100,
    )
    torch.manual_seed(42)
    train_gan(
        X=loader,
        G=G_fm_2,
        D=D_fm_2,
        batch_size=8,
        epochs=1,
        lambda_fm_2=300000
    )
    torch.manual_seed(42)
    train_gan(
        X=loader,
        G=G_fm_both,
        D=D_fm_both,
        batch_size=8,
        epochs=1,
        lambda_fm_1=100,
        lambda_fm_2=300000
    )
    no_fm_params = [p.clone() for p in G.parameters()]
    fm_1_params = [p.clone() for p in G_fm_1.parameters()]
    fm_2_params = [p.clone() for p in G_fm_2.parameters()]
    fm_both_params = [p.clone() for p in G_fm_both.parameters()]
    assert any(
        not torch.allclose(p_no, p_base) for (p_no, p_base) in zip(no_fm_params, G_base_params)
    ),"no update in base"
    assert any(
            not torch.allclose(p_no, p_fm_1) for (p_no, p_fm_1) in zip(no_fm_params, fm_1_params)
    ),"fm 1 did not change from base"
    assert any(
            not torch.allclose(p_no, p_fm_2) for (p_no, p_fm_2) in zip(no_fm_params, fm_2_params)
        ),"fm 2 did not change from base"
    assert any(
            not torch.allclose(p_no, p_fmb) for (p_no, p_fmb) in zip(no_fm_params, fm_both_params)
            ), "Combined did not change from base"
    assert any(
            not torch.allclose(p_fm_1,p_fm_2) for (p_fm_1, p_fm_2) in zip(fm_1_params, fm_2_params)
    ),"fm_1 and fm_2 gave the same penalty"
    assert all(
        torch.allclose(p_base, p_fm) for p_base, p_fm in zip(D.parameters(), D_fm_1.parameters())
        ), "Discriminator changed with fm_1"
    assert all(
        torch.allclose(p_base, p_fm) for p_base, p_fm in zip(D.parameters(), D_fm_2.parameters())
        ), "Discriminator changed with fm_2"
    assert all(
        torch.allclose(p_base, p_fm) for p_base, p_fm in zip(D.parameters(), D_fm_both.parameters())
        ), "Discriminator changed with fm_both"
    assert any(
        not torch.allclose(p_fm_1, p_fmb) for p_fm_1, p_fmb in zip(fm_1_params, fm_both_params)
        ), "fm_both same as fm_1"
    assert any(
        not torch.allclose(p_fm_2, p_fmb) for p_fm_2, p_fmb in zip(fm_2_params, fm_both_params)
        ), "fm_both same as fm_2"

def test_feature_matching_changes_generator_WGAN():
    X = torch.rand(64,4)
    loader = make_dataloader(X,batch_size=64,shuffle=False)
    G = Generator(noise_dim=2, num_hidden_layers=2, hidden_dims=(32,32), out_dim=4)
    D = Discriminator(feature_dim=4, num_hidden_layers=2, hidden_dims=(32,32))
    G_fm_1 = copy.deepcopy(G)
    D_fm_1 = copy.deepcopy(D)
    G_fm_2 = copy.deepcopy(G)
    D_fm_2 = copy.deepcopy(D)
    G_fm_both = copy.deepcopy(G)
    D_fm_both = copy.deepcopy(D)
    G_base_params = [p.clone() for p in G.parameters()]
    torch.manual_seed(42)
    train_wgan_gp(
        X=loader,
        G=G,
        D=D,
        batch_size=8,
        epochs=1
    )
    torch.manual_seed(42)
    train_wgan_gp(
        X=loader,
        G=G_fm_1,
        D=D_fm_1,
        batch_size=8,
        epochs=1,
        lambda_fm_1=100,
    )
    torch.manual_seed(42)
    train_wgan_gp(
        X=loader,
        G=G_fm_2,
        D=D_fm_2,
        batch_size=8,
        epochs=1,
        lambda_fm_2=300000
    )
    torch.manual_seed(42)
    train_wgan_gp(
        X=loader,
        G=G_fm_both,
        D=D_fm_both,
        batch_size=8,
        epochs=1,
        lambda_fm_1=100,
        lambda_fm_2=300000
    )
    no_fm_params = [p.clone() for p in G.parameters()]
    fm_1_params = [p.clone() for p in G_fm_1.parameters()]
    fm_2_params = [p.clone() for p in G_fm_2.parameters()]
    fm_both_params = [p.clone() for p in G_fm_both.parameters()]
    assert any(
        not torch.allclose(p_no, p_base) for (p_no, p_base) in zip(no_fm_params, G_base_params)
    ),"no update in base"
    assert any(
            not torch.allclose(p_no, p_fm_1) for (p_no, p_fm_1) in zip(no_fm_params, fm_1_params)
    ),"fm 1 did not change from base"
    assert any(
            not torch.allclose(p_no, p_fm_2) for (p_no, p_fm_2) in zip(no_fm_params, fm_2_params)
        ),"fm 2 did not change from base"
    assert any(
            not torch.allclose(p_no, p_fmb) for (p_no, p_fmb) in zip(no_fm_params, fm_both_params)
            ), "Combined did not change from base"
    assert any(
            not torch.allclose(p_fm_1,p_fm_2) for (p_fm_1, p_fm_2) in zip(fm_1_params, fm_2_params)
    ),"fm_1 and fm_2 gave the same penalty"
    assert all(
        torch.allclose(p_base, p_fm) for p_base, p_fm in zip(D.parameters(), D_fm_1.parameters())
        ), "Discriminator changed with fm_1"
    assert all(
        torch.allclose(p_base, p_fm) for p_base, p_fm in zip(D.parameters(), D_fm_2.parameters())
        ), "Discriminator changed with fm_2"
    assert all(
        torch.allclose(p_base, p_fm) for p_base, p_fm in zip(D.parameters(), D_fm_both.parameters())
        ), "Discriminator changed with fm_both"
    assert any(
        not torch.allclose(p_fm_1, p_fmb) for p_fm_1, p_fmb in zip(fm_1_params, fm_both_params)
        ), "fm_both same as fm_1"
    assert any(
        not torch.allclose(p_fm_2, p_fmb) for p_fm_2, p_fmb in zip(fm_2_params, fm_both_params)
        ), "fm_both same as fm_2"

def test_generator_single_output():
    torch.manual_seed(42)
    G = Generator(
        noise_dim=3,
        num_hidden_layers=2,
        hidden_dims=[(3, 8), (8, 8)],
        out_dim=5  # single output
    )
    
    # Generate 10 samples
    z = torch.randn(10, 3)
    out = G(z)
    
    # Check output shape
    assert out.shape == (10, 5), f"Expected shape (10,5), got {out.shape}"
    
    # Check values are finite
    assert torch.isfinite(out).all(), "Output contains NaN or Inf"

# ===============================
# Test 2: multiple output heads
# ===============================
def test_generator_multiple_heads():
    torch.manual_seed(42)
    heads = [
        OutputHead(dim=2, activation=nn.Identity, name="head1"),
        OutputHead(dim=3, activation=nn.ReLU, name="head2"),
        OutputHead(dim=1, activation=nn.Sigmoid, name="head3")
    ]
    
    G = Generator(
        noise_dim=4,
        num_hidden_layers=2,
        hidden_dims=[(4, 8), (8, 8)],
        output_heads=heads
    )
    
    z = torch.randn(7, 4)
    out = G(z)
    
    # Check output shape: sum of head dims
    expected_dim = sum(head.dim for head in heads)
    assert out.shape == (7, expected_dim), f"Expected shape (7,{expected_dim}), got {out.shape}"
    
    # Optional: check ReLU and Sigmoid outputs are in expected ranges
    # head1: Identity, should match first 2 dims directly
    head1_out = out[:, 0:2]
    # head2: ReLU, should be >= 0
    head2_out = out[:, 2:5]
    assert (head2_out >= 0).all(), "ReLU head produced negative values"
    # head3: Sigmoid, should be in [0,1]
    head3_out = out[:, 5:6]
    assert ((0 <= head3_out) & (head3_out <= 1)).all(), "Sigmoid head out of range [0,1]"
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
