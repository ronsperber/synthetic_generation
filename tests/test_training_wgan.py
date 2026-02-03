import pytest
import torch
from synthetic_generation.gan.training import train_wgan_gp
from synthetic_generation.gan.models import Generator, Discriminator

def test_train_wgan_gp_parameter_updates(models, sample_data):
    G, D = models
    initial_g_params = [p.clone() for p in G.parameters()]
    initial_d_params = [p.clone() for p in D.parameters()]
    train_wgan_gp(X=sample_data,G=G, D=D, epochs=5, batch_size=16, n_critic=3)
    assert any(not torch.allclose(p0, p1) for p0, p1 in zip(initial_g_params, G.parameters()))
    assert any(not torch.allclose(p0, p1) for p0, p1 in zip(initial_d_params, D.parameters()))

def test_train_wgan_gp_conditional(conditional_models, sample_conditional_data):
    G, D = conditional_models
    X, c = sample_conditional_data
    initial_g_params = [p.clone() for p in G.parameters()]
    train_wgan_gp(X=X, G=G, D=D, c=c, epochs=2, batch_size=16, n_critic=2)
    assert any(not torch.allclose(p0, p1) for p0, p1 in zip(initial_g_params, G.parameters()))

def test_wgan_gp_critic_iterations(models, sample_data):
    G, D = models
    train_wgan_gp(X=sample_data, G=G, D=D, epochs=1, batch_size=16, n_critic=5)

def test_wgan_gp_gradient_penalty_scaling(models, sample_data):
    for lambda_gp in [1.0, 10.0, 100.0]:
        G = Generator(noise_dim=20, num_hidden_layers=2, out_dim=10, hidden_dims=(32, 32))
        D = Discriminator(feature_dim=10, num_hidden_layers=2, hidden_dims=(32, 32))
        train_wgan_gp(X=sample_data, G=G, D=D, epochs=1, batch_size=16, lambda_gp=lambda_gp)

def test_wgan_gp_different_critic_counts(models, sample_data):
    for n_critic in [1, 3, 5]:
        G = Generator(noise_dim=20, num_hidden_layers=2, out_dim=10, hidden_dims=(32, 32))
        D = Discriminator(feature_dim=10, num_hidden_layers=2, hidden_dims=(32, 32))
        train_wgan_gp(X=sample_data, G=G, D=D, epochs=1, batch_size=16, n_critic=n_critic)

def test_wgan_gp_learning_rates(sample_data):
    for lr_g, lr_d in [(1e-4, 1e-4), (5e-5, 5e-5), (1e-3, 1e-3)]:
        G = Generator(noise_dim=20, num_hidden_layers=2, out_dim=10, hidden_dims=(32, 32))
        D = Discriminator(feature_dim=10, num_hidden_layers=2, hidden_dims=(32, 32))
        train_wgan_gp(X=sample_data, G=G, D=D, epochs=1, batch_size=16, lr_G=lr_g, lr_D=lr_d)
