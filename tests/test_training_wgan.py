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

def test_wgan_gp_callback_stops_training(models, sample_data):
    G, D = models
    stop_epoch = 3
    
    def callback(state):
        return state["epoch"] >= stop_epoch
    
    G_losses, D_losses = train_wgan_gp(
        X=sample_data, G=G, D=D, 
        epochs=10, batch_size=16, n_critic=2,
        epoch_callback=callback,
        return_history=True
    )
    assert len(G_losses) == stop_epoch
    assert len(D_losses) == stop_epoch
    
def test_train_wgan_gp_conditional(conditional_models, sample_conditional_data):
    G, D = conditional_models
    X, c = sample_conditional_data
    initial_g_params = [p.clone() for p in G.parameters()]
    train_wgan_gp(X=X, G=G, D=D, c=c, epochs=2, batch_size=16, n_critic=2)
    assert any(not torch.allclose(p0, p1) for p0, p1 in zip(initial_g_params, G.parameters()))

