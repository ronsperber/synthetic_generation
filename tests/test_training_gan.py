import pytest
import torch
from synthetic_generation.gan.training import train_gan
from synthetic_generation.gan.models import Generator, Discriminator

def test_dimension_mismatch_output_feature():
    G = Generator(noise_dim=20, num_hidden_layers=2, out_dim=10, hidden_dims=(32, 32))
    D = Discriminator(feature_dim=5, num_hidden_layers=2, hidden_dims=(32, 32))
    X = torch.randn(32, 10)
    with pytest.raises(ValueError, match="G outputs|dimension"):
        train_gan(X=X, G=G, D=D, epochs=1, batch_size=16)

def test_dimension_mismatch_conditional():
    G = Generator(noise_dim=20, num_hidden_layers=2, out_dim=10, hidden_dims=(32, 32), conditional_dim=3)
    D = Discriminator(feature_dim=10, num_hidden_layers=2, hidden_dims=(32, 32),conditional_dim=5)
    X = torch.randn(32, 10)
    c = torch.nn.functional.one_hot(torch.randint(0, 3, (32,)), num_classes=3).float()
    with pytest.raises(ValueError, match="Conditional"):
        train_gan(X=X, G=G, D=D, c=c, epochs=1, batch_size=16)

def test_invalid_learning_rate():
    G = Generator(noise_dim=20, num_hidden_layers=2, out_dim=10, hidden_dims=(32, 32))
    D = Discriminator(feature_dim=10, num_hidden_layers=2, hidden_dims=(32, 32))
    X = torch.randn(32, 10)
    with pytest.raises(ValueError):
        train_gan(X=X, G=G, D=D, lr_G=-0.001, epochs=1, batch_size=16)

def test_invalid_batch_size():
    G = Generator(noise_dim=20, num_hidden_layers=2, out_dim=10, hidden_dims=(32, 32))
    D = Discriminator(feature_dim=10, num_hidden_layers=2, hidden_dims=(32, 32))
    X = torch.randn(32, 10)
    with pytest.raises((ValueError, RuntimeError)):
        train_gan(X=X, G=G, D=D, epochs=1, batch_size=0)

def test_negative_epochs_raises_error():
    G = Generator(noise_dim=20, num_hidden_layers=2, out_dim=10, hidden_dims=(32, 32))
    D = Discriminator(feature_dim=10, num_hidden_layers=2, hidden_dims=(32, 32))
    X = torch.randn(32, 10)
    with pytest.raises(ValueError):
        train_gan(X=X, G=G, D=D, epochs=-5, batch_size=16)

def test_train_gan_with_tensor_input(models, sample_data):
    G, D = models
    initial_g_params = [p.clone() for p in G.parameters()]
    initial_d_params = [p.clone() for p in D.parameters()]
    train_gan(X=sample_data, G=G, D=D, epochs=2, batch_size=16)
    assert any(not torch.allclose(p0, p1) for p0, p1 in zip(initial_g_params, G.parameters()))
    assert any(not torch.allclose(p0, p1) for p0, p1 in zip(initial_d_params, D.parameters()))

def test_train_gan_with_dataloader(models, dataloader):
    G, D = models
    initial_g_params = [p.clone() for p in G.parameters()]
    train_gan(X=dataloader, G=G, D=D, epochs=2, batch_size=16)
    assert any(not torch.allclose(p0, p1) for p0, p1 in zip(initial_g_params, G.parameters()))

def test_train_gan_conditional(conditional_models, sample_conditional_data):
    G, D = conditional_models
    X, c = sample_conditional_data
    initial_g_params = [p.clone() for p in G.parameters()]
    train_gan(X=X, G=G, D=D, c=c, epochs=2, batch_size=16)
    assert any(not torch.allclose(p0, p1) for p0, p1 in zip(initial_g_params, G.parameters()))

def test_gan_callback_stops_training(models, sample_data):
    G, D = models
    stop_epoch = 3
    
    def callback(state):
        return state["epoch"] >= stop_epoch
    
    G_losses, D_losses = train_gan(
        X=sample_data, G=G, D=D, 
        epochs=10, batch_size=16,
        epoch_callback=callback,
        return_history=True
    )
    assert len(G_losses) == stop_epoch
    assert len(D_losses) == stop_epoch

def test_batch_size_one(models, sample_data):
    G, D = models
    train_gan(X=sample_data, G=G, D=D, epochs=1, batch_size=1)

def test_small_dataset(models):
    G, D = models
    X = torch.randn(8, 10)
    train_gan(X=X, G=G, D=D, epochs=2, batch_size=2)

def test_single_sample_batch(models):
    G, D = models
    X = torch.randn(32, 10)
    train_gan(X=X, G=G, D=D, epochs=2, batch_size=32)