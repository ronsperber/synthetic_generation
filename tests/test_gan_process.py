import torch
import pytest
from synthetic_generation.gan.models import Generator, Discriminator
from synthetic_generation.gan.process import GANProcess, WGANProcess

def test_gan_process_train_save_load(tmp_path):
    # Toy data
    X = torch.randn(10, 2)  # 10 samples, 2 features
    c = None  # no conditional input for simplicity

    # --- Test GANProcess ---
    G_gan = Generator(noise_dim=2, num_hidden_layers=1, hidden_dims=(4,4), out_dim=2)
    D_gan = Discriminator(feature_dim=2, num_hidden_layers=1, hidden_dims=(4,4))
    gan_process = GANProcess(G_gan, D_gan)

    # Tiny training step
    gan_process.train(X, c=c, epochs=1, lr_G=0.01, lr_D=0.01, batch_size=4)

    # Generate samples
    X_gen_before = gan_process.generate_samples(3)

    # Save
    save_path_gan = tmp_path / "gan_process.pt"
    gan_process.save(save_path_gan)
    G_gan_params = [p.clone() for p in gan_process.G.parameters()]
    D_gan_params = [p.clone() for p in gan_process.D.parameters()]
    # Load
    loaded_gan = GANProcess.load(save_path_gan)
    X_gen_after = loaded_gan.generate_samples(3)

    # Checks
    assert X_gen_after.shape[0] == 3
    assert X_gen_after.shape[1] == 2
    assert X_gen_before.shape[1:] == X_gen_after.shape[1:]
    assert all(torch.allclose(p0, p1) for p0, p1 in zip(G_gan_params, loaded_gan.G.parameters()))
    assert all(torch.allclose(p0, p1) for p0, p1 in zip(D_gan_params, loaded_gan.D.parameters()))

    # --- Test WGANProcess ---
    G_wgan = Generator(noise_dim=2, num_hidden_layers=1, hidden_dims=(4,4), out_dim=2)
    D_wgan = Discriminator(feature_dim=2, num_hidden_layers=1, hidden_dims=(4,4))
    wgan_process = WGANProcess(G_wgan, D_wgan)

    # Tiny training step
    wgan_process.train(X, c=c, epochs=1, lr_G=0.01, lr_D=0.01, batch_size=4, lambda_fm_1=0.0, lambda_fm_2=0.0)
    G_wgan_params = [p.clone() for p in wgan_process.G.parameters()]
    D_wgan_params = [p.clone() for p in wgan_process.D.parameters()]
                     
    # Generate samples
    X_wgan_before = wgan_process.generate_samples(2)

    # Save
    save_path_wgan = tmp_path / "wgan_process.pt"
    wgan_process.save(save_path_wgan)

    # Load
    loaded_wgan = WGANProcess.load(save_path_wgan)
    X_wgan_after = loaded_wgan.generate_samples(2)

    # Checks
    assert X_wgan_after.shape[0] == 2
    assert X_wgan_after.shape[1] == 2
    assert X_wgan_before.shape[1:] == X_wgan_after.shape[1:]
    assert all(torch.allclose(p0, p1) for p0, p1 in zip(G_wgan_params, loaded_wgan.G.parameters()))
    assert all(torch.allclose(p0, p1) for p0, p1 in zip(D_wgan_params, loaded_wgan.D.parameters()))
