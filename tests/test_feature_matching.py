import copy
import torch
from synthetic_generation.gan.models import Generator, Discriminator
from synthetic_generation.gan.training import train_gan, train_wgan_gp
from synthetic_generation.data_utils import make_dataloader

def test_feature_matching_changes_generator_GAN():
    X = torch.rand(64, 4)
    loader = make_dataloader(X, batch_size=64, shuffle=False)
    G = Generator(noise_dim=2, num_hidden_layers=2, hidden_dims=(32, 32), out_dim=4)
    D = Discriminator(feature_dim=4, num_hidden_layers=2, hidden_dims=(32, 32))

    G_fm_1 = copy.deepcopy(G)
    D_fm_1 = copy.deepcopy(D)
    G_fm_2 = copy.deepcopy(G)
    D_fm_2 = copy.deepcopy(D)
    G_fm_both = copy.deepcopy(G)
    D_fm_both = copy.deepcopy(D)

    G_base_params = [p.clone() for p in G.parameters()]

    torch.manual_seed(42)
    train_gan(loader, G, D, batch_size=8, epochs=1)

    torch.manual_seed(42)
    train_gan(loader, G_fm_1, D_fm_1, batch_size=8, epochs=1, lambda_fm_1=100)

    torch.manual_seed(42)
    train_gan(loader, G_fm_2, D_fm_2, batch_size=8, epochs=1, lambda_fm_2=300000)

    torch.manual_seed(42)
    train_gan(loader, G_fm_both, D_fm_both, batch_size=8, epochs=1, lambda_fm_1=100, lambda_fm_2=300000)

    no_fm_params = [p.clone() for p in G.parameters()]
    fm_1_params = [p.clone() for p in G_fm_1.parameters()]
    fm_2_params = [p.clone() for p in G_fm_2.parameters()]
    fm_both_params = [p.clone() for p in G_fm_both.parameters()]

    assert any(not torch.allclose(p_no, p_base) for p_no, p_base in zip(no_fm_params, G_base_params))
    assert any(not torch.allclose(p_no, p_fm_1) for p_no, p_fm_1 in zip(no_fm_params, fm_1_params))
    assert any(not torch.allclose(p_no, p_fm_2) for p_no, p_fm_2 in zip(no_fm_params, fm_2_params))
    assert any(not torch.allclose(p_no, p_fmb) for p_no, p_fmb in zip(no_fm_params, fm_both_params))
    assert any(not torch.allclose(p_fm_1, p_fm_2) for p_fm_1, p_fm_2 in zip(fm_1_params, fm_2_params))
    assert all(torch.allclose(p_base, p_fm) for p_base, p_fm in zip(D.parameters(), D_fm_1.parameters()))
    assert all(torch.allclose(p_base, p_fm) for p_base, p_fm in zip(D.parameters(), D_fm_2.parameters()))
    assert all(torch.allclose(p_base, p_fm) for p_base, p_fm in zip(D.parameters(), D_fm_both.parameters()))
    assert any(not torch.allclose(p_fm_1, p_fmb) for p_fm_1, p_fmb in zip(fm_1_params, fm_both_params))
    assert any(not torch.allclose(p_fm_2, p_fmb) for p_fm_2, p_fmb in zip(fm_2_params, fm_both_params))


def test_feature_matching_changes_generator_WGAN():
    X = torch.rand(64, 4)
    loader = make_dataloader(X, batch_size=64, shuffle=False)
    G = Generator(noise_dim=2, num_hidden_layers=2, hidden_dims=(32, 32), out_dim=4)
    D = Discriminator(feature_dim=4, num_hidden_layers=2, hidden_dims=(32, 32))

    G_fm_1 = copy.deepcopy(G)
    D_fm_1 = copy.deepcopy(D)
    G_fm_2 = copy.deepcopy(G)
    D_fm_2 = copy.deepcopy(D)
    G_fm_both = copy.deepcopy(G)
    D_fm_both = copy.deepcopy(D)

    G_base_params = [p.clone() for p in G.parameters()]

    torch.manual_seed(42)
    train_wgan_gp(loader, G, D, batch_size=8, epochs=1)

    torch.manual_seed(42)
    train_wgan_gp(loader, G_fm_1, D_fm_1, batch_size=8, epochs=1, lambda_fm_1=100)

    torch.manual_seed(42)
    train_wgan_gp(loader, G_fm_2, D_fm_2, batch_size=8, epochs=1, lambda_fm_2=300000)

    torch.manual_seed(42)
    train_wgan_gp(loader, G_fm_both, D_fm_both, batch_size=8, epochs=1, lambda_fm_1=100, lambda_fm_2=300000)

    no_fm_params = [p.clone() for p in G.parameters()]
    fm_1_params = [p.clone() for p in G_fm_1.parameters()]
    fm_2_params = [p.clone() for p in G_fm_2.parameters()]
    fm_both_params = [p.clone() for p in G_fm_both.parameters()]

    assert any(not torch.allclose(p_no, p_base) for p_no, p_base in zip(no_fm_params, G_base_params))
    assert any(not torch.allclose(p_no, p_fm_1) for p_no, p_fm_1 in zip(no_fm_params, fm_1_params))
    assert any(not torch.allclose(p_no, p_fm_2) for p_no, p_fm_2 in zip(no_fm_params, fm_2_params))
    assert any(not torch.allclose(p_no, p_fmb) for p_no, p_fmb in zip(no_fm_params, fm_both_params))
    assert any(not torch.allclose(p_fm_1, p_fm_2) for p_fm_1, p_fm_2 in zip(fm_1_params, fm_2_params))
    assert all(torch.allclose(p_base, p_fm) for p_base, p_fm in zip(D.parameters(), D_fm_1.parameters()))
    assert all(torch.allclose(p_base, p_fm) for p_base, p_fm in zip(D.parameters(), D_fm_2.parameters()))
    assert all(torch.allclose(p_base, p_fm) for p_base, p_fm in zip(D.parameters(), D_fm_both.parameters()))
    assert any(not torch.allclose(p_fm_1, p_fmb) for p_fm_1, p_fmb in zip(fm_1_params, fm_both_params))
    assert any(not torch.allclose(p_fm_2, p_fmb) for p_fm_2, p_fmb in zip(fm_2_params, fm_both_params))
