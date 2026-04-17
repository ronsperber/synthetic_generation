import copy
import torch
from synthetic_generation.gan.utils import feature_entropy
from synthetic_generation.data_utils import make_dataloader
from synthetic_generation.gan.models import Generator, Discriminator
from synthetic_generation.gan.training import train_gan

def test_feature_entropy_scalar_and_finite():
    f_fake = torch.randn(32, 8)
    ent = feature_entropy(f_fake)
    assert ent.ndim == 0
    assert torch.isfinite(ent)

def test_feature_entropy_penalizes_collapse():
    f_fake = torch.zeros(32, 64)  # completely collapsed
    ent = feature_entropy(f_fake,eps=1e-12)
    assert ent > 25  # should be very large

def test_feature_entropy_rewards_spread():
    f_narrow = torch.randn(32, 8) * 0.01
    f_wide = torch.randn(32, 8) * 10.0

    ent_narrow = feature_entropy(f_narrow)
    ent_wide = feature_entropy(f_wide)

    assert ent_wide < ent_narrow

def test_feature_entropy_translation_invariant():
    f = torch.randn(32, 8)
    ent1 = feature_entropy(f)
    ent2 = feature_entropy(f + 100.0)
    torch.testing.assert_close(ent1, ent2, rtol=1e-5, atol=1e-6)

def test_feature_entropy_positive_when_det_lt_one():
    f = 0.01 * torch.randn(128, 4)  # small variance → det(cov) << 1
    ent = feature_entropy(f)
    assert ent > 0

def test_feature_entropy_negative_when_det_gt_one():
    f = 10.0 * torch.randn(128, 4)  # large variance → det(cov) >> 1
    ent = feature_entropy(f)
    assert ent < 0

def test_entropy_changes_generator_GAN():
    X = torch.rand(64, 4)
    loader = make_dataloader(X, batch_size=64, shuffle=False)
    G = Generator(noise_dim=2, num_hidden_layers=2, hidden_dims=(32, 32), output_dim=4)
    D = Discriminator(feature_dim=4, num_hidden_layers=2, hidden_dims=(32, 32))
    G_copy = copy.deepcopy(G)
    D_copy = copy.deepcopy(D)
    G_ent = copy.deepcopy(G)
    D_ent = copy.deepcopy(D)

    torch.manual_seed(42)
    train_gan(X=loader, G=G, D=D, epochs=1)
    torch.manual_seed(42)
    train_gan(X=loader, G=G_copy, D=D_copy, epochs=1)
    torch.manual_seed(42)
    train_gan(X=loader, G=G_ent, D=D_ent, lambda_entropy=100, epochs=1)
    no_ent_params = [p.clone() for p in G.parameters()]
    copy_params = [p.clone() for p in G_copy.parameters()]
    ent_params = [p.clone() for p in G_ent.parameters()]
    # verify that no change occurs if we redo the same thing after resetting seed
    assert all(torch.allclose(p_orig, p_copy) for p_orig, p_copy in zip(no_ent_params, copy_params))
    # verify that changing the lambda_entropy parameter does change the parameters
    assert any(not torch.allclose(p_no, p_fm_1) for p_no, p_fm_1 in zip(no_ent_params, ent_params))

  
