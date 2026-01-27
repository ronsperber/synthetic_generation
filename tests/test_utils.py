import torch
from synthetic_generation.gan.utils import cov_matrix, cov_penalty

def test_cov_matrix_and_penalty():
    f_real = torch.tensor([[1.0, 2.0],
                           [2.0, 4.0],
                           [3.0, 6.0]])
    f_fake = torch.tensor([[2.0, 1.0],
                           [4.0, 2.0],
                           [6.0, 3.0]])

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

    cov_real_fn = cov_matrix(f_real)
    cov_fake_fn = cov_matrix(f_fake)
    assert torch.allclose(cov_real_fn, cov_real_manual)
    assert torch.allclose(cov_fake_fn, cov_fake_manual)

    manual_penalty = ((var_real[0]-var_fake[0])**2 +
                      (var_real[1]-var_fake[1])**2 +
                      (cov_real_off - cov_fake_off)**2) / 3

    penalty_fn = cov_penalty(f_fake, f_real)
    assert torch.isclose(penalty_fn, manual_penalty)
