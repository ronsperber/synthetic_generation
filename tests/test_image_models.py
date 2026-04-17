import pytest
import torch
import torch.nn as nn
from synthetic_generation.gan.models import ImageGenerator, ImageDiscriminator


# ---------------------------------------------------------------------------
# ImageGenerator
# ---------------------------------------------------------------------------

def test_image_generator_basic_forward():
    torch.manual_seed(0)
    G = ImageGenerator(noise_dim=16, output_dim=(1, 8, 8), conv_in_channels=[64, 32])
    z = torch.randn(4, 16)
    out = G(z)
    assert out.shape == (4, 1, 8, 8)
    assert torch.isfinite(out).all()


def test_image_generator_multichannel_output():
    torch.manual_seed(0)
    G = ImageGenerator(noise_dim=16, output_dim=(3, 8, 8), conv_in_channels=[64, 32])
    z = torch.randn(5, 16)
    out = G(z)
    assert out.shape == (5, 3, 8, 8)


def test_image_generator_int_channels_with_num_layers():
    torch.manual_seed(0)
    # conv_in_channels=64 with num_conv_layers=2 → [64, 32]
    G = ImageGenerator(noise_dim=16, output_dim=(1, 8, 8), conv_in_channels=64, num_conv_layers=2)
    z = torch.randn(3, 16)
    out = G(z)
    assert out.shape == (3, 1, 8, 8)


def test_image_generator_output_activation_tanh():
    torch.manual_seed(0)
    G = ImageGenerator(
        noise_dim=16, output_dim=(1, 8, 8),
        conv_in_channels=[64, 32],
        output_activation=nn.Tanh,
    )
    z = torch.randn(6, 16)
    out = G(z)
    assert out.min() >= -1.0 and out.max() <= 1.0


def test_image_generator_conditional_forward():
    torch.manual_seed(0)
    G = ImageGenerator(
        noise_dim=16, output_dim=(1, 8, 8),
        conv_in_channels=[64, 32],
        conditional_dim=4,
    )
    z = torch.randn(5, 16)
    c = torch.randn(5, 4)
    out = G(z, c)
    assert out.shape == (5, 1, 8, 8)


def test_image_generator_generate():
    torch.manual_seed(0)
    G = ImageGenerator(noise_dim=16, output_dim=(1, 8, 8), conv_in_channels=[64, 32])
    out = G.generate(7)
    assert out.shape == (7, 1, 8, 8)
    assert torch.isfinite(out).all()


def test_image_generator_generate_conditional():
    torch.manual_seed(0)
    G = ImageGenerator(
        noise_dim=16, output_dim=(1, 8, 8),
        conv_in_channels=[64, 32],
        conditional_dim=3,
    )
    c = torch.randn(6, 3)
    out = G.generate(6, c=c)
    assert out.shape == (6, 1, 8, 8)


def test_image_generator_generate_samples_shape():
    torch.manual_seed(0)
    G = ImageGenerator(noise_dim=16, output_dim=(1, 8, 8), conv_in_channels=[64, 32])
    out = G.generate_samples(5)
    assert out.shape == (5, 1, 8, 8)
    assert out.device.type == "cpu"


def test_image_generator_generate_samples_restores_train_mode():
    G = ImageGenerator(noise_dim=16, output_dim=(1, 8, 8), conv_in_channels=[64, 32])
    G.train()
    assert G.training is True
    G.generate_samples(3)
    assert G.training is True


def test_image_generator_generate_samples_restores_eval_mode():
    G = ImageGenerator(noise_dim=16, output_dim=(1, 8, 8), conv_in_channels=[64, 32])
    G.eval()
    assert G.training is False
    G.generate_samples(3)
    assert G.training is False


def test_image_generator_backward():
    torch.manual_seed(0)
    G = ImageGenerator(noise_dim=16, output_dim=(1, 8, 8), conv_in_channels=[64, 32])
    z = torch.randn(4, 16)
    loss = G(z).mean()
    loss.backward()
    grads = [p.grad for p in G.parameters() if p.grad is not None]
    assert len(grads) > 0


def test_image_generator_missing_num_conv_layers_raises():
    with pytest.raises(ValueError, match="Number of convolutional layers"):
        ImageGenerator(noise_dim=16, output_dim=(1, 8, 8), conv_in_channels=64)


def test_image_generator_indivisible_image_size_raises():
    # 7 is not divisible by 2^2 = 4
    with pytest.raises(ValueError, match="not divisible"):
        ImageGenerator(noise_dim=16, output_dim=(1, 7, 7), conv_in_channels=[64, 32])


def test_image_generator_missing_conditional_raises():
    G = ImageGenerator(
        noise_dim=16, output_dim=(1, 8, 8),
        conv_in_channels=[64, 32],
        conditional_dim=3,
    )
    z = torch.randn(4, 16)
    with pytest.raises(ValueError, match="[Cc]onditional"):
        G(z)


def test_image_generator_conditional_length_mismatch_raises():
    G = ImageGenerator(
        noise_dim=16, output_dim=(1, 8, 8),
        conv_in_channels=[64, 32],
        conditional_dim=3,
    )
    c = torch.randn(5, 3)  # 5 samples
    with pytest.raises(ValueError, match="[Nn]umber of samples"):
        G.generate(6, c=c)  # 6 samples


# ---------------------------------------------------------------------------
# ImageDiscriminator
# ---------------------------------------------------------------------------

def test_image_discriminator_basic_forward():
    torch.manual_seed(0)
    D = ImageDiscriminator(feature_dim=(1, 8, 8), conv_out_channels=[32, 64])
    x = torch.randn(4, 1, 8, 8)
    out = D(x)
    assert out.shape == (4, 1)
    assert torch.isfinite(out).all()


def test_image_discriminator_multichannel_input():
    torch.manual_seed(0)
    D = ImageDiscriminator(feature_dim=(3, 8, 8), conv_out_channels=[32, 64])
    x = torch.randn(5, 3, 8, 8)
    out = D(x)
    assert out.shape == (5, 1)


def test_image_discriminator_int_channels_with_num_layers():
    torch.manual_seed(0)
    # conv_out_channels=32 with num_conv_layers=2 → [32, 64]
    D = ImageDiscriminator(feature_dim=(1, 8, 8), conv_out_channels=32, num_conv_layers=2)
    x = torch.randn(4, 1, 8, 8)
    out = D(x)
    assert out.shape == (4, 1)


def test_image_discriminator_use_sigmoid_output_range():
    torch.manual_seed(0)
    D = ImageDiscriminator(feature_dim=(1, 8, 8), conv_out_channels=[32, 64], use_sigmoid=True)
    x = torch.randn(6, 1, 8, 8)
    out = D(x)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_image_discriminator_conditional_forward():
    torch.manual_seed(0)
    D = ImageDiscriminator(
        feature_dim=(1, 8, 8),
        conv_out_channels=[32, 64],
        conditional_dim=4,
    )
    x = torch.randn(5, 1, 8, 8)
    c = torch.randn(5, 4)
    out = D(x, c)
    assert out.shape == (5, 1)


def test_image_discriminator_return_features():
    torch.manual_seed(0)
    D = ImageDiscriminator(feature_dim=(1, 8, 8), conv_out_channels=[32, 64])
    x = torch.randn(4, 1, 8, 8)
    out, features = D(x, return_features=True)
    assert out.shape == (4, 1)
    assert features.ndim == 2
    assert features.shape[0] == 4


def test_image_discriminator_backward():
    torch.manual_seed(0)
    D = ImageDiscriminator(feature_dim=(1, 8, 8), conv_out_channels=[32, 64])
    x = torch.randn(4, 1, 8, 8)
    loss = D(x).mean()
    loss.backward()
    grads = [p.grad for p in D.parameters() if p.grad is not None]
    assert len(grads) > 0


def test_image_discriminator_missing_num_conv_layers_raises():
    with pytest.raises(ValueError, match="Number of convolutional layers"):
        ImageDiscriminator(feature_dim=(1, 8, 8), conv_out_channels=32)


def test_image_discriminator_too_many_conv_layers_raises():
    # 4 conv layers on 8x8: 8 // 16 = 0 → final_size < 1
    with pytest.raises(ValueError, match="[Tt]oo many conv layers"):
        ImageDiscriminator(feature_dim=(1, 8, 8), conv_out_channels=[32, 64, 128, 256])


def test_image_discriminator_missing_conditional_raises():
    D = ImageDiscriminator(
        feature_dim=(1, 8, 8),
        conv_out_channels=[32, 64],
        conditional_dim=4,
    )
    x = torch.randn(4, 1, 8, 8)
    with pytest.raises(ValueError, match="[Cc]onditional"):
        D(x)


def test_image_discriminator_conditional_batch_mismatch_raises():
    D = ImageDiscriminator(
        feature_dim=(1, 8, 8),
        conv_out_channels=[32, 64],
        conditional_dim=4,
    )
    x = torch.randn(4, 1, 8, 8)
    c = torch.randn(3, 4)  # batch size 3 ≠ 4
    with pytest.raises(ValueError, match="[Bb]atch size"):
        D(x, c)
