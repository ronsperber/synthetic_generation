import torch
import torch.nn as nn
import pytest
from gan.models import Generator, Discriminator, OutputHead
from gan.activations import GumbelSoftmax, SoftplusShift, TanhShiftScale, BoundedSigmoid, ClampedIdentity
import torch
import torch.nn as nn

def test_generator_forward_runs_with_activation():
    # tests to make sure passing an nn.Module as an activation works properly
    # previous tests passing show that this change is backwards compatible
    G = Generator(
        noise_dim=3,
        num_hidden_layers=1,
        hidden_dims=[(3, 3)],
        hidden_activation=nn.ReLU(),
        output_heads=[OutputHead(dim=2,activation=nn.ReLU())]
    )

    z = torch.randn(4, 3)  # batch of 4 samples

    output = G.forward(z)

    # Basic sanity checks
    assert output.shape == (4, 2), "Output should match the sum of head dimensions"
    assert isinstance(G.activation, nn.ReLU), "Activation attribute should store the instance"
    assert torch.all(output >= 0), "ReLU should zero out negative values"

# test Gumbel Softmax
def test_gumbel_softmax_training_soft():
    torch.manual_seed(0)
    activation = GumbelSoftmax(tau=1.0, hard_inference=True)
    activation.train()

    x = torch.randn(16, 5)
    y = activation(x)

    # Rows should sum to ~1
    row_sums = y.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    # Should not be strictly one-hot
    assert not torch.all((y == 0) | (y == 1)), "Training output should be soft"

def test_gumbel_softmax_eval_hard():
    torch.manual_seed(0)
    activation = GumbelSoftmax(tau=1.0, hard_inference=True)
    activation.eval()

    x = torch.randn(16, 5)
    y = activation(x)

    # Rows should sum to 1
    row_sums = y.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    # Should be exactly one-hot
    assert torch.all((y == 0) | (y == 1)), "Eval output should be hard one-hot"

def test_gumbel_softmax_eval_soft_when_hard_inference_false():
    torch.manual_seed(0)
    activation = GumbelSoftmax(tau=1.0, hard_inference=False)
    activation.eval()

    x = torch.randn(16, 5)
    y = activation(x)

    # Rows should sum to 1
    row_sums = y.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    # Should not be strictly one-hot
    assert not torch.all((y == 0) | (y == 1)), "Eval output should be soft when hard_inference=False"

def test_gumbel_softmax_allows_gradients():
    torch.manual_seed(0)
    activation = GumbelSoftmax(tau=1.0, hard_inference=True)
    activation.train()

    x = torch.randn(8, 4, requires_grad=True)
    y = activation(x)
    loss = y.mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all(), "Gradients should be finite"

# test softplus

def test_softplus_shift_respects_min():
    act = SoftplusShift(min_value=2.5)
    x = torch.tensor([-10.0, 0.0, 10.0])
    y = act(x)
    assert (y >= 2.5).all()


def test_softplus_shift_monotonic():
    act = SoftplusShift(min_value=0.0)
    x = torch.linspace(-10, 10, 100)
    y = act(x)
    assert torch.all(y[1:] >= y[:-1])


def test_softplus_shift_backward():
    act = SoftplusShift(min_value=1.0)
    x = torch.randn(10, requires_grad=True)
    y = act(x).mean()
    y.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

def test_tanh_shift_scale_bounds():
    act = TanhShiftScale(min_value=-2.0, max_value=3.0)
    x = torch.randn(100)
    y = act(x)
    assert (y >= -2.0).all()
    assert (y <= 3.0).all()


def test_tanh_shift_scale_extremes():
    act = TanhShiftScale(min_value=0.0, max_value=1.0)
    x = torch.tensor([-1e6, 1e6])
    y = act(x)
    assert torch.allclose(y[0], torch.tensor(0.0), atol=1e-4)
    assert torch.allclose(y[1], torch.tensor(1.0), atol=1e-4)


def test_tanh_shift_scale_backward():
    act = TanhShiftScale(min_value=0.0, max_value=5.0)
    x = torch.randn(10, requires_grad=True)
    y = act(x).mean()
    y.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

# test ClamedIdentity

def test_clamped_identity_no_bounds():
    act = ClampedIdentity()
    x = torch.randn(10)
    y = act(x)
    assert torch.allclose(x, y)


def test_clamped_identity_lower_bound():
    eps = 1e-6
    act = ClampedIdentity(min_value=0.0, eps=eps)
    x = torch.tensor([-1.0, 0.5, 2.0])
    y = act(x)
    assert (y >= -eps).all()


def test_clamped_identity_upper_bound():
    eps = 1e-6
    act = ClampedIdentity(max_value=1.0, eps=eps)
    x = torch.tensor([0.5, 1.5, 3.0])
    y = act(x)
    assert (y <= 1.0 + eps).all()


def test_clamped_identity_both_bounds():
    eps = 1e-6
    act = ClampedIdentity(min_value=-1.0, max_value=1.0, eps=eps)
    x = torch.tensor([-5.0, -0.5, 0.5, 5.0])
    y = act(x)
    assert (y >= -1.0 - eps).all()
    assert (y <= 1.0 + eps).all()


def test_clamped_identity_backward():
    act = ClampedIdentity(min_value=0.0, max_value=1.0)
    x = torch.randn(10, requires_grad=True)
    y = act(x).mean()
    y.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

# Bounded Sigmoid tests

def test_bounded_sigmoid_bounds():
    act = BoundedSigmoid(min_value=2.0, max_value=5.0)
    x = torch.randn(100)
    y = act(x)
    assert (y >= 2.0).all()
    assert (y <= 5.0).all()


def test_bounded_sigmoid_extremes():
    act = BoundedSigmoid(min_value=0.0, max_value=1.0)
    x = torch.tensor([-1e6, 1e6])
    y = act(x)
    assert torch.allclose(y[0], torch.tensor(0.0), atol=1e-4)
    assert torch.allclose(y[1], torch.tensor(1.0), atol=1e-4)


def test_bounded_sigmoid_monotonic():
    act = BoundedSigmoid(min_value=0.0, max_value=10.0)
    x = torch.linspace(-10, 10, 100)
    y = act(x)
    assert torch.all(y[1:] >= y[:-1])


def test_bounded_sigmoid_backward():
    act = BoundedSigmoid(min_value=0.0, max_value=1.0)
    x = torch.randn(10, requires_grad=True)
    y = act(x).mean()
    y.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


# integration tests
def test_generator_with_gumbel_softmax_head_backward():
    torch.manual_seed(0)
    heads = [OutputHead(dim=4, activation=GumbelSoftmax(), name="cat")]
    G = Generator(
        noise_dim=3,
        num_hidden_layers=1,
        hidden_dims=[(3, 8)],
        output_heads=heads
    )

    z = torch.randn(10, 3, requires_grad=True)
    out = G(z)
    loss = out.mean()
    loss.backward()

    grads = [p.grad for p in G.parameters() if p.requires_grad]
    assert any(g is not None for g in grads), "Gradients should flow through generator"

def test_generator_with_custom_activation_backward():
    from gan.models import Generator
    from gan.activations import TanhShiftScale

    G = Generator(
        noise_dim=5,
        num_hidden_layers=1,
        hidden_dims=[(5, 10)],
        output_heads=[OutputHead(dim=3, activation=TanhShiftScale(-1, 1), name="bounded")]
    )

    z = torch.randn(8, 5)
    out = G(z)
    loss = out.mean()
    loss.backward()

    grads = [p.grad for p in G.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)

def test_generator_forward_runs_with_activation():
    # tests to make sure passing an nn.Module as an activation works properly
    # previous tests passing show that this change is backwards compatible
    G = Generator(
        noise_dim=3,
        num_hidden_layers=1,
        hidden_dims=[(3, 3)],
        hidden_activation=nn.ReLU(),
        output_heads=[OutputHead(dim=2,activation=nn.ReLU())]
    )

    z = torch.randn(4, 3)  # batch of 4 samples

    output = G.forward(z)

    # Basic sanity checks
    assert output.shape == (4, 2), "Output should match the sum of head dimensions"
    assert isinstance(G.activation, nn.ReLU), "Activation attribute should store the instance"
    assert torch.all(output >= 0), "ReLU should zero out negative values"

def test_output_head_custom_activation():
        eps = 1e-6
        G = Generator(
        noise_dim=3,
        num_hidden_layers=1,
        hidden_dims=[(3, 3)],
        hidden_activation=nn.ReLU(),
        output_heads=[OutputHead(dim=2,activation=ClampedIdentity(min_value=0.0, max_value=10.0, eps=eps))]
        )

        z = torch.randn(4,3)
        output=G.forward(z)
        loss = output.mean()
        loss.backward()

        grads = [p.grad for p in G.parameters() if p.requires_grad]
        assert any(g is not None for g in grads), "Gradients should flow through generator"
        assert (output >= 0.0 - eps).all(), "Outputs should be >= min_val - eps"
        assert (output <= 10.0 + eps).all(), "Outputs should be <= max_val + eps"


