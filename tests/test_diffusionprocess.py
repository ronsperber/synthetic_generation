# test_diffusion_process.py

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from synthetic_generation.diffusion.models import DiffusionNet, DiffusionProcess
from synthetic_generation.diffusion.schedules import linear_beta_schedule, cosine_beta_schedule

# ------------------------------
# DiffusionProcess initialization tests
# ------------------------------

def test_diffusion_process_init_with_model_data_dim():
    """Uses model.data_dim if present"""
    model = DiffusionNet(data_dim=5)
    betas = linear_beta_schedule(num_timesteps=100)
    process = DiffusionProcess(model, betas=betas, num_timesteps=100)
    
    assert process.data_dim == 5
    assert process.num_timesteps == 100

def test_diffusion_process_init_with_explicit_data_dim():
    """Uses explicit data_dim if model doesn't have it"""
    model = torch.nn.Linear(10, 10)  # Custom model without data_dim
    betas = linear_beta_schedule(num_timesteps=100)
    process = DiffusionProcess(model, betas=betas, num_timesteps=100, data_dim=10)
    
    assert process.data_dim == 10

def test_diffusion_process_init_missing_data_dim():
    """Raises if model has no data_dim and none provided"""
    model = torch.nn.Linear(10, 10)
    betas = linear_beta_schedule(num_timesteps=100)
    
    with pytest.raises(AttributeError, match="data_dim"):
        DiffusionProcess(model, betas=betas, num_timesteps=100)

def test_diffusion_process_default_betas():
    """Uses default linear schedule if betas=None"""
    model = DiffusionNet(data_dim=2)
    process = DiffusionProcess(model, num_timesteps=1000)
    
    assert process.betas is not None
    assert process.betas.shape == (1000,)

def test_diffusion_process_schedules_computed():
    """All schedule tensors are computed and on correct device"""
    model = DiffusionNet(data_dim=2)
    betas = linear_beta_schedule(num_timesteps=100)
    process = DiffusionProcess(model, betas=betas, num_timesteps=100)
    
    assert process.betas.shape == (100,)
    assert process.alphas.shape == (100,)
    assert process.alphas_cumprod.shape == (100,)
    assert process.alphas_cumprod_prev.shape == (100,)
    assert process.sqrt_alphas_cumprod.shape == (100,)
    assert process.sqrt_one_minus_alphas_cumprod.shape == (100,)
    
    # All should be on same device as model
    device = next(model.parameters()).device
    assert process.betas.device == device
    assert process.alphas.device == device

def test_diffusion_process_custom_schedule():
    """Works with custom beta schedule"""
    model = DiffusionNet(data_dim=2)
    betas = cosine_beta_schedule(num_timesteps=500)
    process = DiffusionProcess(model, betas=betas, num_timesteps=500)
    
    assert torch.allclose(process.betas, betas.to(process.device))

# ------------------------------
# DiffusionProcess.train() tests
# ------------------------------

def test_diffusion_process_train_with_tensor():
    """Train accepts tensor input"""
    model = DiffusionNet(data_dim=2)
    betas = linear_beta_schedule(num_timesteps=100)
    process = DiffusionProcess(model, betas=betas, num_timesteps=100)
    
    X = torch.randn(50, 2)
    process.train(X, epochs=2, batch_size=10, lr=1e-3)
    
    # Should complete without error

def test_diffusion_process_train_with_dataloader():
    """Train accepts DataLoader input"""
    model = DiffusionNet(data_dim=2)
    betas = linear_beta_schedule(num_timesteps=100)
    process = DiffusionProcess(model, betas=betas, num_timesteps=100)
    
    X = torch.randn(50, 2)
    loader = DataLoader(TensorDataset(X), batch_size=10)
    process.train(loader, epochs=2, lr=1e-3)
    
    # Should complete without error

def test_diffusion_process_train_with_conditioning():
    """Train works with conditioning"""
    model = DiffusionNet(data_dim=2, conditional_dim=3)
    betas = linear_beta_schedule(num_timesteps=100)
    process = DiffusionProcess(model, betas=betas, num_timesteps=100)
    
    X = torch.randn(50, 2)
    c = torch.randn(50, 3)
    process.train(X, c=c, epochs=2, batch_size=10, lr=1e-3)
    
    # Should complete without error

def test_diffusion_process_train_dataloader_with_c_warns():
    """Warns if DataLoader provided with c parameter"""
    model = DiffusionNet(data_dim=2)
    betas = linear_beta_schedule(num_timesteps=100)
    process = DiffusionProcess(model, betas=betas, num_timesteps=100)
    
    X = torch.randn(50, 2)
    c = torch.randn(50, 3)
    loader = DataLoader(TensorDataset(X), batch_size=10)
    
    with pytest.warns(UserWarning, match="c parameter is ignored"):
        process.train(loader, c=c, epochs=1, lr=1e-3)

def test_diffusion_process_train_invalid_type_raises():
    """Raises if X is neither tensor nor DataLoader"""
    model = DiffusionNet(data_dim=2)
    betas = linear_beta_schedule(num_timesteps=100)
    process = DiffusionProcess(model, betas=betas, num_timesteps=100)
    
    with pytest.raises(TypeError, match="torch.Tensor or DataLoader"):
        process.train([1, 2, 3], epochs=1)

def test_diffusion_process_train_negative_epochs_raises():
    """Raises if epochs <= 0"""
    model = DiffusionNet(data_dim=2)
    betas = linear_beta_schedule(num_timesteps=100)
    process = DiffusionProcess(model, betas=betas, num_timesteps=100)
    
    X = torch.randn(50, 2)
    
    with pytest.raises(ValueError, match="positive"):
        process.train(X, epochs=0)
    
    with pytest.raises(ValueError, match="positive"):
        process.train(X, epochs=-5)

# ------------------------------
# DiffusionProcess.generate_samples() tests
# ------------------------------

def test_diffusion_process_generate_samples_shape():
    """Generate returns correct shape"""
    model = DiffusionNet(data_dim=3)
    betas = linear_beta_schedule(num_timesteps=50)  # Small for speed
    process = DiffusionProcess(model, betas=betas, num_timesteps=50)
    
    samples = process.generate_samples(num_samples=10)
    assert samples.shape == (10, 3)

def test_diffusion_process_generate_samples_cpu():
    """Generated samples are on CPU"""
    model = DiffusionNet(data_dim=2)
    betas = linear_beta_schedule(num_timesteps=50)
    process = DiffusionProcess(model, betas=betas, num_timesteps=50)
    
    samples = process.generate_samples(num_samples=5)
    assert samples.device == torch.device('cpu')

def test_diffusion_process_generate_with_conditioning():
    """Generate works with conditioning"""
    model = DiffusionNet(data_dim=2, conditional_dim=4)
    betas = linear_beta_schedule(num_timesteps=50)
    process = DiffusionProcess(model, betas=betas, num_timesteps=50)
    
    c = torch.randn(10, 4)
    samples = process.generate_samples(num_samples=10, c=c)
    assert samples.shape == (10, 2)

def test_diffusion_process_generate_conditioning_size_mismatch_raises():
    """Raises if c batch size doesn't match num_samples"""
    model = DiffusionNet(data_dim=2, conditional_dim=3)
    betas = linear_beta_schedule(num_timesteps=50)
    process = DiffusionProcess(model, betas=betas, num_timesteps=50)
    
    c = torch.randn(5, 3)  # 5 samples
    
    with pytest.raises(ValueError, match="length equal"):
        process.generate_samples(num_samples=10, c=c)  # Asking for 10

def test_diffusion_process_generate_no_nans():
    """Generated samples should not contain NaNs"""
    model = DiffusionNet(data_dim=2)
    betas = linear_beta_schedule(num_timesteps=50)
    process = DiffusionProcess(model, betas=betas, num_timesteps=50)
    
    samples = process.generate_samples(num_samples=20)
    assert not torch.any(torch.isnan(samples))

def test_diffusion_process_generate_finite():
    """Generated samples should be finite"""
    model = DiffusionNet(data_dim=2)
    betas = linear_beta_schedule(num_timesteps=50)
    process = DiffusionProcess(model, betas=betas, num_timesteps=50)
    
    samples = process.generate_samples(num_samples=20)
    assert torch.all(torch.isfinite(samples))

# ------------------------------
# Integration tests
# ------------------------------

def test_diffusion_process_train_and_generate():
    """Can train and then generate"""
    model = DiffusionNet(data_dim=2)
    betas = linear_beta_schedule(num_timesteps=50)
    process = DiffusionProcess(model, betas=betas, num_timesteps=50)
    
    # Train on simple data
    X = torch.randn(100, 2)
    process.train(X, epochs=2, batch_size=20, lr=1e-3)
    
    # Generate
    samples = process.generate_samples(num_samples=50)
    assert samples.shape == (50, 2)
    assert torch.all(torch.isfinite(samples))

def test_diffusion_process_conditional_end_to_end():
    """Conditional training and generation work together"""
    model = DiffusionNet(data_dim=2, conditional_dim=3)
    betas = linear_beta_schedule(num_timesteps=50)
    process = DiffusionProcess(model, betas=betas, num_timesteps=50)
    
    # Train
    X = torch.randn(100, 2)
    c_train = torch.randn(100, 3)
    process.train(X, c=c_train, epochs=2, batch_size=20, lr=1e-3)
    
    # Generate with conditioning
    c_gen = torch.randn(30, 3)
    samples = process.generate_samples(num_samples=30, c=c_gen)
    assert samples.shape == (30, 2)

def test_generate_samples_ddim_shape():
    model = DiffusionNet(data_dim=2)
    betas = linear_beta_schedule(num_timesteps=100)
    process = DiffusionProcess(model=model, betas=betas, num_timesteps=100, data_dim=2)

    samples = process.generate_samples_ddim(
        num_samples=10,
        num_inference_steps=20,
        eta=0.0
    )

    assert samples.shape == (10, 2)
