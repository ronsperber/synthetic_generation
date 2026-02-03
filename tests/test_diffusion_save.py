import torch
import pytest
from synthetic_generation.diffusion.models import DiffusionNet, DiffusionProcess, MLPTimeEmbedding, SinusoidalTimeEmbedding
from synthetic_generation.diffusion.schedules import linear_beta_schedule
from synthetic_generation.diffusion.model_saving import save_diffusion_checkpoint, load_diffusion_checkpoint

def test_diffusion_save_load_no_conditional(tmp_path):
    """Test save/load with no conditional and default time embedding."""
    model = DiffusionNet(
        data_dim=4,
        num_hidden_layers=2,
        hidden_dims=(16,16),
        time_embedding=MLPTimeEmbedding(num_timesteps=10)
    )
    process = DiffusionProcess(
        model=model,
        betas=linear_beta_schedule(num_timesteps=10),
        num_timesteps=10,
        data_dim=4
    )

    save_path = tmp_path / "checkpoint.pt"
    save_diffusion_checkpoint(process, save_path)
    loaded_process, _ = load_diffusion_checkpoint(save_path)

    # Forward pass check
    X = torch.randn(2,4)
    t = torch.tensor([0,1])
    with torch.no_grad():
        out_orig = process.model(X, t)
        out_loaded = loaded_process.model(X, t)
    assert torch.allclose(out_orig, out_loaded, atol=1e-6)


def test_diffusion_save_load_identity_conditional(tmp_path):
    """Test save/load with nn.Identity conditional embedding."""
    time_emb = MLPTimeEmbedding(num_timesteps=10)
    model = DiffusionNet(
        data_dim=4,
        conditional_dim=3,
        conditional_embedding=torch.nn.Identity(),
        time_embedding=time_emb,
        num_hidden_layers=2,
        hidden_dims=(16,16)
    )
    process = DiffusionProcess(
        model=model,
        betas=linear_beta_schedule(num_timesteps=10),
        num_timesteps=10,
        data_dim=4
    )
    save_path = tmp_path / "checkpoint_identity.pt"
    save_diffusion_checkpoint(process, save_path)
    loaded_process, _ = load_diffusion_checkpoint(save_path)

    X = torch.randn(2,4)
    c = torch.randn(2,3)
    t = torch.tensor([0,1])
    with torch.no_grad():
        out_orig = process.model(X, t, c)
        out_loaded = loaded_process.model(X, t, c)
    assert torch.allclose(out_orig, out_loaded, atol=1e-6)


def test_diffusion_save_load_sinusoidal_time(tmp_path):
    """Test save/load with a custom SinusoidalTimeEmbedding."""
    time_emb = SinusoidalTimeEmbedding(embedding_dim=8)
    model = DiffusionNet(
        data_dim=4,
        time_embedding=time_emb,
        num_hidden_layers=2,
        hidden_dims=(16,16)
    )
    process = DiffusionProcess(
        model=model,
        betas=linear_beta_schedule(num_timesteps=10),
        num_timesteps=10,
        data_dim=4
    )
    save_path = tmp_path / "checkpoint_sinusoidal.pt"
    save_diffusion_checkpoint(process, save_path)
    loaded_process, _ = load_diffusion_checkpoint(save_path)

    X = torch.randn(2,4)
    t = torch.tensor([0,1])
    with torch.no_grad():
        out_orig = process.model(X, t)
        out_loaded = loaded_process.model(X, t)
    assert torch.allclose(out_orig, out_loaded, atol=1e-6)


def test_diffusion_save_load(tmp_path):
  
    # Time embedding
    time_emb = MLPTimeEmbedding(num_timesteps=10)

    # Model with a conditional embedding
    cond_dim = 3
    cond_emb = torch.nn.Embedding(num_embeddings=5, embedding_dim=cond_dim)
    model = DiffusionNet(
        data_dim=4,
        conditional_dim=cond_dim,
        conditional_embedding=cond_emb,
        time_embedding=time_emb,
        num_hidden_layers=2,
        hidden_dims=(16,16)
    )

    # Diffusion process
    process = DiffusionProcess(
        model=model,
        betas=linear_beta_schedule(num_timesteps=10),
        num_timesteps=10,
        data_dim=4
    )

    # Save checkpoint
    save_path = tmp_path / "checkpoint.pt"
    save_diffusion_checkpoint(process, save_path)

    # Load checkpoint
    loaded_process, config = load_diffusion_checkpoint(save_path)

    # Check basic properties
    assert isinstance(loaded_process, DiffusionProcess)
    assert isinstance(loaded_process.model, DiffusionNet)
    assert loaded_process.model.time_embedding.__class__ == time_emb.__class__
    assert isinstance(loaded_process.model.conditional_embedding, torch.nn.Embedding)

    # Forward pass consistency (optional)
    X_sample = torch.randn(2,4)
    with torch.no_grad():
        orig_out = process.model(X_sample, torch.tensor([0,1]), torch.tensor([0,1]))
        loaded_out = loaded_process.model(X_sample, torch.tensor([0,1]), torch.tensor([0,1]))
    assert torch.allclose(orig_out, loaded_out, atol=1e-6)
