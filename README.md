# Synthetic Data Generation

This repository provides tools for generating synthetic data for tabular or low-dimensional datasets using fully connected neural networks.

Supported generative models include:

GANs (vanilla, conditional)

WGAN-GP (standard and conditional)

Diffusion models (standard and conditional)

A key design feature is the Process API, which provides a unified interface for training, generating samples, and saving/loading models.

## Repository Structure
### GANs

- `src/synthetic_generation/gan/models.py`
Fully connected Generator and Discriminator (Critic) classes, optionally conditional.
Includes `OutputHead` for training/inference activations. The Generator exposes `.generate_samples()` to produce synthetic data easily.

- `src/synthetic_generation/gan/training.py`
Training loops for:

  - Standard GANs (`train_gan`)
  - WGAN-GP (`train_wgan_gp`)

Both can return loss histories via return_history=True.

- `src/synthetic_generation/gan/model_saving.py`
Functions to save/load models:

  - `save_gan_checkpoint()`
  - `load_gan_checkpoint()`

- `src/synthetic_generation/gan/process.py`
Wrapper classes for simplified workflows:

  -`BaseGanProcess`: holds model pair and provides `.save()` and `.load()`

  -`GANProcess` / `WGANProcess`: adds `.train()` using the respective training loops, and `.train_save()`
### Diffusion

- `src/synthetic_generation/diffusion/models.py`

  - `DiffusionNet`: core model to predict noise

  - `DiffusionProcess`: wrapper class with `.train()`, `.generate_samples()`, `.generate_samples_ddim()`, `.save()`, `.load_process()`

  - Time embedding options: `SinusoidalTimeEmbedding`, `MLPTimeEmbedding`

  - `BaseMLP` : Base MLP

- `src/synthetic_generation/diffusion/schedules.py`
Beta schedules: `linear_beta_schedule`, `cosine_beta_schedule`

- `src/synthetic_generation/diffusion/sampling.py`
Sampling functions: `q_sample`, `p_sample`, `ddim_sample`

- `src/synthetic_generation/diffusion/model_saving.py`
Functions to save/load a DiffusionProcess:

  - `save_diffusion_checkpoint()`

  - `load_diffusion_checkpoint()`

### Utilities

`src/synthetic_generation/data_utils.py`
General utilities including make_dataloader() for tensor or conditional datasets.

`Notebooks/`
Contains example notebooks for GANs and Diffusion, illustrating vanilla, conditional, and scaled versions.

## Installation
```bash
git clone https://github.com/ronsperber/synthetic_generation.git
cd synthetic_generation
pip install -r requirements.txt
pip install -e .
```

## Example Usage
### GANs

```python
from synthetic_generation.gan.models import Generator, Discriminator
from synthetic_generation.gan.process import GANProcess
import torch

# Example dataset
X_train = torch.randn(1000, 2)

# Create models
G = Generator(noise_dim=2, num_hidden_layers=2, hidden_dims=(128,128), out_dim=2)
D = Discriminator(feature_dim=2, num_hidden_layers=2, hidden_dims=(128,128))

# Create GAN process
process = GANProcess(G=G, D=D)

# Train
process.train(X_train)

# Generate samples
X_fake = process.generate_samples(num_samples=1000)

# Save process
process.save("gan_checkpoint.pt")

# Load process
process_loaded = GANProcess.load("gan_checkpoint.pt")
```

### Diffusion
```python
from synthetic_generation.diffusion.models import DiffusionNet, DiffusionProcess, MLPTimeEmbedding
from synthetic_generation.diffusion.schedules import linear_beta_schedule
import torch

NUM_TIMESTEPS = 1000
NUM_SAMPLES = 5000

X_train = torch.randn(2000, 2)

# Create model
model = DiffusionNet(data_dim=2, num_hidden_layers=4, hidden_dims=(256,256), time_embedding=MLPTimeEmbedding(NUM_TIMESTEPS))

# Create diffusion process
process = DiffusionProcess(
    model=model,
    betas=linear_beta_schedule(NUM_TIMESTEPS),
    num_timesteps=NUM_TIMESTEPS,
    data_dim=2
)

# Train
process.train(X_train)

# Generate samples
X_fake = process.generate_samples(NUM_SAMPLES)

# Save/load
process.save("diffusion_checkpoint.pt")
process_loaded = DiffusionProcess.load_process("diffusion_checkpoint.pt")
```
## Design Notes and Limitations

  - Fully connected networks only (no convolutions)

  - Conditional information is incorporated via feature concatenation

  - Primarily designed for tabular or low-dimensional data

  - Intended for experimentation and demonstration; production use requires additional validation, monitoring, and constraints

## Extensibility

The framework is designed to be modular, making it straightforward to add new generative models (e.g., VAEs) while keeping the same unified Process API for training, sampling, and checkpointing