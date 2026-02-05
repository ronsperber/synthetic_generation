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

  - `BaseGanProcess`: holds model pair and provides `.save()` and `.load()`

  - `GANProcess` / `WGANProcess`: adds `.train()` using the respective training loops, and `.train_save()`
### Diffusion

- `src/synthetic_generation/diffusion/models.py`

  - `DiffusionNet`: core model to predict noise

  - Time embedding options: `SinusoidalTimeEmbedding`, `MLPTimeEmbedding`

  - `BaseMLP` : Base MLP

- `src/synthetic_generation/diffusion/process.py`
  holds `DiffusionProcess`: wrapper class with `.train()`, `.generate_samples()`, `.generate_samples_ddim()`, `.save()`, `.load_process()`

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
from sklearn.datasets import make_blobs
from synthetic_generation.gan.models import Generator, Discriminator
from synthetic_generation.gan.process import GANProcess
import torch

# Example dataset
X, labels, centers = make_blobs(
    n_samples = 20000,
    n_features = 2,
    return_centers = True
)
# convert to a PyTorch tensor
X_train = torch.tensor(X).float()

# Create models
G = Generator(noise_dim=2, num_hidden_layers=2, hidden_dims=(128,128), out_dim=2)
D = Discriminator(feature_dim=2, num_hidden_layers=2, hidden_dims=(128,128))

# Create GAN process
process = GANProcess(G=G, D=D)

# Train and save
process.train_save(path="gan_checkpoint.pt", X=X_train)

# Generate samples
X_fake = process.generate_samples(num_samples=1000)

# Load process
process_loaded = GANProcess.load("gan_checkpoint.pt")

# a conditional version of it
# one hot encode the labels
one_hot = torch.nn.functional.one_hot(torch.tensor(labels), num_classes=3).float()
# assign each sample its cluster center
centers_per_sample = torch.tensor(centers)[labels]
# combine the centers with encoded labels
c = torch.cat([centers_per_sample, one_hot], dim=1).float()
# get the number of conditional dimensions
conditional_dim = c.shape[1]
G_cond = Generator(
    noise_dim = 2,
    num_hidden_layers = 2,
    out_dim = 2,
    hidden_dims=(128,128),
    conditional_dim=conditional_dim
)

D_cond = Discriminator(
    feature_dim=2,
    num_hidden_layers=2,
    hidden_dims=(128,128),
    conditional_dim=conditional_dim
)

#create conditional GAN process
process_cond = GANProcess(G=G_cond, D=D_cond)

# train
process_cond.train(
  X=X_train,
  c=c
)
# create fake data
# note the length of the conditional must be the same as num_samples
X_fake_cond = process_cond.generate_samples(num_samples=1000, c=c[:1000])
```

Note: training and saving can be done separately, e.g.

```python
process.train(X=X_train)
process.save(path="gan_checkpoint.pt")
```
### Diffusion
```python
from sklearn.datasets import make_blobs
from synthetic_generation.diffusion.models import DiffusionNet, MLPTimeEmbedding
from synthetic_generation.diffusion.process import DiffusionProcess
from synthetic_generation.diffusion.schedules import linear_beta_schedule
import torch

NUM_TIMESTEPS = 1000
NUM_SAMPLES = 5000

# Example dataset
X, labels, centers = make_blobs(
    n_samples = 20000,
    n_features = 2,
    return_centers = True
)
# convert to a PyTorch tensor
X_train = torch.tensor(X).float()


# Create model
model = DiffusionNet(data_dim=2, num_hidden_layers=4, hidden_dims=(256,256), time_embedding=MLPTimeEmbedding(NUM_TIMESTEPS))

# Create diffusion process
process = DiffusionProcess(
    model=model,
    betas=linear_beta_schedule(NUM_TIMESTEPS),
    num_timesteps=NUM_TIMESTEPS,
    data_dim=2
)

# Train and save
process.train_save(path="diffusion_checkpoint.pt",X=X_train)


# Alternately, as with GANs can use process.train() and process.save() separately
# Generate samples
X_fake = process.generate_samples(NUM_SAMPLES)

process_loaded = DiffusionProcess.load_process("diffusion_checkpoint.pt")

# As with GANs we can use a conditional
# one hot encode the labels
one_hot = torch.nn.functional.one_hot(torch.tensor(labels), num_classes=3).float()
# assign each sample its cluster center
centers_per_sample = torch.tensor(centers)[labels]
# combine the centers with encoded labels
c = torch.cat([centers_per_sample, one_hot], dim=1).float()
# get the number of conditional dimensions
conditional_dim = c.shape[1]
# create the process
model = DiffusionNet(data_dim=2, conditional_dim=conditional_dim, num_hidden_layers=4,  hidden_dims=(256,256))
process_cond = DiffusionProcess(
  model=model,
  betas=linear_beta_schedule(NUM_TIMESTEPS),
  num_timesteps=NUM_TIMESTEPS,
  data_dim=2
)

# train the model
process_cond.train(
  X=X_train,
  c=c
)

# generate samples
X_fake_cond = process_cond.generate_samples(num_samples=NUM_SAMPLES, c=c[:NUM_SAMPLES])
```
## Advanced Usage

### GANs : Output Heads

```python
from synthetic_generation.gan.models import Generator, OutputHead
import torch.nn as nn

# Output head for scaled/tanh outputs
heads = [OutputHead(dim=2, activation=nn.Tanh, decode=nn.Tanh, name="scaled_output")]

G = Generator(noise_dim=2, num_hidden_layers=2, hidden_dims=(128,128), output_heads=heads, out_dim=2)
D = Discriminator(feature_dim=2, num_hidden_layers=2, hidden_dims=(128,128))

process = GANProcess(G=G, D=D)

# train with feature matching and entropy penalties
train_args = {
    "lambda_fm_1": 0.1,
    "lambda_fm_2": 0.05,
    "lambda_entropy": 0.01,
    "epochs": 200
}

process.train(X_train, **train_args)
```

### Diffusion: Scaling and using cosine schedule:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
X_scaled_tensor = torch.tensor(X_scaled).float()

# Use a diffusion process with a cosine beta schedule
from synthetic_generation.diffusion.schedules import cosine_beta_schedule
process = DiffusionProcess(
    model=model,
    betas=cosine_beta_schedule(NUM_TIMESTEPS),
    num_timesteps=NUM_TIMESTEPS,
    data_dim=2
)

process.train(X_scaled_tensor)

# Generate scaled samples and invert scaling
X_fake_scaled = process.generate_samples(NUM_SAMPLES)
X_fake = scaler.inverse_transform(X_fake_scaled.detach().cpu().numpy())
```

## Design Notes and Limitations

  - Fully connected networks only (no convolutions)

  - Conditional information is incorporated via feature concatenation

  - Primarily designed for tabular or low-dimensional data

  - Intended for experimentation and demonstration; production use requires additional validation, monitoring, and constraints
### Extending DiffusionProcess with a Custom Model

To use a custom model with `DiffusionProcess`, your class should:

- Accept `data_dim`, `conditional_dim`, `conditional_embedding`, `time_embedding`, `num_hidden_layers`, `hidden_dims`, `activation` in its constructor.
- Have the same-named attributes for checkpointing.
- Optional: include `init_args` in `time_embedding` or `conditional_embedding` to enable automatic reconstruction on load.

This ensures `.save()` and `.load_process()` work correctly.
## Extensibility

The framework is designed to be modular, making it straightforward to add new generative models (e.g., VAEs) while keeping the same unified Process API for training, sampling, and checkpointing