# Synthetic Data Generation

This repository contains implementations of:
- vanilla GANs
- conditional GANs
- WGAN-GP
- conditional WGAN-GP
- Diffusion
- Diffusion with conditionals (flat or embedded)

The focus is on **synthetic data generation for low-dimensional or tabular datasets**, using fully connected neural networks.  

---

## Repository Structure

- `src/synthetic_generation/gan/models.py`  
  Contains classes for a **Generator** and **Discriminator (or Critic)**.  
  These are fully connected feed-forward networks built from linear layers and nonlinear activation functions, with optional conditional inputs. There is also an OutputHead dataclass with dims, activation, decode, and name.
  For the output head, `activation` is used during training, and `decode` used at inference.
  The `Generator` exposes a `.generate()` method that handles noise sampling and device placement automatically

- `src/synthetic_generation/gan/training.py`  
  Contains training loops for:
  - standard GANs (`train_gan`)
  - WGAN-GP (`train_wgan_gp`)
  These both return loss histories for the G and D by default, and it can be turned off with `return_history=False`

- `src/synthetic_generation/gan/utils.py`  
  GAN/WGAN-GP utility functions, including:
  - `gradient_penalty` for WGAN-GP
  - `cov_matrix` and `cov_penalty` for use for feature matching penalties

- `src/synthetic_generation/gan/model_saving.py`
  GAN/WGAN-GP functions to save and load model information
  - `save_gan_checkpoint` to save information on G,D and any desired configs
  - `load_gan_checkpoint` to create a G, D from saved data

- `src/synthetic_generation/gan/process.py`
  Contains wrapper classes to hold model pairs, train, save, load
  - `BaseGanProcess` contains the model info, has load and save methods common to both GAN and WGAP-GP
  - `GANProcess` specific to GANs with a `train` method using `train_gan`
  - `WGANProcess` specific to WGAN-GPs with a `train` method using `train_wgan_gp`
  

- `src/synthetic_generation/diffusion/models.py`
Diffusion models including
  - SinusoidalTimeEmbedding` : to use sinusoidal embedding for time
  - `BaseMLP` : basic multi-layer perceptron class used for the diffusion model and a time embedding
  - `MLPTimeEmbedding` : a class to use a multi-layer perceptron to embed the time dimension
  - `DiffusionModel` : a class to use for diffusion to predict noise
  - `DiffusionProcess` : a class that holds a model attribute and has methods `train` to train the model on data, `generate_samples` to generate samples using DDPM and `generate_samples_ddim` to generate samples using DDIM

- `src/synthetic_generation/diffusion/schedules.py`
Has functions to generate schedules for use with diffusion including `linear_beta_schedule` and `cosine_beta_schedule`

- `src/synthetic_generation/diffusion/sampling.py`
  Functions for sampling in diffusion:
  - `q_sample` : used to sample `x_t` from `x_0` and t
  - `p_sample` : used to sample `x_{t-1}` from `x_t` for DDPM
  - `ddim_sample` : used to sample for DDIM 

- `src/synthetic_generation/diffuision/model_saving.py`
  Contains functions used to save/load a diffusion process
  - `save_diffusion_checkpoint` : used to save a diffusion process
  - `load_diffusion_checkpoint` : used to load a saved diffusion process

- `src/synthetic_generation/data_utils.py`
  General data utilities, including :
  - `make_dataloader` to take a data set or data set + conditional and turn into a data loader

- `Notebooks/GAN`   
  Sample notebooks illustrating
  - Vanilla GAN
  - Conditional GAN
  - WGAN-GP
  - WGAN-GP with conditional
- `Notebooks\Diffusion
Sample notebooks illustrating
  - Simple basic diffusion 
  - Diffusion with conditional using a flat tensor conditional
  - Diffusion with conditional using a conditional embedding
  - Conditional diffusing using a cosine beta schedule (both unscaled and scaled)

---

## Installation
Clone the repository and change directories to it
```bash
git clone https://github.com/ronsperber/synthetic_generation.git
cd synthetic_generation
```

Install required dependencies and the package in editable mode with:

```bash
pip install -r requirements.txt
pip install -e .
```

## Example Usage for GANs

### Training a vanilla GAN
```python
# load libraries
from sklearn.datasets import make_blobs
import torch
from synthetic_generation.gan.models import Generator, Discriminator
from synthetic_generation.gan.training import train_gan

# create a toy dataset
X, labels, centers = make_blobs(
    n_samples = 20000,
    n_features = 2,
    return_centers = True
)
# convert to a PyTorch tensor
X_torch = torch.tensor(X).float()
 # get a Generator with 2 dimensional noise
G = Generator(
    noise_dim=2,
    num_hidden_layers=2,
    out_dim=2,
    hidden_dims=(128,128)
)
# get a discriminator
D = Discriminator(
    feature_dim=2,
    num_hidden_layers=2,
    hidden_dims=(128,128)
)

# train the models
train_gan(
    X=X_torch,
    G=G,
    D=D
)

# create a fake data set
X_fake = G.generate_samples(10000)
```
### Training a conditional GAN
The data is as we saw in the previous example (`X, centers, labels`)

```python
# Now a conditional GAN on the same data
# one hot encode the labels
one_hot = torch.nn.functional.one_hot(torch.tensor(labels), num_classes=3).float()
# assign each sample its cluster center
centers_per_sample = torch.tensor(centers)[labels]
# combine the centers with encoded labels
c = torch.cat([centers_per_sample, one_hot], dim=1).float()
# get the number of conditional dimensions
conditional_dim = c.shape[1]

# create a Generator/Discriminator pair that will include those dimensions
G_cond = Generator(
    noise_dim = 2,
    num_hidden_layers = 2,
    out_dim = 2,
    hidden_dims=(128,128),
    use_conditional=True,
    conditional_dim=conditional_dim
)

D_cond = Discriminator(
    feature_dim=2,
    num_hidden_layers=2,
    hidden_dims=(128,128),
    use_conditional=True,
    conditional_dim=conditional_dim
)
train_gan(
    X=X_torch,
    G=G_cond,
    D=D_cond,
    c=c
)

# generate fake data from this
X_fake_cond = G_cond.generate_samples(20000, c)
```
To save the results of training, use the `save_gan_checkpoint`
```python
from synthetic_generation.gan.model_saving import save_gan_checkpoint
save_gan_checkpoint(
  path='cond_gan.pt',
  G=G,
  D=D
  train_configs={'epochs':200} # add whatever training configurations you like here
)
```

To then load the model, we use the `load_gan_checkpoint` as follows
```python
from synthetic_generation.gan.model_saving import load_gan_checkpoint
G,D,configs = load_gan_checkpoint(
  path = 'cond_gan.pt'
)
```
`G` will be a copy of the trained Generator, `D` will be a copy of the trained Discriminator, and `configs` will be a dictionary of configs saved. 
### WGAN-GP notes

To train a WGAN-GP use `train_wgan_gp()` instead of `train_gan()`

For best results, it is recommended to:
* scale data to be in the range [-1,1]
* use a `tanh` output activation for the Generator

To do this we will need a custom output head
```python
import torch.nn as nn
from synthetic_generation.gan.models import OutputHead
heads = [OutputHead(dim=2,activation=nn.Tanh, decode=nn.Tanh, name="scaled_output")]
G = Generator(
    noise_dim=2,
    num_hidden_layers=2,
    output_heads=heads,
    hidden_dims=(128, 128)
)
```

## Example usage for Diffusion
```python
# load libraries
from sklearn.datasets import make_blobs
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import StandardScaler
from synthetic_generation.diffusion.models import MLPTimeEmbedding, DiffusionNet, DiffusionProcess, MLPTimeEmbedding
from synthetic_generation.diffusion.schedules import linear_beta_schedule, cosine_beta_schedule

# for all of these we will use 1000 time steps
NUM_TIMESTEPS = 1000
# for all samples we will generate 5000 samples
NUM_SAMPLES = 5000
# create data set
X, labels, centers = make_blobs(
    n_samples = 20000,
    n_features = 2,
    return_centers = True
)
# one hot encode the labels
one_hot = F.one_hot(torch.tensor(labels), num_classes=3).float()
# assign each sample its cluster center
centers_per_sample = torch.tensor(centers)[labels]
# combine the centers with encoded labels
c = torch.cat([centers_per_sample, one_hot], dim=1)
# start with an example with no conditional
# note here, when we don't specify the time embedding, it will default to a simple MLPTimeEmbedding
model = DiffusionNet(
  data_dim=2,
  num_hidden_layers=4,
  hidden_dims=(256, 256),
  time_embedding=MLPTimeEmbedding(num_timesteps=NUM_TIMESTEPS)
)
# convert data to a tensor
X_train = torch.Tensor(X).float()
# create a diffusion process
process = DiffusionProcess(
  model=model,
  betas=linear_beta_schedule(num_timesteps=NUM_TIMESTEPS),
  num_timesteps=NUM_TIMESTEPS,
  data_dim=2
)
# train the data with default settings
process.train(X=X_train)

# generate a sample data set
X_fake = process.generate_samples(
  num_samples=NUM_SAMPLES
)

# now we do a conditional version of things
model_cond = DiffusionNet(
  data_dim=2,
  conditional_dim = c.shape[1],
  num_hidden_layers=4,
  time_embedding=MLPTimeEmbedding(num_timesteps=NUM_TIMESTEPS)
)

process_cond = DiffusionProcess(
  model=model_cond,
  betas=linear_beta_schedule(num_timesteps=NUM_TIMESTEPS),
  num_timesteps=NUM_TIMESTEPS,
  data_dim=2
)
# train the data
process_cond.train(
  X=X_train,
  c=c
)
# if we only want NUM_SAMPLES samples generated, we pick a random subset of c to use of that size
idx = np.random.default_rng(42).choice(len(c), size=NUM_SAMPLES, replace=False)
c_sample = c[idx]

X_fake_cond = process_cond.generate_samples(
  num_samples=NUM_SAMPLES,
  c=c_sample
)
```
Note: If a cosine schedule is to be used, it is **strongly recommended** to scale the data. (All data may work better if scaled, but due to the high betas at the end of the cosine schedule, it can really cause poor results if not scaled)

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
centers_scaled = scaler.transform(centers_per_sample)
c_scaled = torch.cat([torch.tensor(centers_scaled), one_hot], dim=1).float()
X_train_scaled = torch.Tensor(X_scaled)
# make the model and process
model_cond_cos = DiffusionNet(
  data_dim=2,
  conditional_dim = c_scaled.shape[1],
  num_hidden_layers=4,
  time_embedding=MLPTimeEmbedding(num_timesteps=NUM_TIMESTEPS)
)

process_cond_cos = DiffusionProcess(
  model=model_cond,
  betas=cosine_beta_schedule(num_timesteps=NUM_TIMESTEPS),
  num_timesteps=NUM_TIMESTEPS,
  data_dim=2
)

# train on the scaled data
process_cond_cos.train(
  X=X_train_scaled,
  c=c_scaled
)
# get scaled sample c
c_scaled_sample = c_scaled[idx]

# create generated data
X_fake_c_cos = process_cond.generate_samples(
  num_samples=NUM_SAMPLES,
  c=c_scaled_sample
)

# to get it back to the correct scale, we convert back to numpy and use the scaler
X_fake_c_cos_np = X_fake_c_cos.detach().cpu().numpy()
X_fake_c_cos_np_gen = scaler.inverse_transform(X_fake_c_cos_np)
```


## Design Notes and Limitations
* All models use fully connected layers, convolutional layers are out of scope
* Conditional information is incorporated via feature concatenation
* While WGAN-GP often improves stability, it may struggle on certain geometric toy datasets; conditional GANs can perform better in those cases
* This repository is designed for experimentation and demonstration of GAN-based synthetic data generation, particularly for tabular data. While not a full production framework, the architecture and training loops are intentionally structured to be extensible toward production use with additional validation, monitoring, and data-specific constraints.
