# GANs for Synthetic Data Generation

This repository contains implementations of:
- vanilla GANs
- conditional GANs
- WGAN-GP
- conditional WGAN-GP

The focus is on **synthetic data generation for low-dimensional or tabular datasets**, using fully connected neural networks.  
This is not an image-focused GAN framework.

---

## Repository Structure

- `gan/models.py`  
  Contains classes for a **Generator** and **Discriminator (or Critic)**.  
  These are fully connected feed-forward networks built from linear layers and nonlinear activation functions, with optional conditional inputs. There is also an OutputHead dataclass with dims, activation, decode, and name.
  For the output head, `activation` is used during training, and `decode` used at inference.
  The `Generator` exposes a `.generate()` method that handles noise sampling and device placement automatically

- `gan/training.py`  
  Contains training loops for:
  - standard GANs (`train_gan`)
  - WGAN-GP (`train_wgan_gp`)
  These both return loss histories for the G and D by default, and it can be turned off with `return_history=False`

- `gan/utils.py`  
  Utility functions, including:
  - `make_dataloader` for wrapping tensors into a `DataLoader`
  - `gradient_penalty` for WGAN-GP
  - `cov_matrix` and `cov_penalty` for use for feature matching penalties
  - `load_gan_checkpoint` to load a saved model

- `Notebooks`   
  Sample notebooks illustrating
  - Vanilla GAN
  - Conditional GAN
  - WGAN-GP
  - WGAN-GP with conditional

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

## Example Usage

### Training a vanilla GAN
```python
# load libraries
from sklearn.datasets import make_blobs
import torch
from gan.models import Generator, Discriminator
from gan.training import train_gan

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
G_history, D_history = train_gan(
    X=X_torch,
    G=G,
    D=D
)

# create a fake data set
X_fake = G.generate_sample(10000)
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
c = torch.cat([torch.tensor(center_per_sample), one_hot], dim=1)
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

G_hist, D_hist = train_gan(
    X=X_torch,
    G=G_cond,
    D=D_cond,
    c=c
)

# generate fake data from this
X_fake_cond = G_cond.generate_samples(20000, c)
```
To save the results of training, add a `save_path` argument, e.g.
```python
G_hist, D_hist = train_gan(
  X=X_torch,
  G=G_cond,
  D=D_cond,
  c=c,
  save_path='cond_gan.pt'
)
```

To then load the model, we use the `load_gan_checkpoint` as follows
```python
from gan.utils import load_gan_checkpoint
G,D,configs = load_gan_checkpoint(
  path = 'cond_gan.pt'
)
```
`G` will be a copy of the trained Generator, `D` will be a copy of the trained Discriminator, and `configs` will be a dictionary of configs used in the train function. 
### WGAN-GP notes

To train a WGAN-GP use `train_wgan_gp()` instead of `train_gan()`

For best results, it is recommended to:
* scale data to be in the range [-1,1]
* use a `tanh` output activation for the Generator

To do this we will need a custom output head
```python
import torch.nn as nn
from gan.models import OutputHead
heads = [OutputHead(dim=2,activation=nn.Tanh, decode=nn.Tanh, name="scaled_output")]
G = Generator(
    noise_dim=2,
    num_hidden_layers=2,
    output_heads=heads,
    hidden_dims=(128, 128)
)
```

## Design Notes and Limitations
* All models use fully connected layers, convolutional layers are out of scope
* Conditional information is incorporated via feature concatenation
* While WGAN-GP often improves stability, it may struggle on certain geometric toy datasets; conditional GANs can perform better in those cases
* This repository is designed for experimentation and demonstration of GAN-based synthetic data generation, particularly for tabular data. While not a full production framework, the architecture and training loops are intentionally structured to be extensible toward production use with additional validation, monitoring, and data-specific constraints.
