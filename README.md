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
  These are fully connected feed-forward networks built from linear layers and nonlinear activation functions, with optional conditional inputs.
  The `Generator` exposes a `.generate()` method that handles noise sampling and device placement automatically

- `gan/training.py`  
  Contains training loops for:
  - standard GANs (`train_gan`)
  - WGAN-GP (`train_wgan_gp`)

- `gan/utils.py`  
  Utility functions, including:
  - `make_dataloader()` for wrapping tensors into a `DataLoader`
  - `gradient_penalty()` for WGAN-GP

- `Notebooks`   
  Sample notebooks illustrating
  - Vanilla GAN
  - Conditional GAN
  - WGAN-GP
  - WGAN-GP with conditional

---

## Installation

Install the required dependencies with:

```bash
pip install -r requirements.txt
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
train_gan(
    X=X_torch,
    G=G,
    D=D
)

# create a fake data set
X_fake = G.generate(10000)
```
### Training a conditional GAN
The data is as we saa in the previous example (`X, centers,labels`)

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

train_gan(
    X=X_torch,
    G=G_cond,
    D=D_cond,
    c=c
)

# generate fake data from this
X_fake_cond = G.generate(20000, c)
```

### WGAN-GP notes

To train a WGAN-GP use `train_wgan_gp() instead of `train_gan()`

For best results, it is recommended to:
* scale data to be in the range [-1,1]
* use a `tanh` output activation for the Generator
```python
import torch.nn as nn

G = Generator(
    noise_dim=2,
    num_hidden_layers=2,
    out_dim=2,
    hidden_dims=(128, 128),
    output_activation=nn.Tanh
)
```

## Design Notes and Limitations
* All models use fully connected layers, convolutional layers are out of scope
* Conditional information is incorporated via feature concatenation
* While WGAN-GP often improves stability, it may struggle on certain geometric toy datasets; conditional GANs can perform better in those cases
* This repository is designed for experimentation and demonstration of GAN-based synthetic data generation, particularly for tabular data. While not a full production framework, the architecture and training loops are intentionally structured to be extensible toward production use with additional validation, monitoring, and data-specific constraints.

