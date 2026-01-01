# GANs for Synthetic Data Generation

This repository contains code to create GANs (conditional or not) and WGAN-GP (conditional or not) to create synthetic data learning the distribution from an existing data set.

## Contents of the repository

`gan/models.py` : contains classes for a Generator and a Discriminator (or Critic). These are simply feed forward networks with linear layers and non-linear activation functions

`gan/training.py` : contains functions to train both a regular GAN (`train_gan`) and a WGAN-GP (`train_gan_wp()`)

`gan/utils.py` : contains utility functions. These include a function to make a dataloader (`make_dataloader()`) and the function used to compute the gradient penalty for WGAN-GP (`gradient_penalty()`)

## Usage

To get the required libraries installed

`pip install -r requirements.txt`



