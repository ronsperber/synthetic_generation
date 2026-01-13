"""
module with functions to train a GAN and a WGAN-GP
"""

from typing import Literal
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from .models import Generator, Discriminator
from .utils import make_dataloader, gradient_penalty


def train_gan(
    X: torch.Tensor | DataLoader,
    G: Generator,
    D: Discriminator,
    lr_G: float = 1e-4,
    lr_D: float = 1e-4,
    lambda_fm_1: float = 0,
    lambda_fm_2: float = 0,
    loss : Literal["bce", "bce_with_logits"] = "bce_with_logits",
    batch_size: int = 64,
    epochs: int = 200,
    c: torch.Tensor | None = None,
    save_path: str | None = None
):
    """
    Function to train a GAN
    Parameters
    ----------
    X : torch.Tensor | DataLoader
        data that the generator will be trying to emulate
    G: Generator
        the generator for the model
    D: Discriminator
        the Discriminator for the model
    lr_G : float
        the learning rate for the Generator
    lr_D : float
        the learning rate for the Discriminator
    lambda_fm_1 :
        weight on loss/penalty to use for E(fake_features) - E(real_features)
    lambda_fm_2 : 
        weight on loss/penalty to use for E(fake_features**2) - E(real_features**2)
    loss : str ('bce' or 'bce_with_logits')
        loss function to be used
    batch_size : int
        when the data is not already a DataLoader, the batch size
        if a DataLoader is passed, this has no effect
    epochs : int
        number of epochs to train over
    c : Optional torch.Tensor
        for conditional GAN, the conditional
    save_path: str | None
        when not None, where to save information about the models
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G.to(device)
    D.to(device)
    if loss == "bce":
        criterion = nn.BCELoss()
    elif loss == "bce_with_logits":
        criterion = nn.BCEWithLogitsLoss()
    else:
        raise ValueError("Loss must be 'bce' or 'bce_with_logits'")
    # make sure loss matches output from Discriminator
    if loss == "bce" and not D.init_args["use_sigmoid"]:
        raise ValueError("BCELoss requires sigmoid discriminator")
    if loss == "bce_with_logits" and D.init_args["use_sigmoid"]:
        raise ValueError("BCEWithLogitsLoss is incompatible with sigmoid discriminator")
    # if the data is not already in a DataLoader, put it in one
    dataloader = (
        X if isinstance(X, DataLoader) else make_dataloader(X, c, batch_size=batch_size)
    )
    if epochs <= 0:
        raise ValueError("Number of epochs must be positive")
    # verify dimensions that must match
    if G.output_dim != D.feature_dim:
        raise ValueError(f"G outputs {G.output_dim}, D expects {D.feature_dim}")
    if G.conditional_dim != D.conditional_dim:
        raise ValueError("Conditional dimensions must match")
    # set the optimizers
    opt_g = torch.optim.Adam(G.parameters(), lr=lr_G, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=lr_D, betas=(0.5, 0.999))
    pbar = tqdm(range(1, epochs+1), desc = "GAN training")
    for _ in pbar:
        epoch_d_losses = []
        epoch_g_losses = []
        for batch in dataloader:
            if len(batch) == 1:
                x_real = batch[0].to(device)
                c_batch = None
            else:
                x_real, c_batch = batch
                x_real = x_real.to(device)
                c_batch = c_batch.to(device)
            # get the size of the batch. normally the batch size of the dataloader
            # but will be different for last batch
            B = x_real.size(0)

            # ====================
            # Train Discriminator
            # ====================
            # generate fake data
            x_fake = G.generate(B, c_batch)
            # Get the the value from the discriminator for the batch of real data
            # and the fake data
            d_real = D(x_real, c_batch)
            d_fake = D(x_fake, c_batch)

            loss_d = criterion(d_real, torch.ones_like(d_real)) + criterion(
                d_fake, torch.zeros_like(d_fake)
            )
            epoch_d_losses.append(loss_d.item())
            opt_d.zero_grad()
            loss_d.backward()
            opt_d.step()

            # ====================
            # Train Generator
            # ====================
            # generate fake data and get the value from the Discriminator for it
            x_fake = G.generate(B, c_batch)
            # determine if we need inputs for feature matching
            need_fm = (lambda_fm_1 != 0.0) or (lambda_fm_2 != 0.0)
            # if we need feature matching get the output in the feature space from
            # both x_real and x_fake
            if need_fm:
                _, f_real = D(x_real, c_batch, return_features=True)
                d_fake, f_fake = D(x_fake, c_batch, return_features=True)
            else:
                d_fake = D(x_fake, c_batch)
            loss_g = criterion(d_fake, torch.ones_like(d_fake))
            if lambda_fm_1 > 0.0:
                loss_g += lambda_fm_1 * ((f_fake.mean(0) - f_real.mean(0))**2).mean()
            if lambda_fm_2 > 0.0:
                loss_g += lambda_fm_2 * (((f_fake**2).mean(0) -(f_real**2).mean(0))**2).mean()
            epoch_g_losses.append(loss_g.item())
            opt_g.zero_grad()
            loss_g.backward()
            opt_g.step()

        pbar.set_postfix(
            {"D": f"{np.mean(epoch_d_losses):.4f}", "G": f"{np.mean(epoch_g_losses):.4f}"}
        )
    pbar.close()
    if save_path is not None:
        # when saving models, note that D_config and G_config contain activation function classes 
        # e.g. nn.LeakyReLU, so when loading, torch.load will need weights_only=False
        torch.save({
            "training_configs": {
                "epochs": epochs,
                "lr_G": lr_G,
                "lr_D": lr_D,
                "batch_size": batch_size,
                "loss": loss
            },
            "G_config": G.init_args,
            "D_config": D.init_args,
            "G_state_dict": G.state_dict(),
            "D_state_dict": D.state_dict()
        }, save_path)


def train_wgan_gp(
    X: torch.Tensor | DataLoader,
    G: Generator,
    D: Discriminator,
    lr_G: float = 1e-4,
    lr_D: float = 1e-4,
    batch_size: int = 64,
    epochs: int = 200,
    c: torch.Tensor | None = None,
    n_critic: int = 5,
    lambda_gp: float = 10.0,
    save_path : str | None = None
):
    """
    Function to train a WGAN-GP
    Parameters
    X : torch.Tensor | DataLoader
        data to train the networks on
    G: Generator
        the Generator for the data
    D: Discriminator
        the Critic for the data
    lr_G : float
        the learning rate for G
    lr_D : float
        the learning rate for D
    batch_size : int
        when the data is not already a DataLoader, the batch size to be used.
        when the data is a DataLoader, this has no effect
    epochs: int
        number of epochs to train
    c : Optional torch.Tensor
        when there is a condtional, this represents the conditional for the training data
    n_critic : int
        number of passes the critic makes each batch
    lambda_gp : float
        lambda used for the gradient penalty
    save_path : str | None:
        when not None, the path to save model information
    
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    G.to(device)
    D.to(device)
    # if the data is not already in a DataLoader, put it in one
    dataloader = (
        X if isinstance(X, DataLoader) else make_dataloader(X, c, batch_size=batch_size)
    )
    if epochs <= 0:
        raise ValueError("Number of epochs must be positive")
    # validate dimensions
    if G.output_dim != D.feature_dim:
        raise ValueError("Generator / Critic dimension mismatch")
    if G.conditional_dim != D.conditional_dim:
        raise ValueError("Conditional dimension mismatch")
    # set optimizers
    opt_g = torch.optim.Adam(G.parameters(), lr=lr_G, betas=(0.0, 0.9))
    opt_d = torch.optim.Adam(D.parameters(), lr=lr_D, betas=(0.0, 0.9))
    pbar = tqdm(range(1, epochs+1), desc = "WGAN-GP training")
    for _ in pbar:
        epoch_d_losses = []
        epoch_g_losses = []
        for batch in dataloader:
            if len(batch) == 1:
                x_real = batch[0].to(device)
                c_batch = None
            else:
                x_real, c_batch = batch
                x_real = x_real.to(device)
                c_batch = c_batch.to(device)

            B = x_real.size(0)

            # ---------------------
            # Train critic
            # ---------------------
            for _ in range(n_critic):
                # generate fake data and compute the loss function
                x_fake = G.generate(B, c_batch)
                d_real = D(x_real, c_batch).mean()
                d_fake = D(x_fake, c_batch).mean()

                gp = gradient_penalty(D, x_real, x_fake, c_batch, lambda_gp)

                loss_d = d_fake - d_real + gp
                epoch_d_losses.append(loss_d.item())
                opt_d.zero_grad()
                loss_d.backward()
                opt_d.step()

            # ---------------------
            # Train generator
            # ---------------------
            # generate fake data and compute the loss function
            x_fake = G.generate(B, c_batch)
            loss_g = -D(x_fake, c_batch).mean()
            epoch_g_losses.append(loss_g.item())
            opt_g.zero_grad()
            loss_g.backward()
            opt_g.step()

        pbar.set_postfix(
            {"D": f"{np.mean(epoch_d_losses):.4f}", "G": f"{np.mean(epoch_g_losses):.4f}"}
        )
    pbar.close()
    if save_path is not None:
        # when saving models, note that D_config and G_config contain activation function classes 
        # e.g. nn.LeakyReLU, so when loading, torch.load will need weights_only=False
        torch.save(
            {
                "training_configs" : {
                    "epochs": epochs,
                    "lr_G": lr_G,
                    "lr_D": lr_D,
                    "batch_size": batch_size,
                    "n_critic": n_critic,
                    "lambda_gp": lambda_gp
                },
                "G_config":G.init_args,
                "D_config":D.init_args,
                "G_state_dict":G.state_dict(),
                "D_state_dict":D.state_dict()
            },save_path
        )