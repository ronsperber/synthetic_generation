"""
module with functions to train a GAN and a WGAN-GP
"""

from typing import Literal, Callable
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from .models import Generator, Discriminator, ImageGenerator, ImageDiscriminator
from .utils import gradient_penalty, cov_penalty, feature_entropy
from synthetic_generation.data_utils import make_dataloader


def train_gan(
    G: Generator | ImageGenerator,
    D: Discriminator | ImageDiscriminator,
    X: torch.Tensor | DataLoader,
    c: torch.Tensor | None = None,
    lr_G: float = 1e-4,
    lr_D: float = 1e-4,
    lambda_fm_1: float = 0,
    lambda_fm_2: float = 0,
    lambda_entropy: float = 0,
    loss : Literal["bce", "bce_with_logits"] = "bce_with_logits",
    batch_size: int = 64,
    epochs: int = 200,
    return_history: bool = False,
    return_configs: bool = False,
    epoch_callback: Callable[[dict], bool] | None = None,
    callback_every: int = 1,
):
    """
    Function to train a GAN
    Parameters
    ----------
    G: Generator
        the generator for the model
    D: Discriminator
        the Discriminator for the model
    X : torch.Tensor | DataLoader
        data that the generator will be trying to emulate
    c : Optional torch.Tensor
        for conditional GAN, the conditional
    lr_G : float
        the learning rate for the Generator
    lr_D : float
        the learning rate for the Discriminator
    lambda_fm_1 : float
        weight on loss/penalty to use for E(fake_features) - E(real_features)
    lambda_fm_2 : float
        weight on loss/penalty to use for second moments
    lambda_entropy : float
        weight on loss/penalty to use for entropy in feature space
    loss : str ('bce' or 'bce_with_logits')
        loss function to be used
    batch_size : int
        when the data is not already a DataLoader, the batch size
        if a DataLoader is passed, this has no effect
    epochs : int
        number of epochs to train over
    save_path: str | None
        when not None, where to save information about the models
    return_history : boolean
        whether or not to return the history of losses
    return_configs: boolean
        whether or not to return the training configs
    epoch_callback : Callable[[dict], bool] | None
        optional callback called every callback_every epochs.
        receives a state dict with keys: epoch, g_loss, d_loss, G, D,
        g_loss_history, d_loss_history.
        return True to stop training early, False to continue.
    callback_every : int
        how often to call epoch_callback (default 1 = every epoch)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G.to(device)
    D.to(device)
    G.train()
    D.train()
    train_configs = {
        "lr_G": lr_G,
        "lr_D": lr_D,
        "lambda_fm_1": lambda_fm_1,
        "lambda_fm_2": lambda_fm_2,
        "lambda_entropy": lambda_entropy,
        "loss": loss,
        "batch_size": batch_size,
        "epochs": epochs
    }
    track_history = return_history or (epoch_callback is not None)
    if track_history:
        G_losses = []
        D_losses = []
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
    for epoch in pbar:
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
            # freeze D parameters during Generator training
            for param in D.parameters():
              param.requires_grad = False
            # generate fake data and get the value from the Discriminator for it
            x_fake = G.generate(B, c_batch)
            # determine if we need inputs for feature matching and/or entropy
            need_f_fake = (lambda_fm_1 > 0) or (lambda_fm_2 > 0) or (lambda_entropy > 0)
            need_f_real = (lambda_fm_1 > 0) or (lambda_fm_2 > 0)
            # if we need feature matching get the output in the feature space from
            # both x_real and x_fake
            if need_f_real:
                _, f_real = D(x_real, c_batch, return_features=True)
            if need_f_fake:
                d_fake, f_fake = D(x_fake, c_batch, return_features=True)
            else:
                d_fake = D(x_fake, c_batch)
            loss_g = criterion(d_fake, torch.ones_like(d_fake))
            if lambda_fm_1 > 0.0:
                mean_penalty = ((f_fake.mean(0) - f_real.mean(0))**2).mean()
                loss_g += lambda_fm_1 * mean_penalty
            if lambda_fm_2 > 0.0:
                cov_pen = cov_penalty(f_fake, f_real)
                loss_g += lambda_fm_2 * cov_pen
            if lambda_entropy > 0.0:
                entropy_penalty = feature_entropy(f_fake)
                loss_g += lambda_entropy * entropy_penalty
            epoch_g_losses.append(loss_g.item())
            opt_g.zero_grad()
            loss_g.backward()
            opt_g.step()
            # Unfreeze D parameters
            for param in D.parameters():
               param.requires_grad = True    
        
        if track_history:
            D_losses.append((epoch, np.mean(epoch_d_losses)))
            G_losses.append((epoch, np.mean(epoch_g_losses)))
        if epoch_callback is not None and epoch % callback_every == 0:
            callback_state = {
                "epoch": epoch,
                "G": G,
                "D": D,
                "g_loss": np.mean(epoch_g_losses),
                "d_loss": np.mean(epoch_d_losses),
                "G_history": G_losses if track_history else None,
                "D_history": D_losses if track_history else None,
            }
            should_stop = epoch_callback(callback_state)
            if should_stop:
                break
        pbar.set_postfix(
            {"D": f"{np.mean(epoch_d_losses):.4f}", "G": f"{np.mean(epoch_g_losses):.4f}"}

        )
    pbar.close()
    if return_history and return_configs:
        return G_losses, D_losses, train_configs
    elif return_history:
        return G_losses, D_losses
    elif return_configs:
        return train_configs
   


def train_wgan_gp(
    G: Generator | ImageGenerator,
    D: Discriminator |ImageDiscriminator,
    X: torch.Tensor | DataLoader,
    c: torch.Tensor | None = None,
    lr_G: float = 1e-4,
    lr_D: float = 1e-4,
    lambda_fm_1: float = 0,
    lambda_fm_2: float = 0,
    lambda_entropy: float = 0,
    batch_size: int = 64,
    epochs: int = 200,
    n_critic: int = 5,
    lambda_gp: float = 10.0,
    return_history : bool = False,
    return_configs: bool = False,
    epoch_callback: Callable[[dict], bool] | None = None,
    callback_every: int = 1,
):
    """
    Function to train a WGAN-GP
    Parameters
    G: Generator
        the Generator for the data
    D: Discriminator
        the Critic for the data
    X : torch.Tensor | DataLoader
        data to train the networks on
    c : Optional torch.Tensor
        when there is a conditional, this represents the conditional for the training data
    lr_G : float
        the learning rate for G
    lr_D : float
        the learning rate for D
    lambda_fm_1 : float
        weight on loss/penalty to use for E(fake_features) - E(real_features)
    lambda_fm_2 : float
        weight on loss/penalty to use for second moments
    lambda_entropy : float
        weight on loss/penalty to use for entropy in the feature space
    batch_size : int
        when the data is not already a DataLoader, the batch size to be used.
        when the data is a DataLoader, this has no effect
    epochs: int
        number of epochs to train
    n_critic : int
        number of passes the critic makes each batch
    lambda_gp : float
        lambda used for the gradient penalty
    return_history : bool
        whether or not to return the history of losses over training
    return_configs : bool
        whether or not to return the training_configs
    epoch_callback : Callable[[dict], bool] | None
        optional callback called every callback_every epochs.
        receives a state dict with keys: epoch, g_loss, d_loss, G, D,
        g_loss_history, d_loss_history.
        return True to stop training early, False to continue.
    callback_every : int
        how often to call epoch_callback (default 1 = every epoch)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    G.to(device)
    D.to(device)
    G.train()
    D.train()
    train_configs = {
        "lr_G": lr_G,
        "lr_D": lr_D,
        "lambda_fm_1": lambda_fm_1,
        "lambda_fm_2": lambda_fm_2,
        "lambda_entropy": lambda_entropy,
        "batch_size": batch_size,
        "epochs": epochs,
        "n_critic": n_critic,
        "lambda_gp": lambda_gp
    }
    track_history = return_history or (epoch_callback is not None)
    if track_history:
        G_losses = []
        D_losses = []
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
    for epoch in pbar:
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
            # first, we freeze D
            for param in D.parameters():
              param.requires_grad = False
            # determine if we need inputs for feature matching
            need_f_real = (lambda_fm_1 > 0.0) or (lambda_fm_2 > 0.0)
            need_f_fake = (lambda_fm_1 > 0.0) or (lambda_fm_2 > 0.0) or (lambda_entropy > 0.0)
            x_fake = G.generate(B, c_batch)
            # if we need feature matching get the output in the feature space from
            # both x_real and x_fake
            if need_f_real:
                _, f_real = D(x_real, c_batch, return_features=True)
            if need_f_fake:
                d_fake, f_fake = D(x_fake, c_batch, return_features=True)
            else:
                d_fake = D(x_fake, c_batch)
            loss_g = -d_fake.mean()
            if lambda_fm_1 > 0.0:
                mean_penalty = ((f_fake.mean(0) - f_real.mean(0))**2).mean()
                loss_g += lambda_fm_1 * mean_penalty
            if lambda_fm_2 > 0.0:
                cov_pen = cov_penalty(f_fake, f_real)
                loss_g += lambda_fm_2 * cov_pen
            if lambda_entropy > 0.0:
                entropy_penalty = feature_entropy(f_fake)
                loss_g += lambda_entropy * entropy_penalty
            epoch_g_losses.append(loss_g.item())
            opt_g.zero_grad()
            loss_g.backward()
            opt_g.step()
            # Unfreeze D parameters
            for param in D.parameters():
               param.requires_grad = True
        if track_history:
            D_losses.append((epoch, np.mean(epoch_d_losses)))
            G_losses.append((epoch, np.mean(epoch_g_losses)))
        if epoch_callback is not None and epoch % callback_every == 0:
            callback_state = {
                "epoch": epoch,
                "G": G,
                "D": D,
                "g_loss": np.mean(epoch_g_losses),
                "d_loss": np.mean(epoch_d_losses),
                "G_history": G_losses if track_history else None,
                "D_history": D_losses if track_history else None,
            }
            should_stop = epoch_callback(callback_state)
            if should_stop:
                break
        pbar.set_postfix(
            {"D": f"{np.mean(epoch_d_losses):.4f}", "G": f"{np.mean(epoch_g_losses):.4f}"}
        )
    pbar.close()
    if return_history and return_configs:
        return G_losses, D_losses, train_configs
    elif return_history:
        return G_losses, D_losses
    elif return_configs:
        return train_configs