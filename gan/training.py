import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from .models import Generator, Discriminator
from .utils import make_dataloader, gradient_penalty


def train_gan(
    X: torch.Tensor | DataLoader,
    G: Generator,
    D: Discriminator,
    lr_G: float = 1e-4,
    lr_D: float = 1e-4,
    criterion=nn.BCEWithLogitsLoss(),
    batch_size: int = 64,
    epochs: int = 200,
    c: torch.Tensor | None = None
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G.to(device)
    D.to(device)

    dataloader = (
        X if isinstance(X, DataLoader)
        else make_dataloader(X, c, batch_size=batch_size)
    )

    # sanity checks
    if G.output_dim != D.feature_dim:
        raise ValueError(
            f"G outputs {G.output_dim}, D expects {D.feature_dim}"
        )

    if G.conditional_dim != D.conditional_dim:
        raise ValueError("Conditional dimensions must match")

    opt_g = torch.optim.Adam(G.parameters(), lr=lr_G, betas=(0.5,0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=lr_D, betas=(0.5,0.999))

    for epoch in range(1, epochs + 1):
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

            # ====================
            # Train Discriminator
            # ====================
            x_fake = G.generate(B, c_batch)

            d_real = D(x_real, c_batch)
            d_fake = D(x_fake, c_batch)

            loss_d = (
                criterion(d_real, torch.ones_like(d_real)) +
                criterion(d_fake, torch.zeros_like(d_fake))
            )
            epoch_d_losses.append(loss_d.item())
            opt_d.zero_grad()
            loss_d.backward()
            opt_d.step()

            # ====================
            # Train Generator
            # ====================
            x_fake = G.generate(B, c_batch)
            d_fake = D(x_fake, c_batch)

            loss_g = criterion(d_fake, torch.ones_like(d_fake))
            epoch_g_losses.append(loss_g.item())
            opt_g.zero_grad()
            loss_g.backward()
            opt_g.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs} | D: {np.mean(epoch_d_losses) :.4f} | G: {np.mean(epoch_g_losses):.4f}")

def train_wgan_gp(
    X: torch.Tensor | DataLoader,
    G: Generator,
    D: Discriminator,
    lr_G=1e-4,
    lr_D=1e-4,
    batch_size=64,
    epochs=200,
    c: torch.Tensor | None = None,
    n_critic=5,
    lambda_gp=10.0,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    G.to(device)
    D.to(device)

    dataloader = (
        X if isinstance(X, DataLoader)
        else make_dataloader(X, c, batch_size=batch_size)
    )

    # sanity checks
    if G.output_dim != D.feature_dim:
        raise ValueError("Generator / Critic dimension mismatch")
    if G.conditional_dim != D.conditional_dim:
        raise ValueError("Conditional dimension mismatch")

    opt_g = torch.optim.Adam(G.parameters(), lr=lr_G, betas=(0.0, 0.9))
    opt_d = torch.optim.Adam(D.parameters(), lr=lr_D, betas=(0.0, 0.9))

    for epoch in range(1, epochs + 1):
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
            x_fake = G.generate(B, c_batch)
            loss_g = -D(x_fake, c_batch).mean()
            epoch_g_losses.append(loss_g.item())
            opt_g.zero_grad()
            loss_g.backward()
            opt_g.step()

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch}/{epochs} | "
                f"D loss: {np.mean(epoch_d_losses):.4f} | "
                f"G loss: {np.mean(epoch_g_losses):.4f}"
            )
