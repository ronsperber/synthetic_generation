import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from .models import Generator, Discriminator

def make_dataloader(X, c=None, batch_size=64, shuffle=True):
    if c is not None:
        dataset = TensorDataset(X, c)
    else:
        dataset = TensorDataset(X)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

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
            z = torch.randn(B, G.noise_dim, device=device)
            x_fake = G(z, c_batch).detach()

            d_real = D(x_real, c_batch)
            d_fake = D(x_fake, c_batch)

            loss_d = (
                criterion(d_real, torch.ones_like(d_real)) +
                criterion(d_fake, torch.zeros_like(d_fake))
            )

            opt_d.zero_grad()
            loss_d.backward()
            opt_d.step()

            # ====================
            # Train Generator
            # ====================
            z = torch.randn(B, G.noise_dim, device=device)
            x_fake = G(z, c_batch)
            d_fake = D(x_fake, c_batch)

            loss_g = criterion(d_fake, torch.ones_like(d_fake))

            opt_g.zero_grad()
            loss_g.backward()
            opt_g.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs} | D: {loss_d.item():.4f} | G: {loss_g.item():.4f}")
