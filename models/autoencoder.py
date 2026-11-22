import torch
import torch.nn as nn


class Decoder(nn.Module):
    # dipakai juga sebagai frozen decoder di eksperimen MLP, supaya arsitekturnya identik dengan decoder AE
    def __init__(self, latent_dim: int = 64, hidden: int = 256, out_dim: int = 784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim), nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).view(-1, 1, 28, 28)


class VanillaAE(nn.Module):
    def __init__(self, latent_dim: int = 64, hidden: int = 256, in_dim: int = 784):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, latent_dim),   # latent linear (tanpa ReLU)
        )
        self.decoder = Decoder(latent_dim, hidden, in_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x.flatten(1))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))
