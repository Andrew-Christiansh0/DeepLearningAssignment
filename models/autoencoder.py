import torch
import torch.nn as nn
import torch.nn.functional as F


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

    def hidden(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder[:2](x.flatten(1))    # aktivasi ReLU hidden layer: (B, 256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


class SparseAE(VanillaAE):
    # arsitektur identik dengan VanillaAE (784-256-64-256-784); bedanya hanya penalti L1 pada hidden layer 256
    def forward_with_hidden(self, x: torch.Tensor):
        h = self.hidden(x)                   # (B, 256), keluaran ReLU -> target penalti sparsity
        z = self.encoder[2](h)               # (B, 64) bottleneck
        return self.decode(z), h

    def loss(self, x: torch.Tensor, lam: float = 1e-3):
        # L = MSE(x, x_hat) + lam * mean|h|  (rata-rata atas batch dan 256 unit)
        x_hat, h = self.forward_with_hidden(x)
        mse = F.mse_loss(x_hat, x)
        l1 = h.abs().mean()
        return mse + lam * l1, {"mse": mse.detach(), "l1": l1.detach()}
