import torch
import torch.nn as nn
import torch.nn.functional as F

from models.autoencoder import Decoder


class VAE(nn.Module):
    # kapasitas sama dengan VanillaAE (784-256-64-256-784); decoder dipakai ulang agar perbandingannya adil
    def __init__(self, latent_dim: int = 64, hidden: int = 256, in_dim: int = 784):
        super().__init__()
        self.latent_dim = latent_dim
        self.enc = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU())
        self.fc_mu = nn.Linear(hidden, latent_dim)
        self.fc_logvar = nn.Linear(hidden, latent_dim)
        self.decoder = Decoder(latent_dim, hidden, in_dim)

    def encode(self, x: torch.Tensor):
        h = self.enc(x.flatten(1))
        return self.fc_mu(h), self.fc_logvar(h)          # masing-masing (B, 64)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # z = mu + sigma * eps, eps ~ N(0, I): sampling dipindah ke eps sehingga gradien mengalir ke mu dan logvar
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        return self.decoder(self.reparameterize(mu, logvar)), mu, logvar

    def loss(self, x: torch.Tensor):
        # -ELBO per citra = BCE (jumlah atas piksel) + KL( N(mu, sigma^2) || N(0, I) )
        x_hat, mu, logvar = self(x)
        b = x.size(0)
        recon = F.binary_cross_entropy(x_hat, x, reduction="sum") / b
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum() / b
        return recon + kl, {"recon": recon.detach(), "kl": kl.detach()}

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        mu, _ = self.encode(x)                           # evaluasi deterministik: pakai mean posterior
        return self.decoder(mu)

    @torch.no_grad()
    def sample(self, n: int) -> torch.Tensor:
        z = torch.randn(n, self.latent_dim, device=self.fc_mu.weight.device)   # z ~ N(0, I)
        return self.decoder(z)                           # (n, 1, 28, 28) di [0, 1]
