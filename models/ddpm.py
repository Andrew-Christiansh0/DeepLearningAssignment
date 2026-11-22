import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Citra masuk/keluar di [-1, 1]. Jaringan memprediksi derau epsilon (bukan x0).


class SinusoidalEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)   # (half,)
        ang = t.float()[:, None] * freqs[None, :]                                            # (B, half)
        return torch.cat([ang.sin(), ang.cos()], dim=1)                                      # (B, dim)


class ResBlock(nn.Module):
    def __init__(self, cin: int, cout: int, tdim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, cin)
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.t_proj = nn.Linear(tdim, cout)
        self.norm2 = nn.GroupNorm(8, cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.t_proj(t_emb)[:, :, None, None]         # (B, cout) -> broadcast ke (B, cout, H, W)
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class UNet(nn.Module):
    # 28x28 -> 14x14 -> 7x7 (strided conv) lalu kembali lewat upsample + skip connection
    def __init__(self, base: int = 32):
        super().__init__()
        tdim = base * 4
        self.t_mlp = nn.Sequential(SinusoidalEmb(tdim), nn.Linear(tdim, tdim), nn.SiLU(), nn.Linear(tdim, tdim))
        self.inp = nn.Conv2d(1, base, 3, padding=1)
        self.down1 = ResBlock(base, base, tdim)                           # 28x28, 32 ch
        self.pool1 = nn.Conv2d(base, base * 2, 3, stride=2, padding=1)    # -> 14x14
        self.down2 = ResBlock(base * 2, base * 2, tdim)                   # 14x14, 64 ch
        self.pool2 = nn.Conv2d(base * 2, base * 4, 3, stride=2, padding=1)  # -> 7x7
        self.mid1 = ResBlock(base * 4, base * 4, tdim)                    # 7x7, 128 ch
        self.mid2 = ResBlock(base * 4, base * 4, tdim)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.up2_conv = nn.Conv2d(base * 4, base * 2, 3, padding=1)
        self.dec2 = ResBlock(base * 4, base * 2, tdim)                    # concat(64 + 64) -> 64
        self.up1_conv = nn.Conv2d(base * 2, base, 3, padding=1)
        self.dec1 = ResBlock(base * 2, base, tdim)                        # concat(32 + 32) -> 32
        self.out = nn.Sequential(nn.GroupNorm(8, base), nn.SiLU(), nn.Conv2d(base, 1, 3, padding=1))

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.t_mlp(t)                                             # (B, 128)
        s1 = self.down1(self.inp(x), t_emb)                               # (B, 32, 28, 28)
        s2 = self.down2(self.pool1(s1), t_emb)                            # (B, 64, 14, 14)
        m = self.mid2(self.mid1(self.pool2(s2), t_emb), t_emb)            # (B, 128, 7, 7)
        u2 = self.up2_conv(self.up(m))                                    # (B, 64, 14, 14)
        d2 = self.dec2(torch.cat([u2, s2], dim=1), t_emb)                 # (B, 64, 14, 14)
        u1 = self.up1_conv(self.up(d2))                                   # (B, 32, 28, 28)
        d1 = self.dec1(torch.cat([u1, s1], dim=1), t_emb)                 # (B, 32, 28, 28)
        return self.out(d1)                                               # (B, 1, 28, 28)


class DDPM(nn.Module):
    def __init__(self, T: int = 1000, base: int = 32):
        super().__init__()
        self.T = T
        self.net = UNet(base)
        scale = 1000 / T                                                  # jadwal linear tetap setara untuk T != 1000
        betas = torch.linspace(scale * 1e-4, scale * 0.02, T)
        alphas = 1.0 - betas
        abar = torch.cumprod(alphas, dim=0)                               # alpha_bar_t
        abar_prev = torch.cat([torch.ones(1), abar[:-1]])
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("abar", abar)
        self.register_buffer("post_var", betas * (1 - abar_prev) / (1 - abar))   # varians posterior q(x_{t-1}|x_t, x_0)

    def loss(self, x0: torch.Tensor) -> torch.Tensor:
        t = torch.randint(0, self.T, (x0.size(0),), device=x0.device)
        noise = torch.randn_like(x0)
        a = self.abar[t].view(-1, 1, 1, 1)
        xt = a.sqrt() * x0 + (1 - a).sqrt() * noise                       # q(x_t | x_0) bentuk tertutup
        return F.mse_loss(self.net(xt, t), noise)                         # supervised noise prediction

    @torch.no_grad()
    def sample(self, n: int, record_steps=()):
        # ancestral sampling x_T -> x_0. record_steps: indeks t yang state-nya (input jaringan) disimpan
        dev = self.betas.device
        record = set(record_steps)
        x = torch.randn(n, 1, 28, 28, device=dev)
        snaps = {}
        for i in reversed(range(self.T)):
            if i in record:
                snaps[i] = x.clone()
            t = torch.full((n,), i, device=dev, dtype=torch.long)
            eps = self.net(x, t)
            # mean posterior: (x_t - beta_t / sqrt(1 - abar_t) * eps) / sqrt(alpha_t)
            mean = (x - self.betas[i] / (1 - self.abar[i]).sqrt() * eps) / self.alphas[i].sqrt()
            if i > 0:
                x = mean + self.post_var[i].sqrt() * torch.randn_like(x)
            else:
                x = mean
        return x.clamp(-1, 1), snaps
