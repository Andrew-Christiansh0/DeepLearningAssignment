from typing import Callable, Dict

import torch
import torch.nn.functional as F


def _gaussian_kernel(size: int, sigma: float, channels: int, ref: torch.Tensor) -> torch.Tensor:
    coords = torch.arange(size, dtype=ref.dtype, device=ref.device) - size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    k2d = g[:, None] * g[None, :]                                # (size, size)
    return k2d.expand(channels, 1, size, size).contiguous()      # depthwise conv: (C, 1, size, size)


def ssim(x: torch.Tensor, y: torch.Tensor, win: int = 11, sigma: float = 1.5, max_val: float = 1.0) -> torch.Tensor:
    # SSIM per citra, jendela gaussian 11x11 sigma 1.5 (Wang et al., 2004). x, y: (B, C, H, W) -> (B,)
    c = x.shape[1]
    k = _gaussian_kernel(win, sigma, c, x)
    mu_x = F.conv2d(x, k, groups=c)
    mu_y = F.conv2d(y, k, groups=c)
    var_x = F.conv2d(x * x, k, groups=c) - mu_x**2
    var_y = F.conv2d(y * y, k, groups=c) - mu_y**2
    cov = F.conv2d(x * y, k, groups=c) - mu_x * mu_y
    c1, c2 = (0.01 * max_val) ** 2, (0.03 * max_val) ** 2
    s = ((2 * mu_x * mu_y + c1) * (2 * cov + c2)) / ((mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2))
    return s.mean(dim=(1, 2, 3))


@torch.no_grad()
def reconstruction_metrics(
    forward: Callable[[torch.Tensor], torch.Tensor],
    loader,
    device: torch.device,
) -> Dict[str, float]:
    # forward: (B, 1, 28, 28) di [0, 1] -> rekonstruksi dengan shape sama
    # PSNR dihitung per citra lalu dirata-rata (bukan dari MSE rata-rata)
    tot = {"mse": 0.0, "psnr": 0.0, "ssim": 0.0}
    n = 0
    for x, _ in loader:
        x = x.to(device)
        x_hat = forward(x).clamp(0, 1)
        mse_i = ((x - x_hat) ** 2).flatten(1).mean(1)            # (B,)
        tot["mse"] += mse_i.sum().item()
        tot["psnr"] += (10 * torch.log10(1.0 / mse_i.clamp_min(1e-10))).sum().item()
        tot["ssim"] += ssim(x, x_hat).sum().item()
        n += x.size(0)
    return {k: v / n for k, v in tot.items()}
