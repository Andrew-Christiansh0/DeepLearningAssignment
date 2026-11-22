from pathlib import Path
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import make_grid


def minmax_per_image(x: torch.Tensor) -> torch.Tensor:
    # peregangan kontras per citra, hanya untuk visualisasi (sigmoid sering jenuh)
    flat = x.flatten(1)
    lo = flat.min(1, keepdim=True).values
    hi = flat.max(1, keepdim=True).values
    return ((flat - lo) / (hi - lo).clamp_min(1e-8)).view_as(x)


def montage(rows: List[torch.Tensor], pad: int = 2) -> np.ndarray:
    # tiap elemen rows: (N, 1, H, W), N sama di semua baris -> satu citra (H', W')
    grid = make_grid(torch.cat(rows).detach().cpu().float().clamp(0, 1), nrow=rows[0].shape[0], padding=pad, pad_value=1.0)
    return grid[0].numpy()


def draw_rows(ax, rows: List[torch.Tensor], labels: List[str], title: Optional[str] = None,
              col_labels: Optional[List[str]] = None, pad: int = 2) -> None:
    h, w = rows[0].shape[-2:]
    ax.imshow(montage(rows, pad), cmap="gray", vmin=0, vmax=1)
    ax.set_yticks([pad + i * (h + pad) + h / 2 for i in range(len(rows))])
    ax.set_yticklabels(labels, fontsize=8)
    if col_labels:
        ax.set_xticks([pad + j * (w + pad) + w / 2 for j in range(rows[0].shape[0])])
        ax.set_xticklabels(col_labels, fontsize=6, rotation=45, ha="right")
    else:
        ax.set_xticks([])
    ax.tick_params(length=0)
    if title:
        ax.set_title(title, fontsize=9)


def save_rows(path, rows: List[torch.Tensor], labels: List[str], title: Optional[str] = None,
              col_labels: Optional[List[str]] = None, dpi: int = 300) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(0.9 * rows[0].shape[0] + 1.5, 0.9 * len(rows) + 0.8))
    draw_rows(ax, rows, labels, title, col_labels)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def draw_grid(ax, imgs: torch.Tensor, nrow: int = 4, title: Optional[str] = None, pad: int = 2) -> None:
    # imgs: (N, 1, H, W) di [0, 1] -> kisi nrow kolom
    grid = make_grid(imgs.detach().cpu().float().clamp(0, 1), nrow=nrow, padding=pad, pad_value=1.0)
    ax.imshow(grid[0].numpy(), cmap="gray", vmin=0, vmax=1)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=10)
