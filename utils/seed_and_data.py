import random
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, random_split
from torchvision import datasets, transforms

SEED = 42
CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _seed_worker(worker_id: int) -> None:
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)


def get_dataloaders(
    root: str = "data",
    batch_size: int = 128,
    num_workers: int = 2,
    seed: int = SEED,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
   
    tfm = transforms.ToTensor()
    train_raw = datasets.FashionMNIST(root, train=True, download=True, transform=tfm)
    test_raw = datasets.FashionMNIST(root, train=False, download=True, transform=tfm)

    
    full = ConcatDataset([train_raw, test_raw])
    n = len(full)
    n_train, n_val = n * 8 // 10, n // 10
    n_test = n - n_train - n_val
    train_ds, val_ds, test_ds = random_split(
        full, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(seed)
    )

    kw = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=_seed_worker,
    )
    train_dl = DataLoader(train_ds, shuffle=True, generator=torch.Generator().manual_seed(seed), **kw)
    val_dl = DataLoader(val_ds, shuffle=False, **kw)
    test_dl = DataLoader(test_ds, shuffle=False, **kw)
    return train_dl, val_dl, test_dl


if __name__ == "__main__":
    set_seed()
    tr, va, te = get_dataloaders()
    print(f"train={len(tr.dataset)} val={len(va.dataset)} test={len(te.dataset)}")
    x, y = next(iter(tr))
    print(x.shape, x.min().item(), x.max().item(), y[:10].tolist())
