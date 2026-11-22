import torch
import torch.nn as nn

# Semua generator keluar di [-1, 1] (Tanh) dengan shape (B, 1, 28, 28);
# semua discriminator menerima citra [-1, 1] dan mengeluarkan logit (B,).


def weights_init(m: nn.Module) -> None:
    name = m.__class__.__name__
    if "Conv" in name:                       # Conv2d dan ConvTranspose2d
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif "BatchNorm" in name:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)


class MLPGenerator(nn.Module):
    def __init__(self, z_dim: int = 100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 512), nn.LeakyReLU(0.2),
            nn.Linear(512, 1024), nn.LeakyReLU(0.2),
            nn.Linear(1024, 784), nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).view(-1, 1, 28, 28)


class MLPDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 1024), nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(1024, 512), nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.flatten(1)).view(-1)


class DCGANGenerator(nn.Module):
    # ConvTranspose2d: out = (in - 1) * stride - 2 * pad + k
    def __init__(self, z_dim: int = 100):
        super().__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 256, 7, 1, 0, bias=False), nn.BatchNorm2d(256), nn.ReLU(True),   # 1x1 -> 7x7
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(True),     # 7 -> 14
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(True),       # 14 -> 28
            nn.ConvTranspose2d(64, 1, 3, 1, 1), nn.Tanh(),                                             # 28 -> 28
        )
        self.apply(weights_init)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z.view(-1, self.z_dim, 1, 1))    # (B, 100) -> (B, 100, 1, 1) -> (B, 1, 28, 28)


class DCGANDiscriminator(nn.Module):
    # Conv2d: out = floor((in + 2 * pad - k) / stride) + 1
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, True),                            # 28 -> 14
            nn.Conv2d(64, 128, 4, 2, 1, bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, True),     # 14 -> 7
            nn.Conv2d(128, 256, 3, 2, 1, bias=False), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, True),    # 7 -> 4
            nn.Conv2d(256, 1, 4, 1, 0),                                                                # 4 -> 1
        )
        self.apply(weights_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1)
