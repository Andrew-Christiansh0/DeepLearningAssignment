import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPClassifier(nn.Module):
    # 784 -> 256 -> 64 -> 10. nn.Linear(in, out).weight punya shape (out, in)
    def __init__(self, in_dim: int = 784, h1: int = 256, h2: int = 64, n_classes: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, n_classes)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        h1 = F.relu(self.fc1(x.flatten(1)))
        return F.relu(self.fc2(h1))          # z_MLP: (B, 64), lapisan sebelum klasifikasi akhir

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc3(self.features(x))    # logits (B, 10)
