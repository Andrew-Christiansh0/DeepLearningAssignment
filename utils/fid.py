import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import Inception_V3_Weights, inception_v3


def fid_from_stats(mu1: np.ndarray, s1: np.ndarray, mu2: np.ndarray, s2: np.ndarray) -> float:
    # FID = |mu1 - mu2|^2 + Tr(S1) + Tr(S2) - 2 Tr( (S1 S2)^(1/2) )
    # Tr((S1 S2)^(1/2)) dihitung stabil lewat eigenvalue dari S1^(1/2) S2 S1^(1/2) yang simetris PSD
    w, v = np.linalg.eigh(s1)
    sq1 = (v * np.sqrt(np.clip(w, 0, None))) @ v.T
    m = sq1 @ s2 @ sq1
    ev = np.linalg.eigvalsh((m + m.T) / 2)
    tr_covmean = np.sqrt(np.clip(ev, 0, None)).sum()
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1) + np.trace(s2) - 2 * tr_covmean)


class InceptionFID:
    # Bobot Inception-v3 ImageNet milik torchvision (diunduh otomatis sekali, ~100 MB).
    # Angkanya konsisten untuk membandingkan antarmodel, tapi TIDAK sebanding dengan FID di paper
    # (paper memakai bobot Inception versi TF).
    def __init__(self, device: torch.device):
        net = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, transform_input=False)
        net.fc = nn.Identity()                       # keluaran = pool feature 2048-d
        self.net = net.eval().to(device)
        self.device = device
        self.mu = None
        self.sigma = None

    @torch.no_grad()
    def features(self, imgs: torch.Tensor, batch: int = 200) -> np.ndarray:
        # imgs: (N, 1, 28, 28) di [0, 1] -> (N, 2048)
        out = []
        for i in range(0, len(imgs), batch):
            x = imgs[i:i + batch].to(self.device).float()
            x = F.interpolate(x.repeat(1, 3, 1, 1), size=(299, 299), mode="bilinear", align_corners=False)
            x = x * 2 - 1                            # transform_input=False -> Inception torchvision mengharapkan [-1, 1]
            out.append(self.net(x).double().cpu())
        return torch.cat(out).numpy()

    def fit_real(self, imgs: torch.Tensor) -> None:
        f = self.features(imgs)
        self.mu, self.sigma = f.mean(0), np.cov(f, rowvar=False)

    def score(self, imgs: torch.Tensor) -> float:
        f = self.features(imgs)
        return fid_from_stats(f.mean(0), np.cov(f, rowvar=False), self.mu, self.sigma)
