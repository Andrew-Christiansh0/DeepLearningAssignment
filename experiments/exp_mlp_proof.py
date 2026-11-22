import argparse
import copy
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from models.autoencoder import Decoder, VanillaAE
from models.mlp_classifier import MLPClassifier
from utils.logger import save_csv, save_json
from utils.metrics import reconstruction_metrics
from utils.seed_and_data import CLASS_NAMES, get_dataloaders, get_device, set_seed
from utils.viz import draw_rows, minmax_per_image, save_rows

LATENT = 64
N_VIS = 10          # satu citra uji per kelas
N_SAMPLES = 20      # sampel prior per model (2 baris x 10)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="data")
    p.add_argument("--out-dir", default="results/mlp_proof")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs-clf", type=int, default=30)
    p.add_argument("--epochs-recon", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-workers", type=int, default=2)
    return p.parse_args()


# ---------- training ----------

@torch.no_grad()
def accuracy(model, loader, device) -> float:
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.numel()
    return correct / total


def train_classifier(model, train_dl, val_dl, epochs, lr, device, log):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_acc, best_state = 0.0, None
    for ep in range(1, epochs + 1):
        model.train()
        t0, run, n = time.time(), 0.0, 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            run += loss.item() * y.size(0)
            n += y.size(0)
        sched.step()
        val_acc = accuracy(model, val_dl, device)
        log.append({"stage": "mlp_clf", "epoch": ep, "train_loss": run / n, "val_acc": val_acc,
                    "epoch_time_s": time.time() - t0})
        print(f"[mlp_clf] ep {ep:02d}/{epochs} loss={run / n:.4f} val_acc={val_acc:.4f}")
        if val_acc > best_acc:
            best_acc, best_state = val_acc, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)


def train_recon(name, module, forward, train_dl, val_dl, epochs, lr, device, log):
    # module = bagian yang dilatih; forward = x -> x_hat (bisa melibatkan modul beku)
    opt = torch.optim.Adam(module.parameters(), lr=lr)
    best_val, best_state = float("inf"), None
    for ep in range(1, epochs + 1):
        module.train()
        t0, run, n = time.time(), 0.0, 0
        for x, _ in train_dl:
            x = x.to(device)
            loss = F.mse_loss(forward(x), x)
            opt.zero_grad()
            loss.backward()
            opt.step()
            run += loss.item() * x.size(0)
            n += x.size(0)
        ep_time = time.time() - t0
        module.eval()
        val_mse = reconstruction_metrics(forward, val_dl, device)["mse"]
        log.append({"stage": name, "epoch": ep, "train_loss": run / n, "val_mse": val_mse, "epoch_time_s": ep_time})
        print(f"[{name}] ep {ep:02d}/{epochs} train_mse={run / n:.5f} val_mse={val_mse:.5f}")
        if val_mse < best_val:
            best_val, best_state = val_mse, copy.deepcopy(module.state_dict())
    module.load_state_dict(best_state)


# ---------- proyeksi balik bobot (tanpa training) ----------

@torch.no_grad()
def project_from_class(clf: MLPClassifier, device) -> torch.Tensor:
    # Tahap 2: x_hat = sigmoid(W1^T relu(W2^T relu(W3^T y_c))), bias diabaikan sesuai rumus panduan
    # weight nn.Linear shape (out, in): W1 (256,784), W2 (64,256), W3 (10,64)
    # W^T v  ==  v @ W untuk v berbentuk baris (B, out)
    W1, W2, W3 = clf.fc1.weight, clf.fc2.weight, clf.fc3.weight
    y = torch.eye(10, device=device)                 # one-hot tiap kelas (10, 10)
    z_hat = F.relu(y @ W3)                           # (10,10) @ (10,64)  -> (10, 64)
    h_hat = F.relu(z_hat @ W2)                       # (10,64) @ (64,256) -> (10, 256)
    return torch.sigmoid(h_hat @ W1).view(-1, 1, 28, 28)   # (10,256) @ (256,784) -> (10, 784)


@torch.no_grad()
def project_from_latent(clf: MLPClassifier, z: torch.Tensor) -> torch.Tensor:
    # Tahap 3: x_hat = sigmoid(W1^T relu(W2^T z_MLP)), z: (B, 64)
    W1, W2 = clf.fc1.weight, clf.fc2.weight
    h_hat = F.relu(z @ W2)                           # (B,64) @ (64,256) -> (B, 256)
    return torch.sigmoid(h_hat @ W1).view(-1, 1, 28, 28)   # (B,256) @ (256,784) -> (B, 784)


# ---------- helper ----------

def frozen_decode(clf: MLPClassifier, dec: Decoder):
    def fwd(x):
        with torch.no_grad():
            z = clf.features(x)                      # z_MLP beku, (B, 64)
        return dec(z)
    return fwd


@torch.no_grad()
def collect_latents(encode, loader, device) -> torch.Tensor:
    return torch.cat([encode(x.to(device)) for x, _ in loader])      # (N, 64)


def mean_image(loader, device) -> torch.Tensor:
    total, n = 0.0, 0
    for x, _ in loader:
        total = total + x.sum(0, keepdim=True).to(device)
        n += x.size(0)
    return total / n                                 # (1, 1, 28, 28)


def first_per_class(loader, device) -> torch.Tensor:
    found = {}
    for x, y in loader:
        for xi, yi in zip(x, y.tolist()):
            found.setdefault(yi, xi)
        if len(found) == 10:
            break
    return torch.stack([found[c] for c in range(10)]).to(device)     # (10, 1, 28, 28)


def main():
    args = parse_args()
    device = get_device()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    set_seed()
    train_dl, val_dl, test_dl = get_dataloaders(args.data_root, args.batch_size, args.num_workers)
    print(f"device={device} train={len(train_dl.dataset)} val={len(val_dl.dataset)} test={len(test_dl.dataset)}")

    log, res = [], {}

    # Tahap 1: latih MLP classifier, target test acc >= 88%
    set_seed()
    clf = MLPClassifier().to(device)
    train_classifier(clf, train_dl, val_dl, args.epochs_clf, args.lr, device, log)
    test_acc = accuracy(clf, test_dl, device)
    res["stage1_mlp_classifier"] = {"test_acc": test_acc, "target_88_met": test_acc >= 0.88}
    print(f"\n[tahap 1] test acc = {test_acc:.4f}")
    if test_acc < 0.88:
        print("PERINGATAN: test acc < 88%, naikkan --epochs-clf lalu jalankan ulang")
    torch.save(clf.state_dict(), out / "mlp_classifier.pt")

    clf.eval()
    for p in clf.parameters():
        p.requires_grad_(False)                      # bekukan W1, W2, W3 (dan bias)

    x_vis = first_per_class(test_dl, device)         # (10, 1, 28, 28)

    # Tahap 2: W^T dari one-hot kelas
    proj_cls = project_from_class(clf, device)
    res["stage2_class_projection"] = {
        "raw_min": proj_cls.min().item(), "raw_max": proj_cls.max().item(), "raw_mean": proj_cls.mean().item(),
    }

    # Tahap 3: W^T dari z_MLP citra uji + pembanding mean-image
    wt_forward = lambda x: project_from_latent(clf, clf.features(x))
    mimg = mean_image(train_dl, device)
    res["stage3_latent_projection"] = {
        "wt_projection": reconstruction_metrics(wt_forward, test_dl, device),
        "mean_image_baseline": reconstruction_metrics(lambda x: mimg.expand_as(x), test_dl, device),
    }
    print(f"[tahap 3] {res['stage3_latent_projection']}")

    # Tahap 4: frozen latent decoder vs Vanilla AE (latent sama = 64)
    set_seed()
    dec = Decoder(LATENT).to(device)
    train_recon("frozen_decoder", dec, frozen_decode(clf, dec), train_dl, val_dl, args.epochs_recon, args.lr, device, log)
    set_seed()
    ae = VanillaAE(LATENT).to(device)
    train_recon("vanilla_ae", ae, ae, train_dl, val_dl, args.epochs_recon, args.lr, device, log)
    res["stage4_frozen_decoding"] = {
        "frozen_decoder_z_mlp": reconstruction_metrics(frozen_decode(clf, dec), test_dl, device),
        "vanilla_ae_z64": reconstruction_metrics(ae, test_dl, device),
    }
    print(f"[tahap 4] {res['stage4_frozen_decoding']}")
    torch.save(dec.state_dict(), out / "frozen_decoder.pt")
    torch.save(ae.state_dict(), out / "vanilla_ae.pt")

    # Tahap 5: sampling z ~ N(mu_z, diag(sigma_z^2)); AE ikut diuji sebagai kontrol
    set_seed()
    dec.eval()
    ae.eval()
    z_mlp = collect_latents(clf.features, test_dl, device)           # (N, 64)
    mu, sigma = z_mlp.mean(0), z_mlp.std(0)                          # (64,)
    z_rand = mu + sigma * torch.randn(N_SAMPLES, LATENT, device=device)   # (20, 64)

    z_ae = collect_latents(ae.encode, test_dl, device)
    z_rand_ae = z_ae.mean(0) + z_ae.std(0) * torch.randn(N_SAMPLES, LATENT, device=device)

    with torch.no_grad():
        samples_mlp = dec(z_rand)
        samples_ae = ae.decode(z_rand_ae)
        rec_wt = project_from_latent(clf, clf.features(x_vis))
        rec_frozen = dec(clf.features(x_vis))
        rec_ae = ae(x_vis)

    res["stage5_prior_sampling"] = {
        "z_mlp_frac_exact_zero": (z_mlp == 0).float().mean().item(),    # ReLU -> latent non-negatif & jarang
        "z_rand_frac_negative": (z_rand < 0).float().mean().item(),     # sampel Gaussian di luar support latent asli
    }

    # figur: tiap panel disimpan terpisah + satu komposit 4 panel (Figur 3)
    panels = {
        "fig3a_class_projection": (
            r"Tahap 2: proyeksi $W^T$ dari one-hot kelas",
            [proj_cls, minmax_per_image(proj_cls)], ["sigmoid mentah", "kontras diregangkan"], CLASS_NAMES),
        "fig3b_latent_projection": (
            r"Tahap 3: proyeksi $W^T$ dari $z_{MLP}$",
            [x_vis, rec_wt], ["Asli", r"$W^T z_{MLP}$"], None),
        "fig3c_frozen_decoder_vs_ae": (
            "Tahap 4: frozen decoder vs Vanilla AE",
            [x_vis, rec_frozen, rec_ae], ["Asli", r"Dec. beku ($z_{MLP}$)", r"Vanilla AE ($z_{AE}$)"], None),
        "fig3d_prior_sampling": (
            "Tahap 5: sampling dari Gaussian fit pada latent",
            [samples_mlp[:10], samples_mlp[10:], samples_ae[:10], samples_ae[10:]],
            [r"$z_{MLP}$", "", r"$z_{AE}$ (kontrol)", ""], None),
    }
    for key, (title, rows, labels, cols) in panels.items():
        save_rows(out / f"{key}.png", rows, labels, title, cols)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, (title, rows, labels, cols) in zip(axes.flat, panels.values()):
        draw_rows(ax, rows, labels, title, cols)
    fig.tight_layout()
    fig.savefig(out / "fig3_mlp_proof.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    save_json(res, out / "metrics.json")
    save_csv(log, out / "train_log.csv")
    print(f"\nselesai. hasil tersimpan di {out}/")


if __name__ == "__main__":
    main()
