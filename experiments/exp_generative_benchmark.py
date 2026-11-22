import argparse
import copy
import shutil
import statistics
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from experiments.exp_mlp_proof import first_per_class
from models.autoencoder import SparseAE, VanillaAE
from models.ddpm import DDPM
from models.gan import DCGANDiscriminator, DCGANGenerator, MLPDiscriminator, MLPGenerator
from models.vae import VAE
from utils.fid import InceptionFID
from utils.logger import save_csv, save_json
from utils.metrics import reconstruction_metrics
from utils.seed_and_data import get_dataloaders, get_device, set_seed, to_signed, to_unit
from utils.viz import draw_grid, draw_rows

LATENT = 64                 # AE / Sparse AE / VAE
Z_GAN = 100                 # prior noise GAN / DCGAN
DPI = 300
ALPHAS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
CLS_A, CLS_B = 1, 9         # Trouser vs Ankle boot untuk interpolasi
ORDER = ["Vanilla AE", "Sparse AE", "VAE", "Vanilla GAN", "DCGAN", "DDPM"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="data")
    p.add_argument("--out-dir", default="results/benchmark")
    p.add_argument("--mlp-fig", default="results/mlp_proof/fig3_mlp_proof.png")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs-ae", type=int, default=30)
    p.add_argument("--epochs-gan", type=int, default=40)
    p.add_argument("--epochs-ddpm", type=int, default=40)
    p.add_argument("--lr-ae", type=float, default=1e-3)
    p.add_argument("--lr-gan", type=float, default=2e-4)
    p.add_argument("--lr-ddpm", type=float, default=2e-4)
    p.add_argument("--sparse-lambda", type=float, default=1e-3)
    p.add_argument("--ddpm-T", type=int, default=1000)
    p.add_argument("--n-fid", type=int, default=5000)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--smoke", action="store_true", help="1 epoch, FID 256 sampel: cek pipeline end-to-end")
    return p.parse_args()


# ---------- util ----------

def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def latency_ms(fn, device, warmup=5, repeats=30) -> float:
    for _ in range(warmup):
        fn()
    sync(device)
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    sync(device)
    return (time.perf_counter() - t0) / repeats * 1000


@torch.no_grad()
def ema_update(ema, model, decay=0.999):
    for pe, p in zip(ema.parameters(), model.parameters()):
        pe.lerp_(p, 1 - decay)                       # pe <- decay * pe + (1 - decay) * p


@torch.no_grad()
def evaluate(model, loss_fn, loader, prep, device):
    model.eval()
    acc, n = {}, 0
    for x, _ in loader:
        x = prep(x.to(device))
        loss, parts = loss_fn(x)
        for k, v in {"loss": loss, **parts}.items():
            acc[k] = acc.get(k, 0.0) + v.detach() * x.size(0)
        n += x.size(0)
    return {k: (v / n).item() for k, v in acc.items()}


def fit(name, model, loss_fn, train_dl, val_dl, epochs, lr, device, hist,
        prep=lambda x: x, ema=None, clip=None, keep_best=True):
    # loss_fn(x) -> (loss, {nama: komponen}); prep menyesuaikan rentang input ([0,1] -> [-1,1] untuk DDPM)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best, best_state = float("inf"), None
    for ep in range(1, epochs + 1):
        model.train()
        sync(device)
        t0 = time.time()
        acc, n = {}, 0
        for x, _ in train_dl:
            x = prep(x.to(device))
            loss, parts = loss_fn(x)
            opt.zero_grad()
            loss.backward()
            if clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            if ema is not None:
                ema_update(ema, model)
            for k, v in {"loss": loss, **parts}.items():
                acc[k] = acc.get(k, 0.0) + v.detach() * x.size(0)
            n += x.size(0)
        sync(device)
        row = {"model": name, "epoch": ep, "epoch_time_s": time.time() - t0}
        row.update({f"train_{k}": (v / n).item() for k, v in acc.items()})
        row.update({f"val_{k}": v for k, v in evaluate(model, loss_fn, val_dl, prep, device).items()})
        hist.append(row)
        print(f"[{name}] ep {ep:02d}/{epochs} train={row['train_loss']:.5f} val={row['val_loss']:.5f} ({row['epoch_time_s']:.1f}s)")
        if keep_best and row["val_loss"] < best:
            best, best_state = row["val_loss"], copy.deepcopy(model.state_dict())
    if keep_best:
        model.load_state_dict(best_state)


def train_gan(name, G, D, train_dl, epochs, lr, device, hist):
    optG = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    optD = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    bce = F.binary_cross_entropy_with_logits
    for ep in range(1, epochs + 1):
        G.train()
        D.train()
        sync(device)
        t0 = time.time()
        acc, n = {}, 0
        for x, _ in train_dl:
            real = to_signed(x.to(device))                       # (B, 1, 28, 28) di [-1, 1], cocok dengan Tanh generator
            b = real.size(0)
            ones, zeros = torch.ones(b, device=device), torch.zeros(b, device=device)
            fake = G(torch.randn(b, Z_GAN, device=device))       # (B, 1, 28, 28)

            d_real, d_fake = D(real), D(fake.detach())           # logit (B,)
            d_loss = bce(d_real, ones) + bce(d_fake, zeros)
            optD.zero_grad()
            d_loss.backward()
            optD.step()

            g_loss = bce(D(fake), ones)                          # non-saturating: maksimalkan log D(G(z))
            optG.zero_grad()
            g_loss.backward()
            optG.step()

            stats = {"d_loss": d_loss, "g_loss": g_loss,
                     "d_real": torch.sigmoid(d_real).mean(), "d_fake": torch.sigmoid(d_fake).mean()}
            for k, v in stats.items():
                acc[k] = acc.get(k, 0.0) + v.detach() * b
            n += b
        sync(device)
        row = {"model": name, "epoch": ep, "epoch_time_s": time.time() - t0}
        row.update({k: (v / n).item() for k, v in acc.items()})
        hist.append(row)
        print(f"[{name}] ep {ep:02d}/{epochs} D={row['d_loss']:.3f} G={row['g_loss']:.3f} "
              f"D(x)={row['d_real']:.2f} D(G(z))={row['d_fake']:.2f} ({row['epoch_time_s']:.1f}s)")


@torch.no_grad()
def generate(fn, n, chunk=500) -> torch.Tensor:
    return torch.cat([fn(min(chunk, n - i)).cpu() for i in range(0, n, chunk)])    # (n, 1, 28, 28) di [0, 1]


@torch.no_grad()
def hidden_stats(model, loader, device):
    h = torch.cat([model.hidden(x.to(device)) for x, _ in loader])                 # (N, 256)
    return {"hidden_zero_frac": (h == 0).float().mean().item(), "hidden_mean_act": h.mean().item()}


def last_std(hist, name, key, k=10):
    v = [r[key] for r in hist if r["model"] == name][-k:]
    return statistics.stdev(v) if len(v) > 1 else None


def mean_epoch_time(hist, name):
    return statistics.mean(r["epoch_time_s"] for r in hist if r["model"] == name)


def save_fig(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def print_table(rows):
    f = lambda v: "N/A" if v is None else (f"{v:.4g}" if isinstance(v, float) else str(v))
    cols = list(rows[0].keys())
    print("\n" + " | ".join(cols))
    for r in rows:
        print(" | ".join(f(r[c]) for c in cols))


def main():
    args = parse_args()
    if args.smoke:
        args.epochs_ae = args.epochs_gan = args.epochs_ddpm = 1
        args.n_fid = 256
    device = get_device()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    set_seed()
    train_dl, val_dl, test_dl = get_dataloaders(args.data_root, args.batch_size, args.num_workers)
    print(f"device={device} train={len(train_dl.dataset)} val={len(val_dl.dataset)} test={len(test_dl.dataset)}")
    hist, res = [], {}
    x64 = next(iter(test_dl))[0][:64].to(device)                 # batch tetap (64) untuk ukur latency

    # ---------- training ----------
    set_seed()
    ae = VanillaAE(LATENT).to(device)
    fit("Vanilla AE", ae, lambda x: (F.mse_loss(ae(x), x), {}), train_dl, val_dl, args.epochs_ae, args.lr_ae, device, hist)

    set_seed()
    sae = SparseAE(LATENT).to(device)
    fit("Sparse AE", sae, lambda x: sae.loss(x, args.sparse_lambda), train_dl, val_dl, args.epochs_ae, args.lr_ae, device, hist)

    set_seed()
    vae = VAE(LATENT).to(device)
    fit("VAE", vae, vae.loss, train_dl, val_dl, args.epochs_ae, args.lr_ae, device, hist)

    set_seed()
    g1, d1 = MLPGenerator(Z_GAN).to(device), MLPDiscriminator().to(device)
    train_gan("Vanilla GAN", g1, d1, train_dl, args.epochs_gan, args.lr_gan, device, hist)

    set_seed()
    g2, d2 = DCGANGenerator(Z_GAN).to(device), DCGANDiscriminator().to(device)
    train_gan("DCGAN", g2, d2, train_dl, args.epochs_gan, args.lr_gan, device, hist)

    set_seed()
    ddpm = DDPM(T=args.ddpm_T).to(device)
    ema = copy.deepcopy(ddpm).requires_grad_(False)              # bobot EMA dipakai untuk sampling
    fit("DDPM", ddpm, lambda x: (ddpm.loss(x), {}), train_dl, val_dl, args.epochs_ddpm, args.lr_ddpm, device, hist,
        prep=to_signed, ema=ema, clip=1.0, keep_best=False)

    for m in (ae, sae, vae, g1, d1, g2, d2, ddpm, ema):
        m.eval()
    torch.save({"vanilla_ae": ae.state_dict(), "sparse_ae": sae.state_dict(), "vae": vae.state_dict(),
                "gan_g": g1.state_dict(), "dcgan_g": g2.state_dict(), "ddpm_ema": ema.state_dict()}, out / "models.pt")

    # ---------- metrik rekonstruksi (Test Set, rentang [0, 1]) ----------
    blank = {"test_mse": None, "test_psnr": None, "test_ssim": None, "fid": None}
    recon_fns = {"Vanilla AE": ae, "Sparse AE": sae, "VAE": vae.reconstruct}
    for name, fn in recon_fns.items():
        m = reconstruction_metrics(fn, test_dl, device)
        res[name] = {**blank, "test_mse": m["mse"], "test_psnr": m["psnr"], "test_ssim": m["ssim"]}
    res["Vanilla AE"].update(hidden_stats(ae, test_dl, device))
    res["Sparse AE"].update(hidden_stats(sae, test_dl, device))
    for name in ("Vanilla GAN", "DCGAN", "DDPM"):
        res[name] = dict(blank)

    # ---------- FID: sampel prior vs Test Set ----------
    fid = InceptionFID(device)
    fid.fit_real(torch.cat([x for x, _ in test_dl]))              # (7000, 1, 28, 28) di [0, 1]
    gens = {                                                      # semua mengembalikan [0, 1]
        "VAE": lambda n: vae.sample(n),
        "Vanilla GAN": lambda n: to_unit(g1(torch.randn(n, Z_GAN, device=device))),
        "DCGAN": lambda n: to_unit(g2(torch.randn(n, Z_GAN, device=device))),
        "DDPM": lambda n: to_unit(ema.sample(n)[0]),
    }
    for name, fn in gens.items():
        set_seed()
        res[name]["fid"] = fid.score(generate(fn, args.n_fid))
        print(f"[FID] {name}: {res[name]['fid']:.2f} (n={args.n_fid})")

    # ---------- latency per batch 64 ----------
    lat = {
        "Vanilla AE": ("reconstruction", lambda: ae(x64)),
        "Sparse AE": ("reconstruction", lambda: sae(x64)),
        "VAE": ("generation", lambda: vae.sample(64)),
        "Vanilla GAN": ("generation", lambda: g1(torch.randn(64, Z_GAN, device=device))),
        "DCGAN": ("generation", lambda: g2(torch.randn(64, Z_GAN, device=device))),
        "DDPM": ("generation", lambda: ema.sample(64)),
    }
    for name, (kind, fn) in lat.items():
        slow = name == "DDPM"
        res[name]["latency_ms_batch64"] = latency_ms(fn, device, warmup=1 if slow else 5, repeats=3 if slow else 30)
        res[name]["latency_kind"] = kind
        res[name]["train_time_per_epoch_s"] = mean_epoch_time(hist, name)
    res["VAE"]["latency_recon_ms_batch64"] = latency_ms(lambda: vae.reconstruct(x64), device)

    dims = {"Vanilla AE": "64", "Sparse AE": "64 (bottleneck) / 256 (sparse)", "VAE": "64",
            "Vanilla GAN": "100 (prior)", "DCGAN": "100 (prior)", "DDPM": "28x28"}
    for name in ORDER:
        res[name]["latent_dim"] = dims[name]
    for name in ("Vanilla GAN", "DCGAN"):                         # indikator kestabilan untuk Q3
        res[name]["g_loss_std_last10"] = last_std(hist, name, "g_loss")
        res[name]["d_loss_std_last10"] = last_std(hist, name, "d_loss")
        res[name]["d_real_last"] = [r["d_real"] for r in hist if r["model"] == name][-1]
        res[name]["d_fake_last"] = [r["d_fake"] for r in hist if r["model"] == name][-1]
    res["DDPM"]["loss_std_last10"] = last_std(hist, "DDPM", "train_loss")

    cols = ["latent_dim", "train_time_per_epoch_s", "latency_ms_batch64", "latency_kind",
            "test_mse", "test_psnr", "test_ssim", "fid"]
    table = [{"model": n, **{c: res[n][c] for c in cols}} for n in ORDER]
    print_table(table)
    save_json(res, out / "metrics.json")
    save_csv(table, out / "metrics_table.csv")
    save_csv(hist, out / "train_log.csv")

    # ---------- Figur 1: grid rekonstruksi ----------
    x_vis = first_per_class(test_dl, device)                      # (10, 1, 28, 28), satu per kelas
    with torch.no_grad():
        rows = [x_vis, ae(x_vis), sae(x_vis), vae.reconstruct(x_vis)]
    fig, ax = plt.subplots(figsize=(10, 4.4))
    draw_rows(ax, rows, ["Asli", "Vanilla AE", "Sparse AE", "VAE"], "Figur 1: rekonstruksi citra uji")
    save_fig(fig, out / "fig1_reconstruction.png")

    # ---------- Figur 2: random sampling 4x4 ----------
    set_seed()
    fig, axes = plt.subplots(2, 2, figsize=(9, 9.4))
    for ax, (name, fn), tag in zip(axes.flat, gens.items(), "ABCD"):
        draw_grid(ax, generate(fn, 16), nrow=4, title=f"Panel {tag}: {name}")
    save_fig(fig, out / "fig2_random_sampling.png")

    # ---------- Figur 3: bukti batasan MLP (dibuat oleh exp_mlp_proof.py) ----------
    if Path(args.mlp_fig).exists():
        shutil.copy(args.mlp_fig, out / "fig3_mlp_proof.png")
    else:
        print(f"PERINGATAN: {args.mlp_fig} tidak ada, jalankan experiments.exp_mlp_proof dulu untuk Figur 3")

    # ---------- Figur 4: interpolasi linear ----------
    alphas = torch.tensor(ALPHAS, device=device).view(-1, 1)      # (6, 1)
    lerp = lambda zA, zB: (1 - alphas) * zA + alphas * zB         # zA, zB: (1, D) -> (6, D)

    def contrast_pair(G, n=32):                                   # GAN tanpa encoder: pilih dua z dengan keluaran paling berbeda
        z = torch.randn(n, Z_GAN, device=device)
        img = G(z).flatten(1)
        i, j = divmod(torch.cdist(img, img).argmax().item(), n)
        return z[i:i + 1], z[j:j + 1]

    set_seed()
    with torch.no_grad():
        pair = x_vis[[CLS_A, CLS_B]]                              # (2, 1, 28, 28)
        zae, zsp = ae.encode(pair), sae.encode(pair)              # (2, 64)
        mu, _ = vae.encode(pair)
        r_ae = ae.decode(lerp(zae[:1], zae[1:]))
        r_sp = sae.decode(lerp(zsp[:1], zsp[1:]))
        r_vae = vae.decoder(lerp(mu[:1], mu[1:]))
        r_g1 = to_unit(g1(lerp(*contrast_pair(g1))))
        r_g2 = to_unit(g2(lerp(*contrast_pair(g2))))
    fig, ax = plt.subplots(figsize=(8, 5.6))
    draw_rows(ax, [r_ae, r_sp, r_vae, r_g1, r_g2], ["Vanilla AE", "Sparse AE", "VAE", "Vanilla GAN", "DCGAN"],
              "Figur 4: interpolasi linear $z(\\alpha)=(1-\\alpha)z_A+\\alpha z_B$", [f"α={a}" for a in ALPHAS])
    save_fig(fig, out / "fig4_interpolation.png")

    # ---------- Figur 5: kurva loss + lintasan denoising DDPM ----------
    T = args.ddpm_T
    steps = sorted({round((T - 1) * f) for f in (1, .8, .6, .5, .4, .3, .2, .1, 0)}, reverse=True)
    set_seed()
    x_fin, snaps = ema.sample(4, record_steps=steps)
    traj = [torch.stack([to_unit(snaps[i][s]) for i in steps] + [to_unit(x_fin[s])]) for s in range(4)]   # 4 x (len+1, 1, 28, 28)

    panels = [
        ("Vanilla AE", [("train_loss", "train"), ("val_loss", "val")], "MSE"),
        ("Sparse AE", [("train_loss", "train (MSE + L1)"), ("train_mse", "train MSE"), ("val_loss", "val")], "loss"),
        ("VAE", [("train_loss", "train -ELBO"), ("val_loss", "val -ELBO"), ("train_recon", "BCE"), ("train_kl", "KL")], "nats / citra"),
        ("Vanilla GAN", [("g_loss", "Generator"), ("d_loss", "Discriminator")], "BCE"),
        ("DCGAN", [("g_loss", "Generator"), ("d_loss", "Discriminator")], "BCE"),
        ("DDPM", [("train_loss", "train"), ("val_loss", "val")], r"MSE derau $\epsilon$"),
    ]
    fig = plt.figure(figsize=(15, 13))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.9])
    for k, (name, keys, ylabel) in enumerate(panels):
        ax = fig.add_subplot(gs[k // 3, k % 3])
        rows_k = [r for r in hist if r["model"] == name]
        for key, label in keys:
            ax.plot([r["epoch"] for r in rows_k], [r[key] for r in rows_k], label=label)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    ax = fig.add_subplot(gs[2, :])
    draw_rows(ax, traj, [f"sampel {i + 1}" for i in range(4)], "DDPM reverse diffusion $x_T \\to x_0$ (bobot EMA)",
              [f"t={i}" for i in steps] + ["x0"])
    save_fig(fig, out / "fig5_loss_curves_ddpm_trajectory.png")

    print(f"\nselesai. hasil tersimpan di {out}/")


if __name__ == "__main__":
    main()
