# Studi Komparatif Model Generatif (Fashion-MNIST)

Deep Learning (D6421) - Sesi 07 Take-Home. Vanilla AE, Sparse AE, VAE, Vanilla GAN, DCGAN, DDPM,
plus eksperimen pembuktian batasan MLP Classifier untuk image generation.

## Struktur

```
.
├── data/                 # Fashion-MNIST diunduh otomatis ke sini
├── models/
│   ├── autoencoder.py    # Decoder, VanillaAE
│   └── mlp_classifier.py # MLPClassifier 784-256-64-10
├── utils/
│   ├── seed_and_data.py  # seed 42, split 80/10/10, DataLoader
│   ├── metrics.py        # MSE, PSNR, SSIM (pure torch)
│   ├── viz.py            # helper grid gambar
│   └── logger.py         # simpan json / csv
├── experiments/
│   └── exp_mlp_proof.py  # Bagian 2.1 (Tahap 1-5)
├── results/              # figur + log metrik
└── requirements.txt
```

## Menjalankan

```bash
pip install -r requirements.txt
python -m experiments.exp_mlp_proof
```

Jalankan dari root repo (pakai `-m`, bukan `python experiments/...`) supaya import antar-folder jalan.
Opsi: `--epochs-clf`, `--epochs-recon`, `--batch-size`, `--lr`, `--num-workers` (set 0 kalau di Windows bermasalah).

## Output `results/mlp_proof/`

- `metrics.json`: akurasi MLP, metrik rekonstruksi (MSE/PSNR/SSIM) tiap tahap
- `train_log.csv`: loss per epoch + waktu per epoch
- `fig3_mlp_proof.png`: figur komposit 4 panel; `fig3a..d_*.png`: tiap panel terpisah
- `*.pt`: checkpoint (tidak ikut git)

## Catatan

- Seed 42 dikunci di `set_seed()`; split dibuat dari gabungan train+test resmi (70k) lalu 56k/7k/7k.
- Piksel di [0, 1]. Untuk GAN/DCGAN/DDPM nanti rescale ke [-1, 1] di script masing-masing.
