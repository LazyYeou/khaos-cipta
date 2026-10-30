# 🎨 Batik Diffusion — DDPM from Scratch

Implementasi **Denoising Diffusion Probabilistic Model (DDPM)** dari nol menggunakan PyTorch, dilatih untuk menghasilkan **motif batik Nusantara** yang baru.

Semua komponen inti — noise scheduler, U-Net, training loop, dan sampler — ditulis sendiri tanpa library diffusion siap pakai (tanpa `diffusers`), dengan tujuan memahami cara kerja model diffusion secara mendalam.

<p align="center">
  <img src="assets/reverse_process.gif" width="400" alt="Noise berubah menjadi motif batik"/>
  <br/><em>Proses reverse: noise acak → motif batik (placeholder)</em>
</p>

---

## ✨ Fitur

- **DDPM from scratch** — forward process, reverse process, dan loss prediksi noise (ε-prediction)
- **U-Net custom** — ResBlock, GroupNorm, sinusoidal time embedding, self-attention, skip connections
- **Noise schedule** linear & cosine (bisa dipilih lewat config)
- **EMA (Exponential Moving Average)** bobot model untuk sampling yang lebih bersih
- **DDIM sampler** — sampling 50 langkah, ~20× lebih cepat dari DDPM 1000 langkah
- **Pipeline dataset batik** — potong foto kain menjadi patch 64×64 dengan augmentasi
- **Evaluasi FID** dan visualisasi forward/reverse process
- **Interpolasi latent** antar motif (misalnya parang ↔ kawung)

---

## 📁 Struktur Proyek

```
batik-diffusion/
├── configs/
│   ├── cifar10.yaml          # sanity check di CIFAR-10 (32×32)
│   └── batik64.yaml          # konfigurasi utama batik (64×64)
├── data/
│   ├── raw/                  # foto kain batik asli (tidak di-commit)
│   └── patches/              # hasil potongan 64×64 (tidak di-commit)
├── src/
│   ├── diffusion.py          # noise schedule, q_sample, loss
│   ├── unet.py               # arsitektur U-Net
│   ├── modules.py            # ResBlock, Attention, TimeEmbedding
│   ├── dataset.py            # loader + augmentasi
│   ├── ema.py                # Exponential Moving Average
│   ├── samplers.py           # DDPM & DDIM sampler
│   └── utils.py              # seed, logging, save grid
├── scripts/
│   ├── prepare_patches.py    # foto mentah → patch 64×64
│   ├── train.py              # training loop
│   ├── sample.py             # generate gambar baru
│   ├── evaluate_fid.py       # hitung FID
│   ├── interpolate.py        # interpolasi antar motif
│   └── make_gif.py           # GIF reverse process
├── tests/                    # unit test shape & schedule
├── notebooks/
│   └── colab_train.ipynb     # versi siap jalan di Google Colab
├── assets/                   # gambar & GIF untuk README
├── runs/                     # checkpoint & log (tidak di-commit)
├── requirements.txt
├── AGENT.md
└── README.md
```

---

## 🧠 Cara Kerja Singkat

**1. Forward process** — gambar asli dirusak dengan noise Gaussian bertahap selama `T = 1000` langkah. Berkat trik reparameterisasi, kita bisa langsung lompat ke langkah ke-t:

```
x_t = √ᾱ_t · x_0 + √(1 − ᾱ_t) · ε,     ε ~ N(0, I)
```

**2. Training** — U-Net `ε_θ(x_t, t)` belajar menebak noise yang ditambahkan:

```
L = ‖ ε − ε_θ(x_t, t) ‖²
```

**3. Sampling** — mulai dari noise murni `x_T ~ N(0, I)`, lalu hilangkan noise langkah demi langkah:

```
x_{t−1} = 1/√α_t · ( x_t − β_t/√(1 − ᾱ_t) · ε_θ(x_t, t) ) + σ_t · z
```

---

## 🚀 Quick Start

### 1. Instalasi

```bash
git clone https://github.com/<username>/batik-diffusion.git
cd batik-diffusion
pip install -r requirements.txt
```

### 2. Sanity check di CIFAR-10

Sebelum masuk ke batik, pastikan pipeline benar dengan CIFAR-10 (diunduh otomatis):

```bash
python scripts/train.py --config configs/cifar10.yaml
```

### 3. Siapkan dataset batik

Letakkan foto kain batik di `data/raw/`, lalu potong menjadi patch:

```bash
python scripts/prepare_patches.py \
  --input data/raw \
  --output data/patches \
  --size 64 \
  --stride 32 \
  --min-std 0.05      # buang patch polos / tanpa motif
```

### 4. Training

```bash
python scripts/train.py --config configs/batik64.yaml
```

### 5. Generate motif baru

```bash
# DDPM (1000 langkah, kualitas maksimal)
python scripts/sample.py --ckpt runs/batik64/ema_last.pt --n 64 --sampler ddpm

# DDIM (50 langkah, jauh lebih cepat)
python scripts/sample.py --ckpt runs/batik64/ema_last.pt --n 64 --sampler ddim --steps 50
```

### 6. Evaluasi & visualisasi

```bash
python scripts/evaluate_fid.py --ckpt runs/batik64/ema_last.pt --n 10000
python scripts/make_gif.py --ckpt runs/batik64/ema_last.pt --out assets/reverse_process.gif
python scripts/interpolate.py --ckpt runs/batik64/ema_last.pt --steps 10
```

---

## ⚙️ Konfigurasi Default (`batik64.yaml`)

| Parameter | Nilai |
|---|---|
| Resolusi | 64 × 64 |
| Timesteps (T) | 1000 |
| Noise schedule | cosine |
| Base channels | 128 |
| Channel multiplier | (1, 2, 2, 2) |
| Attention di resolusi | 16 × 16 |
| ResBlock per level | 2 |
| Dropout | 0.1 |
| Optimizer | AdamW, lr 2e-4 |
| Batch size | 64 |
| EMA decay | 0.9999 |
| Mixed precision | fp16 |

---

## 🖼️ Dataset

- **Sumber:** foto kain batik milik sendiri dan dataset terbuka dengan lisensi yang jelas. Daftar sumber dan lisensinya dicatat di `data/SOURCES.md`.
- **Preprocessing:** setiap foto dipotong menjadi patch 64×64 dengan stride 32. Patch yang terlalu polos dibuang berdasarkan standar deviasi piksel.
- **Augmentasi:** random horizontal/vertical flip dan rotasi 90°. Augmentasi warna **tidak** dipakai agar palet khas batik tetap terjaga.
- **Normalisasi:** piksel diskalakan ke rentang `[-1, 1]`.

> ⚠️ Folder `data/` tidak ikut di-commit. Hormati hak cipta pengrajin dan desainer batik.

---

## 📊 Hasil

> Bagian ini diisi setelah eksperimen selesai.

| Eksperimen | Schedule | Attention | Params | Steps | FID ↓ |
|---|---|---|---|---|---|
| Baseline | linear | ✓ | – | – | – |
| Cosine | cosine | ✓ | – | – | – |
| No attention | cosine | ✗ | – | – | – |
| DDIM 50 steps | cosine | ✓ | – | – | – |

**Visualisasi yang disediakan:**
- `assets/forward_process.png` — satu motif yang perlahan berubah menjadi noise
- `assets/reverse_process.gif` — noise yang perlahan menjadi batik
- `assets/samples_grid.png` — 64 motif hasil generate
- `assets/interpolation.png` — transisi antar dua motif

---

## 🗺️ Roadmap

- [x] Rancangan arsitektur & dokumentasi
- [ ] DDPM + U-Net, lolos sanity check CIFAR-10
- [ ] Pipeline dataset batik
- [ ] Training batik 64×64 + EMA
- [ ] DDIM sampler
- [ ] Evaluasi FID & eksperimen ablation
- [ ] Generate motif *tileable* (bisa diulang tanpa sambungan)
- [ ] Class-conditional (per jenis motif: parang, kawung, mega mendung, dll.) dengan classifier-free guidance
- [ ] Demo interaktif di Hugging Face Spaces
- [ ] Lanjutan: fine-tune LoRA Stable Diffusion di dataset yang sama sebagai pembanding

---

## 📚 Referensi

- Ho, Jain & Abbeel (2020) — *Denoising Diffusion Probabilistic Models*
- Nichol & Dhariwal (2021) — *Improved Denoising Diffusion Probabilistic Models* (cosine schedule)
- Song, Meng & Ermon (2021) — *Denoising Diffusion Implicit Models* (DDIM)
- Ho & Salimans (2022) — *Classifier-Free Diffusion Guidance*

---

## 📄 Lisensi

Kode dirilis di bawah lisensi MIT. Dataset dan gambar hasil generate mengikuti lisensi sumber datanya masing-masing.
