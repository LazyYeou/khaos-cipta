# AGENT.md — Panduan untuk AI Coding Agent

Dokumen ini adalah instruksi untuk AI agent (Claude Code, Cursor, Copilot, dll.) yang bekerja di repo **batik-diffusion**. Baca seluruhnya sebelum menulis atau mengubah kode.

---

## 1. Konteks Proyek

- **Tujuan:** implementasi DDPM **dari nol** di PyTorch untuk menghasilkan motif batik 64×64.
- **Sifat proyek:** proyek portofolio dan pembelajaran. **Keterbacaan dan kebenaran lebih penting daripada kecepatan atau kecanggihan.**
- **Target hardware:** Google Colab (GPU T4, 16 GB VRAM) dan GPU lokal kelas konsumen. Semua konfigurasi default harus muat di T4.

---

## 2. Aturan Utama (WAJIB)

1. **Jangan memakai library diffusion siap pakai.** Dilarang `diffusers`, `denoising_diffusion_pytorch`, `k-diffusion`, atau sejenisnya untuk komponen inti (scheduler, U-Net, sampler, loss). Inti proyek ini adalah menulis semuanya sendiri.
   - Boleh: `torch`, `torchvision`, `numpy`, `pillow`, `pyyaml`, `tqdm`, `einops`, `matplotlib`, `imageio`, `pytest`, dan `torch-fidelity` / `clean-fid` **hanya** untuk menghitung FID.
2. **Jangan mengubah formula matematika** di `src/diffusion.py` atau `src/samplers.py` tanpa menuliskan referensi persamaan dari paper di komentar dan menjelaskannya di ringkasan perubahan.
3. **Jangan commit** isi `data/`, `runs/`, checkpoint (`*.pt`, `*.pth`, `*.safetensors`), atau gambar dataset. Pastikan semuanya ada di `.gitignore`.
4. **Jangan mengunduh atau scraping gambar batik** dari internet secara otomatis. Dataset disediakan manual oleh pemilik repo dengan sumber dan lisensi yang jelas.
5. **Jangan menjalankan training panjang** (lebih dari ~5 menit) tanpa diminta. Untuk verifikasi, pakai mode smoke test (lihat bagian 6).
6. **Jangan menghapus fitur yang sudah ada** atau mengubah API publik antar modul tanpa diminta.

---

## 3. Konvensi Teknis

### Tensor & data
- Gambar selalu berformat **`(B, C, H, W)`**, `float32`, dengan rentang **`[-1, 1]`** selama training dan sampling.
- Konversi ke `[0, 1]` atau `uint8` **hanya** dilakukan saat menyimpan atau menampilkan gambar (di `src/utils.py`).
- Timestep `t` berupa tensor `long` berukuran `(B,)`, dengan indeks **0-based**: `0 … T-1`.

### Noise schedule
- Semua buffer schedule (`betas`, `alphas`, `alpha_bar`, `sqrt_alpha_bar`, `sqrt_one_minus_alpha_bar`, `posterior_variance`) dihitung **sekali** di awal dan disimpan sebagai buffer, bukan dihitung ulang tiap langkah.
- Gunakan fungsi helper `extract(buf, t, shape)` untuk mengambil nilai sesuai `t` dan me-reshape ke `(B, 1, 1, 1)`.
- Schedule dihitung dalam `float64`, lalu dikonversi ke `float32`.

### Model
- U-Net menerima `(x_t, t)` dan mengembalikan prediksi noise **dengan shape identik** dengan `x_t`.
- Normalisasi memakai **GroupNorm** (32 grup), aktivasi **SiLU**.
- Time embedding: sinusoidal → MLP → disuntikkan ke setiap ResBlock.
- Konvolusi terakhir di setiap ResBlock dan output layer diinisialisasi nol (*zero-init*) untuk stabilitas.

### Reprodusibilitas
- Setiap script menerima `--seed` dan memanggil `utils.set_seed(seed)`.
- Setiap run menyimpan salinan config yang dipakai ke `runs/<nama_run>/config.yaml`.

### Konfigurasi
- Semua hyperparameter berasal dari file YAML di `configs/`. **Jangan hardcode** angka seperti learning rate, ukuran batch, atau jumlah channel di dalam kode.
- Argumen CLI boleh meng-override nilai YAML.

---

## 4. Gaya Kode

- Python ≥ 3.10, PyTorch ≥ 2.1.
- Type hints di semua fungsi publik.
- Docstring singkat yang menyebutkan **shape input dan output**, contoh:
  ```python
  def q_sample(x0: Tensor, t: Tensor, noise: Tensor) -> Tensor:
      """Forward process (Ho et al. 2020, Eq. 4).
      x0: (B, C, H, W) dalam [-1, 1], t: (B,) long → x_t: (B, C, H, W)
      """
  ```
- Komentar matematika merujuk ke nomor persamaan di paper.
- Format dengan `ruff format`, lint dengan `ruff check`.
- Nama variabel konsisten dengan notasi paper: `x0`, `xt`, `eps`, `alpha_bar`, `beta`.
- Fungsi pendek dan fokus. Hindari abstraksi berlebihan, karena kode ini juga dibaca sebagai materi belajar.

---

## 5. Perintah Penting

```bash
# Instalasi
pip install -r requirements.txt

# Lint & format
ruff check src scripts tests
ruff format src scripts tests

# Unit test
pytest -q

# Smoke test training (CPU-friendly, < 2 menit)
python scripts/train.py --config configs/cifar10.yaml --smoke-test

# Training penuh
python scripts/train.py --config configs/batik64.yaml

# Sampling
python scripts/sample.py --ckpt runs/batik64/ema_last.pt --sampler ddim --steps 50 --n 16
```

`--smoke-test` berarti: model kecil (base channels 32), 2 batch berukuran 4, T = 50, 20 langkah training, lalu sampling 2 gambar. Tujuannya memastikan pipeline jalan dari awal sampai akhir tanpa error.

---

## 6. Definisi "Selesai" untuk Setiap Perubahan

Sebelum menyatakan tugas selesai, agent **wajib**:

1. Menjalankan `ruff check` tanpa error.
2. Menjalankan `pytest -q` dan semua test lulus.
3. Menjalankan smoke test jika perubahan menyentuh `src/` atau `scripts/train.py` / `scripts/sample.py`.
4. Menambah atau memperbarui test untuk logika baru.
5. Memperbarui `README.md` jika ada perubahan perintah, konfigurasi, atau fitur.
6. Menulis ringkasan singkat: apa yang diubah, kenapa, dan cara memverifikasinya.

---

## 7. Test yang Harus Ada

| Test | Yang dicek |
|---|---|
| `test_schedule.py` | `alpha_bar` menurun monoton, `alpha_bar[0] ≈ 1`, `alpha_bar[-1] ≈ 0`, semua nilai di `(0, 1)` |
| `test_q_sample.py` | Statistik `x_t` sesuai teori: di t besar, mean ≈ 0 dan std ≈ 1 |
| `test_unet.py` | Output shape sama dengan input untuk resolusi 32 dan 64, berbagai batch size, dan dengan `t` acak |
| `test_time_embedding.py` | Shape benar dan embedding untuk `t` berbeda tidak identik |
| `test_overfit.py` | Model kecil mampu menurunkan loss secara signifikan pada 1 batch dalam ≤ 200 langkah |
| `test_sampler.py` | DDPM & DDIM menghasilkan output dengan shape benar dan nilai dalam `[-1, 1]` |

---

## 8. Debugging Checklist

Jika hasil sampling hanya noise, abu-abu polos, atau warnanya aneh, periksa urutan berikut:

1. Apakah data benar-benar dinormalisasi ke `[-1, 1]` (bukan `[0, 1]` atau `[0, 255]`)?
2. Apakah indeks `t` konsisten 0-based di training dan sampling?
3. Apakah `t` benar-benar diteruskan ke U-Net dan time embedding tersambung ke ResBlock?
4. Apakah sampler memakai bobot **EMA**? Di awal training, EMA dengan decay 0.9999 butuh warm-up; pakai bobot biasa untuk pengecekan cepat.
5. Apakah `extract()` me-reshape ke `(B, 1, 1, 1)` dengan benar (bug broadcasting sering terjadi di sini)?
6. Apakah langkah terakhir sampling (t = 0) **tidak** menambahkan noise?
7. Apakah `model.eval()` dan `torch.no_grad()` dipakai saat sampling?
8. Jalankan `test_overfit.py`. Jika model tidak bisa overfit satu batch, masalahnya ada di model atau loss, bukan di data.

---

## 9. Logging Eksperimen

- Setiap run menyimpan ke `runs/<nama_run>/`:
  - `config.yaml` — salinan config
  - `log.csv` — step, loss, learning rate, waktu
  - `samples/step_XXXXXX.png` — grid sampel berkala (pakai bobot EMA, seed tetap agar bisa dibandingkan)
  - `last.pt` dan `ema_last.pt` — checkpoint terbaru (berisi model, EMA, optimizer, step, config)
- Training harus bisa **dilanjutkan** dari checkpoint dengan `--resume`. Ini penting karena sesi Colab bisa terputus.
- Hasil eksperimen untuk tabel README dicatat di `runs/EXPERIMENTS.md`.

---

## 10. Hal yang Perlu Ditanyakan ke Pemilik Repo Dulu

Jangan memutuskan sendiri; tanyakan dulu jika:
- Ingin menambah dependency baru.
- Ingin mengubah resolusi default, arsitektur U-Net secara besar, atau format checkpoint.
- Ingin mengubah struktur folder.
- Ada ambiguitas tentang sumber atau lisensi data.
- Perubahan akan membuat checkpoint lama tidak bisa dimuat lagi.
