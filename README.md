# Image Processing Project (Pengolahan Citra)

Aplikasi web pengolahan citra berbasis **Flask + OpenCV** dengan antarmuka Bootstrap.
Setiap metode menampilkan **histogram citra asli dan citra hasil** (kanal R, G, B, dan Gray).

## Input Source
- **Upload Image** – pilih file gambar dari komputer.
- **Use Camera** – *Start Camera* → *Capture* untuk mengambil foto, lalu foto diproses.

## 6 Metode (masing-masing dengan histogram)
| # | Metode | Isi |
|---|--------|-----|
| 1 | **Adjustment** | Brightness, Contrast, Saturation, Sharpness, Blur (Gaussian, Median, Box, Bilateral) |
| 2 | **Noise** | Gaussian, Salt & Pepper, Speckle, Uniform, Poisson + opsi denoise (Median, Gaussian, Bilateral, Non-local Means) |
| 3 | **Filtering** | Gray Scale, Biner (threshold manual / Otsu), Negative, Red, Green, Blue |
| 4 | **Transformation** | Rotate, Scale, Translasi (geser X/Y), Flip horizontal/vertikal |
| 5 | **Equalization** | Histogram Equalization (luminance & per kanal RGB), CLAHE, Contrast Stretching |
| 6 | **Compression** | JPEG, WebP (lossy, atur quality), PNG (lossless, atur level) + ukuran file, rasio, PSNR |

Fitur lain:
- **Pipeline (Apply / Undo / Redo)** – klik **Apply** untuk menjadikan hasil sebagai input metode berikutnya (misal Noise → Median denoise → Equalization). Riwayat langkah tampil sebagai chip, dan bisa di-Undo/Redo.
- **Bandingkan dengan Original** – saklar (aktif secara default) agar panel kiri, histogram, statistik, dan slider before/after selalu membandingkan hasil pipeline dengan gambar asli. Jika dimatikan, perbandingan memakai input langkah sebelumnya.
- **Statistik histogram** – mean, median, std, min, max (kanal gray) di bawah tiap histogram.
- **Kurva CDF** – saklar untuk menampilkan distribusi kumulatif pada histogram (berguna untuk menjelaskan Equalization).
- **Slider before/after** – geser garis untuk membandingkan gambar sebelum dan sesudah.
- Tombol **Reset** (mengembalikan pengaturan), **Download Edited Image**, dan histogram **skala log**.

## Menjalankan
```bash
python -m venv venv
venv\Scripts\activate          # Windows   (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt
python app.py
```
Buka http://127.0.0.1:5000

> Bootstrap dimuat dari CDN, jadi perlu koneksi internet saat membuka halaman.

## Struktur
```
app.py                  # backend: 6 metode + perhitungan histogram (endpoint /api/process)
templates/index.html    # halaman utama
static/css/style.css
static/js/app.js        # UI, kontrol tiap metode, upload/kamera, gambar histogram
requirements.txt
Procfile                # untuk deploy (gunicorn)
```

## Catatan teknis
- Gambar kerja diperkecil otomatis (maks. 1024 px) agar slider tetap responsif; hasil download mengikuti ukuran ini.
- Noise memakai *seed* tetap, jadi hasilnya stabil saat slider digeser.
- Pada Transformation, area latar hitam akibat rotasi **tidak** dihitung dalam histogram.
- Hasil Compression yang diunduh adalah file terkompresi aslinya (.jpg / .webp / .png).
