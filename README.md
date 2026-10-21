# Pengolahan Citra Digital

Proyek Python sederhana untuk mencoba beberapa teknik pengolahan citra menggunakan OpenCV, NumPy, dan Matplotlib. Program menambahkan noise pada gambar, menerapkan filter, lalu menampilkan atau menyimpan hasilnya.

## Fitur

- Membandingkan tiga **High-Pass Filter (HPF)** untuk deteksi tepi dan penajaman.
- Membandingkan **Low-Pass Filter (LPF)** dengan beberapa ukuran kernel.
- Menerapkan Gaussian blur, grayscale, deteksi tepi Canny, dan sharpening.
- Menyimpan sejumlah hasil pengolahan di folder `output`.

## Persiapan

Gunakan Python 3.10 atau lebih baru. Dari folder proyek, buat dan aktifkan virtual environment, lalu pasang dependensi:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install opencv-python numpy matplotlib
```

Jika PowerShell tidak mengizinkan aktivasi script, jalankan Python dan `pip` melalui `venv\Scripts\python.exe` tanpa mengaktifkan environment:

```powershell
.\venv\Scripts\python.exe -m pip install opencv-python numpy matplotlib
```

## Menjalankan

Jalankan perintah dari folder utama proyek:

```powershell
python hpf_comparison.py
python image_filtering.py
```

HPF membaca `image/foto3.jpg`. Program filter umum membaca `image/foto.jpg` dan `image/foto2.jpg`. Gambar perbandingan akan ditampilkan; tutup jendela plot untuk mengakhiri program. Hasil yang disimpan tersedia di `output`.

`lpf_comparison.py` berisi perbandingan LPF, tetapi saat ini konfigurasi `IMAGE_PATH` di dalamnya masih perlu diperbaiki sebelum script dapat dijalankan.

`buat_gambar.py` membuat gambar contoh di `image/foto.jpg`. `main.py` adalah contoh input dan sapaan sederhana.

## Struktur proyek

```text
.
├── image/                  # Gambar sumber
├── output/                 # Hasil pengolahan citra
├── buat_gambar.py          # Membuat gambar contoh
├── hpf_comparison.py       # Perbandingan High-Pass Filter
├── image_filtering.py      # Blur, grayscale, Canny, dan sharpening
├── lpf_comparison.py       # Perbandingan Low-Pass Filter
└── main.py                 # Contoh program sapaan
```
