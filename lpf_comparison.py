
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ========================================
# KONFIGURASI FOLDER
# ========================================
BASE_DIR = Path(__file__).resolve().parent
IMAGE_DIR = BASE_DIR / "image"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Nama gambar sumber
IMAGE_PATH = IMAGE_DIR / "foto.jpg",""

# ========================================
# MEMBACA CITRA ASLI
# ========================================
if not IMAGE_PATH.exists():
    raise FileNotFoundError(
        f"Gambar tidak ditemukan: {IMAGE_PATH}\n"
        "Masukkan gambar valid ke folder image."
    )

citra_asli = cv2.imread(str(IMAGE_PATH))

if citra_asli is None:
    raise ValueError(
        f"Gambar tidak bisa dibaca: {IMAGE_PATH}\n"
        "Pastikan file bukan kosong atau rusak."
    )

# Ubah BGR menjadi RGB untuk Matplotlib
citra_asli = cv2.cvtColor(citra_asli, cv2.COLOR_BGR2RGB)

print("Citra asli berhasil dimuat:", IMAGE_PATH.name)

# ========================================
# 1. MEMBUAT CITRA NOISY
# ========================================
np.random.seed(42)

noise = np.random.normal(
    loc=0,
    scale=25,
    size=citra_asli.shape
)

citra_noisy = np.clip(
    citra_asli.astype(np.float32) + noise,
    0,
    255
).astype(np.uint8)

# ========================================
# 2. LOW PASS FILTER (LPF)
# ========================================
# LPF 1: kernel 3x3
lpf1 = cv2.blur(citra_noisy, (3, 3))

# LPF 2: kernel 5x5
lpf2 = cv2.blur(citra_noisy, (5, 5))

# LPF 3: kernel 9x9
lpf3 = cv2.blur(citra_noisy, (9, 9))

# ========================================
# 3. MENYIMPAN HASIL GAMBAR
# ========================================
hasil = {
    "citra_asli.jpg": citra_asli,
    "citra_noisy.jpg": citra_noisy,
    "lpf1.jpg": lpf1,
    "lpf2.jpg": lpf2,
    "lpf3.jpg": lpf3,
}

for nama, gambar in hasil.items():
    path = OUTPUT_DIR / nama

    # RGB ke BGR sebelum disimpan dengan OpenCV
    gambar_bgr = cv2.cvtColor(
        gambar, cv2.COLOR_RGB2BGR
    )

    berhasil = cv2.imwrite(str(path), gambar_bgr)

    if not berhasil:
        raise IOError(f"Gagal menyimpan: {path}")

    print(f"Berhasil disimpan: {path}")

# ========================================
# 4. MENAMPILKAN 5 CITRA
# ========================================
judul = [
    "Citra Asli",
    "Citra Noisy",
    "LPF 1 (3x3)",
    "LPF 2 (5x5)",
    "LPF 3 (9x9)"
]

gambar_list = [
    citra_asli,
    citra_noisy,
    lpf1,
    lpf2,
    lpf3
]

fig, axes = plt.subplots(1, 5, figsize=(20, 6))

for ax, gambar, nama in zip(
    axes, gambar_list, judul
):
    ax.imshow(gambar)
    ax.set_title(nama, fontsize=12)
    ax.axis("off")

plt.tight_layout()

# Simpan perbandingan sebagai satu gambar
plt.savefig(
    OUTPUT_DIR / "perbandingan_lpf.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()

print("\nSemua proses selesai!")
print("Hasil tersedia di folder output.")
