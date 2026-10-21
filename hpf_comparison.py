
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ======================================
# 1. FOLDER DAN GAMBAR
# ======================================
BASE_DIR = Path(__file__).resolve().parent
IMAGE_DIR = BASE_DIR / "image"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

IMAGE_PATH = IMAGE_DIR / "foto3.jpg"

# ======================================
# 2. MEMBACA CITRA ASLI GRAYSCALE
# ======================================
citra_asli = cv2.imread(
    str(IMAGE_PATH), cv2.IMREAD_GRAYSCALE
)

if citra_asli is None:
    raise ValueError(
        f"Gambar gagal dibaca: {IMAGE_PATH}\n"
        "Pastikan foto.jpg ada dan tidak kosong."
    )

print("Citra asli berhasil dibaca!")

# ======================================
# 3. MEMBUAT CITRA NOISY
# ======================================
np.random.seed(42)

noise = np.random.normal(
    0, 20, citra_asli.shape
)

citra_noisy = np.clip(
    citra_asli.astype(np.float32) + noise,
    0, 255
).astype(np.uint8)

# ======================================
# 4. HIGH PASS FILTER 1
# Deteksi tepi (edge detection)
# ======================================
kernel_hpf1 = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
], dtype=np.float32)

hasil_hpf1 = cv2.filter2D(
    citra_noisy, cv2.CV_32F, kernel_hpf1
)

# Normalisasi agar tepi terlihat jelas
hasil_hpf1 = np.abs(hasil_hpf1)
hasil_hpf1 = cv2.normalize(
    hasil_hpf1, None, 0, 255,
    cv2.NORM_MINMAX
).astype(np.uint8)

# ======================================
# 5. HIGH PASS FILTER 2
# Penajaman standar
# ======================================
kernel_hpf2 = np.array([
    [-1, -1, -1],
    [-1,  9, -1],
    [-1, -1, -1]
], dtype=np.float32)

hasil_hpf2 = cv2.filter2D(
    citra_noisy, -1, kernel_hpf2
)

# ======================================
# 6. HIGH PASS FILTER 3
# Penajaman lebih kuat
# ======================================
kernel_hpf3 = np.array([
    [-2, -2, -2],
    [-2, 17, -2],
    [-2, -2, -2]
], dtype=np.float32)

hasil_hpf3 = cv2.filter2D(
    citra_noisy, -1, kernel_hpf3
)

# ======================================
# 7. MENYIMPAN HASIL
# ======================================
hasil = {
    "citra_asli.jpg": citra_asli,
    "citra_noisy.jpg": citra_noisy,
    "hasil_hpf1.jpg": hasil_hpf1,
    "hasil_hpf2.jpg": hasil_hpf2,
    "hasil_hpf3.jpg": hasil_hpf3
}

for nama, gambar in hasil.items():
    cv2.imwrite(str(OUTPUT_DIR / nama), gambar)
    print(f"Tersimpan: {nama}")

# ======================================
# 8. MENAMPILKAN 5 GAMBAR SEJAJAR
# ======================================
judul = [
    "Citra Asli",
    "Citra Noisy",
    "Hasil HPF 1",
    "Hasil HPF 2",
    "Hasil HPF 3"
]

gambar_list = [
    citra_asli,
    citra_noisy,
    hasil_hpf1,
    hasil_hpf2,
    hasil_hpf3
]

fig, axes = plt.subplots(1, 5, figsize=(18, 5))

for ax, gambar, nama in zip(
    axes, gambar_list, judul
):
    ax.imshow(gambar, cmap="gray", vmin=0, vmax=255)
    ax.set_title(nama, fontsize=12)
    ax.axis("off")

plt.tight_layout()

plt.savefig(
    str(OUTPUT_DIR / "perbandingan_hpf.png"),
    dpi=150,
    bbox_inches="tight"
)

plt.show()

print("\nSemua proses HPF selesai!")
print("Hasil ada di folder output.")
