import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

Path("outputs").mkdir(exist_ok=True)

#membaca citra
img = cv2.imread("image/foto.jpg", cv2.IMREAD_GRAYSCALE)
if img is None:
   raise FileNotFoundError(
      "Citra tidak ditemukan. "
      "Pastikan foto berada di folder yang sama"
      )

#menambahkan noise pada citra
noisy = img.copy()
rng = np.random.default_rng(42)
noise_mask = rng.random(img.shape[:2])
noisy[noise_mask < 0.02] = (0)
noisy[noise_mask > 0.98] = (255)

print("Ukuran citra: ", noisy.shape)
print("Tipe data citra: ", noisy.dtype)

#Menerapkan Median Filter
median1 = cv2.medianBlur(noisy, 3)
median2 = cv2.medianBlur(noisy, 5)
median3 = cv2.medianBlur(noisy, 7)

#Menerapkan Gaussian Filter

# Menerapkan Gaussian Filter menggunakan kernel manual

# Kernel Gaussian 3x3
kernel3 = np.array([
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1]
], dtype=np.float32) / 16

hasil_gaussian1 = cv2.filter2D(noisy, -1, kernel3)

# Kernel Gaussian 5x5
kernel5 = np.array([
    [1,  4,  6,  4, 1],
    [4, 16, 24, 16, 4],
    [6, 24, 36, 24, 6],
    [4, 16, 24, 16, 4],
    [1,  4,  6,  4, 1]
], dtype=np.float32) / 256

hasil_gaussian2 = cv2.filter2D(noisy, -1, kernel5)

# Kernel Gaussian 7x7
kernel7 = np.array([
    [1,  6,  15,  20,  15,  6, 1],
    [6, 36,  90, 120,  90, 36, 6],
    [15, 90, 225, 300, 225, 90, 15],
    [20, 120, 300, 400, 300, 120, 20],
    [15, 90, 225, 300, 225, 90, 15],
    [6, 36,  90, 120,  90, 36, 6],
    [1,  6,  15,  20,  15,  6, 1]
], dtype=np.float32) / 4096

hasil_gaussian3 = cv2.filter2D(noisy, -1, kernel7)


#menampilkan citra asli
plt.figure(figsize=(20, 5))

plt.subplot(1, 8, 1)
plt.imshow(img, cmap="gray", vmin=0, vmax=255)
plt.title("Citra Asli")
plt.axis("off")

plt.subplot(1, 8, 2)
plt.imshow(noisy, cmap="gray", vmin=0, vmax=255)
plt.title("Citra noisy")
plt.axis("off")

plt.subplot(1, 8, 3)
plt.imshow(median1, cmap="gray", vmin=0, vmax=255)
plt.title("Citra Median 1")
plt.axis("off")

plt.subplot(1, 8, 4)
plt.imshow(median2, cmap="gray", vmin=0, vmax=255)
plt.title("Citra Median 2")
plt.axis("off")

plt.subplot(1, 8, 5)
plt.imshow(median3, cmap="gray", vmin=0, vmax=255)
plt.title("Citra Median 3")
plt.axis("off")


# Gaussian Filter 3x3
plt.subplot(1, 8, 6)
plt.imshow(hasil_gaussian1, cmap="gray", vmin=0, vmax=255)
plt.title("Gaussian 3x3")
plt.axis("off")

# Gaussian Filter 5x5
plt.subplot(1, 8, 7)
plt.imshow(hasil_gaussian2, cmap="gray", vmin=0, vmax=255)
plt.title("Gaussian 5x5")
plt.axis("off")

# Gaussian Filter 7x7
plt.subplot(1, 8, 8)
plt.imshow(hasil_gaussian3, cmap="gray", vmin=0, vmax=255)
plt.title("Gaussian 7x7")
plt.axis("off")


plt.tight_layout()
plt.show()
