import numpy as np
import cv2
from pathlib import Path

# Pastikan folder image ada
Path('image').mkdir(exist_ok=True)

# Membuat gambar acuan berwarna-warni (ukuran 640x480)
width, height = 640, 480
img = np.zeros((height, width, 3), dtype=np.uint8)

# Tambahkan pola kotak-kotak warna
for i in range(0, width, 80):
    for j in range(0, height, 80):
        # Warna acak RGB
        color = tuple(int(np.random.randint(50, 255)) for _ in range(3))
        cv2.rectangle(img, (i, j), (i+79, j+79), color, -1)

# Simpan ke dalam folder image/foto.jpg
cv2.imwrite('image/foto.jpg', img)
print("✅ Gambar 'image/foto.jpg' berhasil dibuat!")