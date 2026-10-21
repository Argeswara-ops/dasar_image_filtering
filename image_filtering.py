import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- KONFIGURASI ---
# Masukkan nama gambar di sini (maksimal 2 untuk tampilan yang rapi)
daftar_gambar = ['foto.jpg', 'foto2.jpg'] 
# -------------------

# Siapkan variabel untuk menampung hasil
hasil_filter = []
nama_gambar_list = []

for img_name in daftar_gambar:
    # 1. Tentukan path gambar (sesuaikan dengan folder 'image/')
    img_path = Path('image') / img_name 

    # 2. Cek apakah gambar ada
    if not img_path.exists():
        print(f"⚠️ Gambar '{img_name}' tidak ditemukan.")
        continue
        
    # 3. Load gambar dengan OpenCV
    img = cv2.imread(str(img_path))

    if img is None:
        print(f"❌ Gagal memuat gambar '{img_name}'!")
        continue

    # OpenCV menggunakan BGR, ubah ke RGB untuk matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # --- Berbagai Filter ---
    
    # a. Gaussian Blur
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    blur_rgb = cv2.cvtColor(blur, cv2.COLOR_BGR2RGB)

    # b. Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # c. Edge Detection (Canny)
    edges = cv2.Canny(gray, 100, 200)

    # d. Sharpening
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(img, -1, kernel)
    sharpened_rgb = cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB)

    # Simpan hasil proses untuk gambar ini ke dalam list
    hasil_filter.append({
        'nama': img_name,
        'asli': img_rgb,
        'blur': blur_rgb,
        'gray': gray,
        'edges': edges,
        'sharpened': sharpened_rgb
    })

# --- Tampilkan Hasil ---
if hasil_filter:
    # Tampilkan gambar asli dari semua yang diproses
    plt.figure(figsize=(15, 5 * len(hasil_filter)))
    
    for i, data in enumerate(hasil_filter):
        # Gambar Asli
        plt.subplot(len(hasil_filter), 5, i * 5 + 1)
        plt.imshow(data['asli'])
        plt.title(f"Original: {data['nama']}")
        plt.axis('off')

        # Gaussian Blur
        plt.subplot(len(hasil_filter), 5, i * 5 + 2)
        plt.imshow(data['blur'])
        plt.title('Gaussian Blur')
        plt.axis('off')

        # Sharpen
        plt.subplot(len(hasil_filter), 5, i * 5 + 3)
        plt.imshow(data['sharpened'])
        plt.title('Sharpened')
        plt.axis('off')

        # Grayscale
        plt.subplot(len(hasil_filter), 5, i * 5 + 4)
        plt.imshow(data['gray'], cmap='gray')
        plt.title('Grayscale')
        plt.axis('off')

        # Edge Detection
        plt.subplot(len(hasil_filter), 5, i * 5 + 5)
        plt.imshow(data['edges'], cmap='gray')
        plt.title('Edge Detection')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

    # --- Simpan Hasil ke File ---
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)

    for data in hasil_filter:
        # Simpan blur
        cv2.imwrite(str(output_dir / f"{data['nama']}_blur.jpg"), cv2.cvtColor(data['blur'], cv2.COLOR_RGB2BGR))
        print(f"💾 Blur saved for {data['nama']}")
        
        # Simpan sharpened
        cv2.imwrite(str(output_dir / f"{data['nama']}_sharpened.jpg"), cv2.cvtColor(data['sharpened'], cv2.COLOR_RGB2BGR))
        print(f"💾 Sharpened saved for {data['nama']}")

    print("\n✅ Semua filter selesai!")
else:
    print("❌ Tidak ada gambar yang berhasil diproses.")