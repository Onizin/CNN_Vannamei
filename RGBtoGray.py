import numpy as np

def rgb2gray(rgb):
    h, w, _ = rgb.shape
    
    # Inisialisasi array untuk gambar grayscale
    gray = np.zeros((h, w), dtype=np.float32)
    
    # Bobot untuk konversi RGB ke Grayscale
    r_weight = 0.2989
    g_weight = 0.5870
    b_weight = 0.1140
    
    # Melakukan perkalian manual untuk setiap piksel
    for i in range(h):
        for j in range(w):
            r, g, b = rgb[i, j]
            gray[i, j] = r * r_weight + g * g_weight + b * b_weight
    
    return gray