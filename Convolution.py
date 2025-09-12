from torch import nn
from activation import Activation_Conv2d as AC
import torch
import numpy as np
import psutil
import gc
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# cross corelation untuk image
# modul untuk memanggil perkalian pada image dengan kernel yang dipilih berdasarkan lampiran filter
def corr2d(X, K, padding=0, max_memory_usage=0.6, check_interval=1000):
    batch_size, in_channels, height, width = X.shape
    out_channels, in_channels_k, kernel_height, kernel_width = K.shape

    # Validasi jumlah channel input dan kernel
    if in_channels != in_channels_k:
        raise ValueError(f"Input channels ({in_channels}) must match kernel channels ({in_channels_k}).")

    # Dimensi output
    out_height = height - kernel_height + 1
    out_width = width - kernel_width + 1

    # Inisialisasi output tensor
    Y = torch.zeros((batch_size, out_channels, out_height, out_width), dtype=X.dtype)

    # Fungsi untuk memantau penggunaan memori
    def check_memory_usage():
        memory_info = psutil.virtual_memory()
        return memory_info.percent / 100.0

    # Perhitungan cross-correlation
    counter = 0
    for b in range(batch_size):  # Iterasi batch
        for oc in range(out_channels):  # Iterasi output channels
            for i in range(out_height):
                for j in range(out_width):
                    for ic in range(in_channels):  # Iterasi input channels
                        Y[b, oc, i, j] += (X[b, ic, i:i + kernel_height, j:j + kernel_width] * K[oc, ic]).sum()
                    counter += 1
                    if counter % check_interval == 0:
                        logging.info(f"Processed {counter} elements")
                        if check_memory_usage() > max_memory_usage:
                            gc.collect()  # Mengelola memori secara manual
                            logging.warning("Memory usage exceeded, extending render time...")

                    # Terapkan ReLU custom jika diperlukan
                    if Y[b, oc, i, j] > 255:
                        Y[b, oc, i, j] = 255
                    else:
                        Y[b, oc, i, j] = AC.Relu_custom(Y[b, oc, i, j])

    if padding > 0:
        Y = torch.nn.functional.pad(Y, (padding, padding, padding, padding))
    return Y

# Class Conv2D menggunakan NumPy
class Conv2D(nn.Module):
    def __init__(self, kernel, padding):
        """
        Parameters:
        - kernel: 4D tensor (out_channels, in_channels, kernel_height, kernel_width)
        """
        super().__init__()
        self.weight = kernel  # Kernel sebagai parameter
        self.padding = padding

    def forward(self, x):
        """
        Parameters:
        - x: Input tensor (batch_size, channels, height, width)
        """
        return corr2d(x, self.weight, self.padding)  # Panggil corr2d dengan padding

def pool2d(X, pool_size, mode='max', stride=(2, 2), max_memory_usage=0.6, check_interval=1000):
    batch_size, in_channels, height, width = X.shape

    p_h, p_w = pool_size  # Extract pool_height and pool_width
    stride_h, stride_w = stride  # Extract stride_height and stride_width
 # Stride untuk tinggi dan lebar
    
    # Menghitung ukuran output
    out_h = (height - p_h) // stride_h + 1
    out_w = (width - p_w) // stride_w + 1
    
    # Tensor kosong untuk hasil pooling
    Y = torch.zeros((batch_size, in_channels, out_h, out_w), dtype=X.dtype)
    
    def check_memory_usage():
        memory_info = psutil.virtual_memory()
        return memory_info.percent / 100.0
    
    counter = 0
    # Looping dengan stride
    for b in range(batch_size):  # Iterasi batch
        for c in range(in_channels):  # Iterasi channel
            # Loop untuk tinggi dan lebar output
            for i in range(0, out_h):
                for j in range(0, out_w):
                    # Region pooling
                    region = X[b, c, i * stride_h:i * stride_h + p_h, j * stride_w:j * stride_w + p_w]
                    if mode == 'max':
                        Y[b, c, i, j] = region.max()
                    elif mode == 'avg':
                        Y[b, c, i, j] = region.mean()
                    else:
                        raise ValueError(f"Mode pooling '{mode}' tidak valid. Gunakan 'max' atau 'avg'.")
                    counter += 1
                    if counter % check_interval == 0:
                        logging.info(f"Processed {counter} elements")
                        if check_memory_usage() > max_memory_usage:
                            gc.collect()  # Mengelola memori secara manual
                            logging.warning("Memory usage exceeded, extending render time...")
    return Y

def flatten(X):
    """
    Parameters:
    - X: Input tensor with shape (batch_size, channels, height, width)

    Returns:
    - Flattened tensor with shape (batch_size, flattened_size)
    """
    batch_size, in_channels, height, width = X.shape  # Dapatkan dimensi tensor
    
    flattened_size = in_channels * height * width  # Hitung ukuran flattened
    Y = torch.zeros((batch_size, flattened_size), dtype=X.dtype)  # Inisialisasi tensor output
    
    # Iterasi untuk setiap elemen dalam batch
    for b in range(batch_size):  # Iterasi batch
        idx = 0  # Indeks untuk elemen flattened
        for c in range(in_channels):  # Iterasi channel
            for i in range(height):  # Iterasi tinggi
                for j in range(width):  # Iterasi lebar
                    # Salin nilai dari tensor asli ke tensor flattened
                    Y[b, idx] = X[b, c, i, j]
                    idx += 1  # Perbarui indeks flattened
        # Cetak informasi setelah satu gambar selesai diproses
        print(f"Processed image {b+1}/{batch_size}")           
    return Y
