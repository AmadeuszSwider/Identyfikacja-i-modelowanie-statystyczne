import os
import math
import cv2
import numpy as np
import rawpy
from scipy.io import wavfile
from PIL import Image
from pydub import AudioSegment
from scipy.signal.windows import hann

# Stała PI
pi = math.pi

QUANTIZATION_MATRIX = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])


# ======================== ZOPTYMALIZOWANA FUNKCJA DCT ========================
def precompute_dct_matrix(N):
    #"""Oblicza macierz DCT tylko raz dla danej wielkości bloku"""
    factor = np.array([(1 / np.sqrt(N)) if i == 0 else np.sqrt(2 / N) for i in range(N)])
    indices = np.arange(N)
    cos_values = np.cos(((2 * indices[:, None] + 1) * indices * pi) / (2 * N))
    return factor[:, None] * cos_values

def pprecompute_dct_matrix_1d(N):
    return np.array([
        [np.sqrt(1/N) if k == 0 else np.sqrt(2/N) * np.cos(np.pi * (2*n + 1) * k / (2 * N))
         for n in range(N)]
        for k in range(N)
    ])

def precompute_dct_matrix_1d(N):
    k = np.arange(N).reshape(-1, 1)
    n = np.arange(N).reshape(1, -1)
    factor = np.sqrt(2 / N) * np.cos(np.pi * (2 * n + 1) * k / (2 * N))
    factor[0, :] *= 1 / np.sqrt(2)
    return factor

def dctTransform(matrix, dct_m):
    return np.dot(np.dot(dct_m, matrix), dct_m.T)

def idctTransform(matrix, dct_m):
    return np.dot(np.dot(dct_m.T, matrix), dct_m)

def dct1d(vector, dct_m):
    return dct_m @ vector

def iidct1d(vector, dct_m):
    return dct_m.T @ vector


def idct1d(vector, dct_m):
    inv_dct = np.linalg.inv(dct_m)
    return inv_dct @ vector


def quantize(block, quality=50):
    #""" Zastosowanie kwantyzacji do bloku DCT (jakość od 1-100, wyższa = lepsza jakość) """
    scale = max(1, min(100, quality)) / 50.0
    q_matrix = np.maximum(QUANTIZATION_MATRIX * scale, 1)
    quantized = np.round(block / q_matrix).astype(np.int16)
    return quantized# Kwantyzacja DCT


def dequantize(block, quality=50):
    #""" Odtwarzanie wartości DCT po kwantyzacji """
    scale = max(1, min(100, quality)) / 50.0
    q_matrix = np.maximum(QUANTIZATION_MATRIX * scale, 1)
    dequantized = (block * q_matrix).astype(np.float32)  # Odtwarzanie wartości DCT


    return dequantized
# ======================== DCT DLA OBRAZÓW ========================
def compress_image(image_path, quality):
    #"""Wczytuje obraz, dzieli na bloki 8x8, wykonuje zoptymalizowaną DCT i zapisuje wynik"""
    #image = cv2.imread(image_path,cv2.IMREAD_COLOR)
    #if image is None:
    #    raise ValueError("Nie można wczytać obrazu!")

    with rawpy.imread(image_path) as raw:
        # Konwersja do formatu RGB
        image = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=False,output_bps=8)
        img = Image.fromarray(image)
       # img.show()

   # pixel = image[100, 100]  # Sprawdź piksel w pozycji (100, 100)
  #  print("RGB:", pixel)
    img = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
    img = np.clip(img, 0, 255)
    #img = np.float32(img)*255.0
    h, w, c = img.shape
    img -=128




    print(img.min())
    print(img.max())
    #image -= 0.5
    h_new = (h // 8 + (h % 8 != 0)) * 8
    w_new = (w // 8 + (w % 8 != 0)) * 8
    image_padded = cv2.copyMakeBorder(img, 0, h_new - h, 0, w_new - w, cv2.BORDER_CONSTANT, value=0)

    compressed_dct = np.zeros((h_new, w_new, c), dtype=np.float32)

    DCT_MAT = precompute_dct_matrix(8)

    # DCT na blokach 8x8
    for ch in range(c):
        for i in range(0, h_new, 8):
            for j in range(0, w_new, 8):
                block = image_padded[i:i+8, j:j+8, ch]
                dct_block = dctTransform(block, DCT_MAT)
                compressed_dct[i:i + 8, j:j + 8, ch] = quantize(dct_block, quality)



    print(compressed_dct.min())
    print(compressed_dct.max())

    return compressed_dct, image_padded.shape

def decompress_image(compressed_dct, shape, output_path, quality):
    h_new, w_new, c = shape

        # Macierz na zrekonstruowany obraz
    reconstructed = np.zeros((h_new, w_new, c), dtype=np.float32)

    DCT_MAT = precompute_dct_matrix(8)

    for ch in range(c):
        for i in range(0, h_new, 8):
            for j in range(0, w_new, 8):
                block = compressed_dct[i:i + 8, j:j + 8, ch]
                dequantized_block = dequantize(block, quality)
                reconstructed[i:i + 8, j:j + 8, ch] = idctTransform(dequantized_block, DCT_MAT)

    reconstructed = np.clip(reconstructed + 128, 0, 255)  # Przesunięcie poziomów
    reconstructed = reconstructed.astype(np.uint8)  # Konwersja do 8-bitowej skali
    reconstructed = cv2.cvtColor(reconstructed, cv2.COLOR_LAB2BGR)  # Popraw kolory
    cv2.imwrite(output_path, reconstructed)



# ======================== URUCHAMIANIE ========================
if __name__ == "__main__":
    compressed_dct, shape = compress_image("D:/Pobrane/kot.ARW", quality=30)
    decompress_image(compressed_dct, shape, "dct_image_output.jpg", quality=70)

    original_size = os.path.getsize("D:/Pobrane/kot.ARW")
    decompressed_size = os.path.getsize("dct_image_output.jpg")

    print(f"Oryginalny rozmiar: {original_size / 1024:.2f} KB")
    print(f"Rozmiar zdekompresowanego obrazu JPEG: {decompressed_size / 1024:.2f} KB")