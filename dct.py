import rawpy
import imageio
import numpy as np
from PIL import Image
import io
import os


# 1. Wczytaj plik ARW i skonwertuj do RGB
def load_raw_image(path):
    with rawpy.imread(path) as raw:
        rgb = raw.postprocess(
            use_camera_wb=True,  # Użyj WB z aparatu
            user_flip=0,
            output_color=rawpy.ColorSpace.sRGB,
            output_bps=8
        )
    return rgb

def ycbcr_to_rgb(img):
    img = img.astype(np.float32)
    rgb = np.empty_like(img)
    rgb[..., 0] = img[..., 0] + 1.402 * (img[..., 2] - 128)
    rgb[..., 1] = img[..., 0] - 0.344136 * (img[..., 1] - 128) - 0.714136 * (img[..., 2] - 128)
    rgb[..., 2] = img[..., 0] + 1.772 * (img[..., 1] - 128)
    return np.clip(rgb, 0, 255)

def rgb_to_ycbcr(img, saturation=1.0):
    img = img.astype(np.float32)
    YCbCr = np.empty_like(img)

    # Luminancja (jasność)
    YCbCr[..., 0] = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]

    # Chrominancje (kolor), z regulacją nasycenia
    Cb = -0.168736 * img[..., 0] - 0.331264 * img[..., 1] + 0.5 * img[..., 2] + 128
    Cr = 0.5 * img[..., 0] - 0.418688 * img[..., 1] - 0.081312 * img[..., 2] + 128

    # Regulacja nasycenia względem środka (128)
    Cb = 128 + saturation * (Cb - 128)
    Cr = 128 + saturation * (Cr - 128)

    # Złożenie kanałów
    YCbCr[..., 1] = np.clip(Cb, 16, 240)
    YCbCr[..., 2] = np.clip(Cr, 16, 240)

    return YCbCr


def dct_matrix(N):
    M = np.zeros((N, N))
    for k in range(N):
        for n in range(N):
            alpha = np.sqrt(1.0 / N) if k == 0 else np.sqrt(2.0 / N)
            M[k, n] = alpha * np.cos(np.pi * (2 * n + 1) * k / (2 * N))
    return M

D = dct_matrix(8)

def dct2_matrix(block):
    return D @ block @ D.T


def idct2_matrix(block):
    return D.T @ block @ D
    
# 3. Podział na bloki 8x8 i DCT
def block_process(channel, block_size=8):
    h, w = channel.shape
    dct_blocks = []
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = channel[i:i + block_size, j:j + block_size]
            if block.shape == (block_size, block_size):
                dct = dct2_matrix(block)
                #dct = scipy.fftpack.dct(scipy.fftpack.dct(block.T, norm='ortho').T, norm='ortho')
                dct_blocks.append(dct)
    return dct_blocks

def blocks_to_image(blocks, h, w, block_size=8):
    image = np.zeros((h, w), dtype=np.float32)
    idx = 0
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            if idx < len(blocks):
                image[i:i+block_size, j:j+block_size] = blocks[idx]
                idx += 1
    return image

# 4. Kwantyzacja (dla uproszczenia używamy standardowej macierzy)
QY = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])


def quantize(blocks, Q=QY):
    quantized = [np.round(block / Q).astype(np.int32) for block in blocks]
    return quantized


# 5. Prosta Huffman (dla demonstracji - nieoptymalna)
from collections import Counter
import heapq


class Node:
    def __init__(self, freq, symbol=None, left=None, right=None):
        self.freq = freq
        self.symbol = symbol
        self.left = left
        self.right = right

    def __lt__(self, other):
        return self.freq < other.freq


def build_huffman_tree(data):
    freq = Counter(data)
    heap = [Node(f, sym) for sym, f in freq.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        n1 = heapq.heappop(heap)
        n2 = heapq.heappop(heap)
        new_node = Node(n1.freq + n2.freq, left=n1, right=n2)
        heapq.heappush(heap, new_node)

    return heap[0]


def generate_codes(node, prefix="", codebook=None):
    if codebook is None:
        codebook = {}
    if node.symbol is not None:
        codebook[node.symbol] = prefix
    else:
        generate_codes(node.left, prefix + "0", codebook)
        generate_codes(node.right, prefix + "1", codebook)
    return codebook




def reconstruct_and_save_custom_dct(arw_path, out_path="rekonstrukcja.jpg"):
    rgb = load_raw_image(arw_path)
    ycbcr = rgb_to_ycbcr(rgb)

    y = ycbcr[..., 0]
    cb = ycbcr[..., 1]
    cr = ycbcr[..., 2]

    h, w = y.shape
    y_blocks = block_process(y)
    y_quant = quantize(y_blocks)

    # Dekwantyzacja
    y_dequant = [block * QY for block in y_quant]

    # IDCT
    y_reconstructed_blocks = [idct2_matrix(block) for block in y_dequant]

    # Składanie kanału Y
    y_rec = blocks_to_image(y_reconstructed_blocks, h, w)

    # Składanie obrazu YCbCr
    ycbcr_rec = np.stack([y_rec, cb, cr], axis=-1)

    # YCbCr → RGB
    rgb_rec = ycbcr_to_rgb(ycbcr_rec)
    rgb_rec = np.clip(rgb_rec, 0, 255).astype(np.uint8)

    # Zapisz wynik
    Image.fromarray(rgb_rec).save(out_path)
    print(f"Zapisano zrekonstruowany obraz z własnej kompresji jako: {out_path}")
    original_size = os.path.getsize(arw_path)
    jpeg_size = os.path.getsize(out_path)
    print(f"Rozmiar oryginalnego pliku ARW: {original_size / (1024 * 1024):.2f} MB")
    print(f"Rozmiar przetworzonego pliku JPEG: {jpeg_size / (1024 * 1024):.2f} MB")
    print(f"Stopień kompresji: {( 100 - (100*jpeg_size/original_size))} %")


if __name__ == '__main__':
    reconstruct_and_save_custom_dct("D:/Pobrane/kot.ARW", "rekonstrukcja.jpg")

    # Wczytaj ponownie zapisany JPEG
    jpeg_loaded = np.array(Image.open("rekonstrukcja.jpg")).astype(np.float32)
    rgb = load_raw_image("D:/Pobrane/kot.ARW")
    # Oblicz różnicę i wzmocnij
    diff = np.abs(rgb.astype(np.float32) - jpeg_loaded)
    diff *= 30  # wzmocnienie różnicy dla wizualizacji
    diff = np.clip(diff, 0, 255).astype(np.uint8)

    # Zapisz różnicę jako obraz
    Image.fromarray(diff).save("roznica.jpg")
    print("Zapisano obraz różnicy: roznica.jpg")

