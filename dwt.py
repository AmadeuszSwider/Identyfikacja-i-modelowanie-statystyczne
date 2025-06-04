import rawpy
import numpy as np
import pywt
import matplotlib.pyplot as plt
import imageio
import os

def load_arw_image_color(path):
    with rawpy.imread(path) as raw:
        rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=True)
        return rgb.astype(np.uint8)

def compress_dwt_color(image_rgb, wavelet='haar', level=2, threshold_ratio=0.05):
    compressed_channels = []
    for c in range(3):  # R, G, B
        channel = image_rgb[:, :, c]
        coeffs = pywt.wavedec2(channel, wavelet=wavelet, level=level)
        coeffs_arr, coeffs_slices = pywt.coeffs_to_array(coeffs)

        threshold = threshold_ratio * np.max(coeffs_arr)
        coeffs_arr[np.abs(coeffs_arr) < threshold] = 0

        coeffs_compressed = pywt.array_to_coeffs(coeffs_arr, coeffs_slices, output_format='wavedec2')
        reconstructed = pywt.waverec2(coeffs_compressed, wavelet=wavelet)
        compressed_channels.append(np.clip(reconstructed, 0, 255).astype(np.uint8))

    return np.stack(compressed_channels, axis=-1)

def save_as_jpeg2000(image_rgb, output_path):
    imageio.imwrite(output_path, image_rgb, format='jp2')

def format_size(bytes_size):
    return f"{bytes_size / (1024 * 1024):.2f} MB"

def show_images_with_difference(original_rgb, compressed_rgb):
    diff = np.abs(original_rgb.astype(np.int16) - compressed_rgb.astype(np.int16))
    diff = np.clip(diff * 5, 0, 255).astype(np.uint8)  # Wzmocnienie różnic 5x dla lepszej widoczności

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    axes[0].imshow(original_rgb)
    axes[0].set_title("Oryginalny obraz")
    axes[0].axis('off')

    axes[1].imshow(compressed_rgb)
    axes[1].set_title("Po kompresji DWT")
    axes[1].axis('off')

    axes[2].imshow(diff)
    axes[2].set_title("Różnica (wzmocniona)")
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

# === Przykład użycia ===
if __name__ == "__main__":
    path_to_arw = "1718863107_DSC00343.ARW"      # <- Podaj swoją ścieżkę do pliku .ARW
    output_jp2 = "wynik_kompresji.jp2"

    image_rgb = load_arw_image_color(path_to_arw)
    compressed_rgb = compress_dwt_color(image_rgb, wavelet='haar', level=3, threshold_ratio=0.04)

    save_as_jpeg2000(compressed_rgb, output_jp2)

    show_images_with_difference(image_rgb, compressed_rgb)

    original_size = os.path.getsize(path_to_arw)
    compressed_size = os.path.getsize(output_jp2)

    print(f"Rozmiar oryginalnego pliku (.ARW): {format_size(original_size)}")
    print(f"Rozmiar pliku po kompresji (.JP2):  {format_size(compressed_size)}")
