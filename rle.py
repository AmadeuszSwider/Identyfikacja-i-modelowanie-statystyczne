import rawpy
import numpy as np
from PIL import Image
from scipy.io import wavfile


def rle_compress(image):
    """Kompresja obrazu za pomocą algorytmu RLE (Run-Length Encoding)."""
    compressed = []
    prev_pixel = image[0]
    count = 1

    for pixel in image[1:]:
        if pixel == prev_pixel:
            count += 1
        else:
            compressed.append((count, prev_pixel))
            prev_pixel = pixel
            count = 1
    compressed.append((count, prev_pixel))  # Dodanie ostatniej sekwencji
    return compressed

def rle_decompress(compressed):
    """Dekompresja obrazu skompresowanego algorytmem RLE."""
    decompressed = []
    for count, pixel in compressed:
        decompressed.extend([pixel] * count)
    return decompressed



# Wczytanie pliku ARW
def load_arw_image(file_path):
    """Wczytuje plik ARW i zwraca obraz w formacie numpy."""
    with rawpy.imread(file_path) as raw:
        # Pobranie danych obrazu (w formacie RGGB Bayer)
        rgb_image = raw.postprocess()

      #  if save_binary_path:
      #      np.save(save_binary_path, rgb_image)
        return rgb_image

# Zapis obrazu do pliku PNG
def save_image_to_file(image, filename):
    """Zapisuje obraz do pliku w formacie PNG."""
    pil_image = Image.fromarray(image)
    pil_image.save(filename)



def save_all_channels_rle_to_file(rle_red, rle_green, rle_blue, filename):
    with open(filename, 'wb') as f:
        for channel in (rle_red, rle_green, rle_blue):
            length = len(channel)
            f.write(np.uint32(length).tobytes())  # 4 bajty: ile par
            for count, value in channel:
                f.write(np.uint16(count).tobytes())  # 2 bajty
                f.write(np.uint8(value).tobytes())   # 1 bajt




def save_compression_info_to_file(image, compressed_red, compressed_green, compressed_blue, filename):
    with open(filename, 'wb') as f:
        # Zapisujemy liczbę pikseli w obrazie
        num_pixels = image.size  # Wartość liczby pikseli (height * width * 3)
        f.write(np.uint32(num_pixels).tobytes())  # Zapisujemy liczbę pikseli jako 4 bajty

        # Zapisujemy liczbę par (count, value) w skompresowanym obrazie
        num_red_pairs = len(compressed_red)
        num_green_pairs = len(compressed_green)
        num_blue_pairs = len(compressed_blue)

        f.write(np.uint32(num_red_pairs).tobytes())  # Zapisujemy liczbę par dla kanału czerwonego
        f.write(np.uint32(num_green_pairs).tobytes())  # Zapisujemy liczbę par dla kanału zielonego
        f.write(np.uint32(num_blue_pairs).tobytes())  # Zapisujemy liczbę par dla kanału niebieskiego

        # Zapisujemy skompresowane dane RLE dla każdego kanału
        # Zapisujemy kanał czerwony
        for count, value in compressed_red:
            f.write(np.uint16(count).tobytes())  # 2 bajty: count
            f.write(np.uint8(value).tobytes())   # 1 bajt: value

        # Zapisujemy kanał zielony
        for count, value in compressed_green:
            f.write(np.uint16(count).tobytes())  # 2 bajty: count
            f.write(np.uint8(value).tobytes())   # 1 bajt: value

        # Zapisujemy kanał niebieski
        for count, value in compressed_blue:
            f.write(np.uint16(count).tobytes())  # 2 bajty: count
            f.write(np.uint8(value).tobytes())   # 1 bajt: value





def load_compression_info_from_file(filename):
    with open(filename, 'rb') as f:
        # Odczytujemy liczbę pikseli w obrazie
        num_pixels = int.from_bytes(f.read(4), byteorder='little')

        # Odczytujemy liczbę par (count, value) dla każdego kanału
        num_red_pairs = int.from_bytes(f.read(4), byteorder='little')
        num_green_pairs = int.from_bytes(f.read(4), byteorder='little')
        num_blue_pairs = int.from_bytes(f.read(4), byteorder='little')

        # Odczytujemy skompresowane dane RLE dla każdego kanału
        compressed_red = []
        for _ in range(num_red_pairs):
            count = int.from_bytes(f.read(2), byteorder='little')
            value = int.from_bytes(f.read(1), byteorder='little')
            compressed_red.append((count, value))

        compressed_green = []
        for _ in range(num_green_pairs):
            count = int.from_bytes(f.read(2), byteorder='little')
            value = int.from_bytes(f.read(1), byteorder='little')
            compressed_green.append((count, value))

        compressed_blue = []
        for _ in range(num_blue_pairs):
            count = int.from_bytes(f.read(2), byteorder='little')
            value = int.from_bytes(f.read(1), byteorder='little')
            compressed_blue.append((count, value))

    return num_pixels, num_red_pairs, num_green_pairs, num_blue_pairs, compressed_red, compressed_green, compressed_blue




image_pil = Image.open("D:/Pobrane/aaafff.png")
image_mode = image_pil.mode
print(f"Obraz jest w trybie: {image_mode}")
if image_mode == 'P':  # 'L' oznacza tryb szarości
    image_pil = image_pil.convert('RGB')  # Konwertujemy na RGB

image = np.array(image_pil)


red_channel = image[:, :, 0].flatten()  # Zbieramy tylko dane z kanału R
green_channel = image[:, :, 1].flatten()  # Zbieramy dane z kanału G
blue_channel = image[:, :, 2].flatten()  # Zbieramy dane z kanału B

print("Obraz przed kompresją - Red:", red_channel)
compressed_red = rle_compress(red_channel)
#print("Obraz po kompresji - Red:", compressed_red)

print("Obraz przed kompresją - Green:", green_channel)
compressed_green = rle_compress(green_channel)
#print("Obraz po kompresji - Green:", compressed_green)

print("Obraz przed kompresją - Blue:", blue_channel)
compressed_blue = rle_compress(blue_channel)
#print("Obraz po kompresji - Blue:", compressed_blue)

# Dekompresja każdego kanału
decompressed_red = rle_decompress(compressed_red)
decompressed_green = rle_decompress(compressed_green)
decompressed_blue = rle_decompress(compressed_blue)

# Zrekonstruowanie obrazu (powrót do 2D)
decompressed_red_2d = np.array(decompressed_red).reshape(image.shape[0], image.shape[1])
decompressed_green_2d = np.array(decompressed_green).reshape(image.shape[0], image.shape[1])
decompressed_blue_2d = np.array(decompressed_blue).reshape(image.shape[0], image.shape[1])



save_compression_info_to_file(image, compressed_red, compressed_green, compressed_blue, "compression_info.bin")
(num_pixels, num_red_pairs, num_green_pairs, num_blue_pairs,
 compressed_red, compressed_green, compressed_blue) = load_compression_info_from_file("compression_info.bin")

# Wyświetlanie wyników
print(f"Liczba pikseli przed kompresją: {num_pixels}")
print(f"Liczba par (count, value) po kompresji dla kanału czerwonego: {num_red_pairs}")
print(f"Liczba par (count, value) po kompresji dla kanału zielonego: {num_green_pairs}")
print(f"Liczba par (count, value) po kompresji dla kanału niebieskiego: {num_blue_pairs}")
# Zrekonstruowanie obrazu RGB z dekompresowanych kanałów
reconstructed_image = np.stack((decompressed_red_2d, decompressed_green_2d, decompressed_blue_2d), axis=2)

# Zapisanie obrazu do pliku PNG

save_image_to_file(reconstructed_image.astype(np.uint8), "reconstructed_image.png")

