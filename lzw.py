from PIL import Image
import array

# --- LZW Kompresja ---
def lzw_compress(data):
    dictionary = {bytes([i]): i for i in range(256)}
    w = b""
    compressed = []
    dict_size = 256
    max_dict_size = 256

    for c in data:
        wc = w + bytes([c])
        if wc in dictionary:
            w = wc
        else:
            compressed.append(dictionary[w])
            dictionary[wc] = dict_size
            dict_size += 1
            max_dict_size = max(max_dict_size, dict_size)
            w = bytes([c])
    if w:
        compressed.append(dictionary[w])
    return compressed, max_dict_size

# --- LZW Dekompresja ---
def lzw_decompress(compressed):
    dictionary = {i: bytes([i]) for i in range(256)}
    dict_size = 256
    result = bytearray()

    w = bytes([compressed.pop(0)])
    result.extend(w)

    for k in compressed:
        if k in dictionary:
            entry = dictionary[k]
        elif k == dict_size:
            entry = w + bytes([w[0]])
        else:
            raise ValueError("Błąd dekompresji.")
        result.extend(entry)

        dictionary[dict_size] = w + bytes([entry[0]])
        dict_size += 1
        w = entry
    return result

# --- Zapis skompresowanych kanałów jako binarny plik ---
def save_rgb_binary_lzw(r, g, b, size, filename):
    with open(filename, "wb") as f:
        # Zapisz rozmiar obrazu (szerokość i wysokość)
        f.write(array.array('H', size).tobytes())

        for channel in (r, g, b):
            f.write(array.array('I', [len(channel)]).tobytes())  # długość kanału
            f.write(array.array('I', channel).tobytes())         # dane kanału

# --- Odczyt skompresowanych danych RGB ---
def load_rgb_binary_lzw(filename):
    with open(filename, "rb") as f:
        size = tuple(array.array('H', f.read(4)))  # szerokość, wysokość

        channels = []
        for _ in range(3):
            length = array.array('I', f.read(4))[0]  # długość kanału
            data = array.array('I')
            data.frombytes(f.read(length * 4))
            channels.append(data.tolist())
        return size, channels

# --- Kompresja obrazu RGB ---
def compress_image_rgb(input_path, output_path):
    img = Image.open(input_path).convert("RGB")
    r, g, b = img.split()

    r_data = list(r.getdata())
    g_data = list(g.getdata())
    b_data = list(b.getdata())

    r_comp, r_dict = lzw_compress(r_data)
    g_comp, g_dict = lzw_compress(g_data)
    b_comp, b_dict = lzw_compress(b_data)

    save_rgb_binary_lzw(r_comp, g_comp, b_comp, img.size, output_path)

    original_size = len(r_data) * 3
    compressed_size = (len(r_comp) + len(g_comp) + len(b_comp)) * 4

    print(f"Kompresja zakończona.")
    print(f"Rozmiar oryginału: {original_size} bajtów")
    print(f"Rozmiar skompresowany: {compressed_size} bajtów")
    print(f"Maksymalny rozmiar słownika:")
    print(f"  R: {r_dict} wpisów")
    print(f"  G: {g_dict} wpisów")
    print(f"  B: {b_dict} wpisów")

# --- Dekompresja obrazu RGB ---
def decompress_image_rgb(compressed_path, output_path):
    size, (r_comp, g_comp, b_comp) = load_rgb_binary_lzw(compressed_path)

    r_data = lzw_decompress(r_comp)
    g_data = lzw_decompress(g_comp)
    b_data = lzw_decompress(b_comp)

    r_img = Image.new("L", size)
    g_img = Image.new("L", size)
    b_img = Image.new("L", size)

    r_img.putdata(r_data)
    g_img.putdata(g_data)
    b_img.putdata(b_data)

    img = Image.merge("RGB", (r_img, g_img, b_img))
    img.save(output_path)
    print(f"Dekompresja zakończona. Obraz zapisany jako: {output_path}")

# --- Przykład użycia ---
if __name__ == "__main__":
    original_image = "obraz2.png"
    compressed_file = "obraz_rgb.lzw"
    decompressed_image = "obraz_rgb_decompressed.png"

    compress_image_rgb(original_image, compressed_file)
    decompress_image_rgb(compressed_file, decompressed_image)
