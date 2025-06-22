import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import rawpy
import numpy as np
import os
import array
import collections
import heapq
from bitarray import bitarray
import pywt
import imageio

# ==============================================================================
# SEKCJA 1: Klasy i logika dla algorytmów (Huffman, etc.)
# ==============================================================================

class HuffmanNode:
    """Węzeł drzewa Huffmana."""
    def __init__(self, byte=None, freq=0):
        self.byte = byte
        self.freq = freq
        self.left = None
        self.right = None
    
    def __lt__(self, other):
        return self.freq < other.freq
    def __eq__(self, other):
        return self.freq == other.freq

class HuffmanCodec:
    """Klasa do kompresji i dekompresji z użyciem kanonicznego kodu Huffmana."""
    def __init__(self):
        self._codes = {}
        self._root = None

    def _make_tree(self, freq_tab):
        heap = [HuffmanNode(key, value) for key, value in freq_tab.items()]
        heapq.heapify(heap)
        while len(heap) > 1:
            right, left = heapq.heappop(heap), heapq.heappop(heap)
            node = HuffmanNode(None, left.freq + right.freq)
            node.right, node.left = right, left
            heapq.heappush(heap, node)
        return heapq.heappop(heap) if heap else None

    def _make_codes_from_tree(self, node, current_code=bitarray()):
        if node is None: return
        if node.byte is not None:
            self._codes[node.byte] = current_code.copy()
            return
        self._make_codes_from_tree(node.left, current_code + bitarray('0'))
        self._make_codes_from_tree(node.right, current_code + bitarray('1'))

    def _sort_and_make_canonical(self):
        if not self._codes: return
        sorted_items = sorted({k: len(v) for k, v in self._codes.items()}.items(), key=lambda item: (item[1], item[0]))
        self._codes.clear()
        current_code_val, current_len = 0, 0
        for byte, length in sorted_items:
            if current_len != 0:
                current_code_val = (current_code_val + 1) << (length - current_len)
            current_len = length
            self._codes[byte] = bitarray(format(current_code_val, f'0{length}b'))

    def build(self, freq_tab):
        self._root = self._make_tree(freq_tab)
        self._codes.clear()
        self._make_codes_from_tree(self._root)
        self._sort_and_make_canonical()

    def compress(self, data):
        return bitarray().join(self._codes[byte] for byte in data)

    def decompress(self, bit_stream, output_size):
        reversed_codes = {v.to01(): k for k, v in self._codes.items()}
        decompressed, current_code = bytearray(), ""
        for bit in bit_stream:
            current_code += '1' if bit else '0'
            if current_code in reversed_codes:
                decompressed.append(reversed_codes[current_code])
                current_code = ""
                if len(decompressed) == output_size: break
        return decompressed
    
    def save_to_file(self, bit_stream, filename):
        with open(filename, 'wb') as f:
            f.write(bytes([len(self._codes) if len(self._codes) < 256 else 0]))
            for byte, code in self._codes.items(): f.write(bytes([byte, len(code)]))
            bit_stream.tofile(f)

    def load_from_file(self, filename):
        with open(filename, 'rb') as f:
            num_codes = int.from_bytes(f.read(1), 'big') or 256
            self._codes = {int.from_bytes(f.read(1),'big'):bitarray('0'*int.from_bytes(f.read(1),'big')) for _ in range(num_codes)}
            self._sort_and_make_canonical()
            bit_stream = bitarray(); bit_stream.fromfile(f)
        return bit_stream

# ==============================================================================
# SEKCJA 2: Główne funkcje przetwarzające dla wszystkich algorytmów
# ==============================================================================
D = dct_matrix(8) if 'dct_matrix' in globals() else None

def process_dct_and_get_results(arw_path, quality, out_path="rekonstrukcja_dct.jpg"):
    # (Implementation is unchanged, omitted for brevity)
    return None, None, None, "Implementacja DCT pominięta dla zwięzłości."

def process_rle_and_get_results(image_path, simplification_level, out_path="rekonstrukcja_rle.png", bin_path="skompresowany.rle"):
    # (Implementation is unchanged, omitted for brevity)
    return None, None, None, "Implementacja RLE pominięta dla zwięzłości."

def process_lzw_and_get_results(image_path, simplification_level, out_path="rekonstrukcja_lzw.png", bin_path="skompresowany.lzw"):
    # (Implementation is unchanged, omitted for brevity)
    return None, None, None, "Implementacja LZW pominięta dla zwięzłości."

def process_huffman_and_get_results(image_path, simplification_level, out_path="rekonstrukcja_huff.png", bin_path="skompresowany.huff"):
    try:
        original_pil = Image.open(image_path).convert('RGB')
        rgb_original = np.array(original_pil)
        rgb_processed = np.clip((rgb_original.astype(np.int32) // simplification_level) * simplification_level, 0, 255).astype(np.uint8) if simplification_level > 1 else rgb_original.copy()
        img_bytes = Image.fromarray(rgb_processed).tobytes()

        huff = HuffmanCodec(); huff.build(collections.Counter(img_bytes)); huff.save_to_file(huff.compress(img_bytes), bin_path)
        huff_dec = HuffmanCodec(); decompressed = huff_dec.decompress(huff_dec.load_from_file(bin_path), len(img_bytes))
        
        rgb_reconstructed = np.frombuffer(decompressed, dtype=np.uint8).reshape(rgb_original.shape)
        Image.fromarray(rgb_reconstructed).save(out_path)
        diff = np.clip(np.abs(rgb_original.astype(float) - rgb_reconstructed.astype(float)) * 10, 0, 255).astype(np.uint8)
        
        orig_size, comp_size = os.path.getsize(image_path), os.path.getsize(bin_path)
        stats = f"Oryginał: {orig_size/1024:.2f} KB | Po kompresji: {comp_size/1024:.2f} KB | Stopień kompresji: {(1 - comp_size/orig_size)*100:.2f}%"
        return rgb_original, rgb_reconstructed, diff, stats
    except Exception as e: return None, None, None, f"Błąd w Huffman: {e}"

def process_dwt_and_get_results(arw_path, wavelet, level, threshold_ratio, out_path="rekonstrukcja_dwt.jp2"):
    """Główna funkcja przetwarzająca dla algorytmu DWT."""
    try:
        # 1. Wczytanie obrazu
        with rawpy.imread(arw_path) as raw:
            rgb_original = raw.postprocess(use_camera_wb=True, no_auto_bright=True).astype(np.uint8)

        # 2. Kompresja DWT dla każdego kanału
        compressed_channels = []
        for c in range(3):  # R, G, B
            channel = rgb_original[:, :, c]
            coeffs = pywt.wavedec2(channel, wavelet=wavelet, level=level)
            coeffs_arr, coeffs_slices = pywt.coeffs_to_array(coeffs)

            # Progowanie (główny mechanizm kompresji)
            threshold = threshold_ratio * np.max(np.abs(coeffs_arr))
            coeffs_arr[np.abs(coeffs_arr) < threshold] = 0

            # Rekonstrukcja kanału
            coeffs_compressed = pywt.array_to_coeffs(coeffs_arr, coeffs_slices, output_format='wavedec2')
            reconstructed_channel = pywt.waverec2(coeffs_compressed, wavelet=wavelet)
            # Przytnij do oryginalnych wymiarów
            h, w = channel.shape
            reconstructed_channel = reconstructed_channel[:h, :w]
            compressed_channels.append(np.clip(reconstructed_channel, 0, 255).astype(np.uint8))

        rgb_reconstructed = np.stack(compressed_channels, axis=-1)

        # 3. Zapis i statystyki
        imageio.imwrite(out_path, rgb_reconstructed, format='jp2')
        diff = np.clip(np.abs(rgb_original.astype(float) - rgb_reconstructed.astype(float)) * 10, 0, 255).astype(np.uint8)
        
        orig_size, comp_size = os.path.getsize(arw_path), os.path.getsize(out_path)
        stats = f"Oryginał: {orig_size/1024/1024:.2f} MB | Po kompresji: {comp_size/1024:.2f} KB | Stopień kompresji: {(1 - comp_size/orig_size)*100:.2f}%"
        
        return rgb_original, rgb_reconstructed, diff, stats
    except Exception as e:
        return None, None, None, f"Błąd w DWT: {e}"

# ==============================================================================
# SEKCJA 3: Klasa Aplikacji GUI
# ==============================================================================

class AlgorithmGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("GUI dla Algorytmów Kompresji")
        self.root.geometry("1400x900")

        style = ttk.Style(); style.configure("TNotebook.Tab", padding=[10, 5], font=('Helvetica', 10))
        main_frame = ttk.Frame(self.root, padding="10"); main_frame.pack(fill=tk.BOTH, expand=True)
        self.notebook = ttk.Notebook(main_frame); self.notebook.pack(fill=tk.BOTH, expand=True)

        self.widgets = {'dct':{},'rle':{},'lzw':{},'huffman':{},'dwt':{}}
        self.paths = {key: None for key in self.widgets}

        # Stworzenie zakładek
        self._create_lossy_tab('dct', 'Kompresja DCT', ("Sony RAW", "*.ARW"), "Jakość Kompresji", 1, 100, 50)
        self._create_lossless_tab('rle', 'Kompresja RLE', "Poziom Upraszczania", 1, 32, 1)
        self._create_lossless_tab('lzw', 'Kompresja LZW', "Poziom Upraszczania", 1, 32, 1)
        self._create_lossless_tab('huffman', 'Kompresja Huffman', "Poziom Upraszczania", 1, 32, 1)
        self._create_dwt_tab()

    def _create_base_tab(self, key, title):
        tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(tab, text=title)
        widgets = self.widgets[key]
        
        ctrl_frame = ttk.Frame(tab); ctrl_frame.pack(fill=tk.X, pady=5)
        img_frame = ttk.Frame(tab); img_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        widgets['image_panels'] = self._create_image_panels(img_frame, ["Oryginał", "Zrekonstruowany", "Różnica"])
        return ctrl_frame, widgets

    def _create_lossless_tab(self, key, title, slider_text, from_, to, initial):
        ctrl_frame, widgets = self._create_base_tab(key, title)
        self._add_file_selector(ctrl_frame, key, "Wybierz plik obrazu", ("Pliki Obrazów", "*.png *.jpg *.bmp"))
        self._add_slider(ctrl_frame, widgets, slider_text, from_, to, initial, row=1)
        self._add_process_button(ctrl_frame, key, row=2)

    def _create_lossy_tab(self, key, title, file_types, slider_text, from_, to, initial):
        ctrl_frame, widgets = self._create_base_tab(key, title)
        self._add_file_selector(ctrl_frame, key, "Wybierz plik ARW", file_types)
        self._add_slider(ctrl_frame, widgets, slider_text, from_, to, initial, row=1)
        self._add_process_button(ctrl_frame, key, row=2)
        
    def _create_dwt_tab(self):
        key = 'dwt'
        ctrl_frame, widgets = self._create_base_tab(key, 'Kompresja DWT')
        self._add_file_selector(ctrl_frame, key, "Wybierz plik ARW", ("Sony RAW", "*.ARW"))
        
        # Parametry DWT
        param_frame = ttk.Frame(ctrl_frame); param_frame.grid(row=1, column=0, columnspan=2, sticky='ew', pady=5)
        
        # Wybór Falki
        ttk.Label(param_frame, text="Falka:").pack(side=tk.LEFT, padx=5)
        widgets['wavelet_var'] = tk.StringVar(value='haar')
        wavelets = pywt.wavelist(kind='discrete')
        ttk.Combobox(param_frame, textvariable=widgets['wavelet_var'], values=wavelets, state='readonly').pack(side=tk.LEFT, padx=5)

        # Poziom Dekompozycji
        widgets['level_var'] = tk.IntVar(value=2)
        self._add_slider(param_frame, widgets, "Poziom:", 1, 5, 2, key_suffix='level')

        # Próg
        widgets['thresh_var'] = tk.DoubleVar(value=0.1)
        self._add_slider(param_frame, widgets, "Próg:", 0.01, 1.0, 0.1, key_suffix='thresh', is_double=True)
        
        self._add_process_button(ctrl_frame, key, row=2)

    def _add_file_selector(self, parent, key, btn_text, file_types):
        ttk.Button(parent, text=btn_text, command=lambda k=key, ft=file_types: self._select_file(k, ft)).grid(row=0, column=0, padx=5, pady=5)
        self.widgets[key]['path_label'] = ttk.Label(parent, text="Nie wybrano pliku", width=70)
        self.widgets[key]['path_label'].grid(row=0, column=1, sticky="ew", padx=5)

    def _add_slider(self, parent, widgets, text, from_, to, initial, row=0, key_suffix='param', is_double=False):
        frame = ttk.Frame(parent); frame.grid(row=row, column=0, columnspan=2, sticky='ew', pady=2) if row > 0 else frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(frame, text=text).pack(side=tk.LEFT, padx=5)
        var = widgets.get(f'{key_suffix}_var', tk.DoubleVar(value=initial) if is_double else tk.IntVar(value=initial))
        widgets[f'{key_suffix}_var'] = var
        label = ttk.Label(frame, text=f"{initial:.2f}" if is_double else str(initial), width=4)
        cmd = (lambda v, w=label: w.config(text=f"{float(v):.2f}")) if is_double else (lambda v, w=label: w.config(text=str(int(float(v)))))
        ttk.Scale(frame, from_=from_, to=to, variable=var, orient=tk.HORIZONTAL, command=cmd).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        label.pack(side=tk.LEFT)
        return frame

    def _add_process_button(self, parent, key, row):
        widgets = self.widgets[key]
        widgets['process_button'] = ttk.Button(parent, text=f"Przetwarzaj {key.upper()}", command=lambda k=key: self._process(k), state=tk.DISABLED)
        widgets['process_button'].grid(row=row, column=0, columnspan=2, pady=10)
        widgets['status_label'] = ttk.Label(parent, text="Wybierz plik i kliknij 'Przetwarzaj'.", wraplength=1200)
        widgets['status_label'].grid(row=row+1, column=0, columnspan=2, pady=5, sticky='ew')

    def _create_image_panels(self, parent, titles):
        panels = {}; parent.columnconfigure(list(range(len(titles))), weight=1)
        for i, title in enumerate(titles):
            frame = ttk.LabelFrame(parent, text=title); frame.grid(row=0, column=i, sticky="nsew", padx=5)
            frame.rowconfigure(0, weight=1); frame.columnconfigure(0, weight=1)
            canvas = ttk.Label(frame); canvas.grid(sticky="nsew")
            panels[title] = {'canvas': canvas, 'photo': None}
        return panels

    def _select_file(self, key, file_types):
        path = filedialog.askopenfilename(title="Wybierz plik", filetypes=(file_types,))
        if path:
            self.paths[key] = path
            self.widgets[key]['path_label'].config(text=os.path.basename(path))
            self.widgets[key]['process_button'].config(state=tk.NORMAL)

    def _process(self, key):
        if not self.paths[key]: return
        widgets = self.widgets[key]; path = self.paths[key]
        widgets['status_label'].config(text=f"Przetwarzanie {key.upper()}... Czekaj."); self.root.update_idletasks()
        
        results = None
        try:
            if key in ['rle', 'lzw', 'huffman']:
                param = widgets['param_var'].get()
                if key == 'rle': results = process_rle_and_get_results(path, param)
                elif key == 'lzw': results = process_lzw_and_get_results(path, param)
                elif key == 'huffman': results = process_huffman_and_get_results(path, param)
            elif key == 'dct':
                results = process_dct_and_get_results(path, widgets['param_var'].get())
            elif key == 'dwt':
                results = process_dwt_and_get_results(path, widgets['wavelet_var'].get(), widgets['level_var'].get(), widgets['thresh_var'].get())
        except Exception as e:
            results = (None, None, None, f"Krytyczny błąd wykonania: {e}")

        orig, rec, diff, stats = results if results else (None, None, None, f"Błąd przetwarzania {key.upper()}.")
        widgets['status_label'].config(text=stats)
        if orig is not None:
            self._display_image(orig, widgets['image_panels']["Oryginał"])
            self._display_image(rec, widgets['image_panels']["Zrekonstruowany"])
            self._display_image(diff, widgets['image_panels']["Różnica"])

    def _display_image(self, np_array, panel_dict):
        canvas = panel_dict['canvas']
        max_w = canvas.winfo_width() if canvas.winfo_width() > 20 else 400
        max_h = canvas.winfo_height() if canvas.winfo_height() > 20 else 400
        try:
            img_pil = Image.fromarray(np_array)
            img_pil.thumbnail((max_w - 10, max_h - 10), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img_pil)
            panel_dict['photo'] = photo
            canvas.config(image=photo)
        except Exception as e: messagebox.showerror("Błąd wyświetlania", f"Nie można było wyświetlić obrazu: {e}")

if __name__ == '__main__':
    # Sprawdzenie, czy wymagane biblioteki są dostępne
    try:
        import rawpy, pywt, imageio
        from bitarray import bitarray
    except ImportError as e:
        print(f"Brakująca biblioteka: {e.name}. Zainstaluj ją używając 'pip install {e.name}'")
        exit()
        
    root = tk.Tk()
    app = AlgorithmGUI(root)
    root.mainloop()
