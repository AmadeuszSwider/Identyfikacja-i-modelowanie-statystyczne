# -*- coding: utf-8 -*-

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import numpy as np
import os
import array
import collections
import heapq
import rawpy
import pywt
import imageio
from bitarray import bitarray

# ==============================================================================
# SEKCJA 1: KLASY I LOGIKA ALGORYTMÓW
# ==============================================================================

# ------------------------------------------------------------------------------
# 1.1. Logika dla algorytmu Huffmana
# ------------------------------------------------------------------------------
class HuffmanNode:
    """Węzeł drzewa Huffmana."""
    def __init__(self, byte=None, freq=0):
        self.byte = byte
        self.freq = freq
        self.left = None
        self.right = None
    
    def __lt__(self, other):
        return self.freq < other.freq

class HuffmanCodec:
    """Klasa do kompresji i dekompresji z użyciem kanonicznego kodu Huffmana."""
    def __init__(self):
        self._codes = {}
        self._root = None

    def _make_tree(self, freq_tab):
        heap = [HuffmanNode(key, value) for key, value in freq_tab.items()]
        heapq.heapify(heap)
        while len(heap) > 1:
            right = heapq.heappop(heap)
            left = heapq.heappop(heap)
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
        bit_stream = bitarray()
        for byte in data:
            if byte in self._codes:
                bit_stream.extend(self._codes[byte])
        return bit_stream

    def decompress(self, bit_stream, output_size):
        if not self._codes:
            return bytearray()
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
            num_symbols = len(self._codes)
            f.write(bytes([num_symbols if num_symbols < 256 else 0]))
            for byte, code in self._codes.items():
                f.write(bytes([byte, len(code)]))
            bit_stream.tofile(f)

    def load_from_file(self, filename):
        with open(filename, 'rb') as f:
            num_codes = int.from_bytes(f.read(1), 'big')
            if num_codes == 0: num_codes = 256
            
            self._codes = {int.from_bytes(f.read(1),'big'):bitarray('0'*int.from_bytes(f.read(1),'big')) for _ in range(num_codes)}
            self._sort_and_make_canonical()
            
            bit_stream = bitarray()
            bit_stream.fromfile(f)
        return bit_stream

# ------------------------------------------------------------------------------
# 1.2. Logika dla algorytmu DCT
# ------------------------------------------------------------------------------
QY_BASE = np.array([
    [16,11,10,16,24,40,51,61],[12,12,14,19,26,58,60,55],[14,13,16,24,40,57,69,56],
    [14,17,22,29,51,87,80,62],[18,22,37,56,68,109,103,77],[24,35,55,64,81,104,113,92],
    [49,64,78,87,103,121,120,101],[72,92,95,98,112,100,103,99]
])

def dct_matrix(N=8):
    M = np.zeros((N, N))
    for k in range(N):
        for n in range(N):
            alpha = np.sqrt(1.0/N) if k==0 else np.sqrt(2.0/N)
            M[k,n] = alpha * np.cos(np.pi * (2*n+1) * k / (2*N))
    return M
D = dct_matrix()

def process_dct_and_get_results(arw_path, quality, out_path="rekonstrukcja_dct.jpg"):
    def load_raw_image(path):
        with rawpy.imread(path) as raw:
            return raw.postprocess(use_camera_wb=True, output_color=rawpy.ColorSpace.sRGB, output_bps=8)
    def rgb_to_ycbcr(img):
        x = np.array([.299, .587, .114, -.168736, -.331264, .5, .5, -.418688, -.081312]).reshape(3,3)
        ycbcr = img.dot(x)
        ycbcr[:,:,[1,2]] += 128
        return ycbcr
    def ycbcr_to_rgb(img):
        x = np.array([1, 0, 1.402, 1, -0.344136, -0.714136, 1, 1.772, 0]).reshape(3,3)
        rgb = img.copy()
        rgb[:,:,[1,2]] -= 128
        rgb = rgb.dot(x)
        return np.clip(rgb, 0, 255)
    def dct2(block): return D @ (block - 128) @ D.T
    def idct2(block): return (D.T @ block @ D) + 128
    def process_channel(channel):
        h,w=channel.shape; h_pad=(h+7)//8*8; w_pad=(w+7)//8*8
        padded=np.pad(channel,((0,h_pad-h),(0,w_pad-w)),'edge')
        blocks=[dct2(padded[i:i+8,j:j+8]) for i in range(0,h_pad,8) for j in range(0,w_pad,8)]
        return blocks,h,w
    def unprocess_channel(blocks, h, w):
        h_pad=(h+7)//8*8; w_pad=(w+7)//8*8; img=np.zeros((h_pad,w_pad)); idx=0
        for i in range(0,h_pad,8):
            for j in range(0,w_pad,8):
                if idx<len(blocks): img[i:i+8,j:j+8]=blocks[idx]; idx+=1
        return img[:h,:w]
    def get_q_matrix(q):
        s = 5000/q if q < 50 else 200 - 2*q
        return np.clip((QY_BASE * s + 50) / 100, 1, 255).astype(np.int32)
    try:
        rgb_original = load_raw_image(arw_path)
        ycbcr = rgb_to_ycbcr(rgb_original)
        q_matrix = get_q_matrix(quality)
        y_blocks, h, w = process_channel(ycbcr[..., 0])
        y_quant = [np.round(block/q_matrix) for block in y_blocks]
        y_dequant = [block*q_matrix for block in y_quant]
        y_rec_blocks = [idct2(block) for block in y_dequant]
        y_rec = unprocess_channel(y_rec_blocks, h, w)
        rgb_reconstructed = ycbcr_to_rgb(np.stack([y_rec, ycbcr[...,1], ycbcr[...,2]], axis=-1)).astype(np.uint8)
        diff = np.clip(np.abs(rgb_original.astype(float) - rgb_reconstructed.astype(float)) * 10, 0, 255).astype(np.uint8)
        imageio.imwrite(out_path, rgb_reconstructed, quality=95)
        orig_s, comp_s=os.path.getsize(arw_path),os.path.getsize(out_path)
        stats = f"Oryginał: {orig_s/1024/1024:.2f} MB | Po kompresji: {comp_s/1024:.2f} KB | Kompresja: {(1-comp_s/orig_s)*100:.2f}%"
        return rgb_original, rgb_reconstructed, diff, stats
    except Exception as e: return None, None, None, f"Błąd w DCT: {e}"

# ------------------------------------------------------------------------------
# 1.3. Logika dla algorytmu RLE
# ------------------------------------------------------------------------------
def process_rle_and_get_results(image_path, simplification_level, out_path="rekonstrukcja_rle.png", bin_path="skompresowany.rle"):
    def rle_compress(data):
        if len(data)==0: return []
        c, p, n = [], data[0], 1
        for x in data[1:]:
            if x==p and n<65535: n+=1
            else: c.append((n,p)); p=x; n=1
        c.append((n,p)); return c
    def rle_decompress(comp): return [p for n,p in comp for _ in range(n)]
    def save_rle(ch, fn):
        with open(fn,'wb') as f:
            for c in ch:
                f.write(np.uint32(len(c)).tobytes())
                for n,p in c: f.write(np.uint16(n).tobytes()); f.write(np.uint8(p).tobytes())
    try:
        orig_pil = Image.open(image_path).convert('RGB'); rgb_original = np.array(orig_pil)
        rgb_proc = np.clip((rgb_original.astype(np.int32)//simplification_level)*simplification_level,0,255).astype(np.uint8) if simplification_level>1 else rgb_original.copy()
        ch = [rle_compress(rgb_proc[:,:,i].flatten()) for i in range(3)]; save_rle(ch,bin_path)
        d_ch=[rle_decompress(c) for c in ch];
        if any(len(c)!=rgb_original.size//3 for c in d_ch): raise ValueError("Błąd dekompresji RLE.")
        rgb_rec = np.stack([np.array(c).reshape(rgb_original.shape[:2]) for c in d_ch],axis=2)
        diff = np.clip(np.abs(rgb_original.astype(float)-rgb_rec.astype(float))*10,0,255).astype(np.uint8)
        imageio.imwrite(out_path,rgb_rec)
        orig_s,comp_s=os.path.getsize(image_path),os.path.getsize(bin_path)
        stats = f"Oryginał: {orig_s/1024:.2f} KB | Po kompresji: {comp_s/1024:.2f} KB | Kompresja: {(1-comp_s/orig_s)*100:.2f}%"
        return rgb_original, rgb_rec, diff, stats
    except Exception as e: return None,None,None,f"Błąd w RLE: {e}"

# ------------------------------------------------------------------------------
# 1.4. Logika dla algorytmu LZW (Całkowicie Poprawiona)
# ------------------------------------------------------------------------------
def process_lzw_and_get_results(image_path, simplification_level, out_path="rekonstrukcja_lzw.png", bin_path="skompresowany.lzw"):
    
    def lzw_compress(uncompressed_data):
        """Kompresuje listę bajtów (0-255) do listy kodów LZW."""
        dict_size = 256
        dictionary = {bytes([i]): i for i in range(dict_size)}
        
        w = bytes()
        result = []
        for c_val in uncompressed_data:
            c = bytes([c_val])
            wc = w + c
            if wc in dictionary:
                w = wc
            else:
                result.append(dictionary[w])
                dictionary[wc] = dict_size
                dict_size += 1
                w = c
        if w:
            result.append(dictionary[w])
        return result

    def lzw_decompress(compressed_codes):
        """Dekompresuje listę kodów LZW do bytearray."""
        if not compressed_codes:
            return bytearray()
            
        dict_size = 256
        dictionary = {i: bytes([i]) for i in range(dict_size)}
        
        result = bytearray()
        
        try:
            w = dictionary[compressed_codes.pop(0)]
        except (IndexError, KeyError):
            return bytearray() # Pusta lub nieprawidłowa sekwencja
        
        result.extend(w)

        for k in compressed_codes:
            if k in dictionary:
                entry = dictionary[k]
            elif k == dict_size:
                entry = w + w[:1]
            else:
                # Jeśli kod jest poza zakresem, zwróć to co udało się zdekodować do tej pory
                # To zapobiega crashowi, choć wynik może być niekompletny.
                print(f"Ostrzeżenie LZW: napotkano nieprawidłowy kod {k}")
                break 
                
            result.extend(entry)
            dictionary[dict_size] = w + entry[:1]
            dict_size += 1
            w = entry
            
        return result

    def save_lzw(ch,sz,fn):
        with open(fn,"wb") as f: 
            f.write(array.array('H',sz).tobytes())
            for c in ch:
                # Użyj 'I' dla kodów, które mogą być większe niż 65535
                f.write(array.array('I',[len(c)]).tobytes())
                code_array = array.array('I', c)
                f.write(code_array.tobytes())
                
    def load_lzw(fn):
        with open(fn,"rb") as f:
            sz=tuple(array.array('H',f.read(4)))
            ch=[]
            for _ in range(3):
                # Użyj 'I' dla kodów
                byte_count = array.array('I',f.read(4))[0] * 4 
                d=array.array('I')
                d.frombytes(f.read(byte_count))
                ch.append(d.tolist())
        return sz, ch

    try:
        orig_pil = Image.open(image_path).convert('RGB')
        rgb_original = np.array(orig_pil)
        rgb_proc = np.clip((rgb_original.astype(np.int32)//simplification_level)*simplification_level,0,255).astype(np.uint8) if simplification_level>1 else rgb_original.copy()
        
        img_comp = Image.fromarray(rgb_proc)
        # Przekaż dane jako listę intów
        ch_comp = [lzw_compress(list(c.getdata())) for c in img_comp.split()]
        save_lzw(ch_comp, img_comp.size, bin_path)
        
        sz,(r,g,b)=load_lzw(bin_path)
        r_data,g_data,b_data=[lzw_decompress(c) for c in (r,g,b)]
        
        r_img,g_img,b_img=[Image.new("L",sz) for _ in range(3)]
        r_img.putdata(r_data)
        g_img.putdata(g_data)
        b_img.putdata(b_data)
        
        rgb_rec = np.array(Image.merge("RGB",(r_img,g_img,b_img)))
        diff = np.clip(np.abs(rgb_original.astype(float)-rgb_rec.astype(float))*10,0,255).astype(np.uint8)
        imageio.imwrite(out_path,rgb_rec)
        orig_s,comp_s=os.path.getsize(image_path),os.path.getsize(bin_path)
        stats = f"Oryginał: {orig_s/1024:.2f} KB | Po kompresji: {comp_s/1024:.2f} KB | Kompresja: {(1-comp_s/orig_s)*100:.2f}%"
        return rgb_original, rgb_rec, diff, stats
    except Exception as e:
        return None,None,None,f"Błąd w LZW: {e}"

# ------------------------------------------------------------------------------
# 1.5. Logika dla algorytmu Huffmana (funkcja główna)
# ------------------------------------------------------------------------------
def process_huffman_and_get_results(image_path, simplification_level, out_path="rekonstrukcja_huff.png", bin_path="skompresowany.huff"):
    try:
        orig_pil = Image.open(image_path).convert('RGB'); rgb_original = np.array(orig_pil)
        rgb_proc = np.clip((rgb_original.astype(np.int32)//simplification_level)*simplification_level,0,255).astype(np.uint8) if simplification_level>1 else rgb_original.copy()
        img_bytes = Image.fromarray(rgb_proc).tobytes()
        huff = HuffmanCodec(); huff.build(collections.Counter(img_bytes)); huff.save_to_file(huff.compress(img_bytes), bin_path)
        huff_dec = HuffmanCodec(); decomp = huff_dec.decompress(huff_dec.load_from_file(bin_path), len(img_bytes))
        rgb_rec = np.frombuffer(decomp, dtype=np.uint8).reshape(rgb_original.shape)
        imageio.imwrite(out_path, rgb_rec)
        diff = np.clip(np.abs(rgb_original.astype(float)-rgb_rec.astype(float))*10,0,255).astype(np.uint8)
        orig_s,comp_s=os.path.getsize(image_path),os.path.getsize(bin_path)
        stats = f"Oryginał: {orig_s/1024:.2f} KB | Po kompresji: {comp_s/1024:.2f} KB | Kompresja: {(1-comp_s/orig_s)*100:.2f}%"
        return rgb_original, rgb_rec, diff, stats
    except Exception as e: return None,None,None,f"Błąd w Huffman: {e}"

# ------------------------------------------------------------------------------
# 1.6. Logika dla algorytmu DWT
# ------------------------------------------------------------------------------
def process_dwt_and_get_results(arw_path, wavelet, level, threshold_ratio, out_path="rekonstrukcja_dwt.jp2"):
    try:
        with rawpy.imread(arw_path) as raw:
            rgb_original = raw.postprocess(use_camera_wb=True, no_auto_bright=True).astype(np.uint8)
        ch_comp = []
        for c in range(3):
            coeffs = pywt.wavedec2(rgb_original[:,:,c], wavelet=wavelet, level=level)
            c_arr, c_slices = pywt.coeffs_to_array(coeffs)
            thresh = threshold_ratio * np.max(np.abs(c_arr))
            c_arr[np.abs(c_arr) < thresh] = 0
            c_comp = pywt.array_to_coeffs(c_arr, c_slices, output_format='wavedec2')
            rec_ch = pywt.waverec2(c_comp, wavelet=wavelet)
            h, w = rgb_original.shape[:2]
            ch_comp.append(np.clip(rec_ch[:h,:w], 0, 255).astype(np.uint8))
        rgb_rec = np.stack(ch_comp, axis=-1)
        imageio.imwrite(out_path, rgb_rec, format='jp2')
        diff = np.clip(np.abs(rgb_original.astype(float)-rgb_rec.astype(float))*10,0,255).astype(np.uint8)
        orig_s, comp_s=os.path.getsize(arw_path),os.path.getsize(out_path)
        stats = f"Oryginał: {orig_s/1024/1024:.2f} MB | Po kompresji: {comp_s/1024:.2f} KB | Kompresja: {(1 - comp_s/orig_s)*100:.2f}%"
        return rgb_original, rgb_rec, diff, stats
    except Exception as e: return None, None, None, f"Błąd w DWT: {e}"

# ==============================================================================
# SEKCJA 2: APLIKACJA GUI
# ==============================================================================
class AlgorithmGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("GUI dla Algorytmów Kompresji")
        self.root.geometry("1400x900")

        style = ttk.Style(); style.configure("TNotebook.Tab", padding=[10,5], font=('Helvetica',10))
        main_frame = ttk.Frame(root, padding="10"); main_frame.pack(fill=tk.BOTH, expand=True)
        self.notebook = ttk.Notebook(main_frame); self.notebook.pack(fill=tk.BOTH, expand=True)

        self.widgets = {k:{} for k in ['dct','rle','lzw','huffman','dwt']}
        self.paths = {k:None for k in self.widgets}

        # Definicje typów plików dla okien dialogowych
        arw_ft = [("Sony RAW", "*.ARW"), ("Wszystkie pliki", "*.*")]
        img_ft = [("Pliki Obrazów", "*.png *.jpg *.jpeg *.bmp"), ("Wszystkie pliki", "*.*")]

        # Tworzenie zakładek
        self._create_tab('dct', 'Kompresja DCT', arw_ft, "Jakość", 1, 100, 50)
        self._create_tab('rle', 'Kompresja RLE', img_ft, "Upraszczanie", 1, 32, 1)
        self._create_tab('lzw', 'Kompresja LZW', img_ft, "Upraszczanie", 1, 32, 1)
        self._create_tab('huffman', 'Kompresja Huffman', img_ft, "Upraszczanie", 1, 32, 1)
        self._create_dwt_tab()

    def _create_base_tab(self, key, title):
        tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(tab, text=title)
        widgets = self.widgets[key]
        ctrl_frame = ttk.Frame(tab); ctrl_frame.pack(fill=tk.X, pady=5)
        img_frame = ttk.Frame(tab); img_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        widgets['image_panels'] = self._create_image_panels(img_frame)
        return ctrl_frame, widgets

    def _create_tab(self, key, title, file_types, slider_text, from_, to, initial):
        ctrl_frame, widgets = self._create_base_tab(key, title)
        self._add_file_selector(ctrl_frame, key, "Wybierz plik", file_types)
        self._add_slider(ctrl_frame, widgets, f"{slider_text}:", from_, to, initial, row=1)
        self._add_process_button(ctrl_frame, key, row=2)

    def _create_dwt_tab(self):
        key = 'dwt'
        ctrl_frame, widgets = self._create_base_tab(key, 'Kompresja DWT')
        self._add_file_selector(ctrl_frame, key, "Wybierz plik ARW", [("Sony RAW", "*.ARW"), ("Wszystkie pliki", "*.*")])
        
        param_frame = ttk.Frame(ctrl_frame); param_frame.grid(row=1, column=0, columnspan=2, sticky='ew', pady=5)
        
        ttk.Label(param_frame, text="Falka:").pack(side=tk.LEFT, padx=5)
        widgets['wavelet_var'] = tk.StringVar(value='haar')
        ttk.Combobox(param_frame, textvariable=widgets['wavelet_var'], values=pywt.wavelist(kind='discrete'), state='readonly').pack(side=tk.LEFT, padx=5)

        self._add_slider(param_frame, widgets, "Poziom:", 1, 5, 2, key_suffix='level')
        self._add_slider(param_frame, widgets, "Próg:", 0.01, 1.0, 0.1, key_suffix='thresh', is_double=True)
        
        self._add_process_button(ctrl_frame, key, row=2)

    def _add_file_selector(self, parent, key, btn_text, file_types):
        ttk.Button(parent, text=btn_text, command=lambda k=key, ft=file_types: self._select_file(k, ft)).grid(row=0, column=0, padx=5, pady=5)
        self.widgets[key]['path_label'] = ttk.Label(parent, text="Nie wybrano pliku", width=70)
        self.widgets[key]['path_label'].grid(row=0, column=1, sticky="ew", padx=5)

    def _add_slider(self, parent, widgets, text, from_, to, initial, row=0, key_suffix='param', is_double=False):
        frame = ttk.Frame(parent);
        if row > 0: frame.grid(row=row, column=0, columnspan=2, sticky='ew', pady=2)
        else: frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(frame, text=text).pack(side=tk.LEFT, padx=5)
        var = tk.DoubleVar(value=initial) if is_double else tk.IntVar(value=initial)
        widgets[f'{key_suffix}_var'] = var
        label = ttk.Label(frame, text=f"{initial:.2f}" if is_double else str(initial), width=4)
        cmd = (lambda v, w=label: w.config(text=f"{float(v):.2f}")) if is_double else (lambda v, w=label: w.config(text=str(int(float(v)))))
        ttk.Scale(frame, from_=from_, to=to, variable=var, orient=tk.HORIZONTAL, command=cmd).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        label.pack(side=tk.LEFT)

    def _add_process_button(self, parent, key, row):
        widgets = self.widgets[key]
        widgets['process_button'] = ttk.Button(parent, text=f"Przetwarzaj {key.upper()}", command=lambda k=key: self._process(k), state=tk.DISABLED)
        widgets['process_button'].grid(row=row, column=0, columnspan=2, pady=10)
        widgets['status_label'] = ttk.Label(parent, text="Wybierz plik i kliknij 'Przetwarzaj'.", wraplength=1200)
        widgets['status_label'].grid(row=row+1, column=0, columnspan=2, pady=5, sticky='ew')

    def _create_image_panels(self, parent):
        panels = {}; titles = ["Oryginał", "Zrekonstruowany", "Różnica"]
        parent.columnconfigure(list(range(len(titles))), weight=1)
        for i, title in enumerate(titles):
            frame = ttk.LabelFrame(parent, text=title); frame.grid(row=0, column=i, sticky="nsew", padx=5)
            frame.rowconfigure(0, weight=1); frame.columnconfigure(0, weight=1)
            canvas = ttk.Label(frame); canvas.grid(sticky="nsew")
            panels[title] = {'canvas': canvas, 'photo': None}
        return panels

    def _select_file(self, key, file_types):
        path = filedialog.askopenfilename(parent=self.root, title="Wybierz plik", filetypes=file_types)
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
            if key in ['rle', 'lzw', 'huffman', 'dct']:
                param = widgets['param_var'].get()
                if key == 'dct': results = process_dct_and_get_results(path, param)
                elif key == 'rle': results = process_rle_and_get_results(path, param)
                elif key == 'lzw': results = process_lzw_and_get_results(path, param)
                elif key == 'huffman': results = process_huffman_and_get_results(path, param)
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
            img_pil.thumbnail((max_w-10, max_h-10), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img_pil)
            panel_dict['photo'] = photo # Zachowaj referencję
            canvas.config(image=photo)
        except Exception as e: messagebox.showerror("Błąd wyświetlania", f"Nie można wyświetlić obrazu: {e}")

if __name__ == '__main__':
    try:
        import rawpy, pywt, imageio
        from bitarray import bitarray
    except ImportError as e:
        # Tworzenie tymczasowego okna do wyświetlenia błędu, jeśli główne jeszcze nie istnieje
        temp_root = tk.Tk()
        temp_root.withdraw() # Ukryj okno
        messagebox.showerror("Brak biblioteki", f"Brakująca biblioteka: {e.name}.\nZainstaluj ją używając polecenia:\npip install {e.name}")
        temp_root.destroy()
        exit()
        
    root = tk.Tk()
    app = AlgorithmGUI(root)
    root.mainloop()
