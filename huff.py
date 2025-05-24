import collections
import heapq
from bitarray import bitarray

class Node:
    def __init__(self, byte=None, freq=0):
        self._key = byte
        self._val = freq
        self._left = None
        self._right = None
    
    def __lt__(self, obj):
        return self._val < obj._val
    def __eq__(self, other):
        return self._val == other._val
    
    def printnode(self):
        print(f"k:{self._key}|val:{self._val}")

def printheap(h):
    for n in h:
        n.printnode()
    print("-----")


class Huffman:

    def __init__(self, freq_tab):
        self._codes = {}
        self._root = self.make_tree(freq_tab)
        self.make_codes(self._root)
        self.sort_codes()
        self.change_to_canon()

    def make_tree(self, freq_tab):
        heap = []
        base = dict(freq_tab)

        # minheap for sorting
        for key, value in base.items():
            node = Node(key, value)
            heapq.heappush(heap, node)

        while len(heap) >= 2:

            right = heapq.heappop(heap)
            left = heapq.heappop(heap)
            node = Node(None, left._val+right._val)
            
            node._right = right
            node._left = left
            heapq.heappush(heap, node)

        root = heapq.heappop(heap)
        return root
    

    def make_codes(self, node:Node, c=''):

        if node is None:
            return

        if node._right is None and node._left is None:
            self._codes[node._key] = bitarray(c)
            return
        self.make_codes(node._right, c+'1')
        self.make_codes(node._left, c+'0')


    def print_codes16(self):
        s = ''
        for k, v in self._codes.items():
            s += f"0x{k:02X}: '{v}', "
        print(s)
    

    # len -> code (bitarray)
    def make_canon_code(self, l, current_code: bitarray):
        
        current_length = len(current_code)
        # if first code
        if not current_length: 
            current_code.append(0)
            return

        for i in range(1,current_length+1):
            if current_code[-i] == 0:
                current_code[-i]  = 1
                break
            else:
                current_code[-i]  = 0
        for _ in range(l-current_length):
            current_code.append(0)
    

    def sort_codes(self):

        # sort by code length
        sorted_len = {key: len(value) for key, value in self._codes.items()}
        sorted_len = dict(sorted(sorted_len.items(), key=lambda item : item[1]))

        # sort lexicographically
        tmp = {}
        canon_codes = {}
        # get first length
        l = sorted_len[next(iter(sorted_len))]

        for k, v in sorted_len.items():
            if l == v:
                tmp[k] = v
            else:
                l = v
                tmp = dict(sorted(tmp.items(), key=lambda item: item[0]))
                canon_codes |= tmp
                tmp.clear()
                tmp[k] = v
        # last update
        tmp = dict(sorted(tmp.items(), key=lambda item: item[0]))
        canon_codes |= tmp

        # assign new codes
        self._codes = canon_codes


    # changing from byte:code_lenght to byte:canonical_code
    def change_to_canon(self):
        
        # self.sort_codes()
        
        # make canonical codes
        current_code = bitarray()
        for k,v in self._codes.items():
            self.make_canon_code(v, current_code)
            self._codes[k] = current_code.copy()


    def compress(self, path_out: str, data):

        bitarr = bitarray()
        chunk = 8**4 # 4096 (has to be a multiplication of 8)

        with open(path_out, 'wb') as fout:
            
            # Metadata
            fout.write(bytes([len(self._codes)-1])) # how many symbols (0->1 sym, ..., 255->256 sym)
            for k,v in self._codes.items():
                fout.write(bytes([k]))
                fout.write(bytes([len(v)]))

            # Data
            for b in data:
                code = self._codes[b] # as bits
                bitarr += code
                if len(bitarr) > chunk:
                    tmp = bitarr[chunk:]
                    bitarr = bitarr[:chunk]
                    bitarr.tofile(fout)
                    bitarr = tmp

            # padding 0 if needed
            padded = bitarr.fill()
            bitarr.tofile(fout)
            fout.write(bytes([padded])) # how many 0 padded
        
        print(f"padded: {padded}")


    def decompress(self, path_out: str, data, check):
        
        symbols = {}
        N = int(data[0]) + 1 # how many symbols to read
        padded = int(data[-1])

        # read symbols:lenghts
        for i in range(1,N*2,2):
            symbols[data[i]] = int(data[i+1])
        
        # make canon codes from lengths
        self._codes = symbols
        self.change_to_canon()
        
        # read codes and write bytes to file
        code = bitarray()
        B = bitarray()
        vcodes = list(self._codes.values()) # list of values
        kcodes = list(self._codes.keys()) # list of keys

        # x = 0
        # H.print_codes16()
        # print(f"N = {N}, padded = {padded}")

        # if padded read without last byte of data
        if padded:
            last = -2
        else:
            last = -1

        with open(path_out, 'wb') as fout:
            
            for b in data[2*N+1:last]:
                B.frombytes(bytes([b]))
                for i in B:
                    code.append(i)
                    if code in vcodes:
                        idx = vcodes.index(code)
                        fout.write(bytes([kcodes[idx]]))
                        # if check[x] != kcodes[idx]: print(f"check: {check[x]} | write: {kcodes[idx]}")
                        # x+=1
                        code.clear()
                B.clear()

            # last byte minus padded bits if needed
            if padded:
                B.frombytes(bytes([data[-2]]))
                for i in B[:-padded]:
                    code.append(i)
                    if code in vcodes:
                        idx = vcodes.index(code)
                        fout.write(bytes([kcodes[idx]]))
                        code.clear()




# END OF HUFFMAN

def read_file(path_in: str):

    with open(path_in, 'rb') as f:
        data = f.read() # get all data
    freq_table = collections.Counter(data)

    return data, freq_table



if __name__ == "__main__":

    readconsole = True

    if readconsole:
        print("File to compress: ",end='')
        path_in = input()
        print("Name of compressed file (without .*): ",end='')
        path_hf = input() + ".hf"
        print("Name of decompressed file: ",end='')
        path_out = input()
    else:
        path_in = 'green_black.bmp'   #'test.bin'
        path_hf = 'compressed.hf'
        path_out = "out"
    
    data_in, freq_table = read_file(path_in)

    # freq table print
    # s = "{ "
    # for k,v in freq_table.items():
    #     s += f"0x{k:02X}: {v}, "
    # s += "}"
    # print(s)

    H = Huffman(freq_table)
    H.compress(path_hf, data_in)
    H.print_codes16()
    # read .hf
    data_out, _ = read_file(path_hf)
    H.decompress(path_out, data_out, data_in)
    