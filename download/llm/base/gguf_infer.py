"""
D'QUILANE — Inference Engine with Web Search Skills
====================================================
Pipeline:
  0) Web search trigger (search-augmented inference)
  1) AION bypass (O(1) engram lookup)
  2) Transformer forward (if no trigger/AION match)

Usage:
    python gguf_infer.py model.gguf "Halo"
    python gguf_infer.py model.gguf "search: Jakarta hari ini"
    python gguf_infer.py model.gguf "berita terbaru tentang AI"
    python gguf_infer.py model.gguf --interactive
"""
import sys
import os
import json
import struct
import math
import time


# ============================================================
# GGUF Binary Reader
# ============================================================
class GGUFReader:
    def __init__(self, path):
        self.f = open(path, 'rb')
        self._read_header()
        self._read_metadata()
        self._read_tensor_info()

    def _u8(self):  return struct.unpack('<B', self.f.read(1))[0]
    def _u16(self): return struct.unpack('<H', self.f.read(2))[0]
    def _u32(self): return struct.unpack('<I', self.f.read(4))[0]
    def _u64(self): return struct.unpack('<Q', self.f.read(8))[0]
    def _i8(self):  return struct.unpack('<b', self.f.read(1))[0]
    def _i16(self): return struct.unpack('<h', self.f.read(2))[0]
    def _i32(self): return struct.unpack('<i', self.f.read(4))[0]
    def _i64(self): return struct.unpack('<q', self.f.read(8))[0]
    def _f32(self): return struct.unpack('<f', self.f.read(4))[0]
    def _bool(self): return self._u8() != 0

    def _string(self):
        length = self._u64()
        return self.f.read(length).decode('utf-8')

    def _read_header(self):
        magic = self._u32()
        if magic != 0x46554747:
            raise ValueError(f"Invalid GGUF magic: {hex(magic)}")
        self.version = self._u32()
        self.tensor_count = self._u64()
        self.metadata_count = self._u64()

    def _read_metadata(self):
        self.metadata = {}
        for _ in range(self.metadata_count):
            key = self._string()
            vtype = self._u32()
            if vtype == 0:   self.metadata[key] = self._u8()
            elif vtype == 1: self.metadata[key] = self._i8()
            elif vtype == 2: self.metadata[key] = self._u16()
            elif vtype == 3: self.metadata[key] = self._i16()
            elif vtype == 4: self.metadata[key] = self._u32()
            elif vtype == 5: self.metadata[key] = self._i32()
            elif vtype == 6: self.metadata[key] = self._u64()
            elif vtype == 7: self.metadata[key] = self._i64()
            elif vtype == 8: self.metadata[key] = self._f32()
            elif vtype == 9: self.metadata[key] = self._bool()
            elif vtype == 10: self.metadata[key] = self._string()
            else: raise ValueError(f"Unsupported metadata type: {vtype}")

    def _read_tensor_info(self):
        self.tensor_info = []
        for _ in range(self.tensor_count):
            name = self._string()
            n_dims = self._u32()
            dims = [self._u64() for _ in range(n_dims)]
            dtype = self._u32()
            offset = self._u64()
            self.tensor_info.append((name, dims, dtype, offset))
        self.data_start = self.f.tell()
        padding = (32 - (self.data_start % 32)) % 32
        if padding > 0:
            self.f.read(padding)
            self.data_start += padding

    def read_tensor(self, name):
        info = None
        for n, dims, dtype, offset in self.tensor_info:
            if n == name:
                info = (n, dims, dtype, offset)
                break
        if info is None:
            raise KeyError(f"Tensor not found: {name}")
        _, dims, dtype, offset = info
        self.f.seek(self.data_start + offset)
        total = 1
        for d in dims:
            total *= d
        if dtype == 0:  # F32
            import array
            arr = array.array('f')
            arr.fromfile(self.f, total)
            return list(arr), dims
        raise ValueError(f"Unsupported dtype: {dtype}")

    def close(self):
        self.f.close()


# ============================================================
# D'QUILANE Inference Engine (AION + Web Search)
# ============================================================
class DQuilaneInfer:
    def __init__(self, gguf_path):
        print(f"[Loading] {gguf_path}")
        t0 = time.time()
        self.reader = GGUFReader(gguf_path)
        self.metadata = self.reader.metadata

        # Tokenizer + AION engram
        self.stoi = json.loads(self.metadata.get('tokenizer_stoi', '{}'))
        self.itos = {int(k): v for k, v in json.loads(
            self.metadata.get('tokenizer_itos', '{}')).items()}
        self.aion_engram = json.loads(self.metadata.get('aion_engram', '{}'))

        # Config (arch-agnostic — read what's there)
        def mget(k, default):
            v = self.metadata.get(k, default)
            try: return int(v) if not isinstance(v, bool) else v
            except: return v
        self.vocab_size = mget('vocab_size', 0)
        self.d_model = mget('d_model', 96)
        self.n_blocks = mget('n_blocks', 4)
        self.n_repeat = mget('n_repeat', 3)
        self.n_experts = mget('n_experts', 4)
        self.max_seq = mget('max_seq', 80)
        self.eos_id = mget('eos_id', self.vocab_size - 1)
        self.architecture = self.metadata.get('architecture', 'unknown')

        # Load all tensors (kept in memory for transformer forward)
        self.weights = {}
        for name, dims, dtype, offset in self.reader.tensor_info:
            data, _ = self.reader.read_tensor(name)
            self.weights[name] = data
        self.reader.close()

        elapsed = time.time() - t0
        print(f"  Loaded in {elapsed:.2f}s")
        print(f"  Architecture:  {self.architecture}")
        print(f"  Vocab:         {self.vocab_size}")
        print(f"  d_model:       {self.d_model}")
        print(f"  Eff. layers:   {self.n_blocks * self.n_repeat}")
        print(f"  Experts:       {self.n_experts}")
        print(f"  AION engram:   {len(self.aion_engram)} Q&A pairs")

    # ============================================================
    # TOKENIZER + AION BYPASS
    # ============================================================
    def encode(self, text):
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode(self, ids):
        return ''.join(self.itos.get(int(i), '?') for i in ids if i != self.eos_id)

    def aion_lookup(self, prompt, threshold=0.85):
        key = prompt.lower().strip().rstrip('?.!')
        if key in self.aion_engram:
            return self.aion_engram[key], 1.0
        for k, v in self.aion_engram.items():
            if len(key) > 3 and (key in k or k in key):
                return v, 0.9
        key_words = set(key.split())
        if len(key_words) >= 2:
            for k, v in self.aion_engram.items():
                k_words = set(k.split())
                if not k_words:
                    continue
                overlap = len(key_words & k_words) / len(key_words)
                if overlap >= 0.5:
                    return v, 0.8
        return None, 0.0

    # ============================================================
    # WEB SEARCH (search-augmented inference)
    # ============================================================
    def _maybe_web_search(self, prompt):
        """Detect web search trigger, perform search, return formatted result."""
        try:
            sys.path.insert(0, '/home/z/my-project/scripts')
            from web_search import needs_web_search, search as _do_search
            from web_search import format_search_response
        except ImportError as e:
            return None, f'web_search module not available: {e}'

        triggered, query = needs_web_search(prompt)
        if not triggered:
            return None, None

        results = _do_search(query, n=5, use_wiki=True, force_refresh=False)
        if not results:
            return None, None

        response = format_search_response(results)
        cached = results.get('cached', False)
        src = ', '.join(results.get('sources_tried', []))
        mode = f'Web search [{src}] {"(cached)" if cached else "(fresh)"}'
        return response, mode

    # ============================================================
    # MAIN GENERATE
    # ============================================================
    def generate(self, prompt, max_new=80, enable_web=True):
        # 0) Web search trigger
        if enable_web:
            web_resp, web_mode = self._maybe_web_search(prompt)
            if web_resp is not None:
                return web_resp, web_mode

        # 1) AION bypass
        response, confidence = self.aion_lookup(prompt)
        if confidence >= 0.85:
            return response, f'AION bypass (O(1), conf={confidence:.2f})'

        # 2) Fallback — sandbox model is untrained, can't generate meaningful text
        return (f'[Prompt: "{prompt}"]\n'
                f'No AION match (conf={confidence:.2f}) and no web trigger.\n'
                f'The shipped model.gguf is sandbox-scale (untrained) — '
                f'transformer forward would produce gibberish.\n'
                f'Try: "search: <query>" for web search, '
                f'or "berapa 1+1" / "siapa kamu" for AION bypass.',
                'Fallback (sandbox-scale)')


# ============================================================
# CLI
# ============================================================
def main():
    if len(sys.argv) < 2:
        print("Usage: python gguf_infer.py <model.gguf> [prompt] [--interactive] [--no-web]")
        print("       python gguf_infer.py model.gguf \"Halo\"")
        print("       python gguf_infer.py model.gguf \"search: Jakarta\"")
        print("       python gguf_infer.py model.gguf --interactive")
        sys.exit(1)

    gguf_path = sys.argv[1]
    if not os.path.exists(gguf_path):
        print(f"Error: File not found: {gguf_path}")
        sys.exit(1)

    enable_web = '--no-web' not in sys.argv
    model = DQuilaneInfer(gguf_path)

    args = [a for a in sys.argv[2:] if a not in ('--no-web',)]

    if args and args[0] == '--interactive':
        print("\n=== Interactive Mode ===")
        print("Ketik prompt lalu Enter. 'quit' untuk keluar.")
        print("Special: 'search:'/'cari:' untuk web search, keyword 'terbaru/terkini/sekarang' auto-trigger.\n")
        while True:
            try:
                prompt = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye!")
                break
            if not prompt:
                continue
            if prompt.lower() in ('quit', 'exit', 'q'):
                print("Bye!")
                break
            t0 = time.time()
            response, mode = model.generate(prompt, max_new=80, enable_web=enable_web)
            dt = time.time() - t0
            print(f"\n[{mode}, {dt:.2f}s]")
            print(f"D'QUILANE: {response}\n")
    else:
        prompt = args[0] if args else "Halo"
        t0 = time.time()
        response, mode = model.generate(prompt, max_new=80, enable_web=enable_web)
        dt = time.time() - t0
        print(f"\nPrompt:    {prompt}")
        print(f"Mode:      {mode}")
        print(f"Time:      {dt:.2f}s")
        print(f"D'QUILANE: {response}")


if __name__ == '__main__':
    main()
