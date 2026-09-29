"""
============================================================
  D'QUILANE — Standalone Edition (laptop-ready)
============================================================

Cara pakai:
  # Training (GPU/RAM kerja, lalu save ke SSD)
  python3 dquilane.py train

  # Generate text (load dari SSD, generate, exit, RAM bebas)
  python3 dquilane.py generate "Indonesia"

  # Cek info hardware
  python3 dquilane.py info

Yang penting:
- Auto-detect CUDA / MPS (Apple Silicon) / CPU
- GPU & RAM cuma kerja selama script jalan
- Setelah exit, semua memory dibebaskan
- Model + tokenizer tersimpan di SSD (default ./dquilane_files/)

Tidak perlu install library lain selain: torch, numpy
"""
import sys
import os
import re
import json
import math
import time
import gc
from pathlib import Path
from collections import Counter

# ============================================================
# CONFIG
# ============================================================
DATA_DIR = Path(__file__).parent / "dquilane_files"
DATA_DIR.mkdir(exist_ok=True)
MODEL_PATH = DATA_DIR / "mini_llm.pt"
TOK_PATH = DATA_DIR / "tokenizer.json"
CORPUS_PATH = Path(__file__).parent / "corpus.txt"

CFG = {
    "vocab_size": 1024,
    "d_model": 256,
    "n_heads": 8,
    "d_ff": 1024,
    "num_layers": 24,        # virtual depth via parameter sharing
    "max_seq_len": 256,
}
EPOCHS = 2000
BATCH = 32
LR = 3e-4
SEED = 42


# ============================================================
# DEVICE DETECTION
# ============================================================
def detect_device():
    """Deteksi device terbaik yang tersedia."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.device("cuda"), "CUDA (NVIDIA GPU)"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps"), "MPS (Apple Silicon)"
        return torch.device("cpu"), "CPU"
    except ImportError:
        print("[ERROR] PyTorch belum terinstall. Install dulu:")
        print("  pip install torch numpy")
        sys.exit(1)


def cleanup():
    """Bebaskan GPU/RAM secara eksplisit sebelum exit."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
    except Exception:
        pass


def print_resource_stats(label=""):
    """Print RAM/GPU usage (kalau bisa)."""
    try:
        import torch
        import psutil
        ram = psutil.virtual_memory()
        print(f"  [{label}] RAM: {ram.used/1e9:.1f} / {ram.total/1e9:.1f} GB "
              f"({ram.percent:.0f}%)")
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            cached = torch.cuda.memory_reserved() / 1e9
            print(f"  [{label}] GPU mem: {allocated:.2f} GB allocated, "
                  f"{cached:.2f} GB cached")
    except ImportError:
        # psutil gak wajib
        pass


# ============================================================
# TOKENIZER (BPE from scratch)
# ============================================================
PAT_SIMPLE = r"\s+|\w+|\S"


def _get_pairs(word):
    pairs = set()
    prev = word[0]
    for ch in word[1:]:
        pairs.add((prev, ch))
        prev = ch
    return pairs


class MiniBPETokenizer:
    def __init__(self):
        self.merges = {}
        self.vocab = {}
        self.pad_id, self.bos_id, self.eos_id = 0, 1, 2

    def train(self, text, target_vocab=1024, verbose=True):
        words_raw = re.findall(PAT_SIMPLE, text)
        word_freqs = Counter(words_raw)
        words = {w: list(w.encode("utf-8")) for w in word_freqs}
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.merges = {}
        num_merges = target_vocab - 256
        for i in range(num_merges):
            pair_counts = Counter()
            for w, syms in words.items():
                for p in _get_pairs(syms):
                    pair_counts[p] += word_freqs[w]
            if not pair_counts:
                break
            best, _ = pair_counts.most_common(1)[0]
            new_id = 256 + i
            self.merges[best] = new_id
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]
            new_words = {}
            for w, syms in words.items():
                new_words[w] = self._merge_word(syms, best, new_id)
            words = new_words
            if verbose and (i % 100 == 0 or i == num_merges - 1):
                print(f"  [bpe] merge {i+1}/{num_merges}")
        self.vocab_size = max(self.vocab.keys()) + 1
        print(f"[bpe] done. vocab_size={self.vocab_size}")

    @staticmethod
    def _merge_word(syms, pair, new_id):
        new = []
        i = 0
        while i < len(syms):
            if i < len(syms) - 1 and (syms[i], syms[i + 1]) == pair:
                new.append(new_id)
                i += 2
            else:
                new.append(syms[i])
                i += 1
        return new

    def encode(self, text):
        ids = [self.bos_id]
        for w in re.findall(PAT_SIMPLE, text):
            syms = list(w.encode("utf-8"))
            while len(syms) >= 2:
                pairs = _get_pairs(syms)
                best = None
                best_rank = None
                for p in pairs:
                    if p in self.merges:
                        rank = list(self.merges.keys()).index(p)
                        if best_rank is None or rank < best_rank:
                            best = p
                            best_rank = rank
                if best is None:
                    break
                syms = self._merge_word(syms, best, self.merges[best])
            ids.extend(syms)
        ids.append(self.eos_id)
        return ids

    def decode(self, ids):
        out = b""
        for i in ids:
            if i in (self.pad_id, self.bos_id, self.eos_id):
                continue
            if i in self.vocab:
                out += self.vocab[i]
        return out.decode("utf-8", errors="replace")

    def save(self, path):
        Path(path).write_text(json.dumps({
            "merges": [[list(k), v] for k, v in self.merges.items()],
            "vocab_size": self.vocab_size,
            "pad_id": self.pad_id, "bos_id": self.bos_id, "eos_id": self.eos_id,
        }, ensure_ascii=False))

    @classmethod
    def load(cls, path):
        d = json.loads(Path(path).read_text())
        tok = cls()
        tok.merges = {tuple(p): v for p, v in d["merges"]}
        tok.vocab = {i: bytes([i]) for i in range(256)}
        for pair, new_id in sorted(tok.merges.items(), key=lambda x: x[1]):
            tok.vocab[new_id] = tok.vocab[pair[0]] + tok.vocab[pair[1]]
        tok.vocab_size = d["vocab_size"]
        tok.pad_id = d["pad_id"]
        tok.bos_id = d["bos_id"]
        tok.eos_id = d["eos_id"]
        return tok


# ============================================================
# MODEL (MiniGPT dengan parameter sharing)
# ============================================================
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, attn_bias=True):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=attn_bias)
        self.proj = nn.Linear(d_model, d_model, bias=attn_bias)
        self.scale = 1.0 / math.sqrt(self.d_head)

    def forward(self, x, kv_cache=None):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=-1)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        if kv_cache is not None:
            kp, vp = kv_cache
            if kp is not None:
                k = torch.cat([kp, k], dim=2)
                v = torch.cat([vp, v], dim=2)
            new_cache = (k, v)
        else:
            new_cache = None
        T_full = k.size(2)
        causal = torch.tril(torch.ones(T, T_full, device=x.device, dtype=torch.bool))
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.masked_fill(~causal.unsqueeze(0).unsqueeze(0), float("-inf"))
        attn = F.softmax(attn, dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out), new_cache


class SwiGLUMLP(nn.Module):
    def __init__(self, d_model, d_ff=None):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        var = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(var + self.eps) * self.weight


class SharedBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = RMSNorm(cfg["d_model"])
        self.attn = CausalSelfAttention(cfg["d_model"], cfg["n_heads"])
        self.norm2 = RMSNorm(cfg["d_model"])
        self.mlp = SwiGLUMLP(cfg["d_model"], cfg.get("d_ff"))

    def forward(self, x, kv_cache=None):
        a, nc = self.attn(self.norm1(x), kv_cache=kv_cache)
        x = x + a
        x = x + self.mlp(self.norm2(x))
        return x, nc


class MiniGPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["d_model"])
        self.pos_emb = nn.Embedding(cfg["max_seq_len"], cfg["d_model"])
        self.block = SharedBlock(cfg)
        self.norm_f = RMSNorm(cfg["d_model"])
        self.head = nn.Linear(cfg["d_model"], cfg["vocab_size"], bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, kv_caches=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        new_caches = []
        for i in range(self.cfg["num_layers"]):
            cache = kv_caches[i] if kv_caches else None
            x, nc = self.block(x, kv_cache=cache)
            new_caches.append(nc)
        x = self.norm_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.cfg["vocab_size"]),
                targets.view(-1), ignore_index=-100,
            )
        return logits, loss, new_caches

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=50, temperature=1.0,
                top_k=None, eos_id=None):
        self.eval()
        kv_caches = [None] * self.cfg["num_layers"]
        for t in range(max_new_tokens):
            idx_cond = idx if t == 0 else idx[:, -1:]
            logits, _, kv_caches = self.forward(idx_cond, kv_caches=kv_caches)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
            if eos_id is not None and next_id.item() == eos_id:
                break
        return idx

    def save(self, path):
        torch.save({"state_dict": self.state_dict(), "cfg": self.cfg}, path)

    @classmethod
    def load(cls, path, map_location="cpu"):
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        m = cls(ckpt["cfg"])
        m.load_state_dict(ckpt["state_dict"])
        return m

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# ============================================================
# DATASET
# ============================================================
class LangDataset(torch.utils.data.Dataset):
    def __init__(self, text, tokenizer, seq_len):
        ids = tokenizer.encode(text)
        self.samples = []
        slp = seq_len + 1
        step = max(slp // 2, 1)
        for i in range(0, len(ids) - slp, step):
            chunk = ids[i:i + slp]
            if len(chunk) < slp:
                chunk = chunk + [tokenizer.pad_id] * (slp - len(chunk))
            x = chunk[:-1]
            y = [t if t != tokenizer.pad_id else -100 for t in chunk[1:]]
            self.samples.append((x, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        x, y = self.samples[i]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


# ============================================================
# COMMANDS
# ============================================================
def cmd_info():
    device, desc = detect_device()
    print("=" * 60)
    print("  D'QUILANE — Hardware Info")
    print("=" * 60)
    print(f"Device: {device}  ({desc})")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU memory: {total:.1f} GB")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("Apple Silicon detected (MPS available)")
    print()
    print(f"Data dir: {DATA_DIR}")
    print(f"Model exists: {MODEL_PATH.exists()} "
          f"({MODEL_PATH.stat().st_size/1024:.1f} KB)" if MODEL_PATH.exists()
          else "Model not trained yet")
    print(f"Tokenizer exists: {TOK_PATH.exists()}")
    print()
    # estimasi param
    d = CFG["d_model"]; v = CFG["vocab_size"]
    f = CFG["d_ff"]; n = CFG["max_seq_len"]
    uniq = v*d + n*d + 4*d*d + 3*d*f + 3*d
    print(f"Target config:")
    print(f"  d_model={d}, vocab={v}, depth={CFG['num_layers']}, max_seq={n}")
    print(f"  Estimated unique params: {uniq:,}")
    print(f"  Estimated file size:     {uniq*4/1024/1024:.2f} MB")
    print(f"  Effective (no sharing):   {uniq*CFG['num_layers']:,}")


def cmd_train():
    device, desc = detect_device()
    print("=" * 60)
    print("  D'QUILANE — Training")
    print("=" * 60)
    print(f"Device: {desc}")
    print()
    print("[start] Before loading anything:")
    print_resource_stats("before")

    torch.manual_seed(SEED)
    # 1. corpus
    if not CORPUS_PATH.exists():
        print(f"[ERROR] corpus.txt tidak ditemukan di {CORPUS_PATH}")
        print("Bikin dulu file corpus.txt berisi teks Indonesia.")
        sys.exit(1)
    text = CORPUS_PATH.read_text()
    print(f"\nCorpus: {len(text)} chars, {len(text.split())} words")

    # 2. train tokenizer (CPU ringan)
    print("\n[1/4] Train tokenizer...")
    tok = MiniBPETokenizer()
    tok.train(text, target_vocab=CFG["vocab_size"], verbose=False)
    tok.save(str(TOK_PATH))
    print(f"  saved: {TOK_PATH}")

    # 3. dataset
    print("\n[2/4] Build dataset...")
    ds = LangDataset(text, tok, CFG["max_seq_len"])
    print(f"  samples: {len(ds)}")
    dl = torch.utils.data.DataLoader(ds, batch_size=BATCH, shuffle=True)

    # 4. init model
    print("\n[3/4] Init model + move to device...")
    model = MiniGPT(CFG).to(device)
    n_params = model.count_params()
    print(f"  unique params: {n_params:,}  (~{n_params*4/1024/1024:.2f} MB)")
    print(f"  virtual depth: {CFG['num_layers']}  "
          f"(effective: {n_params * CFG['num_layers']:,})")
    print_resource_stats("after load")

    # 5. training loop (use AMP if cuda)
    print(f"\n[4/4] Training ({EPOCHS} epochs)...")
    use_amp = device.type == "cuda"
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    t0 = time.time()
    for epoch in range(EPOCHS):
        model.train()
        total = 0.0
        nb = 0
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            if use_amp:
                with torch.amp.autocast("cuda"):
                    _, loss, _ = model(x, targets=y)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                _, loss, _ = model(x, targets=y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            total += loss.item()
            nb += 1
        sched.step()
        avg = total / max(nb, 1)
        if epoch % 50 == 0 or epoch == EPOCHS - 1:
            elapsed = time.time() - t0
            print(f"  epoch {epoch+1:4d}/{EPOCHS}  loss={avg:.4f}  t={elapsed:.1f}s")

    # 6. save ke SSD
    print("\n[5/5] Saving to SSD...")
    model_cpu = model.cpu()  # move to CPU dulu sebelum save
    model_cpu.save(str(MODEL_PATH))
    print(f"  model:   {MODEL_PATH} ({MODEL_PATH.stat().st_size/1024:.1f} KB)")
    print(f"  tok:     {TOK_PATH} ({TOK_PATH.stat().st_size/1024:.1f} KB)")

    # 7. cleanup — RAM/GPU balik bebas!
    print("\n[done] Cleaning up RAM/GPU...")
    del model, model_cpu, opt, sched
    if use_amp:
        del scaler
    cleanup()
    print_resource_stats("after cleanup")
    print(f"\nDONE in {time.time()-t0:.1f}s. File di {DATA_DIR}")
    print("GPU & RAM sudah bebas. Cek dengan Task Manager / nvidia-smi.")


def cmd_generate(prompt: str, max_new=50, temperature=0.8, top_k=20):
    device, desc = detect_device()
    print("=" * 60)
    print("  D'QUILANE — Generation")
    print("=" * 60)
    print(f"Device: {desc}")
    print(f"Prompt: {prompt!r}")
    print()

    if not MODEL_PATH.exists() or not TOK_PATH.exists():
        print("[ERROR] Model belum ada. Run: python3 dquilane.py train")
        sys.exit(1)

    # LOAD FROM SSD — ini satu-satunya moment RAM/GPU kerja
    print("[load] Loading model + tokenizer from SSD...")
    tok = MiniBPETokenizer.load(str(TOK_PATH))
    model = MiniGPT.load(str(MODEL_PATH), map_location=str(device))
    model = model.to(device)
    model.eval()
    print_resource_stats("loaded")

    # generate
    print(f"\n[generate] generating {max_new} tokens...")
    ids = tok.encode(prompt)
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    t0 = time.time()
    out = model.generate(idx, max_new_tokens=max_new, temperature=temperature,
                         top_k=top_k, eos_id=tok.eos_id)
    out_ids = out[0].tolist()
    result = tok.decode(out_ids)
    print(f"  generation took: {time.time()-t0:.2f}s")
    print()
    print("=== OUTPUT ===")
    print(result)
    print("==============")

    # CLEANUP — RAM/GPU dibebaskan abis generate
    print("\n[done] Cleaning up...")
    del model, out, idx
    cleanup()
    print_resource_stats("after cleanup")
    print("RAM/GPU sudah bebas.")


# ============================================================
# MAIN
# ============================================================
def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help", "help"):
        print(__doc__)
        return

    cmd = sys.argv[1]
    if cmd == "info":
        cmd_info()
    elif cmd == "train":
        cmd_train()
    elif cmd == "generate":
        prompt = sys.argv[2] if len(sys.argv) > 2 else "Indonesia"
        cmd_generate(prompt)
    else:
        print(f"Unknown command: {cmd}")
        print("Available: info, train, generate")
        sys.exit(1)


if __name__ == "__main__":
    main()
