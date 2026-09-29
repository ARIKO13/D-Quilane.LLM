"""
============================================================
  MINI LLM — Kaggle Edition (GPU ready)
============================================================
Cara pakai di Kaggle:
1. Buat New Notebook di kaggle.com/code
2. Set Accelerator: GPU T4 x2 (atau P100) di Settings -> Accelerator
3. Set Internet: ON (kalau mau pip install tambahan, tidak wajib)
4. Paste file ini sebagai satu cell, atau upload file2 di bawah
5. Run All

File-file yang perlu ada:
- corpus.txt        (dataset — pakai versi kaggle yg lebih gede)
- tokenizer.py
- mini_gpt_llm.py
- train.py
- infer.py

Atau cukup paste blok ini langsung di satu cell Kaggle
"""
import sys, os, subprocess, time
from pathlib import Path

# ---------- 0. install torch (kalau belum) ----------
try:
    import torch
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "torch", "numpy", "-q"])
    import torch

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    DEVICE = "cuda"
else:
    print("WARNING: GPU tidak terdeteksi. Pakai CPU (lambat).")
    DEVICE = "cpu"

# ---------- 1. config (naikkan untuk GPU!) ----------
# Versi sandbox: d_model=96, layers=12 (750 KB)
# Versi Kaggle GPU (rekomendasi starter):
CFG = {
    "vocab_size": 1024,        # 3x lipat dari sandbox
    "d_model": 256,            # 2.7x lipat
    "n_heads": 8,              # 8 head, d_head=32
    "d_ff": 1024,              # 4x d_model
    "num_layers": 24,          # virtual depth 24 (masih 1x param!)
    "max_seq_len": 256,        # context 4x lipat
}

EPOCHS = 5_000                # naikkan
BATCH = 64
LR = 3e-4
SEED = 42

# ---------- 2. tulis source code (inline utk kaggle) ----------
# (kalau di kaggle, paste kode dari scripts/mini_gpt_llm.py, tokenizer.py, train.py di sini)
# Untuk demo, kita pakai versi inline ringkas dari model class.

# FILE: tokenizer.py (paste kode lengkap dari file scripts/tokenizer.py)
# FILE: mini_gpt_llm.py (paste kode lengkap dari file scripts/mini_gpt_llm.py)
# FILE: train.py (paste, modifikasi CFG & DEVICE)

# ---------- 3. jalankan training di GPU ----------
import importlib.util
spec = importlib.util.spec_from_file_location("mini_gpt_llm",
        "/kaggle/working/mini_gpt_llm.py")  # sesuaikan path
# ... dst

# ---------- 4. estimasi size ----------
import math
def estimate_size(cfg):
    d = cfg["d_model"]
    v = cfg["vocab_size"]
    f = cfg.get("d_ff", 4*d)
    n = cfg["max_seq_len"]
    emb = v*d + n*d
    attn = 3*d*d + d*d
    mlp = 3*d*f
    norm = 3*d
    total = emb + attn + mlp + norm
    return total, total * 4 / 1024 / 1024

n, mb = estimate_size(CFG)
print(f"\nEstimated unique params: {n:,}")
print(f"Estimated file size:    {mb:.2f} MB (fp32, tanpa quantize)")
print(f"Virtual depth:           {CFG['num_layers']}")
print(f"Effective (if no share): {n * CFG['num_layers']:,}")
