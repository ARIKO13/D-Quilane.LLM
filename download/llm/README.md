# Mini LLM — Dari Nol (Sandbox Edition)

LLM pertama kamu, dibangun dari nol tanpa library transformer siap pakai.

## Apa yang ada di folder ini

| File | Ukuran | Deskripsi |
|------|--------|-----------|
| `mini_llm.pt` | ~750 KB | Bobot model (PyTorch checkpoint, fp32) |
| `tokenizer.json` | ~3 KB | BPE merges (384 vocab) |

## Konsep Kunci: Parameter Sharing (ALBERT-style)

Tujuan kamu: **"parameter seakan besar, size tetap kecil tanpa quantize."**

Triknya: **1 set bobot transformer dipakai berulang untuk N layer virtual.**

```
Model konvensional (12 layer):
    Layer1 (param)  -> Layer2 (param)  -> ... -> Layer12 (param)
    Total param = 12 × size_per_layer  ->  file besar

Model ini (parameter sharing):
    Block (1 set param) -> dipakai 12x berturut-turut
    Total param = 1 × size_per_block  ->  file kecil, depth tetap 12
```

Hasil di model ini:
- **Param unik: 191,136** (~750 KB @ fp32)
- **Depth efektif: 12 layer**
- **"Effective" params jika tidak sharing: ~2.3 juta** (12× lipat)

Tanpa quantize, tanpa kompresi, tanpa 4-bit. Cuma trik arsitektur.

## Arsitektur

| Komponen | Detail |
|----------|--------|
| Tokenizer | BPE buatan sendiri (384 vocab) |
| Embedding | Token + Positional (learned) |
| Block | Pre-norm + Causal MHSA + SwiGLU MLP |
| Norm | RMSNorm (lebih hemat param dari LayerNorm) |
| FFN | SwiGLU (modern, dipakai Llama/Qwen) |
| Output | Weight-tied ke token embedding |
| Depth | 12 (via parameter sharing) |
| d_model | 96 |
| Heads | 6 |

## Cara Pakai

```bash
# generate dari prompt
python3 scripts/infer.py "Indonesia"
python3 scripts/infer.py "Teknologi masa depan"
```

Atau di kode Python:
```python
import sys
sys.path.insert(0, "scripts")
from infer import load_model_and_tokenizer, generate

model, tok = load_model_and_tokenizer()
out = generate(model, tok, "Indonesia", max_new_tokens=50, temperature=0.8)
print(out)
```

## Cara Latih Ulang / Scale Up

Edit `scripts/train.py`:

```python
CFG = {
    "vocab_size": 512,      # lebih besar utk bahasa lebih kaya
    "d_model": 256,         # naikkan
    "n_heads": 8,
    "d_ff": 1024,
    "num_layers": 24,       # virtual depth lebih dalam, file tidak membengkak!
    "max_seq_len": 256,
}
```

Karena parameter sharing, naikkan `num_layers` **tidak** menambah ukuran file — cuma depth efektif yang naik. Itu inti yang kamu mau.

## Keterbatasan Versi Sandbox Ini

- Dataset sangat kecil (~46 baris teks). Output masih semi-acak.
- Training singkat (80 epoch, 17 detik di CPU).
- Loss masih ~4.6 (random ~5.95). Untuk output masuk akal, perlu loss < 3.

## Roadmap Lanjut (kalau mau serius)

1. **Dataset 10MB+** — scraping Wikipedia Indonesia, buku Project Gutenberg, news.
2. **Vocab 8K-32K** — pakai sentencepiece atau turunan tiktoken.
3. **d_model 512-1024**, depth virtual 24-48.
4. **Train 1-7 hari di GPU** (A100/H100).
5. **Fine-tune SFT** dengan data instruction Indonesia.
6. **RLHF/DPO** untuk alignment.

## Struktur Folder

```
/home/z/my-project/
├── scripts/
│   ├── corpus.txt          # dataset training
│   ├── tokenizer.py        # BPE from scratch
│   ├── mini_gpt_llm.py     # arsitektur model dari nol
│   ├── train.py            # training loop
│   └── infer.py            # inference / generation
└── download/
    └── llm/
        ├── mini_llm.pt     # model (750 KB)
        ├── tokenizer.json  # tokenizer (3 KB)
        └── README.md       # file ini
```

Semua kode di `scripts/` ditulis dari nol — tidak pakai HuggingFace, tidak pakai nanogpt. Bisa kamu baca, edit, dan pahami 100%.
