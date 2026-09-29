# D'QUILANE — Laptop Edition

LLM dari nol, jalan di laptop kamu. **GPU & RAM cuma kerja pas training/generation, abis itu balik ke SSD.**

## 📦 Yang Perlu Di-install

```bash
# Untuk Windows/Mac/Linux:
pip install torch numpy

# Optional: utk lihat RAM usage
pip install psutil
```

Kalau punya NVIDIA GPU: install PyTorch versi CUDA dari https://pytorch.org/get-started/locally/
Kalau Apple Silicon (M1/M2/M3/M4): pytorch standar udah support MPS otomatis.

## 🚀 Cara Pakai

### 1. Cek hardware kamu
```bash
python3 dquilane.py info
```
Akan nge-print: device (CUDA/MPS/CPU), GPU memory, dan estimasi param model.

### 2. Training (GPU/RAM kerja)
```bash
python3 dquilane.py train
```
Selama training:
- GPU/RAM dipakai buat training
- Setelah selesai, model + tokenizer otomatis save ke `./dquilane_files/`
- Setelah script exit, **GPU & RAM dibebaskan sepenuhnya**

### 3. Generate text (load dari SSD)
```bash
python3 dquilane.py generate "Indonesia"
python3 dquilane.py generate "Belajar setiap hari"
```
Selama generate:
- Model di-load dari SSD ke GPU/RAM
- Generate text
- Setelah exit, memory dibebaskan

### 4. Keluar
Tinggal close terminal, semua memory otomatis bebas. File `.pt` tetap di SSD.

## ⚙️ Konfigurasi

Edit `dquilane.py` bagian atas:
```python
CFG = {
    "vocab_size": 1024,   # naikkan kalau dataset gede
    "d_model":    256,    # 256 = entry-level GPU; 512 = mid; 768 = GPT-2 scale
    "n_heads":    8,
    "d_ff":       1024,
    "num_layers": 24,     # virtual depth — gak nambah file size!
    "max_seq_len": 256,   # context length
}
EPOCHS = 2000            # lebih banyak = lebih overfit/akurat
BATCH = 32              # turunkan kalau OOM
LR = 3e-4
```

**Trik hemat**: naikkan `num_layers` GRATIS (gak nambah file size). Naikkan `d_model` baru bikin file gede.

## 📊 Estimasi Resource

| Config | RAM/GPU | File Size | Effective Params |
|--------|---------|-----------|------------------|
| d_model=128, depth=12 | 1-2 GB | 1.3 MB | 16M |
| d_model=256, depth=24 | 3-5 GB | 5.3 MB | 127M |
| d_model=512, depth=32 | 8-12 GB | 21 MB | 670M |
| d_model=768, depth=48 | 16-24 GB | 47 MB | 2.3B effective |

## 🔧 Troubleshooting

**CUDA out of memory:**
- Kurangi `BATCH` (32 → 16 → 8)
- Kurangi `d_model` (256 → 128)
- Pakai `max_seq_len` lebih kecil (256 → 128)

**Apple Silicon gak ke-detect MPS:**
- Update PyTorch ke versi >= 2.0
- Cek: `python3 -c "import torch; print(torch.backends.mps.is_available())"`

**Lambat di CPU:**
- Wajar. Untuk dataset kecil masih OK (5-10 menit)
- Untuk production, GPU wajib

## 💡 Cara Kerja Memory

```
[before train]
RAM: 4 GB (normal pakai OS)
GPU: idle

[train running]
RAM: 4 + 2 GB = 6 GB (model + dataset + optimizer)
GPU: 3 GB allocated (kalau pakai GPU)

[train done, save]
RAM: 6 GB (masih)
File .pt: 5 MB written to SSD

[script exit]
RAM: 4 GB (balik normal)
GPU: idle
```

Pattern ini disebut **ephemeral resource** — hanya hidup selama proses berjalan.

## 📁 Struktur Folder

```
download/llm/
├── dquilane.py           # main script (self-contained)
├── corpus.txt            # dataset training (edit/tambahkan ini!)
├── dquilane_files/       # auto-create saat train
│   ├── mini_llm.pt      # model bobot (~5 MB)
│   └── tokenizer.json   # BPE merges
├── MODEL_CARD.md         # kartu identitas D'QUILANE
├── kaggle_notebook.ipynb # versi kaggle
└── README.md
```

## 🎯 Next Step

1. **Tambahan corpus.txt** — scrape Wikipedia Indonesia / Berita / Buku
2. Train ulang pakai dataset lebih besar
3. Kalau output sudah masuk akal, naikkan `d_model` ke 512
4. Bikin SFT dataset (instruction → response) untuk instruction tuning

---

Kalau ada error, paste ke aku — aku bantu debug. 🛠️
