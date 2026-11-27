# Qoolse AI v3 — 1B-Loop Native BF16

Implementasi ulang arsitektur dari blueprint **Qoolse AI v3**: 135 juta parameter
fisik presisi BF16 yang diputar **9 kali** secara rekursif sehingga setara
**1,015 miliar** parameter efektif, plus satu **Shared-KV** yang dipakai seluruh
loop, MoE top-2 di dalam loop, dan **adaptive early exit** (minimal 4 putaran,
maksimal 9).

```
A  factorized tied embedding + LM head        6,34 M
B  stem 2 lapis + Shared-KV + exit refinement 18,66 M
C  core rekursif (C1 gated-delta, C2 retrieval,
   C3 shared expert + halting gate)           14,00 M
D  8 routed SwiGLU expert (top-2 aktif)       96,00 M
--------------------------------------------------------
   total fisik                              135.000.806
   efektif = 25 M + 9 x 110 M             = 1.015 M
```

Angka di atas bukan klaim: `python qoolse_model.py` membangun model dan
mencetak `135,000,806 params`, dan `model/untrained/QoolseAI-base.pt` hasil
build lokal berukuran **±257 MB** (2 byte/parameter, F16).

## Yang benar-benar diimplementasikan

| Blok | Isi |
|---|---|
| **A** | `E: 24000 x 256` → `W_up: 256 x 768` → `RMSNorm_emb`; `W_up` dan `E` dipakai ulang transposenya sebagai head (tied), jadi kamus tidak menambah byte |
| **B1** | 2 lapisan stem non-rekursif: 12 head x 64, sliding-window attention 256 token + RoPE, SwiGLU 768→2048→768 |
| **B2** | satu producer `W_k, W_v: 768→256` (4 head x 64) dengan RoPE θ=500.000; K/V dipakai bersama 9 loop, plus amplop `K_min/K_max` per blok 64 token |
| **B3** | exit refinement SwiGLU 768→1780→768 + gerbang kalibrasi skalar + `RMSNorm_out` |
| **C1** | Gated DeltaNet multi-skala: 12 head x 64, Conv1D kausal k=4, delta rule `S_t = α_t S_{t-1} + β_t (v - S k) kᵀ`; 6 head cepat (retensi ~0,70) + 6 head lambat (~0,997), state tetap 98 KB |
| **C2** | retrieval Shared-KV per loop: LoRA rank-32 pada proyeksi query tiap putaran, QK-Norm + RoPE, skor blok dari amplop min/max, top-32 blok + 64 token lokal (≤ 2.112 token dibaca per loop) |
| **C3/D** | 1 shared expert SwiGLU 768→4736→768 selalu aktif + 8 routed expert 768→5208→768 yang dikirim sparse (top-2 per token) + halting gate |
| **Loop** | step embedding per putaran, residual gate γ zero-init, halting gate mengakumulasi probabilitas; token yang lewat ambang dibekukan (early exit) setelah putaran ke-4 |
| **Head** | `RMSNorm_out(h) @ W_upᵀ @ Eᵀ` — weight tying penuh |

Streaming decode memakai cache yang sesuai blueprint: K/V stem per window,
satu Shared-KV yang tumbuh, state delta per loop (12 x 64 x 64), dan buffer
konvolusi kausal. Logits jalur streaming sama dengan forward penuh
(`max|diff| = 9,5e-07`, argmax identik pada uji 40 token).

## Menjalankan

```bash
python qoolse_model.py                  # self-test + jumlah parameter
python build_qoolse.py --base-only      # tulis model/untrained/QoolseAI-base.pt
python build_qoolse.py --base-only --tier Compact --params 6B   # skala lain
python build_qoolse.py --epochs 5       # training -> model/trained/QoolseAI-trainer.pt
python build_qoolse.py --init model/untrained/QoolseAI-base.pt --epochs 3   # fine-tune
python build_qoolse.py --epochs 5 --export-gguf   # + export inferensi .gguf
python infer_qoolse.py --model model/untrained/QoolseAI-base.pt --prompt "halo"
python playground.py                    # http://127.0.0.1:8000, memindai model/
```

Tanpa flag, build memakai model v3 yang terkunci di 135 M. `--tier`/`--params`
memperbesar/memperkecil bentuk v3 (rasio dipertahankan), `--auto` memakai
setengah RAM bebas. Kelas ukuran: Small 1B–4B, Compact 4B–9B, Medium 9B–15B,
Large 15B–35B, Very Large 35B–70B, Frontier 70B–200B, Large-scale 200B–500B,
Ultra-scale 500B–1T, Beyond 1T >1T.

## Angka terukur (mesin ini, 6 thread CPU)

| Langkah | Hasil |
|---|---|
| bangun model + simpan container | 135.000.806 param, 88 tensor, 257,50 MB, 4 blok di atas |
| load container | 4,6 s |
| generate (greedy, 8 token) | **8,5 tok/s** |
| active/token | 63.005.414 (46,7%) — 2 dari 8 expert |
| state delta per token | 9 loop x 12 x 64 x 64 float = 1,8 MB |

## Format file: `.pt` vs `.gguf`

`.pt` adalah wadah utamanya, karena hanya wadah inilah yang bisa dibuka lagi
untuk training.

| Ekstensi | Isi | Dipakai untuk |
|---|---|---|
| `.pt` (checkpoint PyTorch) | `state_dict` + config + tokenizer + tabel AION + state optimizer (Muon & AdamW) | **trainable / fine-tune** (`--init`), `infer_qoolse.py`, `playground.py` |
| `.gguf` (opsional, `--export-gguf`) | bobot F16 saja, GGUF-shaped (magic `GGUF`, versi 3, alignment 32, arsitektur `qoolse_v3_loop`) | export inferensi. llama.cpp/ollama tetap menolak karena arsitekturnya kustom, bukan arsitektur yang mereka kenal |

GGUF di sini bukan kuantisasi: tensor disimpan F16 apa adanya. Yang membuatnya
hanya cocok untuk inferensi adalah isinya — tidak ada state optimizer dan tidak
ada informasi training, jadi loop training tidak bisa dilanjutkan darinya.
Kalau bobot ingin dilanjutkan dilatih, mulai dari `.pt`:

```bash
python build_qoolse.py --init model/untrained/QoolseAI-base.pt --epochs 3
```

## Yang belum ada (jangan diklaim)

- **Belum ditraining.** `QoolseAI-base.pt` masih inisialisasi acak (seed 42);
  output sampling hanya statistik karakter. `QoolseAI-trainer.pt` belum dibuat.
- **Tokenisasi masih char-level 118 token** di dalam tabel 24.000 slot; BPE
  bilingual 23.984 token dari blueprint belum dibangun.
- **Hybrid DB 30 MB (SQLite FTS5 + indeks Hamming 256-bit)** dan 5 fase runtime
  belum diimplementasikan; yang ada baru tabel retrieval AION di metadata.
- **Target 12–18 tok/s di HP 4 GB** dari blueprint belum diverifikasi di perangkat
  tersebut; angka di atas diukur di CPU desktop.
- Compute berjalan FP32 di CPU (PyTorch CPU tanpa kernel BF16 cepat); presisi
  bobot tetap BF16 seperti spesifikasi.

## Struktur folder model

```
model/
  untrained/   QoolseAI-base.pt      # random init, trainable (`build_qoolse.py --base-only`)
  trained/     QoolseAI-trainer.pt   # hasil training (`build_qoolse.py --epochs N`)
```

File bobot tidak di-commit (257 MB, di atas batas 100 MB GitHub), jadi folder
dibuat lewat `build_qoolse.py` dan hanya `.gitkeep` yang ikut repo. Dua script
lain memakai path default ini, jadi tanpa flag pun sudah benar:
`infer_qoolse.py` membaca `model/untrained/QoolseAI-base.pt`, sedangkan
`playground.py` memindai seluruh `model/` secara rekursif (`.pt` maupun `.gguf`).

## File

| File | Isi |
|---|---|
| `qoolse_model.py` | arsitektur v3, cache streaming, checkpoint `.pt` + container GGUF, auto sizing |
| `build_qoolse.py` | dedup + split + training (Muon + AdamW) + evaluasi |
| `infer_qoolse.py` | inferensi CPU (kalkulator → retrieval AION → model) |
| `playground.py` | UI web lokal (stdlib, tanpa dependensi tambahan) |
| `calculator.py` | kalkulator deterministik (untuk rute aritmatika) |
| `model/untrained/QoolseAI-base.pt` | checkpoint 135.000.806 param, random init, trainable — dibuat `build_qoolse.py`, tidak di-commit |
| `model/trained/QoolseAI-trainer.pt` | hasil training + state optimizer, ditulis `build_qoolse.py --epochs N` |
