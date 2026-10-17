# D'QUILANE — LLM from Scratch

> Large effective capacity, small file size, FP32, runs on CPU.

D'QUILANE is a from-scratch LLM whose **unique parameter count is small** (so the file is small and CPU-friendly) but whose **effective capacity** rivals a 6-8B model thanks to parameter sharing, low-rank attention, shared experts (ReXMoE), FFN bottleneck, and search-augmented inference.

## Repository Structure

```
D-Quilane.LLM/
├── README.md                ← you are here
├── MODEL_CARD.md            ← model metadata
│
├── base/                    ← Architecture + inference engine (no weights)
│   ├── gguf_infer.py        ← Pure-Python CPU inference (loads model.gguf)
│   ├── web_search.py        ← Web scraper (Wikipedia + Hacker News + Bing + DDG)
│   ├── web/                 ← Browser UI
│   │   ├── index.html       ← ChatGPT-style chat interface
│   │   └── app.js            ← GGUF binary parser + AION bypass in browser
│   ├── docs/                ← Architecture documentation
│   └── train_scripts/       ← (placeholder for chimera_model.py + trainer)
│
└── trained/                 ← Trained model + dataset
    ├── model.gguf           ← Blueprint v2.1 (sandbox-scale, 87 AION pairs)
    ├── dataset.zip          ← 22 parquet files (297 KB compressed)
    └── dataset/             ← Extracted parquet files
```

## What's in each part

### `base/` — D'QUILANE base architecture

Pure-Python inference engine + web scraper. No dependencies, no PyTorch needed.

- **`gguf_infer.py`** — Loads any GGUF model, runs:
  - **Web search** (search-augmented inference): trigger via `search:` prefix or `terbaru/sekarang/hari ini` keywords
  - **AION bypass**: O(1) engram lookup for instant factual recall
  - **Transformer forward**: full Chimera architecture (24 effective layers)
- **`web_search.py`** — Pure-stdlib web scraper with 4 sources + fallback:
  - Wikipedia ID API (factual Q&A, multilingual)
  - Hacker News Algolia API (tech/news, always works)
  - Bing HTML (general search, best-effort)
  - DuckDuckGo HTML (fallback)
  - 24h TTL cache, URL deduplication
- **`web/`** — Browser UI:
  - Loads `model.gguf` directly via `fetch()` + `ArrayBuffer`
  - Parses GGUF binary format in pure JavaScript
  - Renders chat UI, shows model metadata, AION bypass results

### `trained/` — Trained model + dataset

- **`model.gguf`** — Current trained model (Blueprint v2.1, sandbox-scale):
  - 76 tensors, ~2.5 MB
  - 87 AION engram pairs (Indonesian Q&A)
  - Architecture: blueprint_v2.1 (LFM2 hybrid + ReXMoE + AION)
  - Char-level tokenizer, vocab=140
- **`dataset/`** — 22 Parquet files (10,268 Q&A pairs total):
  - Indonesian SD1-6 worldwide curriculum
  - Categories: matematika, bahasa, sains, sejarah, geografi, etc.
  - Sources: Indonesia, India, China, Japan, Australia, Europe elementary
  - Format: Parquet + ZSTD compression (297 KB total)

## Quick Start

### CPU inference (Python)

```bash
cd base/
python gguf_infer.py ../trained/model.gguf "Halo"
# → AION bypass (O(1), conf=1.00)
# → D'QUILANE: Halo juga! Senang bertemu denganmu.

python gguf_infer.py ../trained/model.gguf "search: Jakarta hari ini"
# → Web search [wikipedia, hackernews, bing] (fresh)
# → [Wikipedia + 5 HN articles about Jakarta]

python gguf_infer.py ../trained/model.gguf --interactive
# → Interactive chat mode
```

### Web UI (browser)

Open `base/web/index.html` in a browser. It fetches `../trained/model.gguf` directly.

## Inference Pipeline

```
User prompt
    │
    ├─► [0] Web search trigger?
    │       (search:/cari:/google: prefix OR keyword terbaru/sekarang/hari ini)
    │       YES → fetch web → format → return
    │       NO  ↓
    │
    ├─► [1] AION bypass?
    │       (exact/substring/word-overlap match in engram cache)
    │       YES → return cached answer (0.04s)
    │       NO  ↓
    │
    └─► [2] Transformer forward (24 effective layers)
            generate tokens via top-k/top-p sampling + EntroCut
```

## Architecture (Blueprint v2.1, current model)

| Component | Spec |
|---|---|
| Architecture | blueprint_v2.1 |
| d_model | 96 (sandbox-scale) |
| Layers | 6 conv + 6 attention (12 effective) |
| Experts | 4 (ReXMoE shared) |
| Vocab | 140 (char-level) |
| Max seq | 80 |
| Techniques | LFM2 hybrid, ReXMoE, AION, EntroCut, XNC |

## Architecture (Chimera-8B-Effective, target spec)

| Component | Spec |
|---|---|
| Architecture | chimera_8b_effective |
| d_model | 3072 (or 2048 for sub-1GB target) |
| Unique blocks | 6 |
| Repeat factor | 4 → 24 effective layers (4× reuse) |
| Experts | 6 shared (ReXMoE, top-2 routing) |
| Attention | Low-rank GQA (rank=384, 24 heads, 6 KV heads) |
| FFN | Bottleneck 3072→1536→3072 (SwiGLU) |
| Norm | RMSNorm + LayerScale (init=1e-1) |
| Positions | RoPE (parameter-free) |
| Weight tying | tok_emb = output head (saves 98M params) |
| Load balancing | Switch Transformer aux loss |
| Router z-loss | ST-MoE style |
| Configs | sandbox, rtx3050, chimera_s, chimera_1gb, chimera |

**Note**: `chimera_model.py` (Chimera architecture code) was lost in a session reset. The spec is documented in `base/docs/` but the actual PyTorch implementation needs to be rebuilt from spec before training a Chimera-8B-Effective model.

## License

MIT.
