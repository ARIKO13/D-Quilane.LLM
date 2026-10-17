# D'QUILANE

LLM from scratch. Large effective capacity, small file size, FP32, runs on CPU.

## Files

| File | Size | Description |
|---|---|---|
| `D-QUILANE-base.gguf` | 871 KB | Pure architecture (EMPTY — no data, no AION, random init) |
| `D-QUILANE-trainer.gguf` | 1.40 MB | Trained on full SD dataset (10,268 Q&A pairs cached in AION) |
| `dataset/` | 22 parquet files | 10,268 Q&A pairs (SD worldwide curriculum) |
| `dataset_SD.zip` | 497 KB | Same dataset, zipped |
| `build_dquilane.py` | 10 KB | Training script (rebuild base + trainer) |
| `chimera_model.py` | 16 KB | Architecture code (Chimera-8B-Effective) |

## Base vs Trainer

**Same architecture** — both have 215,808 params (training doesn't change count).

| Property | D-QUILANE-base.gguf | D-QUILANE-trainer.gguf |
|---|---|---|
| File size | 871 KB | 1.40 MB |
| Architecture | chimera_8b_effective | chimera_8b_effective |
| Unique params | 215,808 | 215,808 (SAME) |
| Weights | Random init (seed=42) | Trained via progressive stacking |
| AION engram | **0 pairs (EMPTY)** | **7,266 pairs (full SD curriculum)** |
| SHA256 (first 16) | `bd112873b1eff50b` | `f215998e1b445d9c` |

### What is "base"?

`D-QUILANE-base.gguf` is the **bare architecture** — no training, no Q&A cached.
It's the "blank slate" you'd start training from. Random weights (torch seed=42),
empty AION engram. Pure architecture skeleton.

### What is "trainer"?

`D-QUILANE-trainer.gguf` is the **trained version** — progressive stacking training
on the full SD dataset (10,268 Indonesian Q&A pairs). Loss progression:

| Phase | Description | Steps | Loss start → end |
|---|---|---|---|
| 1 | Warmup (1 block, 1× repeat) | 30 | 3.87 → 3.86 |
| 2 | Stacking (1→2→3× repeat) | 20 | 3.43 → 3.28 |
| 3 | Expert activation (ReXMoE + LB) | 40 | 2.75 → 2.57 |
| 4 | Full training (12 eff. layers) | 110 | 2.47 → 1.64 |

Total: 200 steps, ~33s on CPU. Batch size=8.

AION engram contains 7,266 unique Q&A pairs (10,268 raw → dedup by normalized key).
This is the FULL SD curriculum from `dataset/`.

## Architecture (both files)

`chimera_8b_effective` — sandbox-scale config:

| Property | Value |
|---|---|
| d_model | 96 |
| vocab_size | 115 (char-level) |
| n_blocks × n_repeat | 4 × 3 = 12 effective layers |
| n_experts | 4 (ReXMoE, top-2 routing) |
| attn_rank | 32 |
| ffn_hidden | 48 (bottleneck: 96→48→96) |
| n_heads / n_kv | 4 / 1 (4× KV reduction) |
| sliding_window | 32 (on odd-index blocks) |
| **Unique params** | **215,808** |
| **FP32 size** | **0.82 MB** (weights only) |

**Techniques implemented:**
- Low-Rank Attention (rank=32, 37% param reduction vs full QKV)
- FFN Bottleneck (SwiGLU, 96→48→96, 87.5% reduction vs standard 4× expansion)
- ReXMoE (4 experts shared across all 12 effective layers)
- Weight Tying (output head = token embedding, saves 11K params)
- LayerScale (init=0.1, prevents identity collapse)
- RMSNorm (pre-norm)
- RoPE (parameter-free positional encoding)
- Grouped Query Attention (4 Q heads, 1 KV head)
- Load Balancing Loss (Switch Transformer style)
- Router Z-Loss (ST-MoE style)
- EntroCut (stop generation at conf > 0.95)
- AION Adaptive (O(1) engram bypass)

## Target architecture: chimera_1gb (sub-1GB)

The shipped GGUFs use **sandbox config** (215K params, 0.82 MB) for fast verification
+ GitHub hosting (under 100MB per file).

The **target spec** for a real D'QUILANE model would use `chimera_1gb` config:

| Property | Sandbox (shipped) | chimera_1gb (target) |
|---|---|---|
| d_model | 96 | 2048 |
| vocab | 115 (char) | 32,000 (BPE) |
| n_blocks × n_repeat | 4 × 3 = 12 | 6 × 4 = 24 |
| n_experts | 4 | 6 |
| attn_rank | 32 | 256 |
| ffn_hidden | 48 | 1024 |
| Unique params | 215,808 | ~180M (835× larger) |
| FP32 size | 0.82 MB | 688 MB (835× larger) |

Training requirements for `chimera_1gb`:
- 12GB+ GPU (RTX 3060/4070) — sandbox CPU cannot do this
- 1-2 weeks training time
- 100B+ tokens dataset
- Output would be 688 MB (under 1GB target ✅, requires Git LFS for GitHub)

## Dataset

`dataset/` and `dataset_SD.zip` contain 22 Parquet files:
- 10,268 Q&A pairs total
- Indonesian SD1-6 worldwide curriculum
- Categories: matematika, bahasa, sains, sejarah, geografi, multibahasa, logika, etc.
- Sources: Indonesia, India, China, Japan, Australia, Europe elementary
- Format: Parquet + ZSTD compression

## How to rebuild

```bash
# Sandbox-scale (CPU, ~33s training):
python build_dquilane.py

# To scale up to chimera_1gb (180M params, 688 MB FP32):
# 1. Edit build_dquilane.py
#    - Change cfg = CONFIGS['sandbox'] → CONFIGS['chimera_1gb']
#    - Adjust batch_size, steps for larger model
# 2. Run on GPU 12GB+:
#    python build_dquilane.py
# Estimated time: 1-2 weeks
```

## License

MIT.
