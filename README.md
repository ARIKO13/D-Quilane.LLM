# D'QUILANE

LLM from scratch. Large effective capacity, small file size, FP32, runs on CPU.

## Files

| File | Size | Description |
|---|---|---|
| `D-QUILANE-base.gguf` | 871 KB | Base architecture (untrained, random init, 9 AION pairs) |
| `D-QUILANE-trainer.gguf` | 875 KB | Trained version (progressive stacking, 59 AION pairs) |
| `dataset/` | 22 parquet files | 10,268 Q&A pairs (SD worldwide curriculum) |
| `dataset_SD.zip` | 497 KB | Same dataset, zipped |

## Architecture (both files)

Both `D-QUILANE-base.gguf` and `D-QUILANE-trainer.gguf` use the **same** `chimera_8b_effective` architecture. Training does NOT change param count — it only updates weight values.

| Property | Value |
|---|---|
| Architecture | chimera_8b_effective |
| d_model | 96 |
| vocab_size | 115 (char-level) |
| n_blocks × n_repeat | 4 × 3 = 12 effective layers |
| n_experts | 4 (ReXMoE, top-2 routing) |
| attn_rank | 32 |
| ffn_hidden | 48 (bottleneck: 96→48→96) |
| n_heads / n_kv | 4 / 1 (4× KV reduction) |
| sliding_window | 32 (on odd-index blocks) |
| **Unique params** | **215,808** (same for both) |
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

## Base vs Trainer difference

Since both files have the SAME architecture (same param count), the difference is:

| Property | Base | Trainer |
|---|---|---|
| File size | 871,456 bytes (0.83 MB) | 875,584 bytes (0.84 MB) |
| SHA256 (first 16) | `62a2133672ba6317` | `b64ce6025ff0ade2` |
| Weights | Random init (torch.manual_seed=42) | Trained via progressive stacking |
| AION engram | 9 pairs (basic identity Q&A) | 59 pairs (9 base + 50 from dataset) |
| Architecture | chimera_8b_effective | chimera_8b_effective |
| n_blocks × n_repeat | 4 × 3 = 12 eff. layers | 4 × 3 = 12 eff. layers |
| Unique params | 215,808 | 215,808 (SAME) |

The 4,128-byte file size difference comes from the larger AION engram JSON metadata (59 pairs vs 9 pairs = 50 extra Q&A entries stored as strings).

## Training summary

Progressive stacking trainer (4 phases, 60 steps total, ~7s on CPU):

| Phase | Description | Steps | Loss start → end |
|---|---|---|---|
| 1 | Warmup (1 block, 1× repeat) | 9 | 4.46 → 4.13 |
| 2 | Stacking (1→2→3× repeat) | 6 | 4.46 → 3.81 |
| 3 | Expert activation (ReXMoE + load balancing) | 12 | 4.38 → 3.79 |
| 4 | Full training (12 eff. layers) | 33 | 3.20 → 3.20 |

Note: Loss decrease is small because we trained on only 50 samples × 60 steps
(severe underfitting, sandbox-scale demo). This is a PROOF-OF-CONCEPT that the
architecture trains correctly — not a real trained model.

## Target architecture: chimera_1gb (sub-1GB)

The shipped GGUFs use the **sandbox config** (215K params, 0.82 MB) — used for
fast verification + GitHub hosting (under 100MB per file).

The **target spec** for a real D'QUILANE model would use `chimera_1gb` config:

| Property | Value |
|---|---|
| d_model | 2048 (vs 96 in sandbox) |
| vocab | 32,000 (BPE, vs 115 char-level) |
| n_blocks × n_repeat | 6 × 4 = 24 effective layers (vs 4×3=12) |
| n_experts | 6 (vs 4) |
| attn_rank | 256 (vs 32) |
| ffn_hidden | 1024 (vs 48) |
| n_heads / n_kv | 16 / 4 (vs 4/1) |
| **Unique params** | **~180M** (vs 215K — 835× larger) |
| **FP32 size** | **~688 MB** (vs 0.82 MB — 835× larger) |

Training requirements for `chimera_1gb`:
- 12GB+ GPU (RTX 3060/4070) — sandbox CPU cannot do this
- 1-2 weeks training time
- 100B+ tokens dataset
- Output would be 688 MB (under 1GB target ✅, but requires Git LFS for GitHub)

## Dataset

`dataset/` and `dataset_SD.zip` contain 22 Parquet files:
- 10,268 Q&A pairs total
- Indonesian SD1-6 worldwide curriculum
- Categories: matematika, bahasa, sains, sejarah, geografi, multibahasa, logika, etc.
- Sources: Indonesia, India, China, Japan, Australia, Europe elementary
- Format: Parquet + ZSTD compression

## License

MIT.
