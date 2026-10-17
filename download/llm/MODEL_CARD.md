# D'QUILANE — Model Card

## Model details

| Field | Value |
|---|---|
| **Name** | D'QUILANE |
| **Architecture** | blueprint_v2.1 (current shipped model) |
| **Target architecture** | chimera_8b_effective (spec only, code TBD) |
| **Version** | 2.1 (shipped) / 3.2 audit-fix (spec) |
| **Precision** | FP32 (no quantisation) |
| **Unique params** | ~218K (sandbox-scale, shipped) |
| **Size** | ~2.5 MB FP32 (shipped) |
| **License** | MIT |

## Shipped model (`trained/model.gguf`)

Architecture: **blueprint_v2.1** (LFM2 hybrid conv+attention)

| Component | Value |
|---|---|
| d_model | 96 |
| Layers | 6 conv + 6 attention (12 effective) |
| Experts | 4 (ReXMoE shared) |
| Vocab | 140 (char-level) |
| Max seq | 80 |
| AION engram | 87 Q&A pairs (Indonesian) |
| Char tokenizer | Built from dataset |

**Pipeline**:
1. Web search trigger detection (`search:` prefix, `terbaru/sekarang/hari ini` keywords)
2. AION bypass (O(1) engram lookup, 0.04s)
3. Transformer forward (12 effective layers + ReXMoE)
4. Token sampling (top-k, top-p, EntroCut)

## Target architecture (`chimera_8b_effective` spec)

Documented in `base/docs/` — code TBD (chimera_model.py was lost in session reset).

### Configs

| Config | d_model | n_blocks × n_repeat | n_experts | attn_rank | ffn_hidden | vocab | Params | FP32 size |
|---|---|---|---|---|---|---|---|---|
| sandbox | 96 | 4×3 | 4 | 32 | 48 | 144 | 218K | 0.84 MB |
| rtx3050 | 1024 | 6×4 | 6 | 128 | 512 | 4K | ~33M | ~125 MB |
| chimera_s | 2048 | 6×4 | 6 | 256 | 1024 | 16K | ~148M | ~563 MB |
| **chimera_1gb** | **2048** | **6×4** | **6** | **256** | **1024** | **32K** | **~180M** | **~688 MB** ✅ |
| chimera | 3072 | 6×4 | 6 | 384 | 1536 | 32K | ~357M | ~1361 MB |

### Techniques (implemented in spec)

1. Low-Rank Attention (rank=256-384)
2. FFN Bottleneck (d→d/2→d, SwiGLU)
3. ReXMoE (6 experts shared, top-2 routing)
4. Weight Tying (tok_emb = output head)
5. LayerScale (init=1e-1, prevents identity collapse)
6. RMSNorm (pre-norm)
7. RoPE (parameter-free positional)
8. AION Adaptive (substring + word-overlap bypass)
9. EntroCut (stop at conf>0.95)
10. Sliding-Window Attention (1024 on odd blocks)
11. Progressive Stacking (4 phases: warmup→stacking→expert→full+early stop)
12. Grouped Query Attention (24 Q / 6 KV heads, 4× reduction)
13. Load Balancing Loss (Switch Transformer)
14. Router Z-Loss (ST-MoE)
15. Dropout (0.1, training only)
16. Weight Decay (0.1 in AdamW)
17. Gradient Checkpointing (for full chimera on 24GB GPU)

## Dataset

- 10,268 Q&A pairs in `trained/dataset/`
- 22 Parquet files (519 KB uncompressed, 297 KB compressed in dataset.zip)
- Categories: matematika, bahasa, sains, sejarah, geografi, multibahasa, logika, conversations, etc.
- Sources: Indonesian SD curriculum + India/China/Japan/Australia/Europe elementary
- Format: Parquet + ZSTD compression

## Inference modes

| Mode | Trigger | Latency | Quality |
|---|---|---|---|
| Web search | `search:`/`cari:` prefix OR `terbaru/sekarang` keywords | 0.5-2s (fresh), <0.01s (cached) | High (real-time data) |
| AION bypass | Exact/substring/word-overlap match in engram | 0.04s | High (curated Q&A) |
| Transformer | Fallback (no trigger, no AION match) | 1-5s (sandbox) | Low (untrained sandbox) |

## Limitations

- **Shipped model is sandbox-scale** (218K params, untrained, random init for transformer weights)
- Only AION bypass + web search return meaningful responses currently
- To get full Chimera-8B-Effective responses: rebuild `chimera_model.py` from spec, train on GPU 12GB+ for 1-2 weeks
- Char-level tokenizer (vocab=140) — would need BPE for real performance
- Pure-Python inference is slow (0.6 tok/s) — needs native AVX2 kernels for >20 tok/s target

## License

MIT.
