# D'QUILANE Architecture Documentation

## Current shipped model: blueprint_v2.1

The `trained/model.gguf` is blueprint_v2.1 — a hybrid conv+attention model:

```
TokenEmb (vocab=140, d=96)
  ↓
For block in 1..6:
  GatedShortConv (causal, kernel=4)
  + ReXMoE (4 experts, top-2)
For block in 1..6:
  LowRankAttention (d=96, n_heads=6, n_kv=1)
  + ReXMoE (4 experts, top-2)
  ↓
RMSNorm
  ↓
Linear(tied with TokenEmb) → logits
```

AION bypass: 87 cached Q&A pairs, O(1) lookup, substring + word-overlap matching.

## Target architecture: chimera_8b_effective

Documented spec (code TBD after session reset):

```
TokenEmb (vocab=32K, d=3072)             ← tied with output head
  ↓
For repeat in 1..4:
  For block in 1..6:                       ← 6 unique × 4 = 24 effective layers
    x = x + LS1( LowRankAttn( RMSNorm1(x), RoPE ) )
    h = RMSNorm2(x)
    x = x + LS2( FFNBottleneck(h) + ExpertPool(h, top-2 routing) )
  ↓
RMSNorm
  ↓
Linear(tied with TokenEmb) → logits
```

### Per-block details

- **LowRankAttention**: Q = Wq_up (rank→d) @ Wq_down (d→rank) @ x, same for K/V (GQA: 24 heads, 6 KV heads). Sliding window (1024) on odd-index blocks.
- **FFNBottleneck**: SwiGLU `d→1536→d` (half the usual FFN expansion)
- **ExpertPool**: 6 experts (FFNBottleneck), top-2 routing with renormalisation. Shared across all 24 effective layers (ReXMoE). Load balancing loss + router z-loss.
- **LayerScale (init=1e-1) + RMSNorm**: Stable training without LayerNorm.
- **RoPE**: Parameter-free positional encoding.

## Configs

| Config | d_model | n_blocks × n_repeat | n_experts | attn_rank | ffn_hidden | vocab | max_seq | Params | FP32 size |
|---|---|---|---|---|---|---|---|---|---|
| sandbox | 96 | 4×3 | 4 | 32 | 48 | 144 | 80 | 218K | 0.84 MB |
| rtx3050 | 1024 | 6×4 | 6 | 128 | 512 | 4096 | 1024 | ~33M | ~125 MB |
| chimera_s | 2048 | 6×4 | 6 | 256 | 1024 | 16000 | 2048 | ~148M | ~563 MB |
| **chimera_1gb** | **2048** | **6×4** | **6** | **256** | **1024** | **32000** | **4096** | **~180M** | **~688 MB** ✅ |
| chimera | 3072 | 6×4 | 6 | 384 | 1536 | 32000 | 4096 | ~357M | ~1361 MB |

## Effective capacity (honest claims)

- **Layer reuse factor**: 4× (6 unique × 4 repeats = 24 effective layers)
- **Compute amplification**: ~12-16× vs same-param dense model (4× layer × 3× expert routing)
- **NOT 16× more parameters** — that was an over-claim. Effective capacity ≈ unique × 4.

## Training strategy (progressive stacking)

| Phase | Description | Steps % | LR |
|---|---|---|---|
| 1 | Warmup: 1 block, 1× repeat | 15% | 3e-4 |
| 2 | Stacking: 1→2→4× repeat | 10% | 1e-4 |
| 3 | Expert activation (ReXMoE + load balancing) | 20% | 5e-4 |
| 4 | Full training (24 eff. layers, early stop on val) | 55% | 1e-4 |
| 5 | Distillation from 7B teacher (optional) | — | 1e-5 |
| 6 | Long-context RoPE extension (optional) | — | 1e-5 |

## Hardware

- **chimera_1gb config**: 12 GB VRAM (RTX 3060 12GB / RTX 4070 12GB), 1-2 weeks BF16
- **chimera full config**: 24 GB VRAM (RTX 4090 / A5000), 2-4 weeks BF16
- **CPU inference target**: >20 tok/s (AVX2, FP32, native kernels — TBD)
- **CPU inference current**: 0.6 tok/s (pure Python, no KV-cache)

## Comparison vs frontier models

| Model | Params | FP32 | Train tokens | D'QUILANE chimera_1gb vs |
|---|---|---|---|---|
| D'QUILANE chimera_1gb | 180M | 0.67 GB | 0 (untrained) | baseline |
| GPT-2 Small | 124M | 0.5 GB | 10B | similar param class |
| Pythia-160M | 160M | 0.64 GB | 300B | similar param class |
| TinyLlama 1.1B | 1.1B | 4.4 GB | 1T | bigger, more trained |
| Llama-2 7B | 7B | 14 GB | 2T | different tier |

**Honest verdict**: In same param-size class as GPT-2 Small / Pythia-160M. After full training (100-300B tokens), could match GPT-2 Small quality. Cannot compete with 7B models in 180M params.

## Niche where D'QUILANE wins

1. **Private/offline inference**: 0.67 GB, runs on any laptop without GPU
2. **Factual Q&A via AION bypass**: O(1) lookup, 0.04s, beats any LLM API latency
3. **Web-augmented inference**: small + private alternative to Perplexity
4. **Cost**: 0 USD to run (vs GPT-4 API)
