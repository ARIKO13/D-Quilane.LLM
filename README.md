# D'QUILANE

LLM from scratch. Large effective capacity, small file size, FP32, runs on CPU.

## Files

| File | Size | Description |
|---|---|---|
| `D-QUILANE-base.gguf` | 0.83 MB | Base architecture (untrained, random init, 9 AION pairs) |
| `D-QUILANE-trainer.gguf` | 0.84 MB | Trained version (progressive stacking, 59 AION pairs) |
| `dataset/` | 22 parquet | 10,268 Q&A pairs (SD worldwide curriculum) |
| `dataset_SD.zip` | 497 KB | Same dataset, zipped |

Both GGUFs use `chimera_8b_effective` architecture:
- d_model = 96, vocab = 144
- 4 unique blocks × 3 repeat = 12 effective layers
- 4 shared experts (ReXMoE, top-2 routing)
- Low-rank attention (rank=32) + FFN bottleneck (96→48→96)
- RoPE (parameter-free positional)
- Weight tying (output = token embedding)
- LayerScale (init=0.1) + RMSNorm
- Load balancing loss (Switch Transformer) + router z-loss (ST-MoE)
- 215K-218K unique params, 0.83-0.84 MB FP32

## Base vs Trainer difference

| Property | Base | Trainer |
|---|---|---|
| Init | Random (torch.manual_seed=42) | Trained weights |
| Training | None | Progressive stacking (4 phases, 60 steps on 50 samples) |
| AION engram | 9 pairs (basic identity Q&A) | 59 pairs (9 base + 50 from dataset) |
| Architecture | chimera_8b_effective | chimera_8b_effective |
| Params | 218,592 | 215,808 |
| File size | 871,456 bytes | 875,584 bytes |
| SHA256 (first 16) | 62a2133672ba6317 | e7848fd56993e36f |

## Training summary

Progressive stacking trainer (4 phases, 60 total steps, ~7s on CPU):

| Phase | Description | Steps | Loss start → end |
|---|---|---|---|
| 1 | Warmup (1 block, 1× repeat) | 9 | 4.46 → 4.13 |
| 2 | Stacking (1→2→3× repeat) | 6 | 4.46 → 3.81 |
| 3 | Expert activation (ReXMoE + LB) | 12 | 4.38 → 3.79 |
| 4 | Full training (12 eff. layers) | 33 | 3.20 → 3.20 |

Note: Loss decrease is small because we trained on only 50 samples × 60 steps
(severe underfitting, sandbox-scale demo). For real training, run
`chimera_train.py chimera_1gb` on a 12GB+ GPU for 1-2 weeks (50K steps, full
dataset, sub-1GB target = 180M params, 688 MB FP32).

## Target architecture: chimera_1gb (sub-1GB)

| Property | Value |
|---|---|
| d_model | 2048 |
| vocab | 32,000 |
| n_blocks × n_repeat | 6 × 4 = 24 effective layers |
| n_experts | 6 (ReXMoE, top-2 routing) |
| attn_rank | 256 |
| ffn_hidden | 1024 (bottleneck) |
| n_heads / n_kv | 16 / 4 (4× KV reduction) |
| Unique params | ~180M |
| FP32 size | ~688 MB ✅ under 1 GB |

## Dataset

`dataset/` and `dataset_SD.zip` contain 22 Parquet files:
- 10,268 Q&A pairs total
- Indonesian SD1-6 worldwide curriculum
- Categories: matematika, bahasa, sains, sejarah, geografi, multibahasa, logika, etc.
- Sources: Indonesia, India, China, Japan, Australia, Europe elementary
- Format: Parquet + ZSTD compression

## License

MIT.
