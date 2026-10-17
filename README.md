# D'QUILANE

LLM from scratch. Large effective capacity, small file size, FP32, runs on CPU.

## Files

| File | Description |
|---|---|
| `D-QUILANE-base.gguf` | Base architecture GGUF (Blueprint v2.1, sandbox-scale, 2.5 MB FP32) |
| `D-QUILANE-trainer.gguf` | Trained version (with AION engram + progressive stacking training) |
| `dataset/` | 22 Parquet files (10,268 Q&A pairs, SD worldwide curriculum) |
| `dataset_SD.zip` | Same dataset, zipped (497 KB compressed) |

## Architecture

D'QUILANE uses Blueprint v2.1 architecture:
- LFM2 hybrid (gated short conv + attention)
- ReXMoE (4 shared experts, top-2 routing)
- AION adaptive (O(1) engram bypass)
- EntroCut (stop at conf > 0.95)
- XNC compression
- 6 conv + 6 attention blocks (12 effective layers)
- 87 AION engram pairs (Indonesian Q&A)

Target spec (Chimera-8B-Effective, sub-1GB version):
- d_model = 2048, vocab = 32,000
- 6 unique blocks × 4 repeat = 24 effective layers
- 6 shared experts (ReXMoE, top-2 routing + load balancing + router z-loss)
- Low-rank GQA (rank=256, 16 Q heads, 4 KV heads)
- FFN bottleneck (2048→1024→2048, SwiGLU)
- RoPE (parameter-free positional)
- Weight tying (output = token embedding)
- LayerScale + RMSNorm
- ~180M unique params, ~688 MB FP32 ✅ under 1 GB

## Dataset

`dataset/` and `dataset_SD.zip` contain 22 Parquet files:
- 10,268 Q&A pairs total
- Indonesian SD1-6 worldwide curriculum
- Categories: matematika, bahasa, sains, sejarah, geografi, multibahasa, logika, etc.
- Sources: Indonesia, India, China, Japan, Australia, Europe elementary
- Format: Parquet + ZSTD compression

## License

MIT.
