# D'QUILANE Trained

Trained model weights + dataset.

## Files

```
trained/
├── model.gguf     ← Blueprint v2.1 (sandbox-scale, 87 AION pairs)
├── dataset.zip    ← 22 parquet files, 297 KB compressed
└── dataset/       ← Extracted parquet files (10,268 Q&A pairs)
```

## Model: blueprint v2.1

| Field | Value |
|---|---|
| Architecture | blueprint_v2.1 |
| d_model | 96 (sandbox-scale) |
| Layers | 6 conv + 6 attention (12 effective) |
| Experts | 4 (ReXMoE shared) |
| Vocab | 140 (char-level) |
| Max seq | 80 |
| AION engram | 87 Q&A pairs (Indonesian) |
| Size | ~2.5 MB FP32 |

## Dataset

10,268 Q&A pairs covering SD1-6 worldwide curriculum:
- Indonesian SD curriculum (matematika, bahasa, sains, etc.)
- India elementary
- China elementary
- Japan elementary
- Australia elementary
- Europe elementary

22 Parquet files (297 KB ZSTD compressed in dataset.zip).
