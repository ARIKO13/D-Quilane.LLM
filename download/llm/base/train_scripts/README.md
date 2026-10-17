# D'QUILANE Train Scripts (Placeholder)

This folder should contain `chimera_model.py` and `chimera_train.py` for the
Chimera-8B-Effective architecture. **However, both files were lost in a
sandbox session reset and need to be rebuilt.**

## What to put here (TODO)

```
base/train_scripts/
├── chimera_model.py        ← Chimera-8B-Effective architecture (PyTorch)
├── chimera_train.py        ← Progressive stacking trainer
├── build_chimera_gguf.py   ← Build script for sandbox model.gguf
└── chimera_8b_effective.spec.json  ← Architecture spec
```

## Architecture spec

See `../docs/ARCHITECTURE.md` for the full Chimera-8B-Effective spec
(d_model, n_blocks, n_repeat, n_experts, attn_rank, ffn_hidden, etc.).

## How to rebuild (TODO)

```bash
# Once chimera_model.py is restored:
python chimera_model.py sandbox /tmp/test_chimera.gguf  # smoke test
python build_chimera_gguf.py                            # rebuild model.gguf

# Train full chimera_1gb config (sub-1GB target):
python chimera_train.py chimera_1gb \
    ../../trained/dataset/dataset_train.parquet 50000 4
# → trains on 12GB+ GPU for 1-2 weeks, outputs to ../../trained/model.gguf
```

## Why this is empty

During a session reset, the chimera_model.py and chimera_train.py files were
deleted from the sandbox filesystem. The architecture spec (in `../docs/`) was
preserved because it was committed before the reset. To restore the actual
PyTorch implementation, the architecture code needs to be written from spec.
