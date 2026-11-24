# D'QUILANE

A character-level transformer trained from scratch on a 215K-parameter budget.
Small file, FP32, runs on CPU in milliseconds.

**What this is:** a small, fully readable architecture sandbox plus a working
train/load/generate loop. It trains in ~77 s on a laptop CPU.
**What this is not:** a usable chatbot. Nothing here can answer open questions;
see [Honest limitations](#honest-limitations).

## Files

| File | Size | Description |
|---|---|---|
| `chimera_model.py` | 18 KB | Architecture + GGUF-shaped reader/writer |
| `build_dquilane.py` | 12 KB | Trains and rebuilds both containers |
| `infer_dquilane.py` | 5 KB | CPU loader + generation (the way to run these files) |
| `D-QUILANE-base.gguf` | 850 KB | Random init (seed 42), empty AION engram |
| `D-QUILANE-trainer.gguf` | 1.39 MB | Trained 200 steps, AION engram with 7,266 pairs |
| `dataset/` | 22 parquet files | 20,536 rows / 10,268 unique questions |
| `dataset_SD.zip` | 497 KB | Same dataset, zipped |
| `requirements.txt` | — | `torch`, `pyarrow` (CPU build is enough) |

## Quick start

```bash
pip install -r requirements.txt

python chimera_model.py        # parameter count + forward smoke test
python build_dquilane.py       # retrain and rewrite both .gguf files (~80 s CPU)
python infer_dquilane.py --model D-QUILANE-trainer.gguf --prompt "4 - 4 + 7 = ?"
python infer_dquilane.py --model D-QUILANE-trainer.gguf          # interactive
```

## Measured facts (reproduced from this repo, seed 42)

| Property | Value |
|---|---|
| Unique parameters | **215,712** |
| FP32 weight bytes | 0.82 MB |
| Vocab | **114** char-level tokens (112 chars + `<pad>` + `<eos>`) |
| Depth | 4 blocks x 3 weight-reuse passes = 12 passes |
| Experts | 4 SwiGLU experts, dense top-2 router |
| `max_seq` | 80 characters |
| Training | 200 steps, batch 8 = 1,600 samples = **0.22 epochs** |
| Train time | 76.9 s on one CPU core set |
| Final train CE | 2.68 (uniform baseline ln(114) = 4.74) |

### Progressive stacking run (this is what the log prints)

| Phase | Active depth | Steps | Loss |
|---|---|---|---|
| 1 | 1 block, 1 pass | 30 | 4.761 → 3.875 |
| 2a | 1 block, 2 passes | 10 | 3.703 → 3.579 |
| 2b | 1 block, 3 passes | 10 | 3.570 → 3.523 |
| 3 | 4 blocks, 3 passes | 40 | 3.558 → 3.007 |
| 4 | 4 blocks, 3 passes | 110 | 2.872 → 2.684 |

Loss only counts answer characters: prompt tokens and padding are masked with
`-100`.

## Architecture

`chimera_8b_effective` is a *name*, not a size: it is 0.2M parameters, not 8B.

| Property | Value | Note |
|---|---|---|
| `d_model` | 96 | |
| `vocab_size` | 114 | char-level; no BPE |
| `n_blocks` | 4 | weights reused `n_repeat` = 3 times |
| `n_experts` | 4 | top-2 router, **all 4 experts computed per token** |
| `attn_rank` | 32 | low-rank QKV |
| `ffn_hidden` | 48 | SwiGLU bottleneck 96→48→96 |
| `n_heads / n_kv` | 4 / 1 | GQA, 4x KV reduction |
| `sliding_window` | 32 | odd-index blocks only |

Implemented: RMSNorm (pre-norm), RoPE, low-rank GQA, SwiGLU bottleneck, weight
tying, LayerScale, shared experts with load-balancing + router Z-loss,
EntroCut (in `generate`), and a GGUF-shaped container.

Not implemented, despite older descriptions: gradient checkpointing, KV cache,
real sparse MoE dispatch, any "AION adaptive" behaviour inside the model
(AION is plain key→answer retrieval in `infer_dquilane.py`).

## Base vs trainer

| | base | trainer |
|---|---|---|
| Weights | random init (seed 42) | 200 training steps |
| AION engram | empty | 7,266 prompt→answer pairs |
| Params | 215,712 | 215,712 (identical) |
| Size | 850 KB | 1.39 MB |
| sha256 (16) | `8f161513afea3b56` | `052ddcd55e40b09f` |

The 0.54 MB difference is the JSON AION table in the metadata, not extra
weights. When the prompt hits that table, `infer_dquilane.py` prints
`[aion] retrieved` and returns the stored answer — that is retrieval, not
model output. Force the network with `--no-aion`.

Sample network output (`--no-aion`, prompt "jelaskan fotosintesis"):

```
s a pa then buuaganasa te  pemal  bageengi igi di den d b bae aa puenndnn
tasigah anilea d  st e ie t e an ang thtilas b
```

That is the real capability level of the 215K network: character statistics,
not knowledge.

## Dataset

* `dataset/dataset.parquet` — 7,274 rows; this is the file the training script
  loads. `dataset_train.parquet` (8,214 rows) is only a fallback.
* All 22 parquet files together: 20,536 rows, but only **10,268 unique
  questions** — roughly half of the shipped dataset is duplicated across files.
* Same inputs collapse further to 7,266 keys after the AION normalisation, so
  the table holds 7,266 distinct prompts.
* Provenance/licensing of the parquet content is not documented in this repo;
  treat it as research data, not a redistribution-safe corpus.

## Honest limitations

1. **Not a usable LLM.** 215K parameters over 114 characters, trained for
   0.22 epochs, cannot answer questions. The convincing answers come from the
   AION lookup table.
2. **The files are not loadable by llama.cpp or ollama.** The container copies
   the GGUF header layout (magic `GGUF`, version 3, 32-byte alignment) but the
   `chimera_8b_effective` architecture does not exist there. Use
   `infer_dquilane.py`.
3. **"8B effective" is a label.** `n_repeat` reuses the same weights; it gives
   12 sequential passes, not 8 billion parameters. `compute_multiplier()`
   reports the reuse factor; it is deliberately not called a parameter count.
4. **No sparse MoE.** Every expert runs for every token, so the router adds
   parameters and compute without any speed-up.
5. **Partial training.** 1,600 samples out of 7,274 means the model never sees
   most of the dataset once.
6. **Previous numbers were not reproducible.** The earlier 1.64 loss came from
   a run where padded tokens were scored as targets (the model mostly learned
   to emit `<pad>`); `loss` and the reported `215,808 / 115 vocab` also came
   from an older tokenizer revision than the one in this repo. Current numbers
   above are produced by `python build_dquilane.py` on this code.

## Target config: `chimera_1gb`

`CONFIGS['chimera_1gb']` (d_model 2048, 32K BPE vocab, ~180M params, ~688 MB
FP32) is a **spec only** — it has never been trained here. It needs a 12 GB+
GPU, weeks of compute and a corpus far larger than this one. Nothing in this
repo was produced with it.

## License

MIT.
