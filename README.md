# D'QUILANE

A character-level transformer trained from scratch on a 215K-parameter budget,
with a local web playground for trying it.

**What this is:** a small, fully readable architecture sandbox with a real
train/validate/generate loop, measured validation perplexity, and a UI.
**What this is not:** a chatbot. The network learns character statistics and
short answer shapes; stored dataset answers come from a retrieval table, not
from the model. See [Honest limitations](#honest-limitations).

## Files

| File | Size | Description |
|---|---|---|
| `chimera_model.py` | 24 KB | Architecture + GGUF-shaped reader/writer |
| `build_dquilane.py` | 13 KB | Dedup + split + epoch training + evaluation |
| `playground.py` | 16 KB | Local web UI (stdlib HTTP server, no extra deps) |
| `infer_dquilane.py` | 5 KB | CLI loader + generation |
| `D-QUILANE-base.gguf` | 850 KB | Random init (seed 42), empty AION engram |
| `D-QUILANE-trainer.gguf` | 1.58 MB | 5 epochs, val ppl 2.83, AION 9,744 pairs |
| `dataset/` | 22 parquet files | 20,536 rows → 10,259 unique questions |
| `dataset_SD.zip` | 497 KB | Same dataset, zipped |
| `requirements.txt` | — | `torch`, `pyarrow` (CPU build is enough) |

## Quick start

```bash
pip install -r requirements.txt

python playground.py                      # http://127.0.0.1:8000  (try the model)
python build_dquilane.py --epochs 5 --batch 16 --threads 12   # retrain (~41 min CPU)
python infer_dquilane.py --model D-QUILANE-trainer.gguf --prompt "apa itu fotosintesis?"
python chimera_model.py                   # parameter count + forward smoke test
```

## Playground

`python playground.py` serves a single-page UI at `http://127.0.0.1:8000`:

* model picker (base = random init, trainer = trained) with params, vocab,
  depth, validation perplexity, AION table size and file size;
* temperature / top-k / answer-length sliders and an AION on/off switch;
* every reply is tagged **AION retrieval** (answer read from the stored
  question→answer table) or **network sample** (characters sampled by the
  215K-parameter model), with latency;
* turning AION off is the honest test of what the network itself can do.

Models are loaded lazily and cached; the server reads the metrics from GGUF
metadata, so retraining and reloading the page is enough to see new numbers.
Endpoints: `GET /api/models`, `POST /api/chat`
(`{model, prompt, temp, top_k, max_new, use_aion}`).

## Validation perplexity (measured)

Character-level perplexity over the held-out split, computed on **answer
tokens only** (prompt tokens and padding are `-100`, so they never enter the
loss). 512 held-out questions, 19,803 validation tokens, `ppl = exp(CE)`.

| Epoch | Train loss | Train ppl | Val loss | Val ppl |
|---|---|---|---|---|
| 0 (untrained) | — | — | 4.7405 | **114.49** |
| 1 | 1.5339 | 4.64 | 1.5630 | 4.77 |
| 2 | 1.2575 | 3.52 | 1.2857 | 3.62 |
| 3 | 1.1278 | 3.09 | 1.1657 | 3.21 |
| 4 | 1.0333 | 2.81 | 1.0775 | 2.94 |
| 5 | 0.9926 | 2.70 | 1.0414 | **2.83** |

* Best epoch: **5**, final validation perplexity **2.83** = **1.50 bits per
  character** (log2 2.83), against 114.49 (≈ uniform over the 114-token
  vocabulary) before training.
* Train (2.70) and validation (2.83) move together, so the 5 epochs did not
  overfit; the gap at the end is ~5%.
* These numbers are stored in the trainer container as `train_loss`,
  `train_perplexity`, `val_loss`, `val_perplexity`, `val_tokens`, `epochs`,
  `best_epoch`, `train_pairs`, `val_pairs`, `train_steps` and `train_minutes`,
  and the playground displays them.

**What perplexity does and does not say.** 2.83 means the model is ~40× less
surprised by the next character of an answer than a uniform guess, i.e. it
models character-level structure of Indonesian answers well. It does **not**
mean it can answer: it still cannot produce a coherent factual sentence, as
the sample below shows.

Network sample (AION disabled, prompt "jelaskan fotosintesis", temp 0.6):

```
blood in a memanintah bermenten trenciliting prepater in a
```

Same model with the AION table enabled, prompt "apa itu fotosintesis?":

```
Fotosintesis adalah proses tumbuhan membuat makanan dari cahaya matahari.
```

The second answer is retrieved from the table, not generated.

## Training setup

| Property | Value |
|---|---|
| Dataset rows read | 20,536 (all 22 parquet files) |
| Duplicates removed | 10,277 (normalised question key, first occurrence wins) |
| Unique questions | 10,259 |
| Split | 9,747 train / 512 validation (5%, seed 42) |
| Epochs | 5 full passes (610 steps/epoch, batch 16) |
| Steps | 3,050 |
| Peak LR | 3e-3, cosine decay to 3e-4, 100-step depth-1 warmup |
| Keep | best-validation weights (epoch 5) |
| Wall clock | 41.4 min on 12 CPU threads |
| AION table | 9,744 keys built from train only — validation questions are excluded |

Everything is seeded (`torch`, `random`, `random.Random(SEED)`), so
`python build_dquilane.py --epochs 5 --batch 16 --threads 12` reproduces this
run byte-for-byte on the same machine.

## Measured facts

| Property | Value |
|---|---|
| Unique parameters | 215,712 |
| FP32 weight bytes | 0.82 MB |
| Vocab | 114 char-level tokens (112 chars + `<pad>` + `<eos>`) |
| Depth | 4 blocks x 3 weight-reuse passes = 12 passes |
| Experts | 4 SwiGLU experts, dense top-2 router |
| `max_seq` | 80 characters |
| base sha256 (16) | `33c9a287e59e5c87` |
| trainer sha256 (16) | `f09a8448d4c7f5ca` |

The trainer file is 1.58 MB: 0.87 MB of weights plus ~0.7 MB of JSON AION
table. The table, not the weights, is where stored answers come from.

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
| `n_heads / n_kv` | 4 / 1 | GQA, 4× KV reduction |
| `sliding_window` | 32 | odd-index blocks only |

Implemented: RMSNorm (pre-norm), RoPE, low-rank GQA, SwiGLU bottleneck, weight
tying, LayerScale, shared experts with load-balancing + router Z-loss,
EntroCut (in `generate`), progressive depth (`set_active_depth`), and a
GGUF-shaped container with a matching reader.

Not implemented: gradient checkpointing, KV cache, real sparse MoE dispatch,
any "AION adaptive" behaviour inside the model.

## Honest limitations

1. **Retrieval is not understanding.** 9,744 answers come from a lookup table
   keyed by the normalised question; those answers are correct only if the
   question was in the training data.
2. **The network still cannot answer.** Validation perplexity 2.83 shows it
   models characters, not facts. Free questions return word-shaped noise.
3. **The files are not loadable by llama.cpp or ollama.** The container copies
   the GGUF header layout (magic `GGUF`, version 3, 32-byte alignment) but the
   `chimera_8b_effective` architecture does not exist there. Use
   `infer_dquilane.py` or the playground.
4. **"8B effective" is a label.** `n_repeat` reuses the same weights; it gives
   12 sequential passes, not 8 billion parameters.
5. **No sparse MoE.** Every expert runs for every token, so the router adds
   parameters and compute without any speed-up.
6. **Context is 80 characters**, and there is no KV cache: generation is
   `max_new` full forward passes.
7. **Perplexity is per character, not per word or per token of a real
   tokenizer.** It is not comparable to BPE-model perplexity numbers.
8. **Dataset provenance is undocumented.** 22 parquet files, heavily
   duplicated before dedup; treat it as research data, not a redistribution-safe
   corpus.

## Target config: `chimera_1gb`

`CONFIGS['chimera_1gb']` (d_model 2048, 32K BPE vocab, ~180M params, ~688 MB
FP32) is a **spec only** — it has never been trained here. It needs a 12 GB+
GPU, weeks of compute and a corpus far larger than this one.

## License

MIT.
