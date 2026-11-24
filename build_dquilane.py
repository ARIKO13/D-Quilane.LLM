"""
Build D-QUILANE-base.gguf (untrained) and D-QUILANE-trainer.gguf (trained).

Pipeline:
  1. Load every parquet file in dataset/, keep one row per normalised question.
  2. Deterministic 95/5 train/validation split.
  3. Train full epochs (default 5) at full depth, with a short depth-1 warmup,
     cosine LR decay, and validation perplexity after every epoch.
  4. Keep the best-validation weights, build the AION table from TRAIN ONLY,
     and write both containers with the measured numbers in metadata.

    python build_dquilane.py --epochs 5 --batch 16

Requires: torch (CPU is enough), pyarrow.
"""
import argparse
import copy
import glob
import hashlib
import math
import os
import random
import sys
import time

import torch
import torch.nn.functional as F
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from chimera_model import CONFIGS, ChimeraModel, normalize_aion_key

OUT_DIR = HERE
DATASET_DIR = os.path.join(HERE, 'dataset')
SEED = 42
VAL_FRACTION = 0.05
WARMUP_STEPS = 100

# Char vocabulary (must match CONFIGS['sandbox']['vocab_size'])
vocab_chars = (
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    " .,!?;:'\"-_/()[]{}@#$%&*+=<>~`\n"
    "éèêëàâçïôùûüñáéíóúàèìòù"
)
stoi = {c: i for i, c in enumerate(vocab_chars)}
stoi['<pad>'] = len(stoi)
stoi['<eos>'] = len(stoi)
itos = {i: c for c, i in stoi.items()}

pad_id = stoi['<pad>']
eos_id = stoi['<eos>']


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--lr', type=float, default=3e-3)
    ap.add_argument('--val-fraction', type=float, default=VAL_FRACTION)
    ap.add_argument('--threads', type=int, default=0)
    return ap.parse_args()


def find_col(columns, options):
    for o in options:
        if o in columns:
            return o
    return None


def load_dedup_pairs():
    """One row per normalised question, first occurrence wins."""
    paths = sorted(glob.glob(os.path.join(DATASET_DIR, '*.parquet')))
    if not paths:
        print(f"ERROR: no parquet files in {DATASET_DIR}")
        sys.exit(1)
    rows = []
    files = 0
    for path in paths:
        table = pq.read_table(path)
        cols = table.column_names
        c_in = find_col(cols, ('input', 'question', 'prompt', 'instruction'))
        c_out = find_col(cols, ('output', 'answer', 'response', 'completion'))
        if c_in is None or c_out is None:
            print(f"  [skip] {os.path.basename(path)}: columns {cols}")
            continue
        d = table.select([c_in, c_out]).to_pydict()
        rows.extend(zip(d[c_in], d[c_out]))
        files += 1
    print(f"  Read {files} parquet files, {len(rows):,} rows")

    seen = set()
    pairs = []
    for inp, out in rows:
        if not inp or not out:
            continue
        key = normalize_aion_key(str(inp), max_len=10_000)
        if key in seen:
            continue
        seen.add(key)
        pairs.append((str(inp), str(out)))
    print(f"  Deduplicated: {len(rows) - len(pairs):,} rows removed, "
          f"{len(pairs):,} unique questions left")
    return pairs


def encode_pair(inp, out, max_seq):
    """(inputs, targets); -100 on prompt tokens so only answers are scored."""
    prompt_ids = [stoi[c] for c in inp if c in stoi]
    answer_ids = [stoi[c] for c in (' ' + out) if c in stoi]
    ids = (prompt_ids + answer_ids + [eos_id])[:max_seq]
    if len(ids) < 2:
        return None
    if ids[-1] != eos_id:
        ids[-1] = eos_id
    xs = ids[:-1]
    targets = [-100 if j < len(prompt_ids) else ids[j] for j in range(1, len(ids))]
    return xs, targets


def make_batch(samples):
    L = max(len(xs) for xs, _ in samples)
    xb = torch.full((len(samples), L), pad_id, dtype=torch.long)
    yb = torch.full((len(samples), L), -100, dtype=torch.long)
    for i, (xs, tg) in enumerate(samples):
        xb[i, :len(xs)] = torch.tensor(xs, dtype=torch.long)
        yb[i, :len(tg)] = torch.tensor(tg, dtype=torch.long)
    return xb, yb


@torch.no_grad()
def evaluate(model, data, batch_size=32):
    """Token-weighted CE over answer tokens -> (loss, perplexity, tokens)."""
    model.eval()
    total_ce = 0.0
    total_tok = 0
    for i in range(0, len(data), batch_size):
        chunk = data[i:i + batch_size]
        xb, yb = make_batch(chunk)
        logits, _ = model(xb)
        ce = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1),
                             ignore_index=-100, reduction='sum')
        n = int((yb != -100).sum())
        total_ce += float(ce)
        total_tok += n
    loss = total_ce / max(total_tok, 1)
    return loss, math.exp(min(loss, 20)), total_tok


def lr_at(step, total_steps, peak, warmup):
    if step < warmup:
        return peak * (step + 1) / max(warmup, 1)
    p = (step - warmup) / max(total_steps - warmup, 1)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))


def file_hash(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    args = parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    cfg = CONFIGS['sandbox'].copy()
    cfg['vocab_size'] = len(stoi)

    print("=" * 64)
    print("D'QUILANE training")
    print("=" * 64)
    print(f"  d_model          {cfg['d_model']}")
    print(f"  depth            {cfg['n_blocks']} blocks x {cfg['n_repeat']} passes")
    print(f"  experts          {cfg['n_experts']} (dense top-2)")
    print(f"  vocab            {cfg['vocab_size']} char-level")
    print(f"  max_seq          {cfg['max_seq']}")
    print(f"  epochs           {args.epochs}, batch {args.batch}, lr {args.lr}")
    print(f"  threads          {torch.get_num_threads()}")

    torch.manual_seed(SEED)
    random.seed(SEED)
    rng = random.Random(SEED)

    model = ChimeraModel(cfg)
    n_params = model.count_params()
    print(f"\n[Model] {n_params:,} unique params, {n_params*4/1024/1024:.2f} MB FP32")

    ORIG_BLOCKS, ORIG_REPEAT = cfg['n_blocks'], cfg['n_repeat']

    # ---------------------------------------------------------------- data
    print("\n" + "=" * 64)
    print("Dataset")
    print("=" * 64)
    pairs = load_dedup_pairs()
    rng.shuffle(pairs)
    n_val = max(1, int(len(pairs) * args.val_fraction))
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]
    print(f"  Split: {len(train_pairs):,} train / {len(val_pairs):,} validation "
          f"({args.val_fraction*100:.0f}%)")

    max_seq = cfg['max_seq']
    train_data = [d for d in (encode_pair(i, o, max_seq) for i, o in train_pairs) if d]
    val_data = [d for d in (encode_pair(i, o, max_seq) for i, o in val_pairs) if d]
    print(f"  Encoded: {len(train_data):,} train / {len(val_data):,} validation samples")
    lens = [len(x) for x, _ in train_data]
    print(f"  Length: mean {sum(lens)/len(lens):.1f}, max {max(lens)} chars")

    steps_per_epoch = math.ceil(len(train_data) / args.batch)
    total_steps = steps_per_epoch * args.epochs
    print(f"  Steps: {steps_per_epoch:,}/epoch x {args.epochs} = {total_steps:,} "
          f"({total_steps*args.batch/len(train_data):.2f} epochs worth of samples)")

    # ---------------------------------------------------------------- base
    base_aion = {}
    base_path = os.path.join(OUT_DIR, 'D-QUILANE-base.gguf')
    print("\n[Save] D-QUILANE-base.gguf (random init, empty AION)")
    model.save_gguf(base_path, stoi=stoi, itos=itos, eos_id=eos_id,
                    aion_engram=base_aion, architecture='chimera_8b_effective',
                    extra_meta={'trained': 'false', 'val_perplexity': 'n/a'})

    # ------------------------------------------------------------- training
    print("\n" + "=" * 64)
    print(f"Training {args.epochs} epochs")
    print("=" * 64)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=cfg['weight_decay'])
    t0 = time.time()
    history = []
    best = {'ppl': float('inf'), 'epoch': 0, 'state': None}
    step = 0

    val_loss0, val_ppl0, val_tok = evaluate(model, val_data)
    print(f"[Epoch 0] validation before training: loss {val_loss0:.4f}, "
          f"ppl {val_ppl0:.2f} ({val_tok:,} tokens)")

    for epoch in range(1, args.epochs + 1):
        model.train()
        order = list(range(len(train_data)))
        rng.shuffle(order)
        run_loss, run_n = 0.0, 0
        epoch_t0 = time.time()
        for start in range(0, len(order), args.batch):
            idxs = order[start:start + args.batch]
            samples = [train_data[i] for i in idxs]
            xb, yb = make_batch(samples)
            lr = lr_at(step, total_steps, args.lr, WARMUP_STEPS)
            for g in opt.param_groups:
                g['lr'] = lr
            # Short warmup at depth 1, then the full 4x3 depth.
            model.set_active_depth(1, 1) if step < WARMUP_STEPS else \
                model.set_active_depth(ORIG_BLOCKS, ORIG_REPEAT)
            opt.zero_grad()
            _, loss, _ = model(xb, yb, return_aux=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            run_loss += loss.item()
            run_n += 1
            step += 1
            if step % 200 == 0 or step == total_steps:
                done = (step - (epoch - 1) * steps_per_epoch) / steps_per_epoch
                print(f"  epoch {epoch} {done*100:5.1f}%  step {step:,}/{total_steps:,}  "
                      f"loss {run_loss/run_n:.4f}  lr {lr:.2e}  "
                      f"{time.time()-epoch_t0:.0f}s")
                run_loss, run_n = 0.0, 0

        train_loss, train_ppl, train_tok = evaluate(model, train_data)
        val_loss, val_ppl, val_tok = evaluate(model, val_data)
        history.append((epoch, train_loss, train_ppl, val_loss, val_ppl))
        print(f"[Epoch {epoch}] train loss {train_loss:.4f} ppl {train_ppl:.2f} | "
              f"val loss {val_loss:.4f} ppl {val_ppl:.2f} | {time.time()-epoch_t0:.0f}s")
        if val_ppl < best['ppl']:
            best = {'ppl': val_ppl, 'epoch': epoch,
                    'state': copy.deepcopy(model.state_dict())}
            print(f"          new best validation ppl {val_ppl:.2f}")

    train_time = time.time() - t0
    model.load_state_dict(best['state'])
    model.eval()
    train_loss, train_ppl, train_tok = evaluate(model, train_data)
    val_loss, val_ppl, val_tok = evaluate(model, val_data)
    print(f"\n[Training done in {train_time/60:.1f} min, {total_steps:,} steps]")
    print(f"[Best epoch {best['epoch']}] train ppl {train_ppl:.2f} | "
          f"val ppl {val_ppl:.2f}")

    # -------------------------------------------------------------- AION
    print(f"\n[Build AION engram from {len(train_pairs):,} train questions only]")
    trainer_aion = {}
    for inp, out in train_pairs:
        key = normalize_aion_key(inp)
        if key and key not in trainer_aion:
            trainer_aion[key] = out
    print(f"  AION keys: {len(trainer_aion):,} (validation questions excluded)")

    assert model.count_params() == n_params
    trainer_path = os.path.join(OUT_DIR, 'D-QUILANE-trainer.gguf')
    print("\n[Save] D-QUILANE-trainer.gguf")
    model.save_gguf(
        trainer_path, stoi=stoi, itos=itos, eos_id=eos_id,
        aion_engram=trainer_aion, architecture='chimera_8b_effective',
        extra_meta={
            'trained': 'true',
            'train_pairs': str(len(train_pairs)),
            'val_pairs': str(len(val_pairs)),
            'epochs': str(args.epochs),
            'batch_size': str(args.batch),
            'train_steps': str(total_steps),
            'best_epoch': str(best['epoch']),
            'train_loss': f'{train_loss:.4f}',
            'train_perplexity': f'{train_ppl:.2f}',
            'val_loss': f'{val_loss:.4f}',
            'val_perplexity': f'{val_ppl:.2f}',
            'val_tokens': str(val_tok),
            'train_minutes': f'{train_time/60:.1f}',
        })

    print("\n" + "=" * 64)
    print("Summary")
    print("=" * 64)
    print(f"  epoch  train_loss  train_ppl  val_loss  val_ppl")
    for e, tl, tp, vl, vp in history:
        print(f"  {e:5d}  {tl:10.4f}  {tp:9.2f}  {vl:8.4f}  {vp:7.2f}")
    for label, p, aion in (('base', base_path, base_aion),
                           ('trainer', trainer_path, trainer_aion)):
        print(f"  {label:8s} {os.path.getsize(p):,} bytes  AION {len(aion):,}  "
              f"sha256 {file_hash(p)[:16]}")
    print("\nDone.")


if __name__ == '__main__':
    main()
