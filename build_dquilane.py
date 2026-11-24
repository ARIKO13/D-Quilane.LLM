"""
Build D-QUILANE-base.gguf (untrained) and D-QUILANE-trainer.gguf (trained).

Base: chimera sandbox arch, random init, empty AION engram.
Trainer: same arch/params, trained via progressive stacking, AION engram filled
from the dataset.

Paths are relative to this file, so it runs from any checkout:
    python build_dquilane.py
Requires: torch (CPU is enough), pyarrow.
"""
import glob
import hashlib
import os
import random
import sys
import time

import torch
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from chimera_model import CONFIGS, ChimeraModel, normalize_aion_key

OUT_DIR = HERE
DATASET_DIR = os.path.join(HERE, 'dataset')
SEED = 42

# Char vocabulary (must match CONFIGS['sandbox']['vocab_size'] in chimera_model.py)
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

cfg = CONFIGS['sandbox'].copy()
cfg['vocab_size'] = len(stoi)

print("=" * 60)
print("Building D'QUILANE Chimera sandbox")
print("=" * 60)
print(f"  d_model:         {cfg['d_model']}")
print(f"  n_blocks x repeat: {cfg['n_blocks']} x {cfg['n_repeat']} = "
      f"{cfg['n_blocks']*cfg['n_repeat']} weight-reuse passes")
print(f"  n_experts:       {cfg['n_experts']} (dense top-2 router)")
print(f"  vocab:           {cfg['vocab_size']} (char-level)")

torch.manual_seed(SEED)
random.seed(SEED)
model = ChimeraModel(cfg)
n = model.count_params()
print(f"\n[Model] {n:,} unique params, {n*4/1024/1024:.2f} MB FP32")

# Original depth, restored after progressive stacking
ORIG_BLOCKS = cfg['n_blocks']
ORIG_REPEAT = cfg['n_repeat']

# Smoke test
x = torch.randint(0, cfg['vocab_size'], (1, 16))
_, loss, aux = model(x, targets=x, return_aux=True)
print(f"[Smoke] loss={loss.item():.4f} (uniform baseline ln(V)="
      f"{torch.log(torch.tensor(float(cfg['vocab_size']))).item():.4f})")
print(f"[Smoke] aux: {aux}")

# Base has NO retrieval table: pure architecture skeleton.
base_aion = {}
print(f"\n[Base AION] {len(base_aion)} pairs (EMPTY - pure architecture)")

base_path = os.path.join(OUT_DIR, 'D-QUILANE-base.gguf')
print("\n[Save] D-QUILANE-base.gguf (untrained, EMPTY AION, pure architecture)...")
model.save_gguf(base_path, stoi=stoi, itos=itos, eos_id=stoi['<eos>'],
                aion_engram=base_aion, architecture='chimera_8b_effective')

# ============================================================
# Dataset
# ============================================================
print("\n" + "=" * 60)
print("Loading dataset")
print("=" * 60)


def find_col(columns, options):
    for o in options:
        if o in columns:
            return o
    return None


def load_pairs():
    # One canonical file only: the extra per-subject files overlap it.
    paths = []
    for name in ('dataset.parquet', 'dataset_train.parquet'):
        candidate = os.path.join(DATASET_DIR, name)
        if os.path.exists(candidate):
            paths = [candidate]
            break
    if not paths:
        paths = sorted(glob.glob(os.path.join(DATASET_DIR, '*.parquet')))
    if not paths:
        print(f"ERROR: no parquet files in {DATASET_DIR}")
        sys.exit(1)
    pairs = []
    for path in paths:
        table = pq.read_table(path)
        cols = table.column_names
        c_in = find_col(cols, ('input', 'question', 'prompt', 'instruction'))
        c_out = find_col(cols, ('output', 'answer', 'response', 'completion'))
        if c_in is None or c_out is None:
            print(f"  [skip] {os.path.basename(path)}: columns {cols}")
            continue
        d = table.select([c_in, c_out]).to_pydict()
        pairs.extend(zip(d[c_in], d[c_out]))
        print(f"  {os.path.basename(path)}: {table.num_rows:,} pairs")
    return [(str(i), str(o)) for i, o in pairs if i and o]


data = load_pairs()
print(f"  Loaded {len(data):,} Q&A pairs total")

max_seq = cfg['max_seq']
pad_id = stoi['<pad>']
eos_id = stoi['<eos>']


def encode_pair(inp, out):
    """(inputs, targets) without padding.

    Targets are -100 for every prompt token, so the loss only covers the
    answer (and its closing <eos>). Padding is added per batch in make_batch,
    also with -100, so pad tokens are never trained or scored.
    """
    prompt_ids = [stoi[c] for c in inp if c in stoi]
    answer_ids = [stoi[c] for c in (' ' + out) if c in stoi]
    ids = prompt_ids + answer_ids + [eos_id]
    if len(ids) > max_seq:
        ids = ids[:max_seq]
        ids[-1] = eos_id
    if len(ids) < 2:
        return None
    xs = ids[:-1]
    targets = []
    for j in range(1, len(ids)):
        targets.append(-100 if j < len(prompt_ids) else ids[j])
    return xs, targets


encoded = [e for e in (encode_pair(i, o) for i, o in data) if e]
skipped = len(data) - len(encoded)
print(f"  Encoded {len(encoded):,} samples, max_seq={max_seq}"
      + (f", skipped {skipped}" if skipped else ""))


def make_batch(samples):
    L = max(len(xs) for xs, _ in samples)
    xb = torch.full((len(samples), L), pad_id, dtype=torch.long)
    yb = torch.full((len(samples), L), -100, dtype=torch.long)
    for i, (xs, tg) in enumerate(samples):
        xb[i, :len(xs)] = torch.tensor(xs, dtype=torch.long)
        yb[i, :len(tg)] = torch.tensor(tg, dtype=torch.long)
    return xb, yb


# ============================================================
# Progressive stacking training
# ============================================================
print("\n" + "=" * 60)
print("Progressive Stacking Training")
print("=" * 60)

device = torch.device('cpu')
batch_size = 8
total_steps = 200
steps_done = 0
t0 = time.time()
history = []


def run_phase(label, steps, lr, log_every, active_blocks, active_repeat):
    global steps_done
    model.set_active_depth(active_blocks, active_repeat)
    print(f"\n{label} - {steps} steps "
          f"(blocks={model.n_blocks_active}, repeat={model.n_repeat}, "
          f"{steps*batch_size:,} samples)")
    opt = torch.optim.AdamW(model.parameters(), lr=lr,
                            weight_decay=cfg['weight_decay'])
    first = last = None
    for step in range(steps):
        samples = random.sample(encoded, batch_size)
        xb, yb = make_batch(samples)
        opt.zero_grad()
        _, loss, aux = model(xb, yb, return_aux=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        steps_done += 1
        v = loss.item()
        if first is None:
            first = v
        last = v
        if step % log_every == 0 or step == steps - 1:
            print(f"  {label.split(':')[0]} step {step+1}/{steps}  loss={v:.4f}  "
                  f"ce={aux['ce_loss']:.4f}  lb={aux['load_balancing_loss']:.4f}  "
                  f"rz={aux['router_z_loss']:.4f}")
    history.append((label, steps, first, last))
    print(f"  -> loss {first:.3f} -> {last:.3f}")
    return last


# Phase 1: warmup at depth 1 (1 block, 1 pass)
run_phase("Phase 1: Warmup (1 block, 1x)", 30, 3e-4, 10, 1, 1)
# Phase 2: increase weight reuse at depth 1
run_phase("Phase 2a: Stacking (1 block, 2x)", 10, 1e-4, 5, 1, 2)
run_phase("Phase 2b: Stacking (1 block, 3x)", 10, 1e-4, 5, 1, 3)
# Phase 3: restore full depth
run_phase("Phase 3: Full depth (4 blocks, 3x)", 40, 5e-4, 10, ORIG_BLOCKS, ORIG_REPEAT)
# Phase 4: full training
run_phase("Phase 4: Full training (4 blocks, 3x)", 110, 1e-4, 20, ORIG_BLOCKS, ORIG_REPEAT)

assert steps_done == total_steps, (steps_done, total_steps)
train_time = time.time() - t0
print(f"\n[Training done in {train_time:.1f}s, {steps_done} steps, "
      f"{steps_done*batch_size:,} samples = "
      f"{steps_done*batch_size/len(encoded):.2f} epochs]")

# Verify the structure survived the depth mutations
print(f"\n[Verify] n_blocks_active={model.n_blocks_active} (should be {ORIG_BLOCKS})")
print(f"[Verify] n_repeat={model.n_repeat} (should be {ORIG_REPEAT})")
print(f"[Verify] len(model.blocks)={len(model.blocks)} (should be {ORIG_BLOCKS})")
print(f"[Verify] param count: {model.count_params():,} (should match base = {n:,})")
assert model.n_blocks_active == ORIG_BLOCKS, "depth not restored!"
assert model.n_repeat == ORIG_REPEAT, "repeat not restored!"
assert len(model.blocks) == ORIG_BLOCKS, "ModuleList wrong size!"
assert model.count_params() == n, f"Param count changed: {model.count_params()} vs {n}"
print("[Verify] OK: all checks passed")

# AION retrieval table: a flat prompt→answer map, NOT learned capability.
print(f"\n[Build AION engram from {len(data):,} samples]")
trainer_aion = {}
for inp, out in data:
    key = normalize_aion_key(inp)
    if key and key not in trainer_aion:
        trainer_aion[key] = out
collided = len(data) - len(trainer_aion)
print(f"  Trainer AION: {len(trainer_aion):,} pairs "
      f"({collided:,} rows collapsed into existing keys - dataset duplication)")

trainer_path = os.path.join(OUT_DIR, 'D-QUILANE-trainer.gguf')
print("\n[Save] D-QUILANE-trainer.gguf (trained, with AION engram)...")
model.save_gguf(trainer_path, stoi=stoi, itos=itos, eos_id=eos_id,
                aion_engram=trainer_aion, architecture='chimera_8b_effective')


def file_hash(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


print("\n" + "=" * 60)
print("Comparison")
print("=" * 60)
for label, p, aion in (('Base', base_path, base_aion), ('Trainer', trainer_path, trainer_aion)):
    print(f"  {label:8s} {os.path.getsize(p):,} bytes  "
          f"({os.path.getsize(p)/1024/1024:.2f} MB)  AION pairs: {len(aion):,}  "
          f"sha256: {file_hash(p)[:16]}")
print(f"\nDone. Both files at {OUT_DIR}/")
