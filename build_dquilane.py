"""
Build D-QUILANE-base.gguf (untrained) and D-QUILANE-trainer.gguf (trained).

Base: chimera sandbox arch, random init, minimal AION (9 pairs).
Trainer: same arch, trained via progressive stacking on dataset, larger AION.

Both use chimera_8b_effective architecture.
"""
import os
import sys
import json
import random
import time
import torch
import pyarrow.parquet as pq

sys.path.insert(0, '/home/z/my-project')
from chimera_model import CONFIGS, ChimeraModel

OUT_DIR = '/home/z/my-project'

# Char vocabulary
vocab_chars = (
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    " .,!?;:'\"-_/()[]{}@#$%&*+=<>~`\n"
    "éèêëàâçïîôùûüñáéíóúàèìòù"
)
stoi = {c: i for i, c in enumerate(vocab_chars)}
stoi['<pad>'] = len(stoi)
stoi['<eos>'] = len(stoi)
itos = {i: c for c, i in stoi.items()}

cfg = CONFIGS['sandbox'].copy()
cfg['vocab_size'] = len(stoi)

print("=" * 60)
print("Building D-QUILANE Chimera sandbox (218K params, 0.84 MB FP32)")
print("=" * 60)
print(f"  d_model:        {cfg['d_model']}")
print(f"  n_blocks×repeat: {cfg['n_blocks']}×{cfg['n_repeat']} = "
      f"{cfg['n_blocks']*cfg['n_repeat']} eff. layers")
print(f"  n_experts:      {cfg['n_experts']} (ReXMoE shared)")
print(f"  vocab:          {cfg['vocab_size']}")

torch.manual_seed(42)
model = ChimeraModel(cfg)
n = model.count_params()
print(f"\n[Model] {n:,} unique params, {n*4/1024/1024:.2f} MB FP32")

# Save original cfg backup (Phase 1 mutates model.cfg, need this to restore)
ORIG_CFG = dict(cfg)  # shallow copy of cfg dict
ORIG_N_REPEAT = model.n_repeat
print(f"[Backup] original cfg: n_blocks={ORIG_CFG['n_blocks']}, n_repeat={ORIG_N_REPEAT}")

# Smoke test
x = torch.randint(0, cfg['vocab_size'], (1, 16))
_, loss, aux = model(x, targets=x, return_aux=True)
print(f"[Smoke] loss={loss.item():.4f} (expect ~ln({cfg['vocab_size']})="
      f"{torch.log(torch.tensor(float(cfg['vocab_size']))).item():.4f})")
print(f"[Smoke] aux: {aux}")

# Minimal AION for base (EMPTY — base is pure architecture, no data)
base_aion = {}  # empty — base has no Q&A cache, just architecture skeleton
print(f"\n[Base AION] {len(base_aion)} pairs (EMPTY — pure architecture)")

# Save base (untrained, empty AION)
base_path = os.path.join(OUT_DIR, 'D-QUILANE-base.gguf')
print(f"\n[Save] D-QUILANE-base.gguf (untrained, EMPTY AION, pure architecture)...")
model.save_gguf(base_path, stoi=stoi, itos=itos, eos_id=stoi['<eos>'],
                aion_engram=base_aion, architecture='chimera_8b_effective')

# ============================================================
# Load FULL dataset (10,268 Q&A pairs — SD worldwide curriculum)
# ============================================================
print("\n" + "=" * 60)
print("Loading FULL SD dataset (10,268 Q&A pairs)")
print("=" * 60)

dataset_path = '/home/z/my-project/dataset/dataset.parquet'
if not os.path.exists(dataset_path):
    dataset_path = '/home/z/my-project/dataset/dataset_train.parquet'
if not os.path.exists(dataset_path):
    print(f"ERROR: dataset not found")
    sys.exit(1)

table = pq.read_table(dataset_path)
df = table.to_pandas()
print(f"  Loaded {len(df)} Q&A pairs from {dataset_path}")

# Use ALL samples for training (full SD curriculum)
data_full = list(zip(df['input'].tolist(), df['output'].tolist()))
data = data_full  # ALL 10,268 pairs
print(f"  Using ALL {len(data)} samples for training")

# Encode
max_seq = cfg['max_seq']
pad_id = stoi['<pad>']
eos_id = stoi['<eos>']

def encode_pair(inp, out):
    full = inp + ' ' + out
    ids = [stoi[c] for c in full if c in stoi]
    ids.append(eos_id)
    if len(ids) > max_seq:
        ids = ids[:max_seq]
        ids[-1] = eos_id
    else:
        while len(ids) < max_seq:
            ids.append(pad_id)
    prompt_len = sum(1 for c in inp if c in stoi)
    targets = []
    for i in range(max_seq - 1):
        if i < prompt_len - 1:
            targets.append(-100)
        else:
            targets.append(ids[i + 1] if i + 1 < max_seq else pad_id)
    while len(targets) < max_seq - 1:
        targets.append(-100)
    return ids[:-1][:max_seq - 1], targets

encoded = [encode_pair(inp, out) for inp, out in data]
print(f"  Encoded {len(encoded)} samples, max_seq={max_seq}")

# ============================================================
# Progressive stacking training (sandbox, 60 steps total)
# ============================================================
print("\n" + "=" * 60)
print("Progressive Stacking Training")
print("=" * 60)

device = torch.device('cpu')
total_steps = 200  # bigger training run for full dataset
batch_size = 8     # larger batch since we have 10K samples
t0 = time.time()

# Phase 1: Warmup (1 block, 1× repeat) — 15% = 30 steps
print(f"\nPhase 1: Warmup (1 block, 1× repeat) — 30 steps on {len(encoded)} samples")
model.cfg['n_blocks'] = 1
model.n_repeat = 1
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
for step in range(30):
    samples = random.sample(encoded, batch_size)
    xb = torch.tensor([s[0] for s in samples], dtype=torch.long)
    yb = torch.tensor([s[1] for s in samples], dtype=torch.long)
    opt.zero_grad()
    _, loss, aux = model(xb, yb, return_aux=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if step % 10 == 0:
        print(f"  p1 step {step+1}/30  loss={loss.item():.4f}  "
              f"ce={aux['ce_loss']:.4f}  lb={aux['load_balancing_loss']:.4f}")

# Phase 2: Stacking (1→2→3×) — 10% = 20 steps
print(f"\nPhase 2: Stacking (2× → 3× repeat) — 20 steps")
for rep in [2, 3]:
    model.n_repeat = rep
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)
    for step in range(10):
        samples = random.sample(encoded, batch_size)
        xb = torch.tensor([s[0] for s in samples], dtype=torch.long)
        yb = torch.tensor([s[1] for s in samples], dtype=torch.long)
        opt.zero_grad()
        _, loss, aux = model(xb, yb, return_aux=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 5 == 0:
            print(f"  p2 rep={rep} step {step+1}/10  loss={loss.item():.4f}")

# Phase 3: Expert activation — 20% = 40 steps
print(f"\nPhase 3: Expert activation (ReXMoE + load balancing) — 40 steps")
# Restore from backup (model.cfg was mutated by Phase 1)
model.cfg['n_blocks'] = ORIG_CFG['n_blocks']
model.cfg['n_repeat'] = ORIG_CFG['n_repeat']
model.n_repeat = ORIG_N_REPEAT
print(f"  Restored: n_blocks={model.cfg['n_blocks']}, n_repeat={model.n_repeat}")
print(f"  Current param count: {model.count_params():,}")
opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)
for step in range(40):
    samples = random.sample(encoded, batch_size)
    xb = torch.tensor([s[0] for s in samples], dtype=torch.long)
    yb = torch.tensor([s[1] for s in samples], dtype=torch.long)
    opt.zero_grad()
    _, loss, aux = model(xb, yb, return_aux=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if step % 10 == 0:
        print(f"  p3 step {step+1}/40  loss={loss.item():.4f}  "
              f"lb={aux['load_balancing_loss']:.4f}")

# Phase 4: Full training — 55% = 110 steps
print(f"\nPhase 4: Full training ({model.cfg['n_blocks']*model.n_repeat} eff. layers) — 110 steps")
opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)
for step in range(110):
    samples = random.sample(encoded, batch_size)
    xb = torch.tensor([s[0] for s in samples], dtype=torch.long)
    yb = torch.tensor([s[1] for s in samples], dtype=torch.long)
    opt.zero_grad()
    _, loss, aux = model(xb, yb, return_aux=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if step % 20 == 0:
        print(f"  p4 step {step+1}/110  loss={loss.item():.4f}  "
              f"ce={aux['ce_loss']:.4f}  lb={aux['load_balancing_loss']:.4f}  "
              f"rz={aux['router_z_loss']:.4f}")

train_time = time.time() - t0
print(f"\n[Training done in {train_time:.1f}s]")

# Verify model structure intact after training
print(f"\n[Verify] model.cfg['n_blocks']={model.cfg['n_blocks']} (should be {ORIG_CFG['n_blocks']})")
print(f"[Verify] model.n_repeat={model.n_repeat} (should be {ORIG_N_REPEAT})")
print(f"[Verify] len(model.blocks)={len(model.blocks)} (ModuleList should have {ORIG_CFG['n_blocks']} blocks)")
print(f"[Verify] param count: {model.count_params():,} (should match base = {n:,})")
assert model.cfg['n_blocks'] == ORIG_CFG['n_blocks'], "n_blocks not restored!"
assert len(model.blocks) == ORIG_CFG['n_blocks'], "ModuleList wrong size!"
assert model.count_params() == n, f"Param count changed: {model.count_params()} vs {n}"
print(f"[Verify] ✓ All checks passed")

# Build AION engram from ALL 10,268 training samples (full SD curriculum)
print(f"\n[Build AION engram from ALL {len(data)} SD curriculum samples]")
trainer_aion = {}  # start empty — trainer has ALL 10K Q&A from dataset, no manual base
for inp, out in data:
    # Normalize key: lowercase, strip, truncate to 60 chars
    key = inp.lower().strip()[:60]
    if key and key not in trainer_aion:
        trainer_aion[key] = out
print(f"  Trainer AION: {len(trainer_aion)} Q&A pairs (full SD curriculum)")

# Save trainer
trainer_path = os.path.join(OUT_DIR, 'D-QUILANE-trainer.gguf')
print(f"\n[Save] D-QUILANE-trainer.gguf (trained, with AION engram)...")
model.save_gguf(trainer_path, stoi=stoi, itos=itos, eos_id=eos_id,
                aion_engram=trainer_aion, architecture='chimera_8b_effective')

# Compare
print(f"\n" + "=" * 60)
print(f"Comparison")
print(f"=" * 60)
base_size = os.path.getsize(base_path)
trainer_size = os.path.getsize(trainer_path)
print(f"  Base:    {base_size:,} bytes ({base_size/1024/1024:.2f} MB)")
print(f"  Trainer: {trainer_size:,} bytes ({trainer_size/1024/1024:.2f} MB)")
print(f"  Base AION pairs:    {len(base_aion)}")
print(f"  Trainer AION pairs: {len(trainer_aion)}")
print(f"  Same file content? {base_size == trainer_size}")
# Hash check
import hashlib
def file_hash(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        while True:
            chunk = f.read(8192)
            if not chunk: break
            h.update(chunk)
    return h.hexdigest()[:16]
print(f"  Base SHA256:    {file_hash(base_path)}")
print(f"  Trainer SHA256: {file_hash(trainer_path)}")
print(f"\nDone. Both files at {OUT_DIR}/")
