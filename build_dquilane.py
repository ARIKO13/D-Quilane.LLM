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

# Smoke test
x = torch.randint(0, cfg['vocab_size'], (1, 16))
_, loss, aux = model(x, targets=x, return_aux=True)
print(f"[Smoke] loss={loss.item():.4f} (expect ~ln({cfg['vocab_size']})="
      f"{torch.log(torch.tensor(float(cfg['vocab_size']))).item():.4f})")
print(f"[Smoke] aux: {aux}")

# Minimal AION for base (9 pairs)
base_aion = {
    "halo": "Halo! Aku D'QUILANE, model Chimera-8B-Effective. Ada yang bisa aku bantu?",
    "siapa kamu": "Aku D'QUILANE dengan arsitektur Chimera-8B-Effective.",
    "berapa 1+1": "1 + 1 = 2",
    "berapa 2+2": "2 + 2 = 4",
    "berapa 5+3": "5 + 3 = 8",
    "apa itu chimera": "Chimera: 6 blok unik × 4 repeat = 24 layer efektif, 6 expert ReXMoE.",
    "apa itu d'quilane": "D'QUILANE: LLM kecil tapi efektif besar, jalan di CPU FP32.",
    "apa itu rope": "RoPE: Rotary Position Embedding, 0 parameter, ekstrapolasi long context.",
    "apa itu aion": "AION: O(1) engram bypass untuk Q&A faktual.",
}
print(f"\n[Base AION] {len(base_aion)} Q&A pairs")

# Save base (untrained)
base_path = os.path.join(OUT_DIR, 'D-QUILANE-base.gguf')
print(f"\n[Save] D-QUILANE-base.gguf (untrained, random init)...")
model.save_gguf(base_path, stoi=stoi, itos=itos, eos_id=stoi['<eos>'],
                aion_engram=base_aion, architecture='chimera_8b_effective')

# ============================================================
# Load dataset (use first 100 samples for training)
# ============================================================
print("\n" + "=" * 60)
print("Loading dataset for training (sandbox-scale, 50 samples)")
print("=" * 60)

dataset_path = '/home/z/my-project/dataset/dataset_train.parquet'
if not os.path.exists(dataset_path):
    # Try main dataset.parquet
    dataset_path = '/home/z/my-project/dataset/dataset.parquet'
if not os.path.exists(dataset_path):
    print(f"ERROR: dataset not found")
    sys.exit(1)

table = pq.read_table(dataset_path)
df = table.to_pandas()
print(f"  Loaded {len(df)} Q&A pairs from {dataset_path}")

# Take 50 samples (limit to keep training time reasonable)
data_full = list(zip(df['input'].tolist(), df['output'].tolist()))
random.seed(42)
random.shuffle(data_full)
data = data_full[:50]
print(f"  Using first 50 samples for training")

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
total_steps = 60
batch_size = 2
t0 = time.time()

# Phase 1: Warmup (1 block, 1× repeat) — 15% = 9 steps
print(f"\nPhase 1: Warmup (1 block, 1× repeat) — 9 steps")
model.cfg['n_blocks'] = 1
model.n_repeat = 1
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
for step in range(9):
    samples = random.sample(encoded, batch_size)
    xb = torch.tensor([s[0] for s in samples], dtype=torch.long)
    yb = torch.tensor([s[1] for s in samples], dtype=torch.long)
    opt.zero_grad()
    _, loss, aux = model(xb, yb, return_aux=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if step % 3 == 0:
        print(f"  p1 step {step+1}/9  loss={loss.item():.4f}  "
              f"ce={aux['ce_loss']:.4f}  lb={aux['load_balancing_loss']:.4f}")

# Phase 2: Stacking (1→2→4×) — 10% = 6 steps
print(f"\nPhase 2: Stacking (2× → 3× repeat) — 6 steps")
for rep in [2, 3]:
    model.n_repeat = rep
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)
    for step in range(3):
        samples = random.sample(encoded, batch_size)
        xb = torch.tensor([s[0] for s in samples], dtype=torch.long)
        yb = torch.tensor([s[1] for s in samples], dtype=torch.long)
        opt.zero_grad()
        _, loss, aux = model(xb, yb, return_aux=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        print(f"  p2 rep={rep} step {step+1}/3  loss={loss.item():.4f}")

# Phase 3: Expert activation — 20% = 12 steps
print(f"\nPhase 3: Expert activation (ReXMoE + load balancing) — 12 steps")
model.cfg['n_blocks'] = cfg['n_blocks']
model.n_repeat = cfg['n_repeat']
opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)
for step in range(12):
    samples = random.sample(encoded, batch_size)
    xb = torch.tensor([s[0] for s in samples], dtype=torch.long)
    yb = torch.tensor([s[1] for s in samples], dtype=torch.long)
    opt.zero_grad()
    _, loss, aux = model(xb, yb, return_aux=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if step % 3 == 0:
        print(f"  p3 step {step+1}/12  loss={loss.item():.4f}  "
              f"lb={aux['load_balancing_loss']:.4f}")

# Phase 4: Full training — 55% = 33 steps
print(f"\nPhase 4: Full training (12 eff. layers) — 33 steps")
opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)
best_loss = float('inf')
for step in range(33):
    samples = random.sample(encoded, batch_size)
    xb = torch.tensor([s[0] for s in samples], dtype=torch.long)
    yb = torch.tensor([s[1] for s in samples], dtype=torch.long)
    opt.zero_grad()
    _, loss, aux = model(xb, yb, return_aux=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if step % 6 == 0:
        print(f"  p4 step {step+1}/33  loss={loss.item():.4f}  "
              f"ce={aux['ce_loss']:.4f}  lb={aux['load_balancing_loss']:.4f}  "
              f"rz={aux['router_z_loss']:.4f}")

train_time = time.time() - t0
print(f"\n[Training done in {train_time:.1f}s]")

# Build AION engram from ALL training data (50 pairs)
print(f"\n[Build AION engram from {len(data)} training samples]")
trainer_aion = dict(base_aion)  # keep base 9 pairs
for inp, out in data:
    # Normalize key: lowercase, strip, truncate to 60 chars
    key = inp.lower().strip()[:60]
    if key and key not in trainer_aion:
        trainer_aion[key] = out
print(f"  Trainer AION: {len(trainer_aion)} Q&A pairs (base 9 + {len(trainer_aion)-9} from dataset)")

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
