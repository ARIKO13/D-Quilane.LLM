"""
D'QUILANE — Local Tester (untuk laptop)
======================================
Cara pakai:
  1. Download model dari Kaggle (output/dquilane_balita.pt)
  2. Taruh file .pt di folder yang sama dengan script ini
  3. Install: pip install torch
  4. Run:
     python3 dquilane_local_test.py                 # mode prompt default
     python3 dquilane_local_test.py "Indonesia"     # mode prompt custom
     python3 dquilane_local_test.py interactive     # mode interactive
"""
import sys
import os
import math
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    print("[ERROR] PyTorch belum terinstall. Jalankan: pip install torch")
    sys.exit(1)

# auto-detect device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
if DEVICE == 'cpu' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    DEVICE = 'mps'
print(f"Device: {DEVICE}")

# ---------- CARI MODEL ----------
MODEL_PATH = Path(__file__).parent / "dquilane_balita.pt"
if not MODEL_PATH.exists():
    # coba lokasi alternatif
    alt_paths = [
        Path("dquilane_balita.pt"),
        Path("output/dquilane_balita.pt"),
        Path("download/dquilane_balita.pt"),
    ]
    for p in alt_paths:
        if p.exists():
            MODEL_PATH = p
            break
    else:
        print(f"[ERROR] File model gak ketemu: {MODEL_PATH}")
        print("Letakkan dquilane_balita.pt di folder yang sama dengan script ini.")
        sys.exit(1)

print(f"Loading model: {MODEL_PATH}")
ckpt = torch.load(MODEL_PATH, map_location=str(DEVICE), weights_only=False)
CONFIG = ckpt['config']
stoi = ckpt['stoi']
itos = {int(k): v for k, v in ckpt['itos'].items()}


def encode(s):
    return [stoi[c] for c in s if c in stoi]


def decode(ids):
    return ''.join(itos.get(int(i), '?') for i in ids)


# ---------- MODEL CLASS (harus sama dengan training) ----------
class Attention(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        assert d % h == 0
        self.h = h
        self.dh = d // h
        self.qkv = nn.Linear(d, 3*d, bias=False)
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.h, self.dh)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh)
        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        scores = scores.masked_fill(~mask, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.out(out)


class FFN(nn.Module):
    def __init__(self, d, d_ff):
        super().__init__()
        self.gate = nn.Linear(d, d_ff, bias=False)
        self.val = nn.Linear(d, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.val(x))


class Block(nn.Module):
    def __init__(self, d, h, d_ff):
        super().__init__()
        self.norm1 = nn.RMSNorm(d)
        self.attn = Attention(d, h)
        self.norm2 = nn.RMSNorm(d)
        self.ffn = FFN(d, d_ff)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class DQuilane(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg['vocab_size'], cfg['d_model'])
        self.pos_emb = nn.Embedding(cfg['max_seq'], cfg['d_model'])
        self.block = Block(cfg['d_model'], cfg['n_heads'], cfg['d_ff'])
        self.norm_f = nn.RMSNorm(cfg['d_model'])
        self.head = nn.Linear(cfg['d_model'], cfg['vocab_size'], bias=False)
        self.head.weight = self.tok_emb.weight

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for _ in range(self.cfg['n_layers']):
            x = self.block(x)
        x = self.norm_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.cfg['vocab_size']),
                                   targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new=50, temp=0.8, top_k=10):
        self.eval()
        for _ in range(max_new):
            x = idx if idx.size(1) <= self.cfg['max_seq'] else idx[:, -self.cfg['max_seq']:]
            logits, _ = self(x)
            logits = logits[:, -1, :] / temp
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx


# ---------- LOAD MODEL ----------
model = DQuilane(CONFIG).to(DEVICE)
model.load_state_dict(ckpt['state_dict'])
model.eval()
n_params = sum(p.numel() for p in model.parameters())
print(f"Model loaded! {n_params:,} params, depth={CONFIG['n_layers']}")
print()


def generate_text(prompt, max_new=100, temp=0.7, top_k=10):
    """Generate text dari prompt."""
    ids = encode(prompt)
    if not ids:
        return "[ERROR] prompt gak ada karakter yang dikenal vocab"
    idx = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    out = model.generate(idx, max_new=max_new, temp=temp, top_k=top_k)
    return decode(out[0].tolist())


def cmd_default_prompt():
    """Test dengan beberapa prompt default."""
    prompts = ['Indonesia', 'Belajar', 'Padi', 'Teknologi', 'Mari bangun', 'Kerja keras']
    print("=" * 60)
    print("  D'QUILANE — Default Test")
    print("=" * 60)
    for p in prompts:
        out = generate_text(p, max_new=80, temp=0.7, top_k=10)
        print(f"\n[{p}]")
        print(f"  → {out}")
        print("-" * 60)


def cmd_custom_prompt(prompt):
    """Test dengan prompt custom dari argumen."""
    print("=" * 60)
    print(f"  D'QUILANE — Custom Prompt")
    print("=" * 60)
    print(f"\nPrompt: {prompt!r}")
    out = generate_text(prompt, max_new=120, temp=0.7, top_k=10)
    print(f"Output: {out}")


def cmd_interactive():
    """Mode interactive — ketik prompt sendiri berulang."""
    print("=" * 60)
    print("  D'QUILANE — Interactive Mode")
    print("=" * 60)
    print()
    print("Ketik prompt lalu Enter. Ketik 'quit' / 'exit' untuk keluar.")
    print("Tips: 'temp 1.2 Indonesia' (ubah temperature jadi 1.2)")
    print()
    temp = 0.7
    top_k = 10
    max_new = 100
    while True:
        try:
            user = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if not user:
            continue
        if user.lower() in ('quit', 'exit', 'q'):
            print("Bye!")
            break
        # parse commands
        parts = user.split()
        if parts[0] == 'temp' and len(parts) >= 2:
            try:
                temp = float(parts[1])
                print(f"temperature sekarang: {temp}")
            except ValueError:
                print("invalid temp")
            continue
        if parts[0] == 'topk' and len(parts) >= 2:
            try:
                top_k = int(parts[1])
                print(f"top_k sekarang: {top_k}")
            except ValueError:
                print("invalid top_k")
            continue
        if parts[0] == 'len' and len(parts) >= 2:
            try:
                max_new = int(parts[1])
                print(f"max_new sekarang: {max_new}")
            except ValueError:
                print("invalid len")
            continue
        # otherwise generate
        out = generate_text(user, max_new=max_new, temp=temp, top_k=top_k)
        print(f"\n→ {out}")


def main():
    if len(sys.argv) < 2:
        cmd_default_prompt()
    elif sys.argv[1] == 'interactive':
        cmd_interactive()
    elif sys.argv[1] in ('-h', '--help', 'help'):
        print(__doc__)
    else:
        cmd_custom_prompt(sys.argv[1])


if __name__ == "__main__":
    main()
