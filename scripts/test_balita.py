"""Test script dari notebook dquilane_balita.ipynb - extract code cells & run."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import os

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"PyTorch: {torch.__version__}, Device: {DEVICE}")

# ===== CONFIG =====
CONFIG = {
    'vocab_size': 256,
    'd_model':    128,
    'n_heads':    4,
    'd_ff':       512,
    'n_layers':   4,
    'max_seq':    64,
    'epochs':     500,  # lebih dikit biar cepat test
    'batch_size': 16,
    'lr':         3e-4,
}

# ===== DATASET =====
TEXT = """Indonesia adalah negara kepulauan.
Ibu kota Indonesia adalah Jakarta.
Bahasa Indonesia adalah bahasa resmi.
Mata uang Indonesia adalah rupiah.
Indonesia kaya akan sumber daya alam.
Penduduk Indonesia sangat beragam.
Setiap suku memiliki budaya yang unik.
Gunung berapi banyak di Indonesia.
Hutan tropis Indonesia terbesar di dunia.
Laut Indonesia kaya akan ikan.
Iklim Indonesia adalah tropis.
Musim hujan dan kemarau silih berganti.
Pertanian adalah mata pencaharian utama.
Padi ditanam di sawah oleh petani.
Nasi adalah makanan pokok Indonesia.
Pendidikan kunci masa depan bangsa.
Teknologi berkembang pesat di era modern.
Komputer dan internet ubah cara komunikasi.
Kecerdasan buatan menjadi tren global.
Model bahasa besar sedang populer sekarang.
Transformer arsitektur dasar model modern.
Attention mekanisme memahami konteks kalimat.
Parameter sharing hemat ukuran model.
ALBERT contoh model dengan parameter sharing.
Belajar setiap hari bikin hidup bermakna.
Kerja keras kunci pencapaian tujuan.
Indonesia akan terus maju dan berkembang.
Anak muda harapan masa depan bangsa.
Mari bangun negeri dengan semangat pantang menyerah.
"""

# ===== TOKENIZER =====
chars = sorted(set(TEXT))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
stoi['<pad>'] = len(stoi)
itos[len(itos)] = '<pad>'
PAD_ID = stoi['<pad>']
VOCAB = len(stoi)
CONFIG['vocab_size'] = VOCAB

def encode(s):
    return [stoi[c] for c in s if c in stoi]

def decode(ids):
    return ''.join(itos[i] for i in ids if i != PAD_ID)

print(f"Vocab size: {VOCAB} karakter")

# ===== DATA =====
data = torch.tensor(encode(TEXT), dtype=torch.long)

def get_batch():
    idx = torch.randint(0, len(data) - CONFIG['max_seq'] - 1, (CONFIG['batch_size'],))
    x = torch.stack([data[i:i+CONFIG['max_seq']] for i in idx])
    y = torch.stack([data[i+1:i+1+CONFIG['max_seq']] for i in idx])
    return x.to(DEVICE), y.to(DEVICE)

# ===== MODEL =====
class Attention(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        assert d % h == 0, f'd_model ({d}) harus kelipatan n_heads ({h})'
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
        self.val  = nn.Linear(d, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d, bias=False)
    
    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.val(x))

class Block(nn.Module):
    def __init__(self, d, h, d_ff):
        super().__init__()
        self.norm1 = nn.RMSNorm(d)
        self.attn  = Attention(d, h)
        self.norm2 = nn.RMSNorm(d)
        self.ffn   = FFN(d, d_ff)
    
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
        self.apply(self._init)
    
    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0, std=0.02)
    
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

model = DQuilane(CONFIG).to(DEVICE)
n = sum(p.numel() for p in model.parameters())
print(f"D'QUILANE lahir!")
print(f"Unique params: {n:,}  (~{n*4/1024:.1f} KB)")
print(f"Virtual depth: {CONFIG['n_layers']}")
print(f"Effective (no share): {n * CONFIG['n_layers']:,}")

# Smoke test
x, y = get_batch()
logits, loss = model(x, y)
print(f"\nSmoke test OK!")
print(f"  input:  {x.shape}")
print(f"  logits: {logits.shape}")
print(f"  loss:   {loss.item():.4f}  (random = {math.log(VOCAB):.4f})")

# Training
print(f"\n=== TRAINING ({CONFIG['epochs']} epochs) ===")
opt = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=0.01)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CONFIG['epochs'])

model.train()
t0 = time.time()
for epoch in range(CONFIG['epochs']):
    xb, yb = get_batch()
    opt.zero_grad()
    _, loss = model(xb, yb)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    sched.step()
    if epoch % 50 == 0 or epoch == CONFIG['epochs'] - 1:
        print(f"  epoch {epoch+1:4d}/{CONFIG['epochs']}  loss={loss.item():.4f}  t={time.time()-t0:.1f}s")

print(f"\nTraining done in {time.time()-t0:.1f}s")

# Generate
print("\n=== GENERATE TEST ===")
for p in ['Indonesia', 'Belajar', 'Padi']:
    ids = torch.tensor([encode(p)], dtype=torch.long, device=DEVICE)
    out = model.generate(ids, max_new=80, temp=0.7, top_k=10)
    text = decode(out[0].tolist())
    print(f"[{p}] ->")
    print(text)
    print()

# Save
os.makedirs('/home/z/my-project/download/llm/output_test', exist_ok=True)
torch.save({
    'state_dict': model.state_dict(),
    'config': CONFIG,
    'stoi': stoi,
    'itos': {int(k): v for k, v in itos.items()},
}, '/home/z/my-project/download/llm/output_test/dquilane_balita.pt')
size = os.path.getsize('/home/z/my-project/download/llm/output_test/dquilane_balita.pt') / 1024
print(f"\nModel saved: {size:.1f} KB")
print("\nALL OK — notebook siap dipakai di Kaggle!")
