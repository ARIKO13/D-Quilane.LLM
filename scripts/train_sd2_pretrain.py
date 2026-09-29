"""D'QUILANE SD2 — Train from scratch dengan curated corpus"""
import sys, json, os, math, time, random
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/home/z/my-project/scripts")
from corpus_sd2 import CORPUS

DEVICE = 'cpu'
TEXT = CORPUS

CONFIG = {
    'vocab_size': 0,
    'd_model':    144,
    'n_heads':    6,
    'd_ff':       576,
    'n_layers':   10,
    'max_seq':    80,
    'epochs':     1000,
    'batch_size': 24,
    'lr':         3e-4,
}

# Tokenizer
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
    return ''.join(itos.get(int(i), '?') for i in ids if i != PAD_ID)

data = torch.tensor(encode(TEXT), dtype=torch.long)
print(f"Vocab: {VOCAB} chars (SD2 curated)")
print(f"Corpus: {len(TEXT)} chars, {len(data)} tokens")

def get_batch():
    idx = torch.randint(0, len(data) - CONFIG['max_seq'] - 1, (CONFIG['batch_size'],))
    x = torch.stack([data[i:i+CONFIG['max_seq']] for i in idx])
    y = torch.stack([data[i+1:i+1+CONFIG['max_seq']] for i in idx])
    return x.to(DEVICE), y.to(DEVICE)

# Model (same as before)
class Attention(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        assert d % h == 0
        self.h, self.dh = h, d // h
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
    def generate(self, idx, max_new=50, temp=0.5, top_k=10, top_p=0.9, rep_penalty=1.2):
        self.eval()
        for _ in range(max_new):
            x = idx if idx.size(1) <= self.cfg['max_seq'] else idx[:, -self.cfg['max_seq']:]
            logits, _ = self(x)
            logits = logits[:, -1, :] / max(temp, 1e-5)
            if rep_penalty != 1.0:
                for t in idx[0].tolist()[-20:]:
                    logits[0, t] /= rep_penalty
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_mask = cum_probs > top_p
                sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
                sorted_mask[..., 0] = False
                indices_to_remove = sorted_mask.scatter(-1, sorted_idx, sorted_mask)
                logits = logits.masked_fill(indices_to_remove, float('-inf'))
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx

model = DQuilane(CONFIG).to(DEVICE)
n = sum(p.numel() for p in model.parameters())
print(f"\nD'QUILANE SD2 lahir!")
print(f"  Unique params: {n:,}")
print(f"  Virtual depth: {CONFIG['n_layers']}")
print(f"  Vocab: {VOCAB} (smaller than SD1's 563, more efficient)")

# Pre-training
OUT_DIR = "/home/z/my-project/download/llm/web"
os.makedirs(OUT_DIR, exist_ok=True)

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
    if epoch % 100 == 0 or epoch == CONFIG['epochs'] - 1:
        elapsed = time.time() - t0
        eta = elapsed * (CONFIG['epochs'] - epoch - 1) / max(epoch + 1, 1)
        print(f'  epoch {epoch+1:4d}/{CONFIG["epochs"]}  loss={loss.item():.4f}  t={elapsed:.1f}s  eta={eta:.0f}s', flush=True)
    # Checkpoint
    if (epoch + 1) % 250 == 0:
        weights_cp = {
            'config': CONFIG, 'stoi': stoi,
            'itos': {str(k): v for k, v in itos.items()},
            'weights': {
                'tok_emb_weight': model.tok_emb.weight.detach().cpu().tolist(),
                'pos_emb_weight': model.pos_emb.weight.detach().cpu().tolist(),
                'norm1_weight': model.block.norm1.weight.detach().cpu().tolist(),
                'attn_qkv_weight': model.block.attn.qkv.weight.detach().cpu().tolist(),
                'attn_out_weight': model.block.attn.out.weight.detach().cpu().tolist(),
                'norm2_weight': model.block.norm2.weight.detach().cpu().tolist(),
                'ffn_gate_weight': model.block.ffn.gate.weight.detach().cpu().tolist(),
                'ffn_val_weight': model.block.ffn.val.weight.detach().cpu().tolist(),
                'ffn_down_weight': model.block.ffn.down.weight.detach().cpu().tolist(),
                'norm_f_weight': model.norm_f.weight.detach().cpu().tolist(),
            },
        }
        with open(f"{OUT_DIR}/dquilane_model.json", 'w', encoding='utf-8') as f:
            json.dump(weights_cp, f, ensure_ascii=False)
        print(f'  [checkpoint] saved at epoch {epoch+1}', flush=True)
print(f'\nPre-training done in {time.time()-t0:.1f}s. Final loss: {loss.item():.4f}')

# Save final pre-trained
weights = {
    'config': CONFIG, 'stoi': stoi,
    'itos': {str(k): v for k, v in itos.items()},
    'weights': {
        'tok_emb_weight': model.tok_emb.weight.detach().cpu().tolist(),
        'pos_emb_weight': model.pos_emb.weight.detach().cpu().tolist(),
        'norm1_weight': model.block.norm1.weight.detach().cpu().tolist(),
        'attn_qkv_weight': model.block.attn.qkv.weight.detach().cpu().tolist(),
        'attn_out_weight': model.block.attn.out.weight.detach().cpu().tolist(),
        'norm2_weight': model.block.norm2.weight.detach().cpu().tolist(),
        'ffn_gate_weight': model.block.ffn.gate.weight.detach().cpu().tolist(),
        'ffn_val_weight': model.block.ffn.val.weight.detach().cpu().tolist(),
        'ffn_down_weight': model.block.ffn.down.weight.detach().cpu().tolist(),
        'norm_f_weight': model.norm_f.weight.detach().cpu().tolist(),
    },
}
with open(f"{OUT_DIR}/dquilane_model.json", 'w', encoding='utf-8') as f:
    json.dump(weights, f, ensure_ascii=False)
print(f"\nPre-trained model saved: {os.path.getsize(f'{OUT_DIR}/dquilane_model.json')/1024:.1f} KB")

# Test pre-training (before SFT)
print("\n=== POST PRE-TRAINING TEST ===")
for p in ['Indonesia', 'Halo', 'Satu tambah satu?', 'Apa warna langit?', '日本語']:
    ids = torch.tensor([encode(p)], dtype=torch.long)
    out = model.generate(ids, max_new=40, temp=0.5, top_k=10, top_p=0.9, rep_penalty=1.2)
    print(f"[{p}] -> {decode(out[0].tolist())}")
