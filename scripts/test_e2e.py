"""Test end-to-end: train small model, export JSON, then load & run di Node.js."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, os, json, time, subprocess, sys

DEVICE = 'cpu'

# Load multilingual corpus
CORPUS_PATH = "/home/z/my-project/download/llm/dquilane_global.ipynb"
# Extract TEXT from cell 2 of notebook
nb = json.loads(open(CORPUS_PATH).read())
TEXT = ""
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        src = ''.join(cell['source'])
        if "MULTILINGUAL CORPUS" in src:
            # Extract the TEXT string
            start = src.find("TEXT = '''") + len("TEXT = '''")
            end = src.rfind("'''")
            TEXT = src[start:end]
            break

print(f"Multilingual corpus: {len(TEXT)} chars")

# Config
CONFIG = {
    'vocab_size': 0,
    'd_model':    96,
    'n_heads':    4,
    'd_ff':       384,
    'n_layers':   6,
    'max_seq':    64,
    'epochs':     500,  # pendek buat test
    'batch_size': 16,
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
print(f"Vocab: {VOCAB} chars (multilingual)")

def get_batch():
    idx = torch.randint(0, len(data) - CONFIG['max_seq'] - 1, (CONFIG['batch_size'],))
    x = torch.stack([data[i:i+CONFIG['max_seq']] for i in idx])
    y = torch.stack([data[i+1:i+1+CONFIG['max_seq']] for i in idx])
    return x.to(DEVICE), y.to(DEVICE)

# Model
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
print(f"D'QUILANE lahir: {n:,} params, depth={CONFIG['n_layers']}")

# Train
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
        print(f'epoch {epoch+1:4d}/{CONFIG["epochs"]}  loss={loss.item():.4f}  t={time.time()-t0:.1f}s')
model.eval()

# Test PyTorch generation
print("\n=== PYTORCH GENERATE ===")
for p in ['Indonesia', 'English is', 'La famille']:
    ids = torch.tensor([encode(p)], dtype=torch.long)
    out = model.generate(ids, max_new=40, temp=0.7, top_k=10)
    print(f"[{p}] -> {decode(out[0].tolist())}")

# Export JSON
weights = {
    'config': CONFIG,
    'stoi': stoi,
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

OUT_DIR = "/home/z/my-project/download/llm/web"
os.makedirs(OUT_DIR, exist_ok=True)
JSON_PATH = f"{OUT_DIR}/dquilane_model.json"
with open(JSON_PATH, 'w', encoding='utf-8') as f:
    json.dump(weights, f, ensure_ascii=False)
print(f"\nJSON exported: {JSON_PATH} ({os.path.getsize(JSON_PATH)/1024:.1f} KB)")

# Test JS by running Node.js
print("\n=== JS GENERATE (via Node.js) ===")
test_js = """
const fs = require('fs');
const path = '/home/z/my-project/download/llm/web/app.js';
const appCode = fs.readFileSync(path, 'utf8');
eval(appCode);  // defines DQuilane class

const modelData = JSON.parse(fs.readFileSync('/home/z/my-project/download/llm/web/dquilane_model.json', 'utf8'));
const model = new DQuilane(modelData);

(async () => {
    const prompts = ['Indonesia', 'English is', 'La famille'];
    for (const p of prompts) {
        console.time('gen_' + p);
        const out = await model.generate(p, 40, 0.7, 10);
        console.timeEnd('gen_' + p);
        console.log('[' + p + '] -> ' + out);
    }
})();
"""
result = subprocess.run(['node', '-e', test_js], capture_output=True, text=True)
print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr)
