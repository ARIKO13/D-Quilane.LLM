"""
D'QUILANE v0.6 Anak SD Kelas 1 — Continue training + SFT
Load model TK v0.5 (loss 1.80) dari JSON, train 800 more epochs, lalu SFT 200 epochs.
Target: loss < 1.2, bisa ikutin perintah (Halo, Siapa kamu, translate, dll)
"""
import sys, json, os, math, time, random
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/home/z/my-project/scripts")
from corpus_multilingual import CORPUS

DEVICE = 'cpu'

# ===== Load existing TK model dari JSON =====
JSON_PATH = "/home/z/my-project/download/llm/web/dquilane_model.json"
print(f"Loading model from: {JSON_PATH}")
with open(JSON_PATH, 'r', encoding='utf-8') as f:
    model_data = json.load(f)

CONFIG = model_data['config']
stoi = model_data['stoi']
itos = {int(k): v for k, v in model_data['itos'].items()}
PAD_ID = stoi.get('<pad>', 0)
VOCAB = CONFIG['vocab_size']

print(f"Loaded. Config: {CONFIG}")
print(f"Vocab: {VOCAB}")

# Tokenizer (reuse)
def encode(s):
    return [stoi[c] for c in s if c in stoi]
def decode(ids):
    return ''.join(itos.get(int(i), '?') for i in ids if i != PAD_ID)

TEXT = CORPUS
data = torch.tensor(encode(TEXT), dtype=torch.long)
print(f"Corpus: {len(data)} tokens")

def get_batch(batch_size=24, max_seq=None):
    max_seq = max_seq or CONFIG['max_seq']
    idx = torch.randint(0, len(data) - max_seq - 1, (batch_size,))
    x = torch.stack([data[i:i+max_seq] for i in idx])
    y = torch.stack([data[i+1:i+1+max_seq] for i in idx])
    return x.to(DEVICE), y.to(DEVICE)

# ===== Model class (same as TK v0.5) =====
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
    def generate(self, idx, max_new=50, temp=0.7, top_k=40, top_p=0.95, rep_penalty=1.1):
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

# Create model
model = DQuilane(CONFIG).to(DEVICE)

# Load weights dari JSON
def load_weights_from_json(model, weights_data):
    """Load weights dari JSON ke model state_dict."""
    sd = model.state_dict()
    for key, value in weights_data.items():
        # Map JSON key to state_dict key
        target_key = None
        if key == 'tok_emb_weight':
            target_key = 'tok_emb.weight'
        elif key == 'pos_emb_weight':
            target_key = 'pos_emb.weight'
        elif key == 'norm1_weight':
            target_key = 'block.norm1.weight'
        elif key == 'attn_qkv_weight':
            target_key = 'block.attn.qkv.weight'
        elif key == 'attn_out_weight':
            target_key = 'block.attn.out.weight'
        elif key == 'norm2_weight':
            target_key = 'block.norm2.weight'
        elif key == 'ffn_gate_weight':
            target_key = 'block.ffn.gate.weight'
        elif key == 'ffn_val_weight':
            target_key = 'block.ffn.val.weight'
        elif key == 'ffn_down_weight':
            target_key = 'block.ffn.down.weight'
        elif key == 'norm_f_weight':
            target_key = 'norm_f.weight'
        if target_key and target_key in sd:
            tensor = torch.tensor(value, dtype=sd[target_key].dtype)
            assert tensor.shape == sd[target_key].shape, f"{target_key}: {tensor.shape} != {sd[target_key].shape}"
            sd[target_key] = tensor
    model.load_state_dict(sd)

print("Loading weights from JSON...")
load_weights_from_json(model, model_data['weights'])
print(f"  ✓ weights loaded")

n_params = sum(p.numel() for p in model.parameters())
print(f"\nD'QUILANE TK (loaded) — params: {n_params:,}, depth: {CONFIG['n_layers']}")

# ===== Test generate pre-training =====
print("\n=== PRE-CONTINUE TEST ===")
for p in ['Indonesia', 'English is', 'Halo']:
    ids = torch.tensor([encode(p)], dtype=torch.long)
    out = model.generate(ids, max_new=40, temp=0.7, top_k=40, top_p=0.95, rep_penalty=1.1)
    print(f"[{p}] -> {decode(out[0].tolist())}")

# ===== CONTINUE TRAINING (800 more epochs) =====
print(f"\n=== CONTINUE TRAINING (800 more epochs) ===")
CONTINUE_EPOCHS = 800
CONTINUE_LR = 2e-4  # slightly lower than initial

opt = torch.optim.AdamW(model.parameters(), lr=CONTINUE_LR, weight_decay=0.01)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CONTINUE_EPOCHS)

model.train()
t0 = time.time()
for epoch in range(CONTINUE_EPOCHS):
    xb, yb = get_batch()
    opt.zero_grad()
    _, loss = model(xb, yb)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    sched.step()
    if epoch % 100 == 0 or epoch == CONTINUE_EPOCHS - 1:
        elapsed = time.time() - t0
        eta = elapsed * (CONTINUE_EPOCHS - epoch - 1) / max(epoch + 1, 1)
        print(f'  epoch {epoch+1:4d}/{CONTINUE_EPOCHS}  loss={loss.item():.4f}  t={elapsed:.1f}s  eta={eta:.0f}s', flush=True)
    # checkpoint every 200
    if (epoch + 1) % 200 == 0:
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
        OUT_DIR = "/home/z/my-project/download/llm/web"
        with open(f"{OUT_DIR}/dquilane_model.json", 'w', encoding='utf-8') as f:
            json.dump(weights_cp, f, ensure_ascii=False)
        print(f'  [checkpoint] saved at epoch {epoch+1}', flush=True)

print(f'\nContinue training done in {time.time()-t0:.1f}s. Final loss: {loss.item():.4f}')

# ===== TEST after continue training =====
print("\n=== POST-CONTINUE TEST ===")
for p in ['Indonesia', 'English is', 'Русский', '日本語', '한국어', 'Halo']:
    ids = torch.tensor([encode(p)], dtype=torch.long)
    out = model.generate(ids, max_new=40, temp=0.7, top_k=40, top_p=0.95, rep_penalty=1.1)
    print(f"[{p}] -> {decode(out[0].tolist())}")

# ===== SFT (Supervised Fine-Tuning) =====
print(f"\n=== SFT (200 epochs) ===")
SFT_EXAMPLES = [
    ('Halo', 'Halo juga! Senang bertemu denganmu.'),
    ('Hai', 'Hai! Apa kabar?'),
    ('Hi', 'Hi there! Bagaimana kabarmu?'),
    ('Siapa kamu?', 'Aku D\'QUILANE, LLM buatan Indonesia dari nol.'),
    ('Nama kamu siapa?', 'Namaku D\'QUILANE. LLM multibahasa.'),
    ('Apa itu Indonesia?', 'Indonesia adalah negara kepulauan di Asia Tenggara.'),
    ('Ibu kota Indonesia?', 'Ibu kota Indonesia adalah Jakarta.'),
    ('Bahasa resmi Indonesia?', 'Bahasa resmi Indonesia adalah bahasa Indonesia.'),
    ('Translate halo to English', 'Hello'),
    ('Translate hello to Indonesian', 'Halo'),
    ('Translate terima kasih to English', 'Thank you'),
    ('Translate thank you to Indonesian', 'Terima kasih'),
    ('Satu tambah satu?', 'Satu tambah satu sama dengan dua.'),
    ('Berapa dua tambah dua?', 'Dua tambah dua sama dengan empat.'),
    ('Apa itu AI?', 'AI adalah kecerdasan buatan.'),
    ('Apa itu LLM?', 'LLM adalah Large Language Model.'),
    ('Terima kasih', 'Sama-sama! Ada yang bisa aku bantu lagi?'),
    ('Goodbye', 'Sampai jumpa! Semoga harimu menyenangkan.'),
    ('Dadah', 'Dadah! Sampai bertemu lagi.'),
    ('Apa kabar?', 'Aku baik-baik saja, terima kasih sudah bertanya!'),
]

print(f"SFT dataset: {len(SFT_EXAMPLES)} examples")

# Encode SFT examples
encoded_sft = []
for prompt, response in SFT_EXAMPLES:
    full = prompt + ' ' + response
    ids = [stoi[c] for c in full if c in stoi]
    if len(ids) > CONFIG['max_seq']:
        ids = ids[:CONFIG['max_seq']]
    while len(ids) < CONFIG['max_seq']:
        ids.append(PAD_ID)
    prompt_len = len([c for c in prompt if c in stoi])
    targets = []
    for i in range(CONFIG['max_seq'] - 1):
        if i < prompt_len - 1:
            targets.append(-100)
        else:
            next_id = ids[i + 1] if i + 1 < len(ids) else PAD_ID
            targets.append(next_id if next_id != PAD_ID else -100)
    encoded_sft.append((ids[:-1], targets))

SFT_EPOCHS = 200
SFT_LR = 1e-4
sft_opt = torch.optim.AdamW(model.parameters(), lr=SFT_LR, weight_decay=0.01)
sft_sched = torch.optim.lr_scheduler.CosineAnnealingLR(sft_opt, T_max=SFT_EPOCHS)

model.train()
t0 = time.time()
for epoch in range(SFT_EPOCHS):
    random.shuffle(encoded_sft)
    total_loss = 0
    for x, y in encoded_sft:
        x = torch.tensor([x], dtype=torch.long, device=DEVICE)
        y = torch.tensor([y], dtype=torch.long, device=DEVICE)
        sft_opt.zero_grad()
        _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        sft_opt.step()
        total_loss += loss.item()
    sft_sched.step()
    if epoch % 20 == 0 or epoch == SFT_EPOCHS - 1:
        print(f'  sft epoch {epoch+1:3d}/{SFT_EPOCHS}  loss={total_loss/len(encoded_sft):.4f}  t={time.time()-t0:.1f}s', flush=True)
print(f'SFT done in {time.time()-t0:.1f}s')
model.eval()

# ===== FINAL TEST (after continue + SFT) =====
print("\n=== FINAL TEST (post SFT) ===")
for p in ['Halo', 'Siapa kamu?', 'Ibu kota Indonesia?', 'Translate halo to English', 'Indonesia', 'English is']:
    ids = torch.tensor([encode(p)], dtype=torch.long)
    out = model.generate(ids, max_new=50, temp=0.5, top_k=10, top_p=0.9, rep_penalty=1.2)
    print(f"[{p}] -> {decode(out[0].tolist())}")

# ===== EXPORT JSON =====
print("\n=== EXPORT JSON ===")
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
with open(f"{OUT_DIR}/dquilane_model.json", 'w', encoding='utf-8') as f:
    json.dump(weights, f, ensure_ascii=False)
print(f"JSON saved: {os.path.getsize(f'{OUT_DIR}/dquilane_model.json')/1024/1024:.2f} MB")
