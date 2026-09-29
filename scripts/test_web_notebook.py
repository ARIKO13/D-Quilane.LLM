"""Test script untuk verifikasi code di dquilane_web.ipynb jalan.
Skip launch gradio (gak bisa buka browser di sandbox)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, os, time

DEVICE = 'cpu'

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
"""

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

def get_batch():
    idx = torch.randint(0, len(data) - CONFIG['max_seq'] - 1, (CONFIG['batch_size'],))
    x = torch.stack([data[i:i+CONFIG['max_seq']] for i in idx])
    y = torch.stack([data[i+1:i+1+CONFIG['max_seq']] for i in idx])
    return x.to(DEVICE), y.to(DEVICE)

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

# init model
model = DQuilane(CONFIG).to(DEVICE)
n = sum(p.numel() for p in model.parameters())
print(f"D'QUILANE lahir! params={n:,}, depth={CONFIG['n_layers']}")

# train
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
        print(f'epoch {epoch+1:4d}/{CONFIG["epochs"]}  loss={loss.item():.4f}  t={time.time()-t0:.1f}s')

model.eval()

# test generate function
def generate_text(prompt, max_new=100, temp=0.7, top_k=10):
    ids = encode(prompt)
    if not ids:
        return '[ERROR] prompt gak ada karakter yang dikenal'
    idx = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    out = model.generate(idx, max_new=max_new, temp=temp, top_k=top_k)
    return decode(out[0].tolist())

print("\n=== TEST GENERATE ===")
for p in ['Indonesia', 'Belajar', 'Teknologi']:
    print(f"\n[{p}]")
    print(generate_text(p, max_new=80, temp=0.7, top_k=10))

# test gradio UI build (tanpa launch)
print("\n=== BUILD GRADIO UI ===")
import gradio as gr

with gr.Blocks(title="D'QUILANE Web Tester", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# D'QUILANE Web Tester")
    with gr.Row():
        with gr.Column(scale=1):
            prompt = gr.Textbox(label='Prompt', value='Indonesia', lines=2)
            max_new = gr.Slider(20, 300, value=100, step=10, label='Max tokens')
            temp = gr.Slider(0.1, 2.0, value=0.7, step=0.1, label='Temperature')
            top_k = gr.Slider(0, 50, value=10, step=1, label='Top-k')
            btn = gr.Button('Generate', variant='primary')
            clear = gr.Button('Clear')
        with gr.Column(scale=1):
            output = gr.Textbox(label='Output', lines=15)
    btn.click(generate_text, [prompt, max_new, temp, top_k], output)
    clear.click(lambda: '', None, output)
    gr.Examples(
        examples=[['Indonesia'], ['Belajar'], ['Teknologi']],
        inputs=prompt,
    )

print("Gradio UI built OK! (skip launch di sandbox)")
print("\n=== ALL TEST PASSED ===")
