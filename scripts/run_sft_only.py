"""Run only SFT (load model dari JSON, fine-tune with 44 instructions)."""
import sys, json, os, math, time, random
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/home/z/my-project/scripts")
from corpus_sd1 import CORPUS

DEVICE = 'cpu'

# Load existing model
JSON_PATH = "/home/z/my-project/download/llm/web/dquilane_model.json"
with open(JSON_PATH, 'r', encoding='utf-8') as f:
    model_data = json.load(f)

CONFIG = model_data['config']
stoi = model_data['stoi']
itos = {int(k): v for k, v in model_data['itos'].items()}
PAD_ID = stoi.get('<pad>', 0)
VOCAB = CONFIG['vocab_size']

def encode(s):
    return [stoi[c] for c in s if c in stoi]
def decode(ids):
    return ''.join(itos.get(int(i), '?') for i in ids if i != PAD_ID)

# Model class
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
# Load weights
def load_weights_from_json(model, weights_data):
    sd = model.state_dict()
    mapping = {
        'tok_emb_weight': 'tok_emb.weight',
        'pos_emb_weight': 'pos_emb.weight',
        'norm1_weight': 'block.norm1.weight',
        'attn_qkv_weight': 'block.attn.qkv.weight',
        'attn_out_weight': 'block.attn.out.weight',
        'norm2_weight': 'block.norm2.weight',
        'ffn_gate_weight': 'block.ffn.gate.weight',
        'ffn_val_weight': 'block.ffn.val.weight',
        'ffn_down_weight': 'block.ffn.down.weight',
        'norm_f_weight': 'norm_f.weight',
    }
    for key, target_key in mapping.items():
        if key in weights_data:
            tensor = torch.tensor(weights_data[key], dtype=sd[target_key].dtype)
            sd[target_key] = tensor
    model.load_state_dict(sd)

print("Loading model...")
load_weights_from_json(model, model_data['weights'])
print("  ✓ loaded")

# === SFT Examples (44 instructions covering all SD1 topics) ===
SFT_EXAMPLES = [
    ('Halo', 'Halo juga! Senang bertemu denganmu.'),
    ('Hai', 'Hai! Apa kabar?'),
    ('Selamat pagi', 'Selamat pagi juga! Semoga harimu menyenangkan.'),
    ('Selamat malam', 'Selamat malam! Selamat istirahat.'),
    ('Siapa kamu?', 'Aku D\'QUILANE, LLM buatan Indonesia dari nol.'),
    ('Nama kamu siapa?', 'Namaku D\'QUILANE. LLM multibahasa.'),
    ('Satu tambah satu?', 'Satu tambah satu sama dengan dua.'),
    ('Dua tambah dua?', 'Dua tambah dua sama dengan empat.'),
    ('Lima tambah lima?', 'Lima tambah lima sama dengan sepuluh.'),
    ('Tiga tambah empat?', 'Tiga tambah empat sama dengan tujuh.'),
    ('Sepuluh dikurangi lima?', 'Sepuluh dikurangi lima sama dengan lima.'),
    ('Berapa satu tambah satu?', 'Satu tambah satu sama dengan dua.'),
    ('Berapa dua tambah dua?', 'Dua tambah dua sama dengan empat.'),
    ('Berapa lima tambah lima?', 'Lima tambah lima sama dengan sepuluh.'),
    ('Apa itu Indonesia?', 'Indonesia adalah negara kepulauan di Asia Tenggara.'),
    ('Ibu kota Indonesia?', 'Ibu kota Indonesia adalah Jakarta.'),
    ('Bahasa resmi Indonesia?', 'Bahasa resmi Indonesia adalah bahasa Indonesia.'),
    ('Apa warna langit?', 'Warna langit adalah biru.'),
    ('Apa warna rumput?', 'Warna rumput adalah hijau.'),
    ('Apel adalah?', 'Apel adalah buah.'),
    ('Singa adalah?', 'Singa adalah hewan.'),
    ('Mawar adalah?', 'Mawar adalah bunga.'),
    ('Berapa hari dalam seminggu?', 'Ada tujuh hari dalam seminggu.'),
    ('Berapa bulan dalam setahun?', 'Ada dua belas bulan dalam setahun.'),
    ('Berapa jari di dua tangan?', 'Ada sepuluh jari di dua tangan.'),
    ('Setelah Senin adalah?', 'Setelah Senin adalah Selasa.'),
    ('Setelah Januari adalah?', 'Setelah Januari adalah Februari.'),
    ('Translate halo to English', 'Halo in English is Hello.'),
    ('Translate hello to Indonesian', 'Hello in Indonesian is Halo.'),
    ('Translate terima kasih to English', 'Terima kasih in English is Thank you.'),
    ('Translate thank you to Indonesian', 'Thank you in Indonesian is Terima kasih.'),
    ('Translate goodbye to Indonesian', 'Goodbye in Indonesian is Selamat tinggal.'),
    ('Lebih besar gajah atau semut?', 'Gajah lebih besar dari semut.'),
    ('Lebih berat batu atau kapas?', 'Batu lebih berat dari kapas.'),
    ('Bigger elephant or ant?', 'Elephant is bigger than ant.'),
    ('Apa itu AI?', 'AI adalah kecerdasan buatan, mesin yang meniru kecerdasan manusia.'),
    ('Apa itu LLM?', 'LLM adalah Large Language Model, model AI bahasa besar.'),
    ('Terima kasih', 'Sama-sama! Ada yang bisa aku bantu lagi?'),
    ('Goodbye', 'Sampai jumpa! Semoga harimu menyenangkan.'),
    ('Dadah', 'Dadah! Sampai bertemu lagi.'),
    ('Apa kabar?', 'Aku baik-baik saja, terima kasih sudah bertanya!'),
    ('Count to five', 'Satu dua tiga empat lima.'),
    ('Count to ten', 'Satu dua tiga empat lima enam tujuh delapan sembilan sepuluh.'),
]
print(f"SFT examples: {len(SFT_EXAMPLES)}")

# Encode SFT
SFT_EPOCHS = 200
SFT_LR = 1e-4
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

sft_opt = torch.optim.AdamW(model.parameters(), lr=SFT_LR, weight_decay=0.01)
sft_sched = torch.optim.lr_scheduler.CosineAnnealingLR(sft_opt, T_max=SFT_EPOCHS)

OUT_DIR = "/home/z/my-project/download/llm/web"
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
    # checkpoint every 50
    if (epoch + 1) % 50 == 0:
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
        print(f'  [checkpoint] saved at sft epoch {epoch+1}', flush=True)

print(f'\nSFT done in {time.time()-t0:.1f}s')
model.eval()

# === TEST ===
print("\n=== SD1 TEST ===")
test_prompts = [
    'Halo', 'Siapa kamu?', 'Ibu kota Indonesia?',
    'Satu tambah satu?', 'Dua tambah dua?', 'Lima tambah lima?',
    'Translate halo to English', 'Translate thank you to Indonesian',
    'Apa warna langit?', 'Apel adalah?', 'Berapa hari dalam seminggu?',
    'Setelah Senin adalah?',
    'Indonesia', 'English is', '日本語', '한국어',
]
for p in test_prompts:
    ids = torch.tensor([encode(p)], dtype=torch.long)
    out = model.generate(ids, max_new=40, temp=0.5, top_k=10, top_p=0.9, rep_penalty=1.2)
    print(f"[{p}] -> {decode(out[0].tolist())}")

# Final export
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
print(f"\nFinal JSON saved: {os.path.getsize(f'{OUT_DIR}/dquilane_model.json')/1024/1024:.2f} MB")
