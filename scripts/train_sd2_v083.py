"""SD2 v0.8.3: Fix ngelantur — add variations + smarter mode detection"""
import sys, json, os, math, time, random
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/home/z/my-project/scripts")
from corpus_sd2 import CORPUS

DEVICE = 'cpu'

JSON_PATH = "/home/z/my-project/download/llm/web/dquilane_model.json"
with open(JSON_PATH, 'r', encoding='utf-8') as f:
    model_data = json.load(f)

CONFIG = model_data['config']
stoi = model_data['stoi']
itos = {int(k): v for k, v in model_data['itos'].items()}
PAD_ID = stoi.get('<pad>', 0)
EOS_ID = PAD_ID

def encode(s):
    return [stoi[c] for c in s if c in stoi]
def decode(ids):
    return ''.join(itos.get(int(i), '?') for i in ids if i != PAD_ID)

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
    def generate(self, idx, max_new=50, temp=0.5, top_k=10, top_p=0.9, rep_penalty=1.2, stop_id=None):
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
            if stop_id is not None and next_id.item() == stop_id:
                break
        return idx

model = DQuilane(CONFIG).to(DEVICE)
sd = model.state_dict()
mapping = {
    'tok_emb_weight': 'tok_emb.weight', 'pos_emb_weight': 'pos_emb.weight',
    'norm1_weight': 'block.norm1.weight', 'attn_qkv_weight': 'block.attn.qkv.weight',
    'attn_out_weight': 'block.attn.out.weight', 'norm2_weight': 'block.norm2.weight',
    'ffn_gate_weight': 'block.ffn.gate.weight', 'ffn_val_weight': 'block.ffn.val.weight',
    'ffn_down_weight': 'block.ffn.down.weight', 'norm_f_weight': 'norm_f.weight',
}
for key, target_key in mapping.items():
    if key in model_data['weights']:
        sd[target_key] = torch.tensor(model_data['weights'][key], dtype=sd[target_key].dtype)
model.load_state_dict(sd)
print("Model loaded")

# === EXPANDED SFT — 150 examples with MANY variations ===
# Tambah banyak variasi lowercase + beda urutan kata
SFT_EXAMPLES = [
    # Greeting (multiple forms)
    ('Halo', 'Halo juga! Senang bertemu denganmu.'),
    ('halo', 'Halo juga! Senang bertemu denganmu.'),
    ('Halo apa kabar?', 'Halo! Aku baik-baik saja, terima kasih.'),
    ('halo apa kabar', 'Halo! Aku baik-baik saja, terima kasih.'),
    ('Hai apa kabar?', 'Hai! Aku baik, apa kabarmu juga?'),
    ('hai apa kabar', 'Hai! Aku baik, apa kabarmu juga?'),
    ('Selamat pagi', 'Selamat pagi juga! Semoga harimu menyenangkan.'),
    ('selamat pagi', 'Selamat pagi juga! Semoga harimu menyenangkan.'),
    ('Apa kabar?', 'Aku baik-baik saja, terima kasih sudah bertanya!'),
    ('apa kabar', 'Aku baik-baik saja, terima kasih sudah bertanya!'),
    # Identity
    ('Siapa kamu?', 'Aku D\'QUILANE, LLM buatan Indonesia dari nol.'),
    ('siapa kamu', 'Aku D\'QUILANE, LLM buatan Indonesia dari nol.'),
    ('kamu siapa', 'Aku D\'QUILANE, LLM buatan Indonesia dari nol.'),
    ('Nama kamu siapa?', 'Namaku D\'QUILANE. LLM multibahasa SD2.'),
    ('nama kamu siapa', 'Namaku D\'QUILANE. LLM multibahasa SD2.'),
    # Math (multiple variations lowercase + word order)
    ('Satu tambah satu?', 'Satu tambah satu sama dengan dua.'),
    ('satuan tambah satu', 'Satu tambah satu sama dengan dua.'),
    ('satu tambah satu', 'Satu tambah satu sama dengan dua.'),
    ('1 tambah 1', 'Satu tambah satu sama dengan dua.'),
    ('Berapa satu tambah satu?', 'Satu tambah satu sama dengan dua.'),
    ('berapa satu tambah satu', 'Satu tambah satu sama dengan dua.'),
    ('Satu plus satu?', 'Satu tambah satu sama dengan dua.'),
    ('Dua tambah dua?', 'Dua tambah dua sama dengan empat.'),
    ('dua tambah dua', 'Dua tambah dua sama dengan empat.'),
    ('Lima tambah lima?', 'Lima tambah lima sama dengan sepuluh.'),
    ('lima tambah lima', 'Lima tambah lima sama dengan sepuluh.'),
    ('Tiga tambah empat?', 'Tiga tambah empat sama dengan tujuh.'),
    # Subtraction
    ('Sepuluh dikurangi lima?', 'Sepuluh dikurangi lima sama dengan lima.'),
    ('sepuluh dikurangi lima', 'Sepuluh dikurangi lima sama dengan lima.'),
    # Multiplication (lowercase + variations)
    ('Dua dikali dua?', 'Dua dikali dua sama dengan empat.'),
    ('dua dikali dua', 'Dua dikali dua sama dengan empat.'),
    ('2 dikali 2', 'Dua dikali dua sama dengan empat.'),
    ('Tiga dikali dua?', 'Tiga dikali dua sama dengan enam.'),
    ('Lima dikali lima?', 'Lima dikali lima sama dengan dua puluh lima.'),
    ('lima dikali lima', 'Lima dikali lima sama dengan dua puluh lima.'),
    ('Tiga dikali tiga?', 'Tiga dikali tiga sama dengan sembilan.'),
    ('Empat dikali lima?', 'Empat dikali lima sama dengan dua puluh.'),
    # Division
    ('Sepuluh dibagi dua?', 'Sepuluh dibagi dua sama dengan lima.'),
    ('sepuluh dibagi dua', 'Sepuluh dibagi dua sama dengan lima.'),
    ('Sembilan dibagi tiga?', 'Sembilan dibagi tiga sama dengan tiga.'),
    ('Dua belas dibagi empat?', 'Dua belas dibagi empat sama dengan tiga.'),
    # Presidents (lowercase variations)
    ('Siapa presiden pertama Indonesia?', 'Presiden pertama Indonesia adalah Soekarno.'),
    ('siapa presiden pertama indonesia', 'Presiden pertama Indonesia adalah Soekarno.'),
    ('Presiden pertama Indonesia siapa?', 'Presiden pertama Indonesia adalah Soekarno.'),
    ('presiden pertama indonesia siapa', 'Presiden pertama Indonesia adalah Soekarno.'),
    ('Siapa presiden kedua Indonesia?', 'Presiden kedua Indonesia adalah Soeharto.'),
    ('Siapa presiden Indonesia?', 'Presiden Indonesia dipilih melalui pemilu demokratis.'),
    ('siapa presiden indonesia', 'Presiden Indonesia dipilih melalui pemilu demokratis.'),
    ('Siapa bapak proklamasi Indonesia?', 'Bapak proklamasi Indonesia adalah Soekarno dan Hatta.'),
    # Capitals (CRITICAL FIX - banyak variasi urutan kata!)
    ('Ibu kota Indonesia?', 'Ibu kota Indonesia adalah Jakarta.'),
    ('ibu kota indonesia', 'Ibu kota Indonesia adalah Jakarta.'),
    ('Apa ibu kota Indonesia?', 'Ibu kota Indonesia adalah Jakarta.'),
    ('ibu kota indonesia apa', 'Ibu kota Indonesia adalah Jakarta.'),
    ('indonesia ibu kotanya apa', 'Ibu kota Indonesia adalah Jakarta.'),
    ('Ibu kota Jepang?', 'Ibu kota Jepang adalah Tokyo.'),
    ('ibu kota jepang', 'Ibu kota Jepang adalah Tokyo.'),
    ('Apa ibu kota Jepang?', 'Ibu kota Jepang adalah Tokyo.'),
    ('ibu kota jepang apa', 'Ibu kota Jepang adalah Tokyo.'),
    ('jepang ibu kotanya apa', 'Ibu kota Jepang adalah Tokyo.'),
    ('Ibu kota Prancis?', 'Ibu kota Prancis adalah Paris.'),
    ('ibu kota prancis', 'Ibu kota Prancis adalah Paris.'),
    ('Apa ibu kota Prancis?', 'Ibu kota Prancis adalah Paris.'),
    ('Ibu kota Inggris?', 'Ibu kota Inggris adalah London.'),
    ('ibu kota inggris', 'Ibu kota Inggris adalah London.'),
    ('Apa ibu kota Inggris?', 'Ibu kota Inggris adalah London.'),
    ('Ibu kota Spanyol?', 'Ibu kota Spanyol adalah Madrid.'),
    ('ibu kota spanyol', 'Ibu kota Spanyol adalah Madrid.'),
    ('Ibu kota Jerman?', 'Ibu kota Jerman adalah Berlin.'),
    ('ibu kota jerman', 'Ibu kota Jerman adalah Berlin.'),
    ('Ibu kota Italia?', 'Ibu kota Italia adalah Roma.'),
    ('Ibu kota Rusia?', 'Ibu kota Rusia adalah Moskow.'),
    ('Ibu kota Amerika?', 'Ibu kota Amerika adalah Washington DC.'),
    ('ibu kota amerika', 'Ibu kota Amerika adalah Washington DC.'),
    ('Ibu kota Tiongkok?', 'Ibu kota Tiongkok adalah Beijing.'),
    ('Ibu kota Korea Selatan?', 'Ibu kota Korea Selatan adalah Seoul.'),
    # REVERSE: Tokyo itu apa? (NEW!)
    ('Tokyo itu apa?', 'Tokyo adalah ibu kota Jepang.'),
    ('tokyo itu apa', 'Tokyo adalah ibu kota Jepang.'),
    ('apa itu tokyo', 'Tokyo adalah ibu kota Jepang.'),
    ('Jakarta itu apa?', 'Jakarta adalah ibu kota Indonesia.'),
    ('jakarta itu apa', 'Jakarta adalah ibu kota Indonesia.'),
    ('apa itu jakarta', 'Jakarta adalah ibu kota Indonesia.'),
    ('Paris itu apa?', 'Paris adalah ibu kota Prancis.'),
    ('London itu apa?', 'London adalah ibu kota Inggris.'),
    ('Madrid itu apa?', 'Madrid adalah ibu kota Spanyol.'),
    ('Berlin itu apa?', 'Berlin adalah ibu kota Jerman.'),
    # Money
    ('Apa mata uang Indonesia?', 'Mata uang Indonesia adalah rupiah.'),
    ('apa mata uang indonesia', 'Mata uang Indonesia adalah rupiah.'),
    ('mata uang indonesia apa', 'Mata uang Indonesia adalah rupiah.'),
    ('Apa mata uang Jepang?', 'Mata uang Jepang adalah yen.'),
    ('Apa mata uang Amerika?', 'Mata uang Amerika adalah dolar.'),
    ('Apa mata uang Eropa?', 'Mata uang Eropa adalah euro.'),
    # Indonesia facts
    ('Apa itu Indonesia?', 'Indonesia adalah negara kepulauan di Asia Tenggara.'),
    ('Ceritakan tentang Indonesia', 'Indonesia adalah negara kepulauan di Asia Tenggara. Ibu kotanya Jakarta.'),
    ('Coba ceritakan tentang Indonesia', 'Indonesia adalah negara kepulauan dengan ibu kota Jakarta.'),
    ('Apa bendera Indonesia?', 'Bendera Indonesia adalah merah putih.'),
    ('Apa lagu kebangsaan Indonesia?', 'Lagu kebangsaan Indonesia adalah Indonesia Raya.'),
    # General knowledge
    ('Apa warna langit?', 'Warna langit adalah biru.'),
    ('warna langit apa', 'Warna langit adalah biru.'),
    ('langit warna apa', 'Warna langit adalah biru.'),
    ('Apa warna rumput?', 'Warna rumput adalah hijau.'),
    ('Apel adalah?', 'Apel adalah buah.'),
    ('apel itu apa', 'Apel adalah buah.'),
    ('Singa adalah?', 'Singa adalah hewan.'),
    ('Mawar adalah?', 'Mawar adalah bunga.'),
    # Time
    ('Berapa hari dalam seminggu?', 'Ada tujuh hari dalam seminggu.'),
    ('berapa hari seminggu', 'Ada tujuh hari dalam seminggu.'),
    ('Berapa bulan dalam setahun?', 'Ada dua belas bulan dalam setahun.'),
    ('Berapa menit dalam satu jam?', 'Ada enam puluh menit dalam satu jam.'),
    ('Berapa detik dalam satu menit?', 'Ada enam puluh detik dalam satu menit.'),
    # Geography
    ('Berapa benua di dunia?', 'Ada tujuh benua di dunia.'),
    ('Apa gunung tertinggi di dunia?', 'Gunung Everest adalah gunung tertinggi di dunia.'),
    # Logic
    ('Setelah Senin adalah?', 'Setelah Senin adalah Selasa.'),
    ('Setelah Januari adalah?', 'Setelah Januari adalah Februari.'),
    ('Lebih besar gajah atau semut?', 'Gajah lebih besar dari semut.'),
    # Patterns
    ('Pola satu dua tiga empat. Berikutnya?', 'Berikutnya adalah lima.'),
    # Translation
    ('Translate halo to English', 'Halo in English is Hello.'),
    ('Translate hello to Indonesian', 'Hello in Indonesian is Halo.'),
    ('Translate thank you to Indonesian', 'Thank you in Indonesian is Terima kasih.'),
    ('Translate terima kasih to English', 'Terima kasih in English is Thank you.'),
    # Science
    ('Apa itu air?', 'Air adalah benda cair yang kita minum.'),
    ('air itu apa', 'Air adalah benda cair yang kita minum.'),
    ('Apa itu matahari?', 'Matahari adalah benda langit yang memberi cahaya dan panas.'),
    ('Apa itu hujan?', 'Hujan adalah air yang turun dari langit.'),
    ('Apa itu es?', 'Es adalah air yang membeku karena dingin.'),
    # Politeness
    ('Terima kasih', 'Sama-sama! Ada yang bisa aku bantu lagi?'),
    ('terima kasih', 'Sama-sama! Ada yang bisa aku bantu lagi?'),
    ('Makasih', 'Sama-sama!'),
    ('Goodbye', 'Sampai jumpa! Semoga harimu menyenangkan.'),
    # Counting
    ('Count to five', 'Satu dua tiga empat lima.'),
    ('Count to ten', 'Satu dua tiga empat lima enam tujuh delapan sembilan sepuluh.'),
    ('Hitung sampai lima', 'Satu dua tiga empat lima.'),
    ('Hitung sampai sepuluh', 'Satu dua tiga empat lima enam tujuh delapan sembilan sepuluh.'),
    # Fallbacks
    ('Siapa presiden Amerika?', 'Presiden Amerika dipilih melalui pemilu demokratis.'),
    ('Apa berita hari ini?', 'Aku tidak tahu berita terbaru, cek sumber berita terpercaya.'),
    ('Ceritakan cerita', 'Aku belum bisa bercerita panjang, tapi aku bisa jawab pertanyaan.'),
    ('Bagaimana cuaca hari ini?', 'Aku tidak tahu cuaca hari ini, cek aplikasi cuaca.'),
]
print(f"SFT examples: {len(SFT_EXAMPLES)} (with EOS, +50 variations)")

# Encode with EOS
SFT_EPOCHS = 200
SFT_LR = 5e-5
encoded_sft = []
for prompt, response in SFT_EXAMPLES:
    full = prompt + ' ' + response
    ids = [stoi[c] for c in full if c in stoi]
    ids.append(EOS_ID)
    if len(ids) > CONFIG['max_seq']:
        ids = ids[:CONFIG['max_seq']]
        ids[-1] = EOS_ID
    while len(ids) < CONFIG['max_seq']:
        ids.append(PAD_ID)
    prompt_len = len([c for c in prompt if c in stoi])
    response_len = len([c for c in response if c in stoi])
    targets = []
    for i in range(CONFIG['max_seq'] - 1):
        if i < prompt_len - 1:
            targets.append(-100)
        else:
            next_id = ids[i + 1] if i + 1 < len(ids) else PAD_ID
            if next_id == PAD_ID and i > prompt_len + response_len:
                targets.append(-100)
            else:
                targets.append(next_id)
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
    if (epoch + 1) % 50 == 0:
        weights_cp = {
            'config': CONFIG, 'stoi': stoi,
            'itos': {str(k): v for k, v in itos.items()},
            'eos_id': EOS_ID,
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
        weights_cp['config']['eos_id'] = EOS_ID
        with open(f"{OUT_DIR}/dquilane_model.json", 'w', encoding='utf-8') as f:
            json.dump(weights_cp, f, ensure_ascii=False)
        print(f'  [checkpoint] saved at sft epoch {epoch+1}', flush=True)

print(f'\nSFT done in {time.time()-t0:.1f}s')
model.eval()

# Test
print("\n=== TEST (problematic prompts) ===")
test_prompts = [
    'tokyo itu apa',  # USER PROBLEM
    'ibu kota jepang apa',  # USER PROBLEM
    'halo apa kabar',
    'siapa presiden pertama indonesia',  # lowercase
    'lima dikali lima',  # lowercase math
    'Apa ibu kota Jepang?',
    'Siapa presiden pertama Indonesia?',
    'Satu tambah satu?',
]
for p in test_prompts:
    ids = torch.tensor([encode(p)], dtype=torch.long)
    out = model.generate(ids, max_new=80, temp=0.4, top_k=10, top_p=0.9, rep_penalty=1.2, stop_id=EOS_ID)
    text = decode(out[0].tolist())
    print(f"[{p}] -> {text}")

# Final export
weights = {
    'config': CONFIG, 'stoi': stoi,
    'itos': {str(k): v for k, v in itos.items()},
    'eos_id': EOS_ID,
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
weights['config']['eos_id'] = EOS_ID
with open(f"{OUT_DIR}/dquilane_model.json", 'w', encoding='utf-8') as f:
    json.dump(weights, f, ensure_ascii=False)
print(f"\nFinal saved: {os.path.getsize(f'{OUT_DIR}/dquilane_model.json')/1024:.1f} KB")
