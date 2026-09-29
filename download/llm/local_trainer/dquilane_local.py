"""
D'QUILANE Local GPU Trainer
===========================
Auto-detect GPU (CUDA / MPS / CPU) and train D'QUILANE model.
GPU cuma kepake pas training. Setelah selesai, GPU bebas (memory freed).

Target hardware: NVIDIA RTX 3050 4GB+ (or any CUDA GPU)
Fallback: Apple Silicon MPS, atau CPU

Usage:
    python dquilane_local.py
    (atau klik train.bat di Windows)
"""
import sys, json, os, math, time, random
from pathlib import Path

# Cek PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    print("[ERROR] PyTorch belum terinstall.")
    print("Run setup.bat dulu, atau: pip install torch numpy")
    sys.exit(1)

# Auto-detect device
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"Device: CUDA ({gpu_name})")
    print(f"GPU Memory: {gpu_mem:.1f} GB total")
    print(f"GPU akan kepake pas training, abis itu bebas.\n")
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
    print("Device: MPS (Apple Silicon)")
else:
    DEVICE = torch.device('cpu')
    print("Device: CPU (GPU gak ke-detect, training akan lambat)")
    print("Pastikan NVIDIA driver terinstall dan CUDA available.")

# Import corpus
sys.path.insert(0, str(Path(__file__).parent))
try:
    from corpus_v010 import CORPUS
except ImportError:
    print("[ERROR] corpus_v010.py tidak ditemukan.")
    print("Pastikan file ada di folder yang sama dengan script ini.")
    sys.exit(1)

# ============================================================
# CONFIG (LEBIH BESAR untuk GPU!)
# ============================================================
CONFIG = {
    'vocab_size': 0,  # auto
    'd_model':    192,    # GPU kuat, naik dari 160
    'n_heads':    8,
    'd_ff':       768,
    'n_layers':   16,     # virtual depth 16 (naik dari 12, GRATIS via parameter sharing!)
    'max_seq':    96,     # context window naik dari 80
    'epochs':     2000,   # full training (GPU cepet)
    'batch_size': 32,     # GPU bisa handle batch besar
    'lr':         3e-4,
}

# Tokenizer
chars = sorted(set(CORPUS))
stoi = {ch: i for i, ch in enumerate(chars)}
stoi['<pad>'] = len(stoi)
itos = {i: ch for i, ch in enumerate(chars)}
itos[len(itos)] = '<pad>'
PAD_ID = stoi['<pad>']
EOS_ID = PAD_ID
VOCAB = len(stoi)
CONFIG['vocab_size'] = VOCAB

def encode(s):
    return [stoi[c] for c in s if c in stoi]
def decode(ids):
    return ''.join(itos.get(int(i), '?') for i in ids if i != PAD_ID)

def _save_checkpoint(model, CONFIG, stoi, itos, EOS_ID, OUT_DIR, final=False):
    """Save model checkpoint."""
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
    with open(OUT_DIR / 'dquilane_model.json', 'w', encoding='utf-8') as f:
        json.dump(weights, f, ensure_ascii=False)

# Insert EOS after each line
raw_ids = []
for line in CORPUS.split('\n'):
    raw_ids.extend(encode(line))
    raw_ids.append(EOS_ID)
data = torch.tensor(raw_ids, dtype=torch.long)
print(f"Vocab: {VOCAB} chars")
print(f"Corpus: {len(CORPUS)} chars, {len(data)} tokens (with EOS)")
print(f"Config: d={CONFIG['d_model']}, n_layers={CONFIG['n_layers']}, max_seq={CONFIG['max_seq']}")
print()

# Move data to device
data = data.to(DEVICE)

def get_batch():
    idx = torch.randint(0, len(data) - CONFIG['max_seq'] - 1, (CONFIG['batch_size'],))
    x = torch.stack([data[i:i+CONFIG['max_seq']] for i in idx])
    y = torch.stack([data[i+1:i+1+CONFIG['max_seq']] for i in idx])
    return x, y

# ============================================================
# Model
# ============================================================
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
    def generate(self, idx, max_new=50, temp=0.4, top_k=10, top_p=0.9, rep_penalty=1.2, stop_id=None):
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
n = sum(p.numel() for p in model.parameters())
print(f"D'QUILANE v0.10 GPU lahir!")
print(f"  Unique params: {n:,}")
print(f"  Virtual depth: {CONFIG['n_layers']}")
print(f"  Effective (no share): {n * CONFIG['n_layers']:,}")
print(f"  Vocab: {VOCAB}")

# Show GPU memory usage
if DEVICE.type == 'cuda':
    print(f"  GPU mem allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB / {gpu_mem:.1f} GB")
print()

# Output dir
OUT_DIR = Path(__file__).parent / "web"
OUT_DIR.mkdir(exist_ok=True)

# ============================================================
# PRE-TRAINING (with EOS in data!)
# ============================================================
print("=" * 60)
print(f"  PRE-TRAINING ({CONFIG['epochs']} epochs)")
print("=" * 60)
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
        gpu_info = ""
        if DEVICE.type == 'cuda':
            gpu_info = f"  GPU mem: {torch.cuda.memory_allocated()/1e9:.2f} GB"
        print(f'  epoch {epoch+1:4d}/{CONFIG["epochs"]}  loss={loss.item():.4f}  t={elapsed:.1f}s  eta={eta:.0f}s{gpu_info}', flush=True)
    # Checkpoint
    if (epoch + 1) % 500 == 0:
        _save_checkpoint(model, CONFIG, stoi, itos, EOS_ID, OUT_DIR)
        print(f'  [checkpoint] saved at epoch {epoch+1}', flush=True)

print(f'\nPre-training done in {time.time()-t0:.1f}s. Final loss: {loss.item():.4f}')

# ============================================================
# SFT (Supervised Fine-Tuning)
# ============================================================
print("\n" + "=" * 60)
print(f"  SFT (200 epochs, 213 examples)")
print("=" * 60)

SFT_EXAMPLES = [
    # Greetings
    ('Halo', 'Halo juga! Senang bertemu denganmu.'),
    ('halo', 'Halo juga! Senang bertemu denganmu.'),
    ('Hai', 'Hai! Apa kabar?'),
    ('Halo apa kabar?', 'Halo! Aku baik-baik saja, terima kasih.'),
    ('halo apa kabar', 'Halo! Aku baik-baik saja, terima kasih.'),
    ('Apa kabar?', 'Aku baik-baik saja, terima kasih sudah bertanya!'),
    ('apa kabar', 'Aku baik-baik saja, terima kasih sudah bertanya!'),
    ('Selamat pagi', 'Selamat pagi juga! Semoga harimu menyenangkan.'),
    ('selamat pagi', 'Selamat pagi juga! Semoga harimu menyenangkan.'),
    ('Selamat malam', 'Selamat malam! Selamat istirahat.'),
    # Identity
    ('Siapa kamu?', 'Aku D\'QUILANE, LLM buatan Indonesia dari nol.'),
    ('siapa kamu', 'Aku D\'QUILANE, LLM buatan Indonesia dari nol.'),
    ('Kamu siapa?', 'Aku D\'QUILANE, LLM buatan Indonesia dari nol.'),
    ('kamu siapa', 'Aku D\'QUILANE, LLM buatan Indonesia dari nol.'),
    ('Nama kamu siapa?', 'Namaku D\'QUILANE. Aku LLM multibahasa.'),
    ('nama kamu siapa', 'Namaku D\'QUILANE. Aku LLM multibahasa.'),
    ('Kamu dari mana?', 'Aku dari Indonesia.'),
    ('kamu dari mana', 'Aku dari Indonesia.'),
    # Emotions
    ('Apa itu senang?', 'Senang adalah perasaan gembira dan bahagia.'),
    ('Apa itu sedih?', 'Sedih adalah perasaan kecewa atau kehilangan.'),
    ('Apa itu marah?', 'Marah adalah perasaan tidak suka atau kesal.'),
    ('Apa itu takut?', 'Takut adalah perasaan cemas atau khawatir.'),
    ('Apa itu lelah?', 'Lelah adalah perasaan capek atau kekurangan energi.'),
    ('Apa itu lapar?', 'Lapar adalah perasaan ingin makan.'),
    ('Apa itu haus?', 'Haus adalah perasaan ingin minum.'),
    # Body parts
    ('Apa itu kepala?', 'Kepala adalah bagian atas tubuh tempat otak.'),
    ('Apa fungsi mata?', 'Mata adalah alat untuk melihat.'),
    ('Apa fungsi hidung?', 'Hidung adalah alat untuk mencium bau.'),
    ('Apa fungsi mulut?', 'Mulut adalah alat untuk makan dan bicara.'),
    ('Apa fungsi telinga?', 'Telinga adalah alat untuk mendengar.'),
    ('Apa fungsi tangan?', 'Tangan adalah alat untuk memegang.'),
    ('Apa fungsi kaki?', 'Kaki adalah alat untuk berjalan.'),
    ('Berapa jari di tangan?', 'Ada lima jari di setiap tangan.'),
    ('Berapa jari di kaki?', 'Ada lima jari di setiap kaki.'),
    # Colors
    ('Apa warna darah?', 'Warna darah adalah merah.'),
    ('Apa warna langit?', 'Warna langit adalah biru.'),
    ('Apa warna matahari?', 'Warna matahari adalah kuning.'),
    ('Apa warna rumput?', 'Warna rumput adalah hijau.'),
    ('Apa warna awan?', 'Warna awan adalah putih.'),
    ('Apa warna arang?', 'Warna arang adalah hitam.'),
    ('Apa warna anggrek?', 'Warna anggrek adalah ungu.'),
    ('Apa warna jeruk?', 'Warna jeruk adalah oranye.'),
    # Shapes
    ('Apa bentuk bola?', 'Bentuk bola adalah lingkaran.'),
    ('Apa bentuk buku?', 'Bentuk buku adalah persegi.'),
    ('Berapa sisi segitiga?', 'Segitiga punya tiga sisi.'),
    ('Berapa sisi persegi?', 'Persegi punya empat sisi.'),
    # Animals + sounds
    ('Apa suara kucing?', 'Kucing bersuara meong.'),
    ('Apa suara anjing?', 'Anjing bersuara guk guk.'),
    ('Apa suara ayam jantan?', 'Ayam jantan bersuara kukuruyuk.'),
    ('Apa suara bebek?', 'Bebek bersuara kwek kwek.'),
    ('Apa suara sapi?', 'Sapi bersuara moo.'),
    ('Apa suara kambing?', 'Kambing bersuara mbek mbek.'),
    ('Apa suara kuda?', 'Kuda bersuara igih igih.'),
    ('Apa makanan kucing?', 'Kucing suka ikan.'),
    ('Apa makanan sapi?', 'Sapi suka rumput.'),
    ('Apa yang dihasilkan sapi?', 'Sapi memberi susu.'),
    ('Apa hewan terbesar di darat?', 'Gajah adalah hewan terbesar di darat.'),
    ('Siapa raja hutan?', 'Singa adalah raja hutan.'),
    ('Hewan apa yang bisa terbang?', 'Burung bisa terbang.'),
    ('Hewan apa yang hidup di air?', 'Ikan hidup di air.'),
    # Family
    ('Siapa ayah dan ibu?', 'Ayah dan ibu adalah orang tua.'),
    ('Siapa kakak?', 'Kakak adalah saudara yang lebih tua.'),
    ('Siapa adik?', 'Adik adalah saudara yang lebih muda.'),
    ('Siapa kakek?', 'Kakek adalah ayah dari orang tua.'),
    ('Siapa nenek?', 'Nenek adalah ibu dari orang tua.'),
    ('Siapa paman?', 'Paman adalah saudara laki-laki orang tua.'),
    ('Siapa tante?', 'Tante adalah saudara perempuan orang tua.'),
    ('Siapa sepupu?', 'Sepupu adalah anak paman atau tante.'),
    # Social rules
    ('Apa itu antre?', 'Antre adalah menunggu giliran dengan tertib.'),
    ('Apa itu bergiliran?', 'Bergiliran adalah mengambil giliran satu per satu.'),
    ('Apa itu berbagi?', 'Berbagi adalah memberi sebagian ke orang lain.'),
    # Time
    ('Berapa menit dalam satu jam?', 'Ada enam puluh menit dalam satu jam.'),
    ('Berapa detik dalam satu menit?', 'Ada enam puluh detik dalam satu menit.'),
    ('Berapa jam dalam satu hari?', 'Ada dua puluh empat jam dalam satu hari.'),
    ('Berapa hari dalam seminggu?', 'Ada tujuh hari dalam seminggu.'),
    ('Berapa bulan dalam setahun?', 'Ada dua belas bulan dalam setahun.'),
    ('Setelah Senin adalah?', 'Setelah Senin adalah Selasa.'),
    ('Setelah Januari adalah?', 'Setelah Januari adalah Februari.'),
    # Money
    ('Apa mata uang Indonesia?', 'Mata uang Indonesia adalah rupiah.'),
    ('Apa mata uang Jepang?', 'Mata uang Jepang adalah yen.'),
    ('Apa mata uang Amerika?', 'Mata uang Amerika adalah dolar.'),
    ('Apa mata uang Eropa?', 'Mata uang Eropa adalah euro.'),
    # Weather
    ('Apa itu hujan?', 'Hujan adalah air yang turun dari langit.'),
    ('Apa itu panas?', 'Panas adalah cuaca yang sangat hangat.'),
    ('Apa itu dingin?', 'Dingin adalah cuaca yang sangat sejuk.'),
    ('Kapan pelangi muncul?', 'Pelangi muncul setelah hujan dengan sinar matahari.'),
    # Plants
    ('Apa fungsi akar?', 'Akar menyerap air dari tanah.'),
    ('Apa fungsi batang?', 'Batang menjaga tanaman tetap tegak.'),
    ('Apa fungsi daun?', 'Daun membuat makanan untuk tanaman.'),
    ('Apa yang dibutuhkan tanaman?', 'Tanaman butuh air, sinar matahari, dan udara.'),
    # Tech
    ('Apa itu komputer?', 'Komputer adalah alat untuk menghitung dan bekerja.'),
    ('Apa itu tablet?', 'Tablet adalah komputer kecil layar sentuh.'),
    ('Apa itu YouTube?', 'YouTube adalah situs berbagi video.'),
    ('Apa itu Minecraft?', 'Minecraft adalah permainan membangun dengan balok.'),
    ('Apa itu internet?', 'Internet adalah jaringan yang menghubungkan komputer.'),
    # Maps
    ('Matahari terbit di?', 'Matahari terbit di timur.'),
    ('Matahari terbenam di?', 'Matahari terbenam di barat.'),
    ('Apa itu kompas?', 'Kompas adalah alat penunjuk arah.'),
    # Daily routines
    ('Apa itu mandi?', 'Mandi adalah membersihkan tubuh dengan air.'),
    ('Apa itu sarapan?', 'Sarapan adalah makan pagi sebelum aktivitas.'),
    ('Apa itu tidur?', 'Tidur adalah beristirahat di malam hari.'),
    ('Apa itu sekolah?', 'Sekolah adalah tempat belajar.'),
    ('Apa itu PR?', 'PR adalah pekerjaan rumah dari sekolah.'),
    # Reading
    ('Sebutkan huruf A sampai Z', 'A B C D E F G H I J K L M N O P Q R S T U V W X Y Z.'),
    ('Apa itu huruf?', 'Huruf adalah lambang bunyi dalam bahasa.'),
    ('Apa itu alfabet?', 'Alfabet adalah susunan huruf dari A sampai Z.'),
    # Math
    ('Satu tambah satu?', 'Satu tambah satu sama dengan dua.'),
    ('satu tambah satu', 'Satu tambah satu sama dengan dua.'),
    ('Dua tambah dua?', 'Dua tambah dua sama dengan empat.'),
    ('Lima tambah lima?', 'Lima tambah lima sama dengan sepuluh.'),
    ('Dua dikali dua?', 'Dua dikali dua sama dengan empat.'),
    ('Lima dikali lima?', 'Lima dikali lima sama dengan dua puluh lima.'),
    ('Tiga dikali tiga?', 'Tiga dikali tiga sama dengan sembilan.'),
    ('Sepuluh dibagi dua?', 'Sepuluh dibagi dua sama dengan lima.'),
    ('Sembilan dibagi tiga?', 'Sembilan dibagi tiga sama dengan tiga.'),
    ('Sepuluh dikurangi lima?', 'Sepuluh dikurangi lima sama dengan lima.'),
    # Word problems
    ('Andi punya tiga apel. Ibu memberi dua apel. Berapa total?', 'Total apel Andi adalah lima.'),
    ('Sari punya sepuluh permen. Dia makan empat. Berapa sisa?', 'Sisa permennya enam.'),
    # Capitals
    ('Ibu kota Indonesia?', 'Ibu kota Indonesia adalah Jakarta.'),
    ('ibu kota indonesia', 'Ibu kota Indonesia adalah Jakarta.'),
    ('Apa ibu kota Jepang?', 'Ibu kota Jepang adalah Tokyo.'),
    ('ibu kota jepang apa', 'Ibu kota Jepang adalah Tokyo.'),
    ('Apa ibu kota Prancis?', 'Ibu kota Prancis adalah Paris.'),
    ('Apa ibu kota Inggris?', 'Ibu kota Inggris adalah London.'),
    ('Apa ibu kota Spanyol?', 'Ibu kota Spanyol adalah Madrid.'),
    ('Apa ibu kota Jerman?', 'Ibu kota Jerman adalah Berlin.'),
    ('Apa ibu kota Italia?', 'Ibu kota Italia adalah Roma.'),
    ('Apa ibu kota Rusia?', 'Ibu kota Rusia adalah Moskow.'),
    ('Apa ibu kota Amerika?', 'Ibu kota Amerika adalah Washington DC.'),
    ('Apa ibu kota Tiongkok?', 'Ibu kota Tiongkok adalah Beijing.'),
    ('Apa ibu kota Korea Selatan?', 'Ibu kota Korea Selatan adalah Seoul.'),
    # Reverse capitals
    ('Tokyo itu apa?', 'Tokyo adalah ibu kota Jepang.'),
    ('tokyo itu apa', 'Tokyo adalah ibu kota Jepang.'),
    ('Jakarta itu apa?', 'Jakarta adalah ibu kota Indonesia.'),
    ('Paris itu apa?', 'Paris adalah ibu kota Prancis.'),
    ('London itu apa?', 'London adalah ibu kota Inggris.'),
    # Presidents
    ('Siapa presiden pertama Indonesia?', 'Presiden pertama Indonesia adalah Soekarno.'),
    ('siapa presiden pertama indonesia', 'Presiden pertama Indonesia adalah Soekarno.'),
    ('Siapa presiden kedua Indonesia?', 'Presiden kedua Indonesia adalah Soeharto.'),
    ('Siapa presiden Indonesia?', 'Presiden Indonesia dipilih melalui pemilu demokratis.'),
    ('Siapa bapak proklamasi Indonesia?', 'Bapak proklamasi Indonesia adalah Soekarno dan Hatta.'),
    # Geography
    ('Berapa benua di dunia?', 'Ada tujuh benua di dunia.'),
    ('Apa gunung tertinggi di dunia?', 'Gunung Everest adalah gunung tertinggi di dunia.'),
    # Indonesia
    ('Apa itu Indonesia?', 'Indonesia adalah negara kepulauan di Asia Tenggara.'),
    ('Apa bendera Indonesia?', 'Bendera Indonesia adalah merah putih.'),
    ('Apa lagu kebangsaan Indonesia?', 'Lagu kebangsaan Indonesia adalah Indonesia Raya.'),
    ('Kapan Indonesia merdeka?', 'Indonesia merdeka pada tahun seribu sembilan ratus empat puluh lima.'),
    # Translations
    ('Translate halo to English', 'Halo in English is Hello.'),
    ('Translate hello to Indonesian', 'Hello in Indonesian is Halo.'),
    ('Translate thank you to Indonesian', 'Thank you in Indonesian is Terima kasih.'),
    ('Translate terima kasih to English', 'Terima kasih in English is Thank you.'),
    # Patterns
    ('Pola satu dua tiga empat. Berikutnya?', 'Berikutnya adalah lima.'),
    ('Pola dua empat enam delapan. Berikutnya?', 'Berikutnya adalah sepuluh.'),
    # Fallbacks
    ('Siapa presiden Amerika?', 'Presiden Amerika dipilih melalui pemilu demokratis.'),
    ('Siapa presiden Rusia?', 'Presiden Rusia dipilih melalui pemilu.'),
    ('Apa berita hari ini?', 'Aku tidak tahu berita terbaru, cek sumber berita terpercaya.'),
    ('Ceritakan cerita', 'Aku belum bisa bercerita panjang, tapi aku bisa jawab pertanyaan.'),
    ('Bagaimana cuaca hari ini?', 'Aku tidak tahu cuaca hari ini, cek aplikasi cuaca.'),
    ('Jam berapa sekarang', 'Aku tidak tahu jam berapa sekarang, cek jam.'),
    ('kamu bisa apa', 'Aku bisa matematika dasar, terjemahan, dan menjawab pertanyaan sederhana.'),
    ('lagi apa', 'Aku siap membantu. Ada yang bisa aku bantu?'),
    ('coba ngomong', 'Halo! Aku D\'QUILANE. Ada yang bisa aku bantu?'),
    ('test', 'Halo! Aku D\'QUILANE. Ada yang bisa aku bantu?'),
    ('halo dunia', 'Halo! Aku D\'QUILANE, LLM buatan Indonesia.'),
    ('jepang di mana', 'Jepang adalah negara di Asia Timur.'),
    # Politeness
    ('Terima kasih', 'Sama-sama! Ada yang bisa aku bantu lagi?'),
    ('terima kasih', 'Sama-sama! Ada yang bisa aku bantu lagi?'),
    ('Makasih', 'Sama-sama!'),
    ('Goodbye', 'Sampai jumpa! Semoga harimu menyenangkan.'),
    # Counting
    ('Count to five', 'Satu dua tiga empat lima.'),
    ('Count to ten', 'Satu dua tiga empat lima enam tujuh delapan sembilan sepuluh.'),
]
print(f"SFT examples: {len(SFT_EXAMPLES)}")

# Encode SFT
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
    encoded_sft.append((torch.tensor(ids[:-1], dtype=torch.long, device=DEVICE),
                         torch.tensor(targets, dtype=torch.long, device=DEVICE)))

sft_opt = torch.optim.AdamW(model.parameters(), lr=SFT_LR, weight_decay=0.01)
sft_sched = torch.optim.lr_scheduler.CosineAnnealingLR(sft_opt, T_max=SFT_EPOCHS)

model.train()
t0 = time.time()
for epoch in range(SFT_EPOCHS):
    random.shuffle(encoded_sft)
    total_loss = 0
    for x, y in encoded_sft:
        sft_opt.zero_grad()
        _, loss = model(x.unsqueeze(0), y.unsqueeze(0))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        sft_opt.step()
        total_loss += loss.item()
    sft_sched.step()
    if epoch % 20 == 0 or epoch == SFT_EPOCHS - 1:
        elapsed = time.time() - t0
        print(f'  sft epoch {epoch+1:3d}/{SFT_EPOCHS}  loss={total_loss/len(encoded_sft):.4f}  t={elapsed:.1f}s', flush=True)
    if (epoch + 1) % 50 == 0:
        _save_checkpoint(model, CONFIG, stoi, itos, EOS_ID, OUT_DIR)
        print(f'  [checkpoint] saved at sft epoch {epoch+1}', flush=True)

print(f'\nSFT done in {time.time()-t0:.1f}s')
model.eval()

# Test
print("\n=== TEST ===")
test_prompts = ['Halo', 'Satu tambah satu?', 'Apa suara kucing?', 'tokyo itu apa',
                'Apa warna langit?', 'Siapa presiden pertama Indonesia?']
for p in test_prompts:
    ids = torch.tensor([encode(p)], dtype=torch.long, device=DEVICE)
    out = model.generate(ids, max_new=80, temp=0.4, top_k=10, top_p=0.9, rep_penalty=1.2, stop_id=EOS_ID)
    text = decode(out[0].tolist())
    print(f"[{p}] -> {text}")

# Save final model
_save_checkpoint(model, CONFIG, stoi, itos, EOS_ID, OUT_DIR, final=True)
print(f"\nFinal model saved to: {OUT_DIR / 'dquilane_model.json'}")
print(f"File size: {(OUT_DIR / 'dquilane_model.json').stat().st_size / 1024:.1f} KB")

# Release GPU memory
if DEVICE.type == 'cuda':
    del model
    torch.cuda.empty_cache()
    print(f"\nGPU memory released. Current: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print("\nDone! Sekarang run serve.bat buat test local web UI.")
