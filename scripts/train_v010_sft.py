"""D'QUILANE v0.10 SFT - 200+ examples covering ALL topics"""
import sys, json, os, math, time, random
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/home/z/my-project/scripts")
from corpus_v010 import CORPUS

DEVICE = 'cpu'

JSON_PATH = "/home/z/my-project/download/llm/web/dquilane_model.json"
with open(JSON_PATH, 'r', encoding='utf-8') as f:
    model_data = json.load(f)

CONFIG = model_data['config']
stoi = model_data['stoi']
itos = {int(k): v for k, v in model_data['itos'].items()}
PAD_ID = stoi.get('<pad>', 0)
EOS_ID = PAD_ID
print(f"Loaded. Vocab: {CONFIG['vocab_size']}, EOS: {EOS_ID}")

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

# === MASSIVE SFT — 220+ examples covering ALL cognitive dev topics ===
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
    # EMOTIONS (NEW!)
    ('Apa itu senang?', 'Senang adalah perasaan gembira dan bahagia.'),
    ('Apa itu sedih?', 'Sedih adalah perasaan kecewa atau kehilangan.'),
    ('Apa itu marah?', 'Marah adalah perasaan tidak suka atau kesal.'),
    ('Apa itu takut?', 'Takut adalah perasaan cemas atau khawatir.'),
    ('Apa itu lelah?', 'Lelah adalah perasaan capek atau kekurangan energi.'),
    ('Apa itu lapar?', 'Lapar adalah perasaan ingin makan.'),
    ('Apa itu haus?', 'Haus adalah perasaan ingin minum.'),
    ('Kamu sedang apa?', 'Aku siap membantu. Ada yang bisa aku bantu?'),
    # BODY PARTS (NEW!)
    ('Apa itu kepala?', 'Kepala adalah bagian atas tubuh tempat otak.'),
    ('Apa fungsi mata?', 'Mata adalah alat untuk melihat.'),
    ('Apa fungsi hidung?', 'Hidung adalah alat untuk mencium bau.'),
    ('Apa fungsi mulut?', 'Mulut adalah alat untuk makan dan bicara.'),
    ('Apa fungsi telinga?', 'Telinga adalah alat untuk mendengar.'),
    ('Apa fungsi tangan?', 'Tangan adalah alat untuk memegang.'),
    ('Apa fungsi kaki?', 'Kaki adalah alat untuk berjalan.'),
    ('Apa fungsi gigi?', 'Gigi adalah untuk mengunyah makanan.'),
    ('Berapa jari di tangan?', 'Ada lima jari di setiap tangan.'),
    ('Berapa jari di kaki?', 'Ada lima jari di setiap kaki.'),
    # COLORS (NEW!)
    ('Apa warna darah?', 'Warna darah adalah merah.'),
    ('Apa warna langit?', 'Warna langit adalah biru.'),
    ('Apa warna matahari?', 'Warna matahari adalah kuning.'),
    ('Apa warna rumput?', 'Warna rumput adalah hijau.'),
    ('Apa warna awan?', 'Warna awan adalah putih.'),
    ('Apa warna arang?', 'Warna arang adalah hitam.'),
    ('Apa warna anggrek?', 'Warna anggrek adalah ungu.'),
    ('Apa warna jeruk?', 'Warna jeruk adalah oranye.'),
    ('Apa warna apel merah?', 'Warna apel merah adalah merah.'),
    ('Apa warna coklat?', 'Warna coklat adalah coklat.'),
    # SHAPES (NEW!)
    ('Apa bentuk bola?', 'Bentuk bola adalah lingkaran.'),
    ('Apa bentuk buku?', 'Bentuk buku adalah persegi.'),
    ('Apa bentuk telur?', 'Bentuk telur adalah oval.'),
    ('Berapa sisi segitiga?', 'Segitiga punya tiga sisi.'),
    ('Berapa sisi persegi?', 'Persegi punya empat sisi.'),
    # SIZES (NEW!)
    ('Lebih besar gajah atau semut?', 'Gajah lebih besar dari semut.'),
    ('Lebih kecil semut atau gajah?', 'Semut lebih kecil dari gajah.'),
    ('Lebih tinggi gunung atau rumah?', 'Gunung lebih tinggi dari rumah.'),
    ('Lebih pendek rumah atau gunung?', 'Rumah lebih pendek dari gunung.'),
    ('Lebih berat batu atau kapas?', 'Batu lebih berat dari kapas.'),
    # ANIMALS + SOUNDS (NEW!)
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
    ('Apa yang dihasilkan lebah?', 'Lebah membuat madu.'),
    ('Apa hewan terbesar di darat?', 'Gajah adalah hewan terbesar di darat.'),
    ('Siapa raja hutan?', 'Singa adalah raja hutan.'),
    ('Hewan apa yang bisa terbang?', 'Burung bisa terbang.'),
    ('Hewan apa yang hidup di air?', 'Ikan hidup di air.'),
    ('Kupu-kupu berasal dari?', 'Kupu-kupu berasal dari ulat.'),
    # FAMILY (NEW!)
    ('Siapa ayah dan ibu?', 'Ayah dan ibu adalah orang tua.'),
    ('Siapa kakak?', 'Kakak adalah saudara yang lebih tua.'),
    ('Siapa adik?', 'Adik adalah saudara yang lebih muda.'),
    ('Siapa kakek?', 'Kakek adalah ayah dari orang tua.'),
    ('Siapa nenek?', 'Nenek adalah ibu dari orang tua.'),
    ('Siapa paman?', 'Paman adalah saudara laki-laki orang tua.'),
    ('Siapa tante?', 'Tante adalah saudara perempuan orang tua.'),
    ('Siapa sepupu?', 'Sepupu adalah anak paman atau tante.'),
    # PRETEND PLAY (NEW!)
    ('Pura-pura jadi dokter?', 'Dokter memeriksa pasien.'),
    ('Pura-pura jadi guru?', 'Guru mengajar murid.'),
    ('Pura-pura jadi superhero?', 'Superhero menyelamatkan orang.'),
    ('Pura-pura jadi pilot?', 'Pilot menerbangkan pesawat.'),
    ('Pura-pura jadi polisi?', 'Polisi menangkap pencuri.'),
    # SOCIAL RULES (NEW!)
    ('Apa itu antre?', 'Antre adalah menunggu giliran dengan tertib.'),
    ('Apa itu bergiliran?', 'Bergiliran adalah mengambil giliran satu per satu.'),
    ('Apa itu berbagi?', 'Berbagi adalah memberi sebagian ke orang lain.'),
    ('Apa kata untuk minta bantuan?', 'Tolong adalah kata sopan minta bantuan.'),
    ('Apa balasan terima kasih?', 'Sama-sama adalah balasan untuk terima kasih.'),
    ('Apa kata untuk minta maaf?', 'Maaf adalah kata untuk minta permisi.'),
    # TIME
    ('Berapa menit dalam satu jam?', 'Ada enam puluh menit dalam satu jam.'),
    ('Berapa detik dalam satu menit?', 'Ada enam puluh detik dalam satu menit.'),
    ('Berapa jam dalam satu hari?', 'Ada dua puluh empat jam dalam satu hari.'),
    ('Berapa hari dalam seminggu?', 'Ada tujuh hari dalam seminggu.'),
    ('Berapa bulan dalam setahun?', 'Ada dua belas bulan dalam setahun.'),
    ('Setelah Senin adalah?', 'Setelah Senin adalah Selasa.'),
    ('Setelah Januari adalah?', 'Setelah Januari adalah Februari.'),
    ('Apa jarum pendek jam?', 'Jarum pendek jam menunjukkan jam.'),
    ('Apa jarum panjang jam?', 'Jarum panjang jam menunjukkan menit.'),
    # MONEY
    ('Apa mata uang Indonesia?', 'Mata uang Indonesia adalah rupiah.'),
    ('Apa mata uang Jepang?', 'Mata uang Jepang adalah yen.'),
    ('Apa mata uang Amerika?', 'Mata uang Amerika adalah dolar.'),
    ('Apa mata uang Eropa?', 'Mata uang Eropa adalah euro.'),
    ('Apa itu kembalian?', 'Kembalian adalah uang kembali setelah bayar.'),
    ('Sepuluh ribu atau seribu lebih besar?', 'Sepuluh ribu lebih besar dari seribu.'),
    # WEATHER (NEW!)
    ('Apa itu hujan?', 'Hujan adalah air yang turun dari langit.'),
    ('Apa itu panas?', 'Panas adalah cuaca yang sangat hangat.'),
    ('Apa itu dingin?', 'Dingin adalah cuaca yang sangat sejuk.'),
    ('Apa itu mendung?', 'Mendung adalah awan tebal sebelum hujan.'),
    ('Kapan pelangi muncul?', 'Pelangi muncul setelah hujan dengan sinar matahari.'),
    # PLANTS (NEW!)
    ('Apa fungsi akar?', 'Akar menyerap air dari tanah.'),
    ('Apa fungsi batang?', 'Batang menjaga tanaman tetap tegak.'),
    ('Apa fungsi daun?', 'Daun membuat makanan untuk tanaman.'),
    ('Apa yang dibutuhkan tanaman?', 'Tanaman butuh air, sinar matahari, dan udara.'),
    # TECHNOLOGY (NEW!)
    ('Apa itu komputer?', 'Komputer adalah alat untuk menghitung dan bekerja.'),
    ('Apa itu tablet?', 'Tablet adalah komputer kecil layar sentuh.'),
    ('Apa itu swipe?', 'Swipe adalah menggeser layar dengan jari.'),
    ('Apa itu YouTube?', 'YouTube adalah situs berbagi video.'),
    ('Apa itu Minecraft?', 'Minecraft adalah permainan membangun dengan balok.'),
    ('Apa itu Roblox?', 'Roblox adalah platform permainan online.'),
    ('Apa itu internet?', 'Internet adalah jaringan yang menghubungkan komputer.'),
    # MAPS (NEW!)
    ('Arah apa di atas peta?', 'Utara adalah arah ke atas di peta.'),
    ('Arah apa di bawah peta?', 'Selatan adalah arah ke bawah di peta.'),
    ('Matahari terbit di?', 'Matahari terbit di timur.'),
    ('Matahari terbenam di?', 'Matahari terbenam di barat.'),
    ('Apa itu kompas?', 'Kompas adalah alat penunjuk arah.'),
    # DAILY ROUTINES (NEW!)
    ('Apa itu mandi?', 'Mandi adalah membersihkan tubuh dengan air.'),
    ('Apa itu sarapan?', 'Sarapan adalah makan pagi sebelum aktivitas.'),
    ('Apa itu tidur?', 'Tidur adalah beristirahat di malam hari.'),
    ('Apa itu sekolah?', 'Sekolah adalah tempat belajar.'),
    ('Apa itu PR?', 'PR adalah pekerjaan rumah dari sekolah.'),
    ('Apa itu belajar?', 'Belajar adalah menambah pengetahuan.'),
    # READING/WRITING (NEW!)
    ('Apa itu huruf?', 'Huruf adalah lambang bunyi dalam bahasa.'),
    ('Apa itu alfabet?', 'Alfabet adalah susunan huruf dari A sampai Z.'),
    ('Sebutkan huruf A sampai Z', 'A B C D E F G H I J K L M N O P Q R S T U V W X Y Z.'),
    ('Apa fungsi pensil?', 'Pensil adalah alat tulis dari kayu dan grafit.'),
    ('Apa itu buku?', 'Buku adalah kumpulan kertas bertulisan.'),
    ('Apa itu baca?', 'Baca adalah melihat dan memahami tulisan.'),
    ('Apa itu tulis?', 'Tulis adalah membuat tulisan dengan alat.'),
    # COUNTING
    ('Count to five', 'Satu dua tiga empat lima.'),
    ('Count to ten', 'Satu dua tiga empat lima enam tujuh delapan sembilan sepuluh.'),
    ('Hitung sampai lima', 'Satu dua tiga empat lima.'),
    ('Hitung sampai sepuluh', 'Satu dua tiga empat lima enam tujuh delapan sembilan sepuluh.'),
    # MATH
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
    # WORD PROBLEMS
    ('Andi punya tiga apel. Ibu memberi dua apel. Berapa total?', 'Total apel Andi adalah lima.'),
    ('Sari punya sepuluh permen. Dia makan empat. Berapa sisa?', 'Sisa permennya enam.'),
    # CAPITALS
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
    # REVERSE CAPITALS
    ('Tokyo itu apa?', 'Tokyo adalah ibu kota Jepang.'),
    ('tokyo itu apa', 'Tokyo adalah ibu kota Jepang.'),
    ('Jakarta itu apa?', 'Jakarta adalah ibu kota Indonesia.'),
    ('Paris itu apa?', 'Paris adalah ibu kota Prancis.'),
    ('London itu apa?', 'London adalah ibu kota Inggris.'),
    # PRESIDENTS
    ('Siapa presiden pertama Indonesia?', 'Presiden pertama Indonesia adalah Soekarno.'),
    ('siapa presiden pertama indonesia', 'Presiden pertama Indonesia adalah Soekarno.'),
    ('Siapa presiden kedua Indonesia?', 'Presiden kedua Indonesia adalah Soeharto.'),
    ('Siapa presiden Indonesia?', 'Presiden Indonesia dipilih melalui pemilu demokratis.'),
    ('Siapa bapak proklamasi Indonesia?', 'Bapak proklamasi Indonesia adalah Soekarno dan Hatta.'),
    ('Siapa wakil presiden pertama Indonesia?', 'Wakil presiden pertama Indonesia adalah Hatta.'),
    # GEOGRAPHY
    ('Berapa benua di dunia?', 'Ada tujuh benua di dunia.'),
    ('Apa gunung tertinggi di dunia?', 'Gunung Everest adalah gunung tertinggi di dunia.'),
    ('Apa samudra terbesar di dunia?', 'Samudra Pasifik adalah samudra terbesar di dunia.'),
    # INDONESIA FACTS
    ('Apa itu Indonesia?', 'Indonesia adalah negara kepulauan di Asia Tenggara.'),
    ('Apa bendera Indonesia?', 'Bendera Indonesia adalah merah putih.'),
    ('Apa lagu kebangsaan Indonesia?', 'Lagu kebangsaan Indonesia adalah Indonesia Raya.'),
    ('Kapan Indonesia merdeka?', 'Indonesia merdeka pada tahun seribu sembilan ratus empat puluh lima.'),
    # TRANSLATIONS
    ('Translate halo to English', 'Halo in English is Hello.'),
    ('Translate hello to Indonesian', 'Hello in Indonesian is Halo.'),
    ('Translate thank you to Indonesian', 'Thank you in Indonesian is Terima kasih.'),
    ('Translate terima kasih to English', 'Terima kasih in English is Thank you.'),
    ('Hello in Spanish?', 'Hello in Spanish is Hola.'),
    ('Hello in French?', 'Hello in French is Bonjour.'),
    # PATTERNS
    ('Pola satu dua tiga empat. Berikutnya?', 'Berikutnya adalah lima.'),
    ('Pola dua empat enam delapan. Berikutnya?', 'Berikutnya adalah sepuluh.'),
    # FALLBACKS
    ('Siapa presiden Amerika?', 'Presiden Amerika dipilih melalui pemilu demokratis.'),
    ('Siapa presiden Rusia?', 'Presiden Rusia dipilih melalui pemilu.'),
    ('Apa berita hari ini?', 'Aku tidak tahu berita terbaru, cek sumber berita terpercaya.'),
    ('Ceritakan cerita', 'Aku belum bisa bercerita panjang, tapi aku bisa jawab pertanyaan.'),
    ('Bagaimana cuaca hari ini?', 'Aku tidak tahu cuaca hari ini, cek aplikasi cuaca.'),
    ('Jam berapa sekarang', 'Aku tidak tahu jam berapa sekarang, cek jam.'),
    ('kamu bisa apa', 'Aku bisa matematika dasar, terjemahan, dan menjawab pertanyaan sederhana.'),
    ('apa yang kamu bisa', 'Aku bisa matematika dasar, terjemahan, dan menjawab pertanyaan sederhana.'),
    ('lagi apa', 'Aku siap membantu. Ada yang bisa aku bantu?'),
    ('coba ngomong', 'Halo! Aku D\'QUILANE. Ada yang bisa aku bantu?'),
    ('test', 'Halo! Aku D\'QUILANE. Ada yang bisa aku bantu?'),
    ('halo dunia', 'Halo! Aku D\'QUILANE, LLM buatan Indonesia.'),
    ('jepang di mana', 'Jepang adalah negara di Asia Timur.'),
    ('berbicara', 'Aku bisa berbicara dalam banyak bahasa.'),
    # POLITENESS
    ('Terima kasih', 'Sama-sama! Ada yang bisa aku bantu lagi?'),
    ('terima kasih', 'Sama-sama! Ada yang bisa aku bantu lagi?'),
    ('Makasih', 'Sama-sama!'),
    ('Goodbye', 'Sampai jumpa! Semoga harimu menyenangkan.'),
]
print(f"SFT examples: {len(SFT_EXAMPLES)}")

SFT_EPOCHS = 150
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
    if epoch % 15 == 0 or epoch == SFT_EPOCHS - 1:
        print(f'  sft epoch {epoch+1:3d}/{SFT_EPOCHS}  loss={total_loss/len(encoded_sft):.4f}  t={time.time()-t0:.1f}s', flush=True)
    if (epoch + 1) % 30 == 0:
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

# Test - all topic areas
print("\n=== v0.10 TEST (all topics) ===")
test_prompts = [
    # Emotions
    'Apa itu senang?', 'Apa itu takut?',
    # Body parts
    'Apa fungsi mata?', 'Berapa jari di tangan?',
    # Colors
    'Apa warna langit?', 'Apa warna rumput?',
    # Animals
    'Apa suara kucing?', 'Apa suara sapi?',
    # Family
    'Siapa kakek?', 'Siapa paman?',
    # Plants
    'Apa fungsi akar?', 'Apa yang dibutuhkan tanaman?',
    # Tech
    'Apa itu komputer?', 'Apa itu YouTube?',
    # Maps
    'Matahari terbit di?', 'Apa itu kompas?',
    # Daily routines
    'Apa itu mandi?', 'Apa itu PR?',
    # Reading
    'Sebutkan huruf A sampai Z',
    # Standard
    'Halo', 'Satu tambah satu?', 'Siapa presiden pertama Indonesia?',
    'tokyo itu apa', 'ibu kota jepang apa',
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
