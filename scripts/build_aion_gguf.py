"""Quick: add AION engram to model.json + export GGUF"""
import json, os, sys
sys.path.insert(0, '/home/z/my-project/scripts')

# Load current model
MODEL_PATH = "/home/z/my-project/download/llm/web/dquilane_model.json"
with open(MODEL_PATH, 'r', encoding='utf-8') as f:
    model = json.load(f)

print(f"Architecture: {model.get('architecture', 'unknown')}")
print(f"Techniques: {model.get('techniques', [])}")
print(f"Vocab: {model['config']['vocab_size']}")
print(f"d_model: {model['config']['d_model']}")

# SFT examples (same as in training script)
SFT_EXAMPLES = [
    ('Halo', 'Halo juga! Senang bertemu denganmu.'),
    ('halo', 'Halo juga! Senang bertemu denganmu.'),
    ('Hai', 'Hai! Apa kabar?'),
    ('hai', 'Hai! Apa kabar?'),
    ('Halo apa kabar?', 'Halo! Aku baik-baik saja, terima kasih.'),
    ('halo apa kabar', 'Halo! Aku baik-baik saja, terima kasih.'),
    ('Apa kabar?', 'Aku baik-baik saja, terima kasih sudah bertanya!'),
    ('apa kabar', 'Aku baik-baik saja, terima kasih sudah bertanya!'),
    ('Selamat pagi', 'Selamat pagi juga! Semoga harimu menyenangkan.'),
    ('selamat pagi', 'Selamat pagi juga! Semoga harimu menyenangkan.'),
    ('Selamat malam', 'Selamat malam! Selamat istirahat.'),
    ('Siapa kamu?', 'Aku D\'QUILANE Blueprint v2.1, LLM hybrid dengan teknik frontier.'),
    ('siapa kamu', 'Aku D\'QUILANE Blueprint v2.1, LLM hybrid dengan teknik frontier.'),
    ('Kamu siapa?', 'Aku D\'QUILANE Blueprint v2.1, LLM hybrid dengan teknik frontier.'),
    ('kamu siapa', 'Aku D\'QUILANE Blueprint v2.1, LLM hybrid dengan teknik frontier.'),
    ('Nama kamu siapa?', 'Namaku D\'QUILANE. Aku LLM multibahasa.'),
    ('nama kamu siapa', 'Namaku D\'QUILANE. Aku LLM multibahasa.'),
    ('Kamu dari mana?', 'Aku dari Indonesia.'),
    ('kamu dari mana', 'Aku dari Indonesia.'),
    ('Satu tambah satu?', 'Satu tambah satu sama dengan dua.'),
    ('satu tambah satu', 'Satu tambah satu sama dengan dua.'),
    ('Dua tambah dua?', 'Dua tambah dua sama dengan empat.'),
    ('dua tambah dua', 'Dua tambah dua sama dengan empat.'),
    ('Tiga tambah tiga?', 'Tiga tambah tiga sama dengan enam.'),
    ('Lima tambah lima?', 'Lima tambah lima sama dengan sepuluh.'),
    ('Dua dikali dua?', 'Dua dikali dua sama dengan empat.'),
    ('dua dikali dua', 'Dua dikali dua sama dengan empat.'),
    ('Lima dikali lima?', 'Lima dikali lima sama dengan dua puluh lima.'),
    ('lima dikali lima', 'Lima dikali lima sama dengan dua puluh lima.'),
    ('Tiga dikali tiga?', 'Tiga dikali tiga sama dengan sembilan.'),
    ('Sepuluh dibagi dua?', 'Sepuluh dibagi dua sama dengan lima.'),
    ('sepuluh dibagi dua', 'Sepuluh dibagi dua sama dengan lima.'),
    ('Sepuluh dikurangi lima?', 'Sepuluh dikurangi lima sama dengan lima.'),
    ('Ibu kota Indonesia?', 'Ibu kota Indonesia adalah Jakarta.'),
    ('ibu kota indonesia', 'Ibu kota Indonesia adalah Jakarta.'),
    ('Apa ibu kota Indonesia?', 'Ibu kota Indonesia adalah Jakarta.'),
    ('ibu kota indonesia apa', 'Ibu kota Indonesia adalah Jakarta.'),
    ('Ibu kota Jepang?', 'Ibu kota Jepang adalah Tokyo.'),
    ('ibu kota jepang', 'Ibu kota Jepang adalah Tokyo.'),
    ('Apa ibu kota Jepang?', 'Ibu kota Jepang adalah Tokyo.'),
    ('ibu kota jepang apa', 'Ibu kota Jepang adalah Tokyo.'),
    ('Apa ibu kota Prancis?', 'Ibu kota Prancis adalah Paris.'),
    ('Apa ibu kota Inggris?', 'Ibu kota Inggris adalah London.'),
    ('Ibu kota Spanyol?', 'Ibu kota Spanyol adalah Madrid.'),
    ('Ibu kota Jerman?', 'Ibu kota Jerman adalah Berlin.'),
    ('Ibu kota Italia?', 'Ibu kota Italia adalah Roma.'),
    ('Ibu kota Rusia?', 'Ibu kota Rusia adalah Moskow.'),
    ('Ibu kota Amerika?', 'Ibu kota Amerika adalah Washington DC.'),
    ('Ibu kota Tiongkok?', 'Ibu kota Tiongkok adalah Beijing.'),
    ('Ibu kota Korea Selatan?', 'Ibu kota Korea Selatan adalah Seoul.'),
    ('Tokyo itu apa?', 'Tokyo adalah ibu kota Jepang.'),
    ('tokyo itu apa', 'Tokyo adalah ibu kota Jepang.'),
    ('Jakarta itu apa?', 'Jakarta adalah ibu kota Indonesia.'),
    ('Siapa presiden pertama Indonesia?', 'Presiden pertama Indonesia adalah Soekarno.'),
    ('siapa presiden pertama indonesia', 'Presiden pertama Indonesia adalah Soekarno.'),
    ('Siapa presiden kedua Indonesia?', 'Presiden kedua Indonesia adalah Soeharto.'),
    ('Siapa presiden Indonesia?', 'Presiden Indonesia dipilih melalui pemilu demokratis.'),
    ('Siapa bapak proklamasi Indonesia?', 'Bapak proklamasi Indonesia adalah Soekarno dan Hatta.'),
    ('Apa warna langit?', 'Warna langit adalah biru.'),
    ('Apa warna rumput?', 'Warna rumput adalah hijau.'),
    ('Apa warna darah?', 'Warna darah adalah merah.'),
    ('Apa suara kucing?', 'Kucing bersuara meong.'),
    ('Apa suara sapi?', 'Sapi bersuara moo.'),
    ('Apa suara anjing?', 'Anjing bersuara guk guk.'),
    ('Apel adalah?', 'Apel adalah buah.'),
    ('Singa adalah?', 'Singa adalah hewan.'),
    ('Mawar adalah?', 'Mawar adalah bunga.'),
    ('Berapa hari dalam seminggu?', 'Ada tujuh hari dalam seminggu.'),
    ('Berapa bulan dalam setahun?', 'Ada dua belas bulan dalam setahun.'),
    ('Berapa menit dalam satu jam?', 'Ada enam puluh menit dalam satu jam.'),
    ('Berapa jam dalam satu hari?', 'Ada dua puluh empat jam dalam satu hari.'),
    ('Setelah Senin adalah?', 'Setelah Senin adalah Selasa.'),
    ('Lebih besar gajah atau semut?', 'Gajah lebih besar dari semut.'),
    ('Translate halo to English', 'Halo in English is Hello.'),
    ('Translate thank you to Indonesian', 'Thank you in Indonesian is Terima kasih.'),
    ('Apa fungsi akar?', 'Akar menyerap air dari tanah.'),
    ('Apa fungsi daun?', 'Daun membuat makanan untuk tanaman.'),
    ('Apa itu air?', 'Air adalah benda cair yang kita minum.'),
    ('Matahari terbit di?', 'Matahari terbit di timur.'),
    ('Terima kasih', 'Sama-sama! Ada yang bisa aku bantu lagi?'),
    ('terima kasih', 'Sama-sama! Ada yang bisa aku bantu lagi?'),
    ('Makasih', 'Sama-sama!'),
    ('Goodbye', 'Sampai jumpa! Semoga harimu menyenangkan.'),
    ('Pola satu dua tiga empat. Berikutnya?', 'Berikutnya adalah lima.'),
    ('Siapa presiden Amerika?', 'Presiden Amerika dipilih melalui pemilu demokratis.'),
    ('Apa berita hari ini?', 'Aku tidak tahu berita terbaru, cek sumber berita terpercaya.'),
    ('Bagaimana cuaca hari ini?', 'Aku tidak tahu cuaca hari ini, cek aplikasi cuaca.'),
    ('kamu bisa apa', 'Aku bisa matematika dasar, terjemahan, dan menjawab pertanyaan sederhana.'),
    ('lagi apa', 'Aku siap membantu. Ada yang bisa aku bantu?'),
    ('test', 'Halo! Aku D\'QUILANE Blueprint v2.1. Ada yang bisa aku bantu?'),
    ('jepang di mana', 'Jepang adalah negara di Asia Timur.'),
]

# Build AION engram
engram = {}
for prompt, response in SFT_EXAMPLES:
    engram[prompt.lower().strip()] = response

model['aion_engram'] = engram
print(f"\nAION Engram: {len(engram)} Q&A pairs")

# Save updated model
with open(MODEL_PATH, 'w', encoding='utf-8') as f:
    json.dump(model, f, ensure_ascii=False)
print(f"Model saved: {os.path.getsize(MODEL_PATH)/1024:.1f} KB")

# Generate GGUF
print("\n=== Generating GGUF ===")
import subprocess
result = subprocess.run([
    sys.executable, '/home/z/my-project/scripts/gguf_export.py',
    MODEL_PATH, '/home/z/my-project/download/llm/dquilane_v2.gguf'
], capture_output=True, text=True)
print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr[:200])
