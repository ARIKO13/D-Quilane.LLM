"""
D'QUILANE Massive Dataset Generator
====================================
Generates 3000+ Q&A pairs across 20+ fields, organized by:
- Bidang (field)
- Judul (title)
- Subjudul (subtitle)

Output: dataset/ folder with JSONL files per category
Format: {"bidang": "...", "judul": "...", "subjudul": "...", "prompt": "...", "response": "..."}
"""
import json
import os
import random
from pathlib import Path

OUTPUT_DIR = Path("/home/z/my-project/download/llm/dataset")

# Indonesian number words
NUM_WORDS = {
    0: 'nol', 1: 'satu', 2: 'dua', 3: 'tiga', 4: 'empat', 5: 'lima',
    6: 'enam', 7: 'tujuh', 8: 'delapan', 9: 'sembilan', 10: 'sepuluh',
    11: 'sebelas', 12: 'dua belas', 13: 'tiga belas', 14: 'empat belas',
    15: 'lima belas', 16: 'enam belas', 17: 'tujuh belas', 18: 'delapan belas',
    19: 'sembilan belas', 20: 'dua puluh', 30: 'tiga puluh', 40: 'empat puluh',
    50: 'lima puluh', 60: 'enam puluh', 70: 'tujuh puluh', 80: 'delapan puluh',
    90: 'sembilan puluh', 100: 'seratus',
}

def num_to_words(n):
    if n in NUM_WORDS:
        return NUM_WORDS[n]
    if n < 20:
        return NUM_WORDS.get(n, str(n))
    if n <= 100:
        tens = (n // 10) * 10
        ones = n % 10
        if ones == 0:
            return NUM_WORDS[tens]
        return f"{NUM_WORDS[tens]} {NUM_WORDS[ones]}"
    # Handle 100-999
    hundreds = n // 100
    remainder = n % 100
    if hundreds == 1:
        prefix = 'seratus'
    else:
        prefix = f'{NUM_WORDS[hundreds]} ratus'
    if remainder == 0:
        return prefix
    return f'{prefix} {num_to_words(remainder)}'

def write_jsonl(path, items):
    """Write list of Q&A dicts to JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    return len(items)

total = 0

# ============================================================
# 1. MATEMATIKA
# ============================================================
print("Generating: Matematika...")
items = []

# Basic Addition (programmatic: 1+1 through 20+20)
for a in range(1, 21):
    for b in range(1, 21):
        if random.random() < 0.3:  # sample 30%
            result = a + b
            items.append({
                'bidang': 'matematika',
                'judul': 'Aritmatika Dasar',
                'subjudul': 'Penjumlahan',
                'prompt': f'{num_to_words(a)} tambah {num_to_words(b)}?',
                'response': f'{num_to_words(a)} tambah {num_to_words(b)} sama dengan {num_to_words(result)}.'
            })
total += write_jsonl(OUTPUT_DIR / 'matematika' / 'penjumlahan.jsonl', items)

# Subtraction
items = []
for a in range(5, 51):
    for b in range(1, a):
        if random.random() < 0.15:
            result = a - b
            items.append({
                'bidang': 'matematika',
                'judul': 'Aritmatika Dasar',
                'subjudul': 'Pengurangan',
                'prompt': f'{num_to_words(a)} dikurangi {num_to_words(b)}?',
                'response': f'{num_to_words(a)} dikurangi {num_to_words(b)} sama dengan {num_to_words(result)}.'
            })
total += write_jsonl(OUTPUT_DIR / 'matematika' / 'pengurangan.jsonl', items)

# Multiplication
items = []
for a in range(1, 13):
    for b in range(1, 13):
        result = a * b
        items.append({
            'bidang': 'matematika',
            'judul': 'Aritmatika Dasar',
            'subjudul': 'Perkalian',
            'prompt': f'{num_to_words(a)} dikali {num_to_words(b)}?',
            'response': f'{num_to_words(a)} dikali {num_to_words(b)} sama dengan {num_to_words(result)}.'
        })
total += write_jsonl(OUTPUT_DIR / 'matematika' / 'perkalian.jsonl', items)

# Division
items = []
for a in range(2, 101):
    for b in range(2, 13):
        if a % b == 0:
            result = a // b
            items.append({
                'bidang': 'matematika',
                'judul': 'Aritmatika Dasar',
                'subjudul': 'Pembagian',
                'prompt': f'{num_to_words(a)} dibagi {num_to_words(b)}?',
                'response': f'{num_to_words(a)} dibagi {num_to_words(b)} sama dengan {num_to_words(result)}.'
            })
            if len(items) >= 150:
                break
    if len(items) >= 150:
        break
total += write_jsonl(OUTPUT_DIR / 'matematika' / 'pembagian.jsonl', items)

# Word Problems
items = [
    {'bidang': 'matematika', 'judul': 'Soal Cerita', 'subjudul': 'Penjumlahan',
     'prompt': 'Andi punya tiga apel. Ibu memberi dua apel. Berapa total apel Andi?',
     'response': 'Total apel Andi adalah lima.'},
    {'bidang': 'matematika', 'judul': 'Soal Cerita', 'subjudul': 'Pengurangan',
     'prompt': 'Sari punya sepuluh permen. Dia makan empat permen. Berapa sisa permen?',
     'response': 'Sisa permennya enam.'},
    {'bidang': 'matematika', 'judul': 'Soal Cerita', 'subjudul': 'Perkalian',
     'prompt': 'Budi punya lima kantong dengan empat kelereng tiap kantong. Berapa total kelereng?',
     'response': 'Total kelereng Budi adalah dua puluh.'},
    {'bidang': 'matematika', 'judul': 'Soal Cerita', 'subjudul': 'Pembagian',
     'prompt': 'Rina punya dua belas pensil. Dia bagi ke empat teman. Berapa pensil per teman?',
     'response': 'Setiap teman dapat tiga pensil.'},
    {'bidang': 'matematika', 'judul': 'Soal Cerita', 'subjudul': 'Campuran',
     'prompt': 'Toko buka jam tujuh pagi sampai sembilan malam. Berapa jam toko buka?',
     'response': 'Toko buka selama empat belas jam.'},
]
total += write_jsonl(OUTPUT_DIR / 'matematika' / 'soal_cerita.jsonl', items)

# ============================================================
# 2. SAINS
# ============================================================
print("Generating: Sains...")
science_items = {
    'fisika.jsonl': [
        ('Fisika', 'Hukum Gerak', 'Apa hukum Newton pertama?', 'Benda diam tetap diam, benda bergerak tetap bergerak kecuali ada gaya.'),
        ('Fisika', 'Hukum Gerak', 'Apa hukum Newton kedua?', 'Gaya sama dengan massa dikali percepatan.'),
        ('Fisika', 'Hukum Gerak', 'Apa hukum Newton ketiga?', 'Setiap aksi memiliki reaksi yang sama besar dan berlawanan arah.'),
        ('Fisika', 'Konsep Dasar', 'Apa itu gravitasi?', 'Gravitasi adalah gaya tarik bumi ke benda.'),
        ('Fisika', 'Konsep Dasar', 'Apa itu energi?', 'Energi adalah kemampuan untuk melakukan kerja.'),
        ('Fisika', 'Konsep Dasar', 'Apa itu massa?', 'Massa adalah jumlah materi dalam benda.'),
        ('Fisika', 'Konsep Dasar', 'Apa kecepatan cahaya?', 'Kecepatan cahaya sekitar tiga ratus ribu kilometer per detik.'),
        ('Fisika', 'Termodinamika', 'Apa itu suhu?', 'Suhu adalah ukuran panas atau dingin suatu benda.'),
        ('Fisika', 'Termodinamika', 'Berapa titik didih air?', 'Air mendidih pada suhu seratus derajat Celsius.'),
        ('Fisika', 'Termodinamika', 'Berapa titik beku air?', 'Air membeku pada suhu nol derajat Celsius.'),
        ('Fisika', 'Listrik', 'Apa itu arus listrik?', 'Arus listrik adalah aliran elektron melalui konduktor.'),
        ('Fisika', 'Listrik', 'Apa itu tegangan?', 'Tegangan adalah beda potensial listrik antara dua titik.'),
        ('Fisika', 'Gelombang', 'Apa itu gelombang?', 'Gelombang adalah getaran yang merambat.'),
        ('Fisika', 'Gelombang', 'Apa itu frekuensi?', 'Frekuensi adalah jumlah getaran per detik.'),
        ('Fisika', 'Cahaya', 'Apa itu spektrum cahaya?', 'Spektrum cahaya adalah rentang warna dari merah ke ungu.'),
    ],
    'kimia.jsonl': [
        ('Kimia', 'Materi', 'Apa itu atom?', 'Atom adalah satuan terkecil materi yang masih memiliki sifat elemen.'),
        ('Kimia', 'Materi', 'Apa itu molekul?', 'Molekul adalah gabungan dua atau lebih atom.'),
        ('Kimia', 'Materi', 'Apa itu elemen?', 'Elemen adalah zat murni yang tidak bisa dipecah lagi.'),
        ('Kimia', 'Tabel Periodik', 'Berapa jumlah elemen di tabel periodik?', 'Ada seratus delapan belas elemen di tabel periodik.'),
        ('Kimia', 'Tabel Periodik', 'Apa elemen paling ringan?', 'Hidrogen adalah elemen paling ringan.'),
        ('Kimia', 'Tabel Periodik', 'Apa elemen paling banyak di alam?', 'Oksigen adalah elemen paling banyak di kerak bumi.'),
        ('Kimia', 'Reaksi', 'Apa itu reaksi kimia?', 'Reaksi kimia adalah proses perubahan zat menjadi zat baru.'),
        ('Kimia', 'Reaksi', 'Apa itu oksidasi?', 'Oksidasi adalah reaksi dengan oksigen.'),
        ('Kimia', 'Asam Basa', 'Apa itu asam?', 'Asam adalah zat dengan pH di bawah tujuh.'),
        ('Kimia', 'Asam Basa', 'Apa itu basa?', 'Basa adalah zat dengan pH di atas tujuh.'),
        ('Kimia', 'Asam Basa', 'Apa pH air murni?', 'Air murni memiliki pH tujuh.'),
        ('Kimia', 'Senyawa', 'Apa rumus air?', 'Rumus kimia air adalah H2O.'),
        ('Kimia', 'Senyawa', 'Apa rumus garam dapur?', 'Rumus kimia garam dapur adalah NaCl.'),
        ('Kimia', 'Senyawa', 'Apa rumus karbon dioksida?', 'Rumus kimia karbon dioksida adalah CO2.'),
        ('Kimia', 'Senyawa', 'Apa rumus glukosa?', 'Rumus kimia glukosa adalah C6H12O6.'),
    ],
    'biologi.jsonl': [
        ('Biologi', 'Sel', 'Apa itu sel?', 'Sel adalah unit dasar kehidupan.'),
        ('Biologi', 'Sel', 'Apa bagian sel yang menghasilkan energi?', 'Mitokondria menghasilkan energi untuk sel.'),
        ('Biologi', 'Sel', 'Apa bagian sel yang menyimpan DNA?', 'Nukleus menyimpan DNA dalam sel.'),
        ('Biologi', 'Klasifikasi', 'Apa lima kerajaan makhluk hidup?', 'Lima kerajaan: Monera, Protista, Fungi, Plantae, Animalia.'),
        ('Biologi', 'Klasifikasi', 'Apa itu mamalia?', 'Mamalia adalah hewan yang menyusui anaknya.'),
        ('Biologi', 'Klasifikasi', 'Apa itu reptil?', 'Reptil adalah hewan bersisik yang berkembang biak bertelur.'),
        ('Biologi', 'Sistem Tubuh', 'Apa fungsi jantung?', 'Jantung memompa darah ke seluruh tubuh.'),
        ('Biologi', 'Sistem Tubuh', 'Apa fungsi paru-paru?', 'Paru-paru menukar oksigen dan karbon dioksida.'),
        ('Biologi', 'Sistem Tubuh', 'Apa fungsi otak?', 'Otak mengendalikan seluruh fungsi tubuh.'),
        ('Biologi', 'Sistem Tubuh', 'Apa fungsi ginjal?', 'Ginjal menyaring darah dan menghasilkan urine.'),
        ('Biologi', 'Genetika', 'Apa itu DNA?', 'DNA adalah molekul yang menyimpan informasi genetik.'),
        ('Biologi', 'Genetika', 'Apa itu gen?', 'Gen adalah bagian DNA yang menentukan sifat.'),
        ('Biologi', 'Genetika', 'Apa itu evolusi?', 'Evolusi adalah perubahan makhluk hidup dari waktu ke waktu.'),
        ('Biologi', 'Tumbuhan', 'Apa itu fotosintesis?', 'Fotosintesis adalah proses tumbuhan membuat makanan dari cahaya matahari.'),
        ('Biologi', 'Tumbuhan', 'Apa yang dibutuhkan tumbuhan untuk tumbuh?', 'Tumbuhan butuh air, cahaya, udara, dan nutrisi.'),
    ],
    'ilmu_bumi.jsonl': [
        ('Ilmu Bumi', 'Struktur Bumi', 'Apa lapisan bumi?', 'Bumi memiliki tiga lapisan: kerak, mantel, dan inti.'),
        ('Ilmu Bumi', 'Struktur Bumi', 'Apa lapisan terluar bumi?', 'Kerak bumi adalah lapisan terluar.'),
        ('Ilmu Bumi', 'Struktur Bumi', 'Apa lapisan terdalam bumi?', 'Inti bumi adalah lapisan terdalam.'),
        ('Ilmu Bumi', 'Cuaca', 'Apa itu hujan?', 'Hujan adalah air yang turun dari langit.'),
        ('Ilmu Bumi', 'Cuaca', 'Apa itu petir?', 'Petir adalah pelepasan listrik di atmosfer.'),
        ('Ilmu Bumi', 'Cuaca', 'Apa itu angin?', 'Angin adalah udara yang bergerak.'),
        ('Ilmu Bumi', 'Geologi', 'Apa itu gempa bumi?', 'Gempa bumi adalah getaran permukaan bumi.'),
        ('Ilmu Bumi', 'Geologi', 'Apa itu gunung berapi?', 'Gunung berapi adalah bukit yang mengeluarkan magma.'),
        ('Ilmu Bumi', 'Geologi', 'Apa itu tsunami?', 'Tsunami adalah gelombang besar yang disebabkan gempa laut.'),
        ('Ilmu Bumi', 'Luar Angkasa', 'Berapa planet di tata surya?', 'Ada delapan planet di tata surya.'),
        ('Ilmu Bumi', 'Luar Angkasa', 'Apa planet terdekat dengan matahari?', 'Merkurius adalah planet terdekat dengan matahari.'),
        ('Ilmu Bumi', 'Luar Angkasa', 'Apa planet terbesar?', 'Jupiter adalah planet terbesar di tata surya.'),
        ('Ilmu Bumi', 'Luar Angkasa', 'Apa benda langit yang mengelilingi planet?', 'Satelit atau bulan mengelilingi planet.'),
        ('Ilmu Bumi', 'Luar Angkasa', 'Berapa bulan yang dimiliki bumi?', 'Bumi memiliki satu bulan.'),
        ('Ilmu Bumi', 'Luar Angkasa', 'Apa itu galaksi?', 'Galaksi adalah kumpulan miliaran bintang.'),
    ],
}
for filename, qa_list in science_items.items():
    items = [{'bidang': 'sains', 'judul': j, 'subjudul': s, 'prompt': p, 'response': r}
             for (j, s, p, r) in qa_list]
    total += write_jsonl(OUTPUT_DIR / 'sains' / filename, items)

# ============================================================
# 3. SEJARAH
# ============================================================
print("Generating: Sejarah...")
history_items = {
    'sejarah_indonesia.jsonl': [
        ('Sejarah Indonesia', 'Kemerdekaan', 'Kapan Indonesia merdeka?', 'Indonesia merdeka pada tanggal tujuh belas agustus seribu sembilan ratus empat puluh lima.'),
        ('Sejarah Indonesia', 'Kemerdekaan', 'Siapa proklamator Indonesia?', 'Soekarno dan Hatta adalah proklamator Indonesia.'),
        ('Sejarah Indonesia', 'Kemerdekaan', 'Apa teks proklamasi?', 'Teks proklamasi adalah pernyataan kemerdekaan Indonesia.'),
        ('Sejarah Indonesia', 'Presiden', 'Siapa presiden pertama Indonesia?', 'Presiden pertama Indonesia adalah Soekarno.'),
        ('Sejarah Indonesia', 'Presiden', 'Siapa presiden kedua Indonesia?', 'Presiden kedua Indonesia adalah Soeharto.'),
        ('Sejarah Indonesia', 'Presiden', 'Siapa presiden ketiga Indonesia?', 'Presiden ketiga Indonesia adalah Habibie.'),
        ('Sejarah Indonesia', 'Presiden', 'Siapa presiden keempat Indonesia?', 'Presiden keempat Indonesia adalah Abdurrahman Wahid.'),
        ('Sejarah Indonesia', 'Presiden', 'Siapa presiden kelima Indonesia?', 'Presiden kelima Indonesia adalah Megawati.'),
        ('Sejarah Indonesia', 'Presiden', 'Siapa presiden keenam Indonesia?', 'Presiden keenam Indonesia adalah SBY.'),
        ('Sejarah Indonesia', 'Presiden', 'Siapa presiden ketujuh Indonesia?', 'Presiden ketujuh Indonesia adalah Jokowi.'),
        ('Sejarah Indonesia', 'Kerajaan', 'Apa kerajaan Hindu pertama di Indonesia?', 'Kutai adalah kerajaan Hindu pertama di Indonesia.'),
        ('Sejarah Indonesia', 'Kerajaan', 'Apa kerajaan maritim terbesar di Indonesia?', 'Majapahit adalah kerajaan maritim terbesar.'),
        ('Sejarah Indonesia', 'Kerajaan', 'Apa kerajaan Buddha di Indonesia?', 'Sriwijaya adalah kerajaan Buddha terbesar.'),
        ('Sejarah Indonesia', 'Pahlawan', 'Siapa pahlawan nasional dari Aceh?', 'Cut Nyak Dien adalah pahlawan dari Aceh.'),
        ('Sejarah Indonesia', 'Pahlawan', 'Siapa pahlawan dari Jawa Tengah?', 'Pangeran Diponegoro adalah pahlawan dari Jawa Tengah.'),
        ('Sejarah Indonesia', 'Pahlawan', 'Siapa pahlawan emansipasi wanita?', 'R.A. Kartini adalah pahlawan emansipasi wanita.'),
        ('Sejarah Indonesia', 'Pahlawan', 'Siapa Bapak Pendidikan Indonesia?', 'Ki Hajar Dewantara adalah Bapak Pendidikan Indonesia.'),
        ('Sejarah Indonesia', 'Pahlawan', 'Siapa Bapak Koperasi Indonesia?', 'Mohammad Hatta adalah Bapak Koperasi Indonesia.'),
        ('Sejarah Indonesia', 'Era', 'Apa era pra-aksara?', 'Era pra-aksara adalah masa sebelum manusia mengenal tulisan.'),
        ('Sejarah Indonesia', 'Era', 'Apa era kolonial?', 'Era kolonial adalah masa penjajahan Belanda di Indonesia.'),
    ],
    'sejarah_dunia.jsonl': [
        ('Sejarah Dunia', 'Perang', 'Kapan Perang Dunia Pertama?', 'Perang Dunia Pertama terjadi tahun seribu sembilan ratus empat belas hingga delapan belas.'),
        ('Sejarah Dunia', 'Perang', 'Kapan Perang Dunia Kedua?', 'Perang Dunia Kedua terjadi tahun seribu sembilan ratus tiga puluh sembilan hingga empat puluh lima.'),
        ('Sejarah Dunia', 'Perang', 'Siapa pemimpin Jerman di PD2?', 'Hitler adalah pemimpin Jerman di Perang Dunia Kedua.'),
        ('Sejarah Dunia', 'Tokoh', 'Siapa penemu telepon?', 'Alexander Graham Bell menemukan telepon.'),
        ('Sejarah Dunia', 'Tokoh', 'Siapa penemu bohlam?', 'Thomas Edison menemukan bohlam listrik.'),
        ('Sejarah Dunia', 'Tokoh', 'Siapa bapak komputer?', 'Charles Babbage adalah bapak komputer.'),
        ('Sejarah Dunia', 'Tokoh', 'Siapa penemu pesawat?', 'Wright bersaudara menemukan pesawat terbang.'),
        ('Sejarah Dunia', 'Tokoh', 'Siapa ilmuwan teori relativitas?', 'Albert Einstein mengemukakan teori relativitas.'),
        ('Sejarah Dunia', 'Tokoh', 'Siapa penemu vaksin?', 'Edward Jenner menemukan vaksin pertama.'),
        ('Sejarah Dunia', 'Peradaban', 'Apa peradaban tertua di dunia?', 'Sumeria adalah peradaban tertua di dunia.'),
        ('Sejarah Dunia', 'Peradaban', 'Apa peradaban Mesir Kuno?', 'Peradaban Mesir Kuno terkenal dengan piramida.'),
        ('Sejarah Dunia', 'Peradaban', 'Apa peradaban Yunani Kuno?', 'Yunani Kuno terkenal dengan filsafat dan demokrasi.'),
        ('Sejarah Dunia', 'Peradaban', 'Apa peradaban Romawi?', 'Romawi terkenal dengan hukum dan arsitektur.'),
        ('Sejarah Dunia', 'Penemuan', 'Siapa menemukan benua Amerika?', 'Columbus menemukan benua Amerika untuk Eropa.'),
        ('Sejarah Dunia', 'Penemuan', 'Siapa orang pertama ke bulan?', 'Neil Armstrong adalah orang pertama mendarat di bulan.'),
    ],
}
for filename, qa_list in history_items.items():
    items = [{'bidang': 'sejarah', 'judul': j, 'subjudul': s, 'prompt': p, 'response': r}
             for (j, s, p, r) in qa_list]
    total += write_jsonl(OUTPUT_DIR / 'sejarah' / filename, items)

# ============================================================
# 4. GEOGRAFI
# ============================================================
print("Generating: Geografi...")
geography_items = {
    'ibukota.jsonl': [
        ('Geografi', 'Ibu Kota', 'Apa ibu kota Indonesia?', 'Ibu kota Indonesia adalah Jakarta.'),
        ('Geografi', 'Ibu Kota', 'Apa ibu kota Jepang?', 'Ibu kota Jepang adalah Tokyo.'),
        ('Geografi', 'Ibu Kota', 'Apa ibu kota Prancis?', 'Ibu kota Prancis adalah Paris.'),
        ('Geografi', 'Ibu Kota', 'Apa ibu kota Inggris?', 'Ibu kota Inggris adalah London.'),
        ('Geografi', 'Ibu Kota', 'Apa ibu kota Spanyol?', 'Ibu kota Spanyol adalah Madrid.'),
        ('Geografi', 'Ibu Kota', 'Apa ibu kota Jerman?', 'Ibu kota Jerman adalah Berlin.'),
        ('Geografi', 'Ibu Kota', 'Apa ibu kota Italia?', 'Ibu kota Italia adalah Roma.'),
        ('Geografi', 'Ibu Kota', 'Apa ibu kota Rusia?', 'Ibu kota Rusia adalah Moskow.'),
        ('Geografi', 'Ibu Kota', 'Apa ibu kota Amerika?', 'Ibu kota Amerika adalah Washington DC.'),
        ('Geografi', 'Ibu Kota', 'Apa ibu kota Tiongkok?', 'Ibu kota Tiongkok adalah Beijing.'),
        ('Geografi', 'Ibu Kota', 'Apa ibu kota Korea Selatan?', 'Ibu kota Korea Selatan adalah Seoul.'),
        ('Geografi', 'Ibu Kota', 'Apa ibu kota India?', 'Ibu kota India adalah New Delhi.'),
        ('Geografi', 'Ibu Kota', 'Apa ibu kota Brazil?', 'Ibu kota Brazil adalah Brasilia.'),
        ('Geografi', 'Ibu Kota', 'Apa ibu kota Australia?', 'Ibu kota Australia adalah Canberra.'),
        ('Geografi', 'Ibu Kota', 'Apa ibu kota Kanada?', 'Ibu kota Kanada adalah Ottawa.'),
        ('Geografi', 'Ibu Kota', 'Apa ibu kota Mesir?', 'Ibu kota Mesir adalah Kairo.'),
        ('Geografi', 'Ibu Kota', 'Apa ibu kota Thailand?', 'Ibu kota Thailand adalah Bangkok.'),
        ('Geografi', 'Ibu Kota', 'Apa ibu kota Vietnam?', 'Ibu kota Vietnam adalah Hanoi.'),
        ('Geografi', 'Ibu Kota', 'Apa ibu kota Filipina?', 'Ibu kota Filipina adalah Manila.'),
        ('Geografi', 'Ibu Kota', 'Apa ibu kota Malaysia?', 'Ibu kota Malaysia adalah Kuala Lumpur.'),
    ],
    'benua_samudra.jsonl': [
        ('Geografi', 'Benua', 'Berapa benua di dunia?', 'Ada tujuh benua di dunia.'),
        ('Geografi', 'Benua', 'Apa benua terbesar?', 'Asia adalah benua terbesar.'),
        ('Geografi', 'Benua', 'Apa benua terkecil?', 'Australia adalah benua terkecil.'),
        ('Geografi', 'Benua', 'Apa benua terdingin?', 'Antartika adalah benua terdingin.'),
        ('Geografi', 'Samudra', 'Berapa samudra di dunia?', 'Ada lima samudra di dunia.'),
        ('Geografi', 'Samudra', 'Apa samudra terbesar?', 'Samudra Pasifik adalah samudra terbesar.'),
        ('Geografi', 'Samudra', 'Apa samudra terdalam?', 'Samudra Pasifik adalah samudra terdalam.'),
        ('Geografi', 'Pegunungan', 'Apa gunung tertinggi di dunia?', 'Gunung Everest adalah gunung tertinggi di dunia.'),
        ('Geografi', 'Pegunungan', 'Apa gunung tertinggi di Indonesia?', 'Puncak Jaya adalah gunung tertinggi di Indonesia.'),
        ('Geografi', 'Sungai', 'Apa sungai terpanjang di dunia?', 'Sungai Nil adalah sungai terpanjang di dunia.'),
        ('Geografi', 'Sungai', 'Apa sungai terpanjang di Indonesia?', 'Sungai Kapuas adalah sungai terpanjang di Indonesia.'),
        ('Geografi', 'Laut', 'Apa laut terbesar?', 'Laut Filipina adalah laut terbesar.'),
        ('Geografi', 'Gurun', 'Apa gurun terbesar di dunia?', 'Gurun Sahara adalah gurun terbesar di dunia.'),
        ('Geografi', 'Pulau', 'Apa pulau terbesar di dunia?', 'Greenland adalah pulau terbesar di dunia.'),
        ('Geografi', 'Negara', 'Apa negara terbesar di dunia?', 'Rusia adalah negara terbesar di dunia.'),
        ('Geografi', 'Negara', 'Apa negara terkecil di dunia?', 'Vatikan adalah negara terkecil di dunia.'),
        ('Geografi', 'Negara', 'Apa negara dengan penduduk terbanyak?', 'Tiongkok dan India adalah negara dengan penduduk terbanyak.'),
        ('Geografi', 'Iklim', 'Apa iklim Indonesia?', 'Indonesia beriklim tropis.'),
        ('Geografi', 'Iklim', 'Apa iklim di kutub?', 'Kutub beriklim sangat dingin.'),
        ('Geografi', 'Indonesia', 'Berapa pulau di Indonesia?', 'Indonesia memiliki lebih dari tujuh belas ribu pulau.'),
    ],
}
for filename, qa_list in geography_items.items():
    items = [{'bidang': 'geografi', 'judul': j, 'subjudul': s, 'prompt': p, 'response': r}
             for (j, s, p, r) in qa_list]
    total += write_jsonl(OUTPUT_DIR / 'geografi' / filename, items)

# ============================================================
# 5. PENGETAHUAN UMUM
# ============================================================
print("Generating: Pengetahuan Umum...")
general_items = {
    'warna.jsonl': [
        ('Pengetahuan Umum', 'Warna', 'Apa warna langit?', 'Warna langit adalah biru.'),
        ('Pengetahuan Umum', 'Warna', 'Apa warna rumput?', 'Warna rumput adalah hijau.'),
        ('Pengetahuan Umum', 'Warna', 'Apa warna darah?', 'Warna darah adalah merah.'),
        ('Pengetahuan Umum', 'Warna', 'Apa warna matahari?', 'Warna matahari adalah kuning.'),
        ('Pengetahuan Umum', 'Warna', 'Apa warna awan?', 'Warna awan adalah putih.'),
        ('Pengetahuan Umum', 'Warna', 'Apa warna malam?', 'Warna malam adalah hitam.'),
        ('Pengetahuan Umum', 'Warna', 'Apa warna pisang matang?', 'Warna pisang matang adalah kuning.'),
        ('Pengetahuan Umum', 'Warna', 'Apa warna jeruk?', 'Warna jeruk adalah oranye.'),
        ('Pengetahuan Umum', 'Warna', 'Apa warna anggrek?', 'Warna anggrek adalah ungu.'),
        ('Pengetahuan Umum', 'Warna', 'Apa warna bendera Indonesia?', 'Bendera Indonesia adalah merah putih.'),
    ],
    'hewan.jsonl': [
        ('Pengetahuan Umum', 'Hewan', 'Apa suara kucing?', 'Kucing bersuara meong.'),
        ('Pengetahuan Umum', 'Hewan', 'Apa suara anjing?', 'Anjing bersuara guk guk.'),
        ('Pengetahuan Umum', 'Hewan', 'Apa suara sapi?', 'Sapi bersuara moo.'),
        ('Pengetahuan Umum', 'Hewan', 'Apa suara kambing?', 'Kambing bersuara mbek mbek.'),
        ('Pengetahuan Umum', 'Hewan', 'Apa suara ayam jantan?', 'Ayam jantan bersuara kukuruyuk.'),
        ('Pengetahuan Umum', 'Hewan', 'Apa suara bebek?', 'Bebek bersuara kwek kwek.'),
        ('Pengetahuan Umum', 'Hewan', 'Apa suara kuda?', 'Kuda bersuara igih igih.'),
        ('Pengetahuan Umum', 'Hewan', 'Apa hewan terbesar di darat?', 'Gajah adalah hewan terbesar di darat.'),
        ('Pengetahuan Umum', 'Hewan', 'Apa hewan tercepat?', 'Cheetah adalah hewan darat tercepat.'),
        ('Pengetahuan Umum', 'Hewan', 'Siapa raja hutan?', 'Singa adalah raja hutan.'),
        ('Pengetahuan Umum', 'Hewan', 'Hewan apa yang bisa terbang?', 'Burung bisa terbang.'),
        ('Pengetahuan Umum', 'Hewan', 'Hewan apa yang hidup di air?', 'Ikan hidup di air.'),
        ('Pengetahuan Umum', 'Hewan', 'Apa makanan kucing?', 'Kucing suka ikan.'),
        ('Pengetahuan Umum', 'Hewan', 'Apa makanan sapi?', 'Sapi suka rumput.'),
        ('Pengetahuan Umum', 'Hewan', 'Apa yang dihasilkan sapi?', 'Sapi memberi susu.'),
        ('Pengetahuan Umum', 'Hewan', 'Apa yang dihasilkan lebah?', 'Lebah membuat madu.'),
        ('Pengetahuan Umum', 'Hewan', 'Dari mana kupu-kupu berasal?', 'Kupu-kupu berasal dari ulat.'),
        ('Pengetahuan Umum', 'Hewan', 'Apa hewan berkantung?', 'Kanguru adalah hewan berkantung.'),
        ('Pengetahuan Umum', 'Hewan', 'Hewan apa yang memiliki belalai?', 'Gajah memiliki belalai.'),
        ('Pengetahuan Umum', 'Hewan', 'Apa hewan bersisik?', 'Buaya dan ular adalah hewan bersisik.'),
    ],
    'tumbuhan.jsonl': [
        ('Pengetahuan Umum', 'Tumbuhan', 'Apa fungsi akar?', 'Akar menyerap air dari tanah.'),
        ('Pengetahuan Umum', 'Tumbuhan', 'Apa fungsi batang?', 'Batang menjaga tanaman tetap tegak.'),
        ('Pengetahuan Umum', 'Tumbuhan', 'Apa fungsi daun?', 'Daun membuat makanan untuk tanaman.'),
        ('Pengetahuan Umum', 'Tumbuhan', 'Apa fungsi bunga?', 'Bunga berfungsi untuk reproduksi tanaman.'),
        ('Pengetahuan Umum', 'Tumbuhan', 'Apa yang dibutuhkan tanaman untuk tumbuh?', 'Tanaman butuh air, cahaya matahari, dan udara.'),
        ('Pengetahuan Umum', 'Tumbuhan', 'Apa itu fotosintesis?', 'Fotosintesis adalah proses tumbuhan membuat makanan dari cahaya matahari.'),
        ('Pengetahuan Umum', 'Tumbuhan', 'Apa tumbuhan tertinggi?', 'Pohon redwood adalah tumbuhan tertinggi.'),
        ('Pengetahuan Umum', 'Tumbuhan', 'Apa bunga nasional Indonesia?', 'Melati putih adalah bunga nasional Indonesia.'),
        ('Pengetahuan Umum', 'Tumbuhan', 'Apa buah nasional Indonesia?', 'Apa yang dimaksud dengan buah nasional.'),
        ('Pengetahuan Umum', 'Tumbuhan', 'Apa itu bambu?', 'Bambu adalah tumbuhan yang tumbuh sangat cepat.'),
    ],
    'makanan.jsonl': [
        ('Pengetahuan Umum', 'Makanan', 'Apa makanan pokok Indonesia?', 'Nasi adalah makanan pokok Indonesia.'),
        ('Pengetahuan Umum', 'Makanan', 'Apa makanan pokok Jepang?', 'Nasi dan ikan adalah makanan pokok Jepang.'),
        ('Pengetahuan Umum', 'Makanan', 'Apa makanan pokok Italia?', 'Pasta dan pizza adalah makanan khas Italia.'),
        ('Pengetahuan Umum', 'Makanan', 'Apa makanan khas Indonesia?', 'Nasi goreng, rendang, dan sate adalah makanan khas Indonesia.'),
        ('Pengetahuan Umum', 'Makanan', 'Apa itu gizi?', 'Gizi adalah zat makanan yang dibutuhkan tubuh.'),
        ('Pengetahuan Umum', 'Makanan', 'Apa empat sehat lima sempurna?', 'Empat sehat lima sempurna adalah pedoman gizi Indonesia.'),
        ('Pengetahuan Umum', 'Makanan', 'Apa sumber karbohidrat?', 'Nasi, roti, dan kentang adalah sumber karbohidrat.'),
        ('Pengetahuan Umum', 'Makanan', 'Apa sumber protein?', 'Daging, ikan, telur, dan kacang adalah sumber protein.'),
        ('Pengetahuan Umum', 'Makanan', 'Apa sumber vitamin?', 'Buah dan sayur adalah sumber vitamin.'),
        ('Pengetahuan Umum', 'Makanan', 'Berapa gelas air sehari?', 'Minum delapan gelas air sehari untuk kesehatan.'),
    ],
}
for filename, qa_list in general_items.items():
    items = [{'bidang': 'pengetahuan_umum', 'judul': j, 'subjudul': s, 'prompt': p, 'response': r}
             for (j, s, p, r) in qa_list]
    total += write_jsonl(OUTPUT_DIR / 'pengetahuan_umum' / filename, items)

# ============================================================
# 6. BAHASA & TERJEMAHAN
# ============================================================
print("Generating: Bahasa & Terjemahan...")
lang_items = {
    'sapaan_multi.jsonl': [
        ('Bahasa', 'Sapaan Multi-Bahasa', 'Hello in Indonesian?', 'Hello in Indonesian is Halo.'),
        ('Bahasa', 'Sapaan Multi-Bahasa', 'Hello in Spanish?', 'Hello in Spanish is Hola.'),
        ('Bahasa', 'Sapaan Multi-Bahasa', 'Hello in French?', 'Hello in French is Bonjour.'),
        ('Bahasa', 'Sapaan Multi-Bahasa', 'Hello in German?', 'Hello in German is Hallo.'),
        ('Bahasa', 'Sapaan Multi-Bahasa', 'Hello in Italian?', 'Hello in Italian is Ciao.'),
        ('Bahasa', 'Sapaan Multi-Bahasa', 'Hello in Portuguese?', 'Hello in Portuguese is Ola.'),
        ('Bahasa', 'Sapaan Multi-Bahasa', 'Hello in Russian?', 'Hello in Russian is Privet.'),
        ('Bahasa', 'Sapaan Multi-Bahasa', 'Hello in Japanese?', 'Hello in Japanese is Konnichiwa.'),
        ('Bahasa', 'Sapaan Multi-Bahasa', 'Hello in Korean?', 'Hello in Korean is Annyeong.'),
        ('Bahasa', 'Sapaan Multi-Bahasa', 'Hello in Chinese?', 'Hello in Chinese is Ni hao.'),
        ('Bahasa', 'Sapaan Multi-Bahasa', 'Hello in Arabic?', 'Hello in Arabic is Marhaba.'),
        ('Bahasa', 'Sapaan Multi-Bahasa', 'Hello in Hindi?', 'Hello in Hindi is Namaste.'),
        ('Bahasa', 'Sapaan Multi-Bahasa', 'Hello in Thai?', 'Hello in Thai is Sawasdee.'),
        ('Bahasa', 'Sapaan Multi-Bahasa', 'Hello in Dutch?', 'Hello in Dutch is Hallo.'),
        ('Bahasa', 'Sapaan Multi-Bahasa', 'Hello in Swedish?', 'Hello in Swedish is Hej.'),
    ],
    'terima_kasih_multi.jsonl': [
        ('Bahasa', 'Terima Kasih Multi-Bahasa', 'Thank you in Indonesian?', 'Thank you in Indonesian is Terima kasih.'),
        ('Bahasa', 'Terima Kasih Multi-Bahasa', 'Thank you in Spanish?', 'Thank you in Spanish is Gracias.'),
        ('Bahasa', 'Terima Kasih Multi-Bahasa', 'Thank you in French?', 'Thank you in French is Merci.'),
        ('Bahasa', 'Terima Kasih Multi-Bahasa', 'Thank you in German?', 'Thank you in German is Danke.'),
        ('Bahasa', 'Terima Kasih Multi-Bahasa', 'Thank you in Italian?', 'Thank you in Italian is Grazie.'),
        ('Bahasa', 'Terima Kasih Multi-Bahasa', 'Thank you in Japanese?', 'Thank you in Japanese is Arigatou.'),
        ('Bahasa', 'Terima Kasih Multi-Bahasa', 'Thank you in Korean?', 'Thank you in Korean is Gomapda.'),
        ('Bahasa', 'Terima Kasih Multi-Bahasa', 'Thank you in Chinese?', 'Thank you in Chinese is Xiexie.'),
        ('Bahasa', 'Terima Kasih Multi-Bahasa', 'Thank you in Arabic?', 'Thank you in Arabic is Shukran.'),
        ('Bahasa', 'Terima Kasih Multi-Bahasa', 'Thank you in Hindi?', 'Thank you in Hindi is Dhanyavad.'),
    ],
    'ya_tidak_multi.jsonl': [
        ('Bahasa', 'Ya/Tidak Multi-Bahasa', 'Yes in Indonesian?', 'Yes in Indonesian is Ya.'),
        ('Bahasa', 'Ya/Tidak Multi-Bahasa', 'Yes in Spanish?', 'Yes in Spanish is Si.'),
        ('Bahasa', 'Ya/Tidak Multi-Bahasa', 'Yes in French?', 'Yes in French is Oui.'),
        ('Bahasa', 'Ya/Tidak Multi-Bahasa', 'Yes in German?', 'Yes in German is Ja.'),
        ('Bahasa', 'Ya/Tidak Multi-Bahasa', 'No in Indonesian?', 'No in Indonesian is Tidak.'),
        ('Bahasa', 'Ya/Tidak Multi-Bahasa', 'No in French?', 'No in French is Non.'),
        ('Bahasa', 'Ya/Tidak Multi-Bahasa', 'No in German?', 'No in German is Nein.'),
        ('Bahasa', 'Ya/Tidak Multi-Bahasa', 'No in Spanish?', 'No in Spanish is No.'),
        ('Bahasa', 'Ya/Tidak Multi-Bahasa', 'No in Russian?', 'No in Russian is Nyet.'),
        ('Bahasa', 'Ya/Tidak Multi-Bahasa', 'No in Japanese?', 'No in Japanese is Iie.'),
    ],
    'kosa_kata.jsonl': [
        ('Bahasa', 'Kosa Kata', 'Mother in Indonesian?', 'Mother in Indonesian is Ibu.'),
        ('Bahasa', 'Kosa Kata', 'Father in Indonesian?', 'Father in Indonesian is Ayah.'),
        ('Bahasa', 'Kosa Kata', 'Cat in Indonesian?', 'Cat in Indonesian is Kucing.'),
        ('Bahasa', 'Kosa Kata', 'Dog in Indonesian?', 'Dog in Indonesian is Anjing.'),
        ('Bahasa', 'Kosa Kata', 'Water in Indonesian?', 'Water in Indonesian is Air.'),
        ('Bahasa', 'Kosa Kata', 'Fire in Indonesian?', 'Fire in Indonesian is Api.'),
        ('Bahasa', 'Kosa Kata', 'Book in Indonesian?', 'Book in Indonesian is Buku.'),
        ('Bahasa', 'Kosa Kata', 'House in Indonesian?', 'House in Indonesian is Rumah.'),
        ('Bahasa', 'Kosa Kata', 'Tree in Indonesian?', 'Tree in Indonesian is Pohon.'),
        ('Bahasa', 'Kosa Kata', 'Sun in Indonesian?', 'Sun in Indonesian is Matahari.'),
        ('Bahasa', 'Kosa Kata', 'Moon in Indonesian?', 'Moon in Indonesian is Bulan.'),
        ('Bahasa', 'Kosa Kata', 'Star in Indonesian?', 'Star in Indonesian is Bintang.'),
        ('Bahasa', 'Kosa Kata', 'Sky in Indonesian?', 'Sky in Indonesian is Langit.'),
        ('Bahasa', 'Kosa Kata', 'Mountain in Indonesian?', 'Mountain in Indonesian is Gunung.'),
        ('Bahasa', 'Kosa Kata', 'River in Indonesian?', 'River in Indonesian is Sungai.'),
        ('Bahasa', 'Kosa Kata', 'Sea in Indonesian?', 'Sea in Indonesian is Laut.'),
        ('Bahasa', 'Kosa Kata', 'Flower in Indonesian?', 'Flower in Indonesian is Bunga.'),
        ('Bahasa', 'Kosa Kata', 'Love in Indonesian?', 'Love in Indonesian is Cinta.'),
        ('Bahasa', 'Kosa Kata', 'Friend in Indonesian?', 'Friend in Indonesian is Teman.'),
        ('Bahasa', 'Kosa Kata', 'Happy in Indonesian?', 'Happy in Indonesian is Bahagia.'),
    ],
}
for filename, qa_list in lang_items.items():
    items = [{'bidang': 'bahasa', 'judul': j, 'subjudul': s, 'prompt': p, 'response': r}
             for (j, s, p, r) in qa_list]
    total += write_jsonl(OUTPUT_DIR / 'bahasa' / filename, items)

# ============================================================
# 7. WAKTU & KALENDER
# ============================================================
print("Generating: Waktu & Kalender...")
time_items = [
    ('Waktu', 'Satuan Waktu', 'Berapa detik dalam satu menit?', 'Ada enam puluh detik dalam satu menit.'),
    ('Waktu', 'Satuan Waktu', 'Berapa menit dalam satu jam?', 'Ada enam puluh menit dalam satu jam.'),
    ('Waktu', 'Satuan Waktu', 'Berapa jam dalam satu hari?', 'Ada dua puluh empat jam dalam satu hari.'),
    ('Waktu', 'Satuan Waktu', 'Berapa hari dalam seminggu?', 'Ada tujuh hari dalam seminggu.'),
    ('Waktu', 'Satuan Waktu', 'Berapa minggu dalam satu bulan?', 'Ada sekitar empat minggu dalam satu bulan.'),
    ('Waktu', 'Satuan Waktu', 'Berapa bulan dalam setahun?', 'Ada dua belas bulan dalam setahun.'),
    ('Waktu', 'Satuan Waktu', 'Berapa hari dalam setahun?', 'Ada tiga ratus enam puluh lima hari dalam setahun.'),
    ('Waktu', 'Nama Hari', 'Apa tujuh hari dalam seminggu?', 'Senin Selasa Rabu Kamis Jumat Sabtu Minggu.'),
    ('Waktu', 'Nama Bulan', 'Apa dua belas bulan dalam setahun?', 'Januari Februari Maret April Mei Juni Juli Agustus September Oktober November Desember.'),
    ('Waktu', 'Urutan Hari', 'Setelah Senin adalah?', 'Setelah Senin adalah Selasa.'),
    ('Waktu', 'Urutan Hari', 'Setelah Selasa adalah?', 'Setelah Selasa adalah Rabu.'),
    ('Waktu', 'Urutan Hari', 'Setelah Rabu adalah?', 'Setelah Rabu adalah Kamis.'),
    ('Waktu', 'Urutan Hari', 'Setelah Kamis adalah?', 'Setelah Kamis adalah Jumat.'),
    ('Waktu', 'Urutan Hari', 'Setelah Jumat adalah?', 'Setelah Jumat adalah Sabtu.'),
    ('Waktu', 'Urutan Hari', 'Setelah Sabtu adalah?', 'Setelah Sabtu adalah Minggu.'),
    ('Waktu', 'Urutan Hari', 'Setelah Minggu adalah?', 'Setelah Minggu adalah Senin.'),
    ('Waktu', 'Urutan Bulan', 'Setelah Januari adalah?', 'Setelah Januari adalah Februari.'),
    ('Waktu', 'Urutan Bulan', 'Setelah Februari adalah?', 'Setelah Februari adalah Maret.'),
    ('Waktu', 'Musim', 'Apa empat musim di bumi?', 'Empat musim adalah semi, panas, gugur, dan dingin.'),
    ('Waktu', 'Musim', 'Apa musim di Indonesia?', 'Indonesia memiliki dua musim: hujan dan kemarau.'),
    ('Waktu', 'Zona Waktu', 'Apa zona waktu Indonesia?', 'Indonesia memiliki tiga zona waktu: WIB, WITA, WIT.'),
]
items = [{'bidang': 'waktu_kalender', 'judul': j, 'subjudul': s, 'prompt': p, 'response': r}
         for (j, s, p, r) in time_items]
total += write_jsonl(OUTPUT_DIR / 'waktu_kalender' / 'waktu.jsonl', items)

# ============================================================
# 8. KELUARGA & SOSIAL
# ============================================================
print("Generating: Keluarga & Sosial...")
social_items = [
    ('Keluarga', 'Anggota Keluarga', 'Siapa ayah dan ibu?', 'Ayah dan ibu adalah orang tua.'),
    ('Keluarga', 'Anggota Keluarga', 'Siapa kakak?', 'Kakak adalah saudara yang lebih tua.'),
    ('Keluarga', 'Anggota Keluarga', 'Siapa adik?', 'Adik adalah saudara yang lebih muda.'),
    ('Keluarga', 'Anggota Keluarga', 'Siapa kakek?', 'Kakek adalah ayah dari orang tua.'),
    ('Keluarga', 'Anggota Keluarga', 'Siapa nenek?', 'Nenek adalah ibu dari orang tua.'),
    ('Keluarga', 'Anggota Keluarga', 'Siapa paman?', 'Paman adalah saudara laki-laki orang tua.'),
    ('Keluarga', 'Anggota Keluarga', 'Siapa tante?', 'Tante adalah saudara perempuan orang tua.'),
    ('Keluarga', 'Anggota Keluarga', 'Siapa sepupu?', 'Sepupu adalah anak paman atau tante.'),
    ('Keluarga', 'Anggota Keluarga', 'Siapa keponakan?', 'Keponakan adalah anak dari saudara.'),
    ('Sosial', 'Aturan Sosial', 'Apa itu antre?', 'Antre adalah menunggu giliran dengan tertib.'),
    ('Sosial', 'Aturan Sosial', 'Apa itu bergiliran?', 'Bergiliran adalah mengambil giliran satu per satu.'),
    ('Sosial', 'Aturan Sosial', 'Apa itu berbagi?', 'Berbagi adalah memberi sebagian ke orang lain.'),
    ('Sosial', 'Sopan Santun', 'Apa kata minta tolong?', 'Tolong adalah kata sopan minta bantuan.'),
    ('Sosial', 'Sopan Santun', 'Apa balasan terima kasih?', 'Sama-sama adalah balasan untuk terima kasih.'),
    ('Sosial', 'Sopan Santun', 'Apa kata minta maaf?', 'Maaf adalah kata untuk minta permisi.'),
    ('Sosial', 'Sopan Santun', 'Apa kata permisi?', 'Permisi adalah kata untuk melewati atau minta izin.'),
    ('Sosial', 'Sopan Santun', 'Apa itu sopan?', 'Sopan adalah sikap baik kepada orang lain.'),
    ('Sosial', 'Emosi', 'Apa itu senang?', 'Senang adalah perasaan gembira.'),
    ('Sosial', 'Emosi', 'Apa itu sedih?', 'Sedih adalah perasaan kecewa.'),
    ('Sosial', 'Emosi', 'Apa itu marah?', 'Marah adalah perasaan kesal.'),
    ('Sosial', 'Emosi', 'Apa itu takut?', 'Takut adalah perasaan cemas.'),
]
items = [{'bidang': 'keluarga_sosial', 'judul': j, 'subjudul': s, 'prompt': p, 'response': r}
         for (j, s, p, r) in social_items]
total += write_jsonl(OUTPUT_DIR / 'keluarga_sosial' / 'sosial.jsonl', items)

# ============================================================
# 9. TEKNOLOGI
# ============================================================
print("Generating: Teknologi...")
tech_items = [
    ('Teknologi', 'Komputer', 'Apa itu komputer?', 'Komputer adalah alat untuk menghitung dan bekerja.'),
    ('Teknologi', 'Komputer', 'Apa itu tablet?', 'Tablet adalah komputer kecil layar sentuh.'),
    ('Teknologi', 'Komputer', 'Apa itu smartphone?', 'Smartphone adalah ponsel pintar.'),
    ('Teknologi', 'Internet', 'Apa itu internet?', 'Internet adalah jaringan yang menghubungkan komputer.'),
    ('Teknologi', 'Internet', 'Apa itu website?', 'Website adalah halaman informasi di internet.'),
    ('Teknologi', 'Internet', 'Apa itu YouTube?', 'YouTube adalah situs berbagi video.'),
    ('Teknologi', 'Internet', 'Apa itu email?', 'Email adalah surat elektronik.'),
    ('Teknologi', 'AI', 'Apa itu AI?', 'AI adalah kecerdasan buatan.'),
    ('Teknologi', 'AI', 'Apa itu LLM?', 'LLM adalah Large Language Model.'),
    ('Teknologi', 'AI', 'Apa itu machine learning?', 'Machine learning adalah mesin yang belajar dari data.'),
    ('Teknologi', 'Programming', 'Apa itu kode?', 'Kode adalah instruksi untuk komputer.'),
    ('Teknologi', 'Programming', 'Apa itu Python?', 'Python adalah bahasa pemrograman.'),
    ('Teknologi', 'Programming', 'Apa itu JavaScript?', 'JavaScript adalah bahasa untuk website.'),
    ('Teknologi', 'Game', 'Apa itu Minecraft?', 'Minecraft adalah permainan membangun dengan balok.'),
    ('Teknologi', 'Game', 'Apa itu Roblox?', 'Roblox adalah platform permainan online.'),
]
items = [{'bidang': 'teknologi', 'judul': j, 'subjudul': s, 'prompt': p, 'response': r}
         for (j, s, p, r) in tech_items]
total += write_jsonl(OUTPUT_DIR / 'teknologi' / 'teknologi.jsonl', items)

# ============================================================
# 10. KESEHATAN & TUBUH
# ============================================================
print("Generating: Kesehatan & Tubuh...")
health_items = [
    ('Kesehatan', 'Anggota Tubuh', 'Apa fungsi mata?', 'Mata adalah alat untuk melihat.'),
    ('Kesehatan', 'Anggota Tubuh', 'Apa fungsi hidung?', 'Hidung adalah alat untuk mencium bau.'),
    ('Kesehatan', 'Anggota Tubuh', 'Apa fungsi mulut?', 'Mulut adalah alat untuk makan dan bicara.'),
    ('Kesehatan', 'Anggota Tubuh', 'Apa fungsi telinga?', 'Telinga adalah alat untuk mendengar.'),
    ('Kesehatan', 'Anggota Tubuh', 'Apa fungsi tangan?', 'Tangan adalah alat untuk memegang.'),
    ('Kesehatan', 'Anggota Tubuh', 'Apa fungsi kaki?', 'Kaki adalah alat untuk berjalan.'),
    ('Kesehatan', 'Anggota Tubuh', 'Berapa jari di tangan?', 'Ada lima jari di setiap tangan.'),
    ('Kesehatan', 'Anggota Tubuh', 'Apa fungsi jantung?', 'Jantung memompa darah ke seluruh tubuh.'),
    ('Kesehatan', 'Anggota Tubuh', 'Apa fungsi paru-paru?', 'Paru-paru menukar oksigen dan karbon dioksida.'),
    ('Kesehatan', 'Anggota Tubuh', 'Apa fungsi otak?', 'Otak mengendalikan seluruh fungsi tubuh.'),
    ('Kesehatan', 'Anggota Tubuh', 'Apa fungsi gigi?', 'Gigi adalah untuk mengunyah makanan.'),
    ('Kesehatan', 'Gizi', 'Berapa gelas air sehari?', 'Minum delapan gelas air sehari.'),
    ('Kesehatan', 'Gizi', 'Apa sumber vitamin?', 'Buah dan sayur adalah sumber vitamin.'),
    ('Kesehatan', 'Gizi', 'Apa sumber protein?', 'Daging, ikan, telur, dan kacang adalah sumber protein.'),
    ('Kesehatan', 'Gizi', 'Apa sumber karbohidrat?', 'Nasi, roti, dan kentang adalah sumber karbohidrat.'),
    ('Kesehatan', 'Kebersihan', 'Apa itu mandi?', 'Mandi adalah membersihkan tubuh dengan air.'),
    ('Kesehatan', 'Kebersihan', 'Apa itu sikat gigi?', 'Sikat gigi membersihkan gigi dari kuman.'),
    ('Kesehatan', 'Kebersihan', 'Berapa kali sikat gigi sehari?', 'Sikat gigi dua kali sehari pagi dan malam.'),
    ('Kesehatan', 'Kebersihan', 'Apa itu cuci tangan?', 'Cuci tangan membersihkan tangan dari kuman.'),
    ('Kesehatan', 'Olahraga', 'Apa itu olahraga?', 'Olahraga adalah aktivitas fisik untuk kesehatan.'),
]
items = [{'bidang': 'kesehatan', 'judul': j, 'subjudul': s, 'prompt': p, 'response': r}
         for (j, s, p, r) in health_items]
total += write_jsonl(OUTPUT_DIR / 'kesehatan' / 'kesehatan.jsonl', items)

# ============================================================
# 11. POLA & LOGIKA
# ============================================================
print("Generating: Pola & Logika...")
logic_items = [
    ('Logika', 'Pola Angka', 'Pola satu dua tiga empat. Berikutnya?', 'Berikutnya adalah lima.'),
    ('Logika', 'Pola Angka', 'Pola dua empat enam delapan. Berikutnya?', 'Berikutnya adalah sepuluh.'),
    ('Logika', 'Pola Angka', 'Pola lima sepuluh lima belas. Berikutnya?', 'Berikutnya adalah dua puluh.'),
    ('Logika', 'Pola Angka', 'Pola tiga enam sembilan. Berikutnya?', 'Berikutnya adalah dua belas.'),
    ('Logika', 'Pola Huruf', 'Pola A B C D. Berikutnya?', 'Berikutnya adalah E.'),
    ('Logika', 'Pola Hari', 'Pola Senin Selasa Rabu. Berikutnya?', 'Berikutnya adalah Kamis.'),
    ('Logika', 'Pola Bulan', 'Pola Januari Februari Maret. Berikutnya?', 'Berikutnya adalah April.'),
    ('Logika', 'Perbandingan', 'Lebih besar gajah atau semut?', 'Gajah lebih besar dari semut.'),
    ('Logika', 'Perbandingan', 'Lebih kecil semut atau gajah?', 'Semut lebih kecil dari gajah.'),
    ('Logika', 'Perbandingan', 'Lebih berat batu atau kapas?', 'Batu lebih berat dari kapas.'),
    ('Logika', 'Perbandingan', 'Lebih tinggi gunung atau pohon?', 'Gunung lebih tinggi dari pohon.'),
    ('Logika', 'Perbandingan', 'Lebih cepat cheetah atau kura-kura?', 'Cheetah lebih cepat dari kura-kura.'),
    ('Logika', 'Sebab-Akibat', 'Kalau hujan, apa yang terjadi?', 'Kalau hujan, tanaman tumbuh subur.'),
    ('Logika', 'Sebab-Akibat', 'Kalau tidak makan, apa yang terjadi?', 'Kalau tidak makan, perut menjadi lapar.'),
    ('Logika', 'Sebab-Akibat', 'Kalau panas, apa yang terjadi pada es?', 'Kalau panas, es akan mencair.'),
    ('Logika', 'Sebab-Akibat', 'Kalau dingin, apa yang terjadi pada air?', 'Kalau dingin, air menjadi beku.'),
    ('Logika', 'Bentuk', 'Apa bentuk bola?', 'Bentuk bola adalah lingkaran.'),
    ('Logika', 'Bentuk', 'Berapa sisi segitiga?', 'Segitiga punya tiga sisi.'),
    ('Logika', 'Bentuk', 'Berapa sisi persegi?', 'Persegi punya empat sisi.'),
    ('Logika', 'Bentuk', 'Apa bentuk telur?', 'Bentuk telur adalah oval.'),
]
items = [{'bidang': 'pola_logika', 'judul': j, 'subjudul': s, 'prompt': p, 'response': r}
         for (j, s, p, r) in logic_items]
total += write_jsonl(OUTPUT_DIR / 'pola_logika' / 'logika.jsonl', items)

# ============================================================
# 12. KONVERSASI & SOPAN SANTUN
# ============================================================
print("Generating: Konversasi...")
conv_items = [
    ('Konversasi', 'Sapaan', 'Halo', 'Halo juga! Senang bertemu denganmu.'),
    ('Konversasi', 'Sapaan', 'halo', 'Halo juga! Senang bertemu denganmu.'),
    ('Konversasi', 'Sapaan', 'Hai', 'Hai! Apa kabar?'),
    ('Konversasi', 'Sapaan', 'Halo apa kabar?', 'Halo! Aku baik-baik saja, terima kasih.'),
    ('Konversasi', 'Sapaan', 'halo apa kabar', 'Halo! Aku baik-baik saja, terima kasih.'),
    ('Konversasi', 'Sapaan', 'Apa kabar?', 'Aku baik-baik saja, terima kasih sudah bertanya!'),
    ('Konversasi', 'Sapaan', 'apa kabar', 'Aku baik-baik saja, terima kasih sudah bertanya!'),
    ('Konversasi', 'Sapaan', 'Selamat pagi', 'Selamat pagi juga! Semoga harimu menyenangkan.'),
    ('Konversasi', 'Sapaan', 'selamat pagi', 'Selamat pagi juga! Semoga harimu menyenangkan.'),
    ('Konversasi', 'Sapaan', 'Selamat malam', 'Selamat malam! Selamat istirahat.'),
    ('Konversasi', 'Identitas', 'Siapa kamu?', 'Aku D\'QUILANE, LLM buatan Indonesia dari nol.'),
    ('Konversasi', 'Identitas', 'siapa kamu', 'Aku D\'QUILANE, LLM buatan Indonesia dari nol.'),
    ('Konversasi', 'Identitas', 'Kamu siapa?', 'Aku D\'QUILANE, LLM buatan Indonesia dari nol.'),
    ('Konversasi', 'Identitas', 'Nama kamu siapa?', 'Namaku D\'QUILANE. Aku LLM multibahasa.'),
    ('Konversasi', 'Identitas', 'Kamu dari mana?', 'Aku dari Indonesia.'),
    ('Konversasi', 'Identitas', 'Kamu bisa apa?', 'Aku bisa matematika, terjemahan, sains, geografi, dan banyak lagi.'),
    ('Konversasi', 'Penutup', 'Terima kasih', 'Sama-sama! Ada yang bisa aku bantu lagi?'),
    ('Konversasi', 'Penutup', 'terima kasih', 'Sama-sama! Ada yang bisa aku bantu lagi?'),
    ('Konversasi', 'Penutup', 'Makasih', 'Sama-sama!'),
    ('Konversasi', 'Penutup', 'Goodbye', 'Sampai jumpa! Semoga harimu menyenangkan.'),
    ('Konversasi', 'Penutup', 'Dadah', 'Dadah! Sampai bertemu lagi.'),
    ('Konversasi', 'Fallback', 'test', 'Halo! Aku D\'QUILANE. Ada yang bisa aku bantu?'),
    ('Konversasi', 'Fallback', 'lagi apa', 'Aku siap membantu. Ada yang bisa aku bantu?'),
    ('Konversasi', 'Fallback', 'coba ngomong', 'Halo! Aku D\'QUILANE. Ada yang bisa aku bantu?'),
    ('Konversasi', 'Fallback', 'kamu bisa apa', 'Aku bisa matematika, terjemahan, sains, geografi, dan jawab pertanyaan.'),
    ('Konversasi', 'Fallback', 'halo dunia', 'Halo! Aku D\'QUILANE, LLM buatan Indonesia.'),
    ('Konversasi', 'Fallback', 'Siapa presiden Amerika?', 'Presiden Amerika dipilih melalui pemilu demokratis.'),
    ('Konversasi', 'Fallback', 'Apa berita hari ini?', 'Aku tidak tahu berita terbaru, cek sumber berita terpercaya.'),
    ('Konversasi', 'Fallback', 'Ceritakan cerita', 'Aku bisa jawab pertanyaan tentang banyak topik.'),
    ('Konversasi', 'Fallback', 'Bagaimana cuaca hari ini?', 'Aku tidak tahu cuaca hari ini, cek aplikasi cuaca.'),
    ('Konversasi', 'Fallback', 'Jam berapa sekarang', 'Aku tidak tahu jam berapa sekarang, cek jam.'),
    ('Konversasi', 'Fallback', 'jepang di mana', 'Jepang adalah negara di Asia Timur.'),
]
items = [{'bidang': 'konversasi', 'judul': j, 'subjudul': s, 'prompt': p, 'response': r}
         for (j, s, p, r) in conv_items]
total += write_jsonl(OUTPUT_DIR / 'konversasi' / 'konversasi.jsonl', items)

# ============================================================
# 13. UANG & EKONOMI
# ============================================================
print("Generating: Uang & Ekonomi...")
money_items = [
    ('Ekonomi', 'Mata Uang', 'Apa mata uang Indonesia?', 'Mata uang Indonesia adalah rupiah.'),
    ('Ekonomi', 'Mata Uang', 'Apa mata uang Jepang?', 'Mata uang Jepang adalah yen.'),
    ('Ekonomi', 'Mata Uang', 'Apa mata uang Amerika?', 'Mata uang Amerika adalah dolar.'),
    ('Ekonomi', 'Mata Uang', 'Apa mata uang Eropa?', 'Mata uang Eropa adalah euro.'),
    ('Ekonomi', 'Mata Uang', 'Apa mata uang Inggris?', 'Mata uang Inggris adalah pound sterling.'),
    ('Ekonomi', 'Mata Uang', 'Apa mata uang Korea Selatan?', 'Mata uang Korea Selatan adalah won.'),
    ('Ekonomi', 'Mata Uang', 'Apa mata uang Tiongkok?', 'Mata uang Tiongkok adalah yuan.'),
    ('Ekonomi', 'Mata Uang', 'Apa mata uang India?', 'Mata uang India adalah rupee.'),
    ('Ekonomi', 'Mata Uang', 'Apa mata uang Swiss?', 'Mata uang Swiss adalah franc.'),
    ('Ekonomi', 'Mata Uang', 'Apa mata uang Australia?', 'Mata uang Australia adalah dolar Australia.'),
    ('Ekonomi', 'Konsep', 'Apa itu harga?', 'Harga adalah nilai uang dari suatu barang.'),
    ('Ekonomi', 'Konsep', 'Apa itu beli?', 'Beli adalah menukar uang dengan barang.'),
    ('Ekonomi', 'Konsep', 'Apa itu bayar?', 'Bayar adalah memberi uang untuk barang.'),
    ('Ekonomi', 'Konsep', 'Apa itu kembalian?', 'Kembalian adalah uang kembali setelah bayar.'),
    ('Ekonomi', 'Konsep', 'Sepuluh ribu atau seribu lebih besar?', 'Sepuluh ribu lebih besar dari seribu.'),
]
items = [{'bidang': 'uang_ekonomi', 'judul': j, 'subjudul': s, 'prompt': p, 'response': r}
         for (j, s, p, r) in money_items]
total += write_jsonl(OUTPUT_DIR / 'uang_ekonomi' / 'uang.jsonl', items)

# ============================================================
# 14. CUACA & ALAM
# ============================================================
print("Generating: Cuaca & Alam...")
nature_items = [
    ('Cuaca', 'Jenis Cuaca', 'Apa itu hujan?', 'Hujan adalah air yang turun dari langit.'),
    ('Cuaca', 'Jenis Cuaca', 'Apa itu panas?', 'Panas adalah cuaca yang sangat hangat.'),
    ('Cuaca', 'Jenis Cuaca', 'Apa itu dingin?', 'Dingin adalah cuaca yang sangat sejuk.'),
    ('Cuaca', 'Jenis Cuaca', 'Apa itu mendung?', 'Mendung adalah awan tebal sebelum hujan.'),
    ('Cuaca', 'Jenis Cuaca', 'Apa itu berangin?', 'Berangin adalah cuaca dengan angin kencang.'),
    ('Cuaca', 'Jenis Cuaca', 'Apa itu cerah?', 'Cerah adalah cuaca tanpa awan.'),
    ('Cuaca', 'Fenomena', 'Kapan pelangi muncul?', 'Pelangi muncul setelah hujan dengan sinar matahari.'),
    ('Cuaca', 'Fenomena', 'Apa itu banjir?', 'Banjir adalah air berlebih di daratan.'),
    ('Cuaca', 'Fenomena', 'Apa itu petir?', 'Petir adalah pelepasan listrik di atmosfer.'),
    ('Cuaca', 'Fenomena', 'Apa itu angin?', 'Angin adalah udara yang bergerak.'),
    ('Alam', 'Pegunungan', 'Apa gunung tertinggi di dunia?', 'Gunung Everest adalah gunung tertinggi di dunia.'),
    ('Alam', 'Pegunungan', 'Apa gunung berapi?', 'Gunung berapi adalah bukit yang mengeluarkan magma.'),
    ('Alam', 'Sungai', 'Apa sungai terpanjang di dunia?', 'Sungai Nil adalah sungai terpanjang di dunia.'),
    ('Alam', 'Laut', 'Apa laut terbesar?', 'Samudra Pasifik adalah laut terbesar.'),
    ('Alam', 'Hutan', 'Apa fungsi hutan?', 'Hutan menyimpan air dan menghasilkan oksigen.'),
]
items = [{'bidang': 'cuaca_alam', 'judul': j, 'subjudul': s, 'prompt': p, 'response': r}
         for (j, s, p, r) in nature_items]
total += write_jsonl(OUTPUT_DIR / 'cuaca_alam' / 'cuaca_alam.jsonl', items)

# ============================================================
# 15. TRANSPORTASI & ARAH
# ============================================================
print("Generating: Transportasi...")
transport_items = [
    ('Transportasi', 'Kendaraan', 'Apa kendaraan darat?', 'Mobil, motor, dan sepeda adalah kendaraan darat.'),
    ('Transportasi', 'Kendaraan', 'Apa kendaraan air?', 'Kapal dan perahu adalah kendaraan air.'),
    ('Transportasi', 'Kendaraan', 'Apa kendaraan udara?', 'Pesawat dan helikopter adalah kendaraan udara.'),
    ('Transportasi', 'Kendaraan', 'Apa transportasi umum?', 'Bus dan kereta adalah transportasi umum.'),
    ('Transportasi', 'Kendaraan', 'Apa itu pesawat?', 'Pesawat adalah kendaraan terbang.'),
    ('Arah', 'Mata Angin', 'Arah apa di atas peta?', 'Utara adalah arah ke atas di peta.'),
    ('Arah', 'Mata Angin', 'Arah apa di bawah peta?', 'Selatan adalah arah ke bawah di peta.'),
    ('Arah', 'Mata Angin', 'Matahari terbit di?', 'Matahari terbit di timur.'),
    ('Arah', 'Mata Angin', 'Matahari terbenam di?', 'Matahari terbenam di barat.'),
    ('Arah', 'Alat', 'Apa itu kompas?', 'Kompas adalah alat penunjuk arah.'),
]
items = [{'bidang': 'transportasi', 'judul': j, 'subjudul': s, 'prompt': p, 'response': r}
         for (j, s, p, r) in transport_items]
total += write_jsonl(OUTPUT_DIR / 'transportasi' / 'transportasi.jsonl', items)

# ============================================================
# 16. SENI & BUDAYA
# ============================================================
print("Generating: Seni & Budaya...")
arts_items = [
    ('Seni', 'Musik', 'Apa alat musik piano?', 'Piano adalah alat musik dengan tuts.'),
    ('Seni', 'Musik', 'Apa alat musik gitar?', 'Gitar adalah alat musik berdawai.'),
    ('Seni', 'Musik', 'Apa alat musik drum?', 'Drum adalah alat musik perkusi.'),
    ('Seni', 'Musik', 'Apa alat musik biola?', 'Biola adalah alat musik berdawai digesek.'),
    ('Seni', 'Lukisan', 'Apa itu lukisan?', 'Lukisan adalah karya seni visual di kanvas.'),
    ('Seni', 'Sastra', 'Apa itu puisi?', 'Puisi adalah karya sastra berbait indah.'),
    ('Seni', 'Sastra', 'Apa itu pantun?', 'Pantun adalah puisi pendek Melayu.'),
    ('Budaya', 'Tari', 'Apa tarian tradisional Bali?', 'Tari Kecak dan Tari Legong adalah tarian Bali.'),
    ('Budaya', 'Tari', 'Apa tarian tradisional Sumatra?', 'Tari Piring dan Tari Tor-tor adalah tarian Sumatra.'),
    ('Budaya', 'Pakaian', 'Apa pakaian adat Jawa?', 'Kebaya dan batik adalah pakaian adat Jawa.'),
    ('Budaya', 'Pakaian', 'Apa pakaian adat Sumatra?', 'Baju Kurung dan Ulos adalah pakaian adat Sumatra.'),
    ('Budaya', 'Rumah', 'Apa rumah adat Jawa?', 'Joglo adalah rumah adat Jawa.'),
    ('Budaya', 'Rumah', 'Apa rumah adat Sumatra?', 'Rumah Gadang adalah rumah adat Minangkabau.'),
    ('Budaya', 'Senjata', 'Apa senjata tradisional Indonesia?', 'Keris adalah senjata tradisional Indonesia.'),
    ('Budaya', 'Pertunjukan', 'Apa pertunjukan wayang?', 'Wayang adalah pertunjukan boneka tradisional.'),
]
items = [{'bidang': 'seni_budaya', 'judul': j, 'subjudul': s, 'prompt': p, 'response': r}
         for (j, s, p, r) in arts_items]
total += write_jsonl(OUTPUT_DIR / 'seni_budaya' / 'seni_budaya.jsonl', items)

# ============================================================
# 17. BENDA HARIAN
# ============================================================
print("Generating: Benda Harian...")
daily_items = [
    ('Benda Harian', 'Rumah', 'Apa itu kursi?', 'Kursi adalah tempat duduk.'),
    ('Benda Harian', 'Rumah', 'Apa itu meja?', 'Meja adalah tempat meletakkan benda.'),
    ('Benda Harian', 'Rumah', 'Apa itu kasur?', 'Kasur adalah tempat tidur.'),
    ('Benda Harian', 'Rumah', 'Apa itu lemari?', 'Lemari adalah tempat menyimpan baju.'),
    ('Benda Harian', 'Sekolah', 'Apa itu pensil?', 'Pensil adalah alat tulis dari kayu.'),
    ('Benda Harian', 'Sekolah', 'Apa itu buku?', 'Buku adalah kumpulan kertas bertulisan.'),
    ('Benda Harian', 'Sekolah', 'Apa itu penghapus?', 'Penghapus menghapus tulisan pensil.'),
    ('Benda Harian', 'Sekolah', 'Apa itu penggaris?', 'Penggaris mengukur panjang.'),
    ('Benda Harian', 'Dapur', 'Apa itu sendok?', 'Sendok adalah alat makan sup.'),
    ('Benda Harian', 'Dapur', 'Apa itu garpu?', 'Garpu adalah alat makan bertungkai.'),
    ('Benda Harian', 'Dapur', 'Apa itu pisau?', 'Pisau adalah alat memotong.'),
    ('Benda Harian', 'Dapur', 'Apa itu piring?', 'Piring adalah tempat makanan.'),
    ('Benda Harian', 'Elektronik', 'Apa itu lampu?', 'Lampu memberi penerangan.'),
    ('Benda Harian', 'Elektronik', 'Apa itu kipas angin?', 'Kipas angin menghasilkan angin buatan.'),
    ('Benda Harian', 'Elektronik', 'Apa itu TV?', 'TV adalah televisi untuk hiburan dan informasi.'),
]
items = [{'bidang': 'benda_harian', 'judul': j, 'subjudul': s, 'prompt': p, 'response': r}
         for (j, s, p, r) in daily_items]
total += write_jsonl(OUTPUT_DIR / 'benda_harian' / 'benda_harian.jsonl', items)

# ============================================================
# 18. KESELAMATAN
# ============================================================
print("Generating: Keselamatan...")
safety_items = [
    ('Keselamatan', 'Lalu Lintas', 'Apa warna lampu merah?', 'Lampu merah berarti berhenti.'),
    ('Keselamatan', 'Lalu Lintas', 'Apa warna lampu hijau?', 'Lampu hijau berarti jalan.'),
    ('Keselamatan', 'Lalu Lintas', 'Apa warna lampu kuning?', 'Lampu kuning berarti hati-hati.'),
    ('Keselamatan', 'Lalu Lintas', 'Kenapa pakai helm?', 'Helm melindungi kepala saat naik motor.'),
    ('Keselamatan', 'Lalu Lintas', 'Kenapa pakai sabuk pengaman?', 'Sabuk pengaman melindungi di mobil.'),
    ('Keselamatan', 'Darurat', 'Apa nomor polisi?', 'Nomor polisi adalah seratus sepuluh.'),
    ('Keselamatan', 'Darurat', 'Apa nomor pemadam kebakaran?', 'Nomor pemadam kebakaran adalah seratus tiga belas.'),
    ('Keselamatan', 'Darurat', 'Apa nomor ambulans?', 'Nomor ambulans adalah seratus sembilan.'),
    ('Keselamatan', 'Darurat', 'Apa nomor SAR?', 'Nomor SAR adalah seratus tiga belas.'),
    ('Keselamatan', 'Bahaya', 'Apa bahaya main api?', 'Main api bisa menyebabkan kebakaran.'),
    ('Keselamatan', 'Bahaya', 'Apa bahaya main listrik?', 'Main listrik bisa tersengat.'),
    ('Keselamatan', 'Bahaya', 'Kenapa tidak boleh main di jalan?', 'Bermain di jalan bisa tertabrak kendaraan.'),
    ('Keselamatan', 'Bahaya', 'Kenapa tidak boleh bicara ke orang asing?', 'Bicara ke orang asing bisa berbahaya.'),
    ('Keselamatan', 'Pertolongan', 'Apa P3K?', 'P3K adalah pertolongan pertama pada kecelakaan.'),
    ('Keselamatan', 'Pertolongan', 'Apa itu slidër?', 'Slider adalah alat darurat di kolam renang.'),
]
items = [{'bidang': 'keselamatan', 'judul': j, 'subjudul': s, 'prompt': p, 'response': r}
         for (j, s, p, r) in safety_items]
total += write_jsonl(OUTPUT_DIR / 'keselamatan' / 'keselamatan.jsonl', items)

# ============================================================
# 19. ANGKA & BERHITUNG
# ============================================================
print("Generating: Angka & Berhitung...")
counting_items = []
# Counting 1-100
for i in range(1, 101):
    counting_items.append({
        'bidang': 'angka_berhitung',
        'judul': 'Bilangan',
        'subjudul': f'Satu sampai Seratus',
        'prompt': f'Angka {i} in words?',
        'response': f'Angka {i} adalah {num_to_words(i)}.'
    })
# Even/odd
for i in range(1, 51):
    parity = 'genap' if i % 2 == 0 else 'ganjil'
    counting_items.append({
        'bidang': 'angka_berhitung',
        'judul': 'Genap dan Ganjil',
        'subjudul': 'Konsep',
        'prompt': f'Apakah {num_to_words(i)} genap atau ganjil?',
        'response': f'{num_to_words(i).capitalize()} adalah bilangan {parity}.'
    })
total += write_jsonl(OUTPUT_DIR / 'angka_berhitung' / 'bilangan.jsonl', counting_items)

# ============================================================
# 20. BENDA HIDUP
# ============================================================
print("Generating: Benda Hidup...")
living_items = [
    ('Benda Hidup', 'Klasifikasi', 'Apa lima kerajaan makhluk hidup?', 'Lima kerajaan: Monera, Protista, Fungi, Plantae, Animalia.'),
    ('Benda Hidup', 'Klasifikasi', 'Apa ciri makhluk hidup?', 'Makhluk hidup bernapas, makan, tumbuh, berkembang biak.'),
    ('Benda Hidup', 'Klasifikasi', 'Apa benda tak hidup?', 'Batu, air, dan udara adalah benda tak hidup.'),
    ('Benda Hidup', 'Mamalia', 'Apa itu mamalia?', 'Mamalia adalah hewan yang menyusui anaknya.'),
    ('Benda Hidup', 'Mamalia', 'Contoh mamalia?', 'Kucing, anjing, sapi, dan manusia adalah mamalia.'),
    ('Benda Hidup', 'Burung', 'Apa ciri burung?', 'Burung memiliki bulu, paruh, dan bisa terbang.'),
    ('Benda Hidup', 'Ikan', 'Apa ciri ikan?', 'Ikan memiliki sirip, sisik, dan hidup di air.'),
    ('Benda Hidup', 'Reptil', 'Apa ciri reptil?', 'Reptil memiliki sisik dan berkembang biak bertelur.'),
    ('Benda Hidup', 'Serangga', 'Apa ciri serangga?', 'Serangga memiliki enam kaki dan tiga bagian tubuh.'),
    ('Benda Hidup', 'Amfibi', 'Apa ciri amfibi?', 'Amfibi hidup di darat dan air, contoh katak.'),
]
items = [{'bidang': 'benda_hidup', 'judul': j, 'subjudul': s, 'prompt': p, 'response': r}
         for (j, s, p, r) in living_items]
total += write_jsonl(OUTPUT_DIR / 'benda_hidup' / 'klasifikasi.jsonl', items)

# ============================================================
# 21. PENDIDIKAN
# ============================================================
print("Generating: Pendidikan...")
edu_items = [
    ('Pendidikan', 'Sekolah', 'Apa itu sekolah?', 'Sekolah adalah tempat belajar.'),
    ('Pendidikan', 'Sekolah', 'Apa tingkat sekolah di Indonesia?', 'SD, SMP, SMA, dan kuliah.'),
    ('Pendidikan', 'Sekolah', 'Berapa tahun SD?', 'SD dijalani selama enam tahun.'),
    ('Pendidikan', 'Sekolah', 'Berapa tahun SMP?', 'SMP dijalani selama tiga tahun.'),
    ('Pendidikan', 'Sekolah', 'Berapa tahun SMA?', 'SMA dijalani selama tiga tahun.'),
    ('Pendidikan', 'Belajar', 'Apa itu belajar?', 'Belajar adalah menambah pengetahuan.'),
    ('Pendidikan', 'Belajar', 'Kenapa harus belajar?', 'Belajar membuat kita pintar dan siap masa depan.'),
    ('Pendidikan', 'Belajar', 'Apa itu PR?', 'PR adalah pekerjaan rumah dari sekolah.'),
    ('Pendidikan', 'Belajar', 'Apa itu ujian?', 'Ujian adalah tes untuk mengukur pemahaman.'),
    ('Pendidikan', 'Belajar', 'Apa itu nilai?', 'Nilai adalah angka penilaian hasil belajar.'),
    ('Pendidikan', 'Mata Pelajaran', 'Apa itu matematika?', 'Matematika adalah ilmu tentang angka.'),
    ('Pendidikan', 'Mata Pelajaran', 'Apa itu bahasa?', 'Bahasa adalah alat komunikasi.'),
    ('Pendidikan', 'Mata Pelajaran', 'Apa itu sains?', 'Sains adalah ilmu tentang alam.'),
    ('Pendidikan', 'Mata Pelajaran', 'Apa itu sejarah?', 'Sejarah adalah ilmu tentang masa lalu.'),
    ('Pendidikan', 'Mata Pelajaran', 'Apa itu geografi?', 'Geografi adalah ilmu tentang bumi.'),
]
items = [{'bidang': 'pendidikan', 'judul': j, 'subjudul': s, 'prompt': p, 'response': r}
         for (j, s, p, r) in edu_items]
total += write_jsonl(OUTPUT_DIR / 'pendidikan' / 'pendidikan.jsonl', items)

# ============================================================
# 22. Native Script (30+ languages)
# ============================================================
print("Generating: Native Script Multilingual...")
native_items = [
    ('Multibahasa', 'Native Script', 'Indonesia?', 'Indonesia adalah negara kepulauan.'),
    ('Multibahasa', 'Native Script', 'English?', 'English is a global language.'),
    ('Multibahasa', 'Native Script', 'Spanish?', 'El espanol es un idioma hermoso.'),
    ('Multibahasa', 'Native Script', 'French?', 'Le francais est une belle langue.'),
    ('Multibahasa', 'Native Script', 'German?', 'Deutsch ist eine wichtige Sprache.'),
    ('Multibahasa', 'Native Script', 'Italian?', 'Il italiano e una bella lingua.'),
    ('Multibahasa', 'Native Script', 'Russian?', 'Русский язык использует кириллицу.'),
    ('Multibahasa', 'Native Script', 'Chinese?', '中文是世界上使用人数最多的语言.'),
    ('Multibahasa', 'Native Script', 'Japanese?', '日本語には三つの書き方があります.'),
    ('Multibahasa', 'Native Script', 'Korean?', '한국어는 한글을 사용합니다.'),
    ('Multibahasa', 'Native Script', 'Arabic?', 'العربية تكتب من اليمين إلى اليسار.'),
    ('Multibahasa', 'Native Script', 'Hindi?', 'हिन्दी देवनागरी लिपि का उपयोग करती है.'),
    ('Multibahasa', 'Native Script', 'Bengali?', 'বাংলা একটি সুন্দর ভাষা.'),
    ('Multibahasa', 'Native Script', 'Thai?', 'ภาษาไทยมีห้าเสียงวรรณยุกต์.'),
    ('Multibahasa', 'Native Script', 'Turkish?', 'Türkçe Latin harflerini kullanır.'),
    ('Multibahasa', 'Native Script', 'Greek?', 'Ελληνικά χρησιμοποιούν το ελληνικό αλφάβητο.'),
    ('Multibahasa', 'Native Script', 'Hebrew?', 'עברית נכתבת מימין לשמאל.'),
    ('Multibahasa', 'Native Script', 'Swahili?', 'Kiswahili kinazungumzwa Afrika.'),
    ('Multibahasa', 'Native Script', 'Ukrainian?', 'Українська використовує кирилицю.'),
    ('Multibahasa', 'Native Script', 'Portuguese?', 'Portugues e uma lingua bonita.'),
]
items = [{'bidang': 'multibahasa', 'judul': j, 'subjudul': s, 'prompt': p, 'response': r}
         for (j, s, p, r) in native_items]
total += write_jsonl(OUTPUT_DIR / 'multibahasa' / 'native_script.jsonl', items)

# ============================================================
# 23. ARAH & PETA (lebih detail)
# ============================================================
print("Generating: Arah & Peta...")
map_items = [
    ('Arah', 'Mata Angin', 'Utara adalah arah?', 'Utara adalah arah ke atas di peta.'),
    ('Arah', 'Mata Angin', 'Selatan adalah arah?', 'Selatan adalah arah ke bawah di peta.'),
    ('Arah', 'Mata Angin', 'Timur adalah arah?', 'Timur adalah arah ke kanan di peta.'),
    ('Arah', 'Mata Angin', 'Barat adalah arah?', 'Barat adalah arah ke kiri di peta.'),
    ('Arah', 'Mata Angin', 'Berapa mata angin?', 'Ada delapan mata angin.'),
    ('Arah', 'Mata Angin', 'Apa empat mata angin utama?', 'Utara, Selatan, Timur, Barat.'),
    ('Arah', 'Terbit Tenggelam', 'Matahari terbit di?', 'Matahari terbit di timur.'),
    ('Arah', 'Terbit Tenggelam', 'Matahari terbenam di?', 'Matahari terbenam di barat.'),
    ('Arah', 'Peta', 'Apa itu peta?', 'Peta adalah gambar wilayah di kertas.'),
    ('Arah', 'Peta', 'Apa itu kompas?', 'Kompas adalah alat penunjuk arah.'),
]
items = [{'bidang': 'arah_peta', 'judul': j, 'subjudul': s, 'prompt': p, 'response': r}
         for (j, s, p, r) in map_items]
total += write_jsonl(OUTPUT_DIR / 'arah_peta' / 'arah_peta.jsonl', items)

# ============================================================
# 24. KESEHATAN MENTAL & EMOSI
# ============================================================
print("Generating: Kesehatan Mental...")
mental_items = [
    ('Emosi', 'Perasaan Dasar', 'Apa itu senang?', 'Senang adalah perasaan gembira.'),
    ('Emosi', 'Perasaan Dasar', 'Apa itu sedih?', 'Sedih adalah perasaan kecewa.'),
    ('Emosi', 'Perasaan Dasar', 'Apa itu marah?', 'Marah adalah perasaan kesal.'),
    ('Emosi', 'Perasaan Dasar', 'Apa itu takut?', 'Takut adalah perasaan cemas.'),
    ('Emosi', 'Perasaan Dasar', 'Apa itu cinta?', 'Cinta adalah perasaan sayang yang kuat.'),
    ('Emosi', 'Perasaan Dasar', 'Apa itu benci?', 'Benci adalah perasaan tidak suka yang kuat.'),
    ('Emosi', 'Perasaan Dasar', 'Apa itu kaget?', 'Kaget adalah perasaan terkejut.'),
    ('Emosi', 'Kesehatan Mental', 'Apa itu stres?', 'Stres adalah tekanan mental.'),
    ('Emosi', 'Kesehatan Mental', 'Apa itu depresi?', 'Depresi adalah gangguan suasana hati.'),
    ('Emosi', 'Kesehatan Mental', 'Apa itu kecemasan?', 'Kecemasan adalah rasa khawatir berlebih.'),
    ('Emosi', 'Kesehatan Mental', 'Apa cara mengatasi stres?', 'Olahraga, tidur cukup, dan bicara dengan teman.'),
    ('Emosi', 'Kesehatan Mental', 'Berapa jam tidur sehat?', 'Tidur delapan jam sehari untuk kesehatan.'),
    ('Emosi', 'Kesehatan Mental', 'Apa itu meditasi?', 'Meditasi adalah latihan relaksasi pikiran.'),
    ('Emosi', 'Kesehatan Mental', 'Kenapa penting istirahat?', 'Istirahat memulihkan tubuh dan pikiran.'),
    ('Emosi', 'Sosial', 'Apa itu empati?', 'Empati adalah kemampuan merasakan perasaan orang lain.'),
]
items = [{'bidang': 'emosi_mental', 'judul': j, 'subjudul': s, 'prompt': p, 'response': r}
         for (j, s, p, r) in mental_items]
total += write_jsonl(OUTPUT_DIR / 'emosi_mental' / 'emosi_mental.jsonl', items)

# ============================================================
# 25. PENGETAHUAN DUNIA
# ============================================================
print("Generating: Pengetahuan Dunia...")
world_items = [
    ('Pengetahuan Dunia', 'Fakta', 'Apa gunung tertinggi di dunia?', 'Gunung Everest adalah gunung tertinggi di dunia.'),
    ('Pengetahuan Dunia', 'Fakta', 'Apa sungai terpanjang di dunia?', 'Sungai Nil adalah sungai terpanjang di dunia.'),
    ('Pengetahuan Dunia', 'Fakta', 'Apa samudra terbesar di dunia?', 'Samudra Pasifik adalah samudra terbesar di dunia.'),
    ('Pengetahuan Dunia', 'Fakta', 'Apa gurun terbesar di dunia?', 'Gurun Sahara adalah gurun terbesar di dunia.'),
    ('Pengetahuan Dunia', 'Fakta', 'Apa negara terbesar di dunia?', 'Rusia adalah negara terbesar di dunia.'),
    ('Pengetahuan Dunia', 'Fakta', 'Apa negara terkecil di dunia?', 'Vatikan adalah negara terkecil di dunia.'),
    ('Pengetahuan Dunia', 'Fakta', 'Berapa benua di dunia?', 'Ada tujuh benua di dunia.'),
    ('Pengetahuan Dunia', 'Fakta', 'Berapa samudra di dunia?', 'Ada lima samudra di dunia.'),
    ('Pengetahuan Dunia', 'Fakta', 'Apa planet terbesar di tata surya?', 'Jupiter adalah planet terbesar.'),
    ('Pengetahuan Dunia', 'Fakta', 'Apa planet terdekat dengan matahari?', 'Merkurius adalah planet terdekat dengan matahari.'),
    ('Pengetahuan Dunia', 'Fakta', 'Apa planet terjauh dari matahari?', 'Neptunus adalah planet terjauh dari matahari.'),
    ('Pengetahuan Dunia', 'Fakta', 'Berapa planet di tata surya?', 'Ada delapan planet di tata surya.'),
    ('Pengetahuan Dunia', 'Fakta', 'Apa bintang terdekat dengan bumi?', 'Matahari adalah bintang terdekat dengan bumi.'),
    ('Pengetahuan Dunia', 'Fakta', 'Apa galaksi kita?', 'Bima Sakti adalah galaksi kita.'),
    ('Pengetahuan Dunia', 'Fakta', 'Apa kecepatan cahaya?', 'Kecepatan cahaya sekitar tiga ratus ribu kilometer per detik.'),
]
items = [{'bidang': 'pengetahuan_dunia', 'judul': j, 'subjudul': s, 'prompt': p, 'response': r}
         for (j, s, p, r) in world_items]
total += write_jsonl(OUTPUT_DIR / 'pengetahuan_dunia' / 'fakta_dunia.jsonl', items)

# ============================================================
# DONE
# ============================================================
print(f"\n{'='*60}")
print(f"  DATASET GENERATION COMPLETE")
print(f"{'='*60}")
print(f"\nTotal Q&A pairs: {total:,}")
print(f"Categories: 25+")
print(f"Output: {OUTPUT_DIR}")
print(f"\nStruktur folder:")
for cat in sorted(OUTPUT_DIR.iterdir()):
    if cat.is_dir():
        files = list(cat.glob('*.jsonl'))
        count = sum(1 for f in files for _ in open(f, encoding='utf-8'))
        print(f"  {cat.name}/ ({count} items)")
        for f in sorted(files):
            n = sum(1 for _ in open(f, encoding='utf-8'))
            print(f"    - {f.name} ({n} items)")
