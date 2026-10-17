"""
D'QUILANE FINAL DATASET — Reformat + Expand
============================================
1. Load all existing v3 data → convert to input/output format
2. Generate NEW data: syllogisms, analogies, more math, more science, more history
3. Split 80/20 train/validation
4. Create stats.json
5. Zip everything
"""
import json, os, random, zipfile
from pathlib import Path

random.seed(42)

# Load all existing v3 data
V3_DIR = Path("/home/z/my-project/download/llm/dataset_v3")
all_items = []

for jsonl_file in V3_DIR.rglob("*.jsonl"):
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            # Convert prompt/response → input/output
            all_items.append({
                'input': item.get('prompt', item.get('input', '')),
                'output': item.get('response', item.get('output', '')),
                'bidang': item.get('bidang', ''),
                'judul': item.get('judul', ''),
                'subjudul': item.get('subjudul', ''),
            })

print(f"Loaded from v3: {len(all_items)} items")

# ============================================================
# NEW DATA — Syllogisms (200+)
# ============================================================
print("Generating syllogisms...")
SYLLOGISM_TEMPLATES = [
    # Categorical: All A are B. C is A. → C is B.
    ("Semua {A} {verb} {B}.", "{C} adalah {A}.", "Kesimpulan: {C} {verb} {B}."),
    # Negative: No A are B. C is A. → C is not B.
    ("Tidak ada {A} yang {verb} {B}.", "{C} adalah {A}.", "Kesimpulan: {C} tidak {verb} {B}."),
    # Particular: Some A are B. C is A. → C might be B.
    ("Sebagian {A} {verb} {B}.", "{C} adalah {A}.", "Kesimpulan: {C} mungkin {verb} {B}."),
]

SYLLOGISM_DATA = [
    # (A, verb, B, C)
    ("manusia", "bernapas", "udara", "Budi"),
    ("manusia", "butuh", "air", "Siti"),
    ("kucing", "suka", "ikan", "Kitty"),
    ("anjing", "bisa", "menggonggong", "Rex"),
    ("burung", "bisa", "terbang", "Garuda"),
    ("ikan", "hidup di", "air", "Nemo"),
    ("mamalia", "menyusui", "anaknya", "sapi"),
    ("mahasiswa", "punya", "KTM", "Andi"),
    ("siswa", "wajib", "seragam", "Doni"),
    ("dokter", "menyembuhkan", "pasien", "dr. Siti"),
    ("pilot", "bisa", "mendaratkan pesawat", "Kapten Budi"),
    ("guru", "mengajar", "murid", "Pak Ali"),
    ("polisi", "menangkap", "pencuri", "Bripda Andi"),
    ("petani", "menanam", "padi", "Pak Tani"),
    ("nelayan", "menangkap", "ikan", "Pak Nelayan"),
    ("pedagang", "menjual", "barang", "Ibu Penjual"),
    ("penyanyi", "menyanyi", "lagu", "Agnes"),
    ("atlet", "berolahraga", "secara profesional", "Budi"),
    ("ilmuwan", "meneliti", "alam", "Einstein"),
    ("tumbuhan", "butuh", "cahaya", "mawar"),
    ("tumbuhan", "melakukan", "fotosintesis", "pohon"),
    ("hewan", "butuh", "makan", "singa"),
    ("reptil", "bertelur", "telur", "buaya"),
    ("amfibi", "bisa hidup di", "darat dan air", "katak"),
    ("serangga", "punya", "enam kaki", "lebah"),
    ("logam", "menghantarkan", "listrik", "tembaga"),
    ("logam", "menghantarkan", "panas", "besi"),
    ("benda padat", "punya", "bentuk tetap", "batu"),
    ("benda cair", "mengikuti", "bentuk wadah", "air"),
    ("benda gas", "memenuhi", "ruangan", "udara"),
    ("negara", "punya", "pemerintah", "Indonesia"),
    ("negara", "punya", "bendera", "Jepang"),
    ("benua", "punya", "negara", "Asia"),
    ("planet", "mengelilingi", "matahari", "Bumi"),
    ("satelit", "mengelilingi", "planet", "Bulan"),
    ("bintang", "memancarkan", "cahaya", "Matahari"),
    ("galaksi", "terdiri dari", "bintang", "Bima Sakti"),
    ("sel", "adalah", "unit kehidupan", "sel manusia"),
    ("DNA", "menyimpan", "informasi genetik", "DNA manusia"),
    ("atom", "adalah", "unit terkecil materi", "atom hidrogen"),
    ("molekul", "terdiri dari", "atom", "molekul air"),
    ("energy", "tidak bisa", "dimusnahkan", "energi panas"),
    ("energi", "bisa", "berubah wujud", "energi listrik"),
    ("air", "mendidih pada", "seratus derajat", "air di gelas"),
    ("air", "membeku pada", "nol derajat", "air di freezer"),
    ("magnet", "menarik", "besi", "magnet batang"),
    ("gravitasi", "menarik", "benda ke bumi", "bola jatuh"),
    ("cahaya", "merambat", "lurus", "sinar matahari"),
    ("bunyi", "merambat", "melalui medium", "suara musik"),
]

syllogism_items = []
for A, verb, B, C in SYLLOGISM_DATA:
    for template_p1, template_p2, template_concl in SYLLOGISM_TEMPLATES:
        p1 = template_p1.format(A=A, verb=verb, B=B)
        p2 = template_p2.format(C=C, A=A)
        concl = template_concl.format(C=C, verb=verb, B=B)
        syllogism_items.append({
            'input': f"{p1} {p2}",
            'output': concl,
            'bidang': 'logika',
            'judul': 'Silogisme',
            'subjudul': 'Deduktif',
        })

all_items.extend(syllogism_items)
print(f"  Syllogisms: +{len(syllogism_items)}")

# ============================================================
# NEW DATA — Analogies (100+)
# ============================================================
print("Generating analogies...")
ANALOGIES = [
    ("Gajah : besar :: semut : ?", "kecil"),
    ("Api : panas :: es : ?", "dingin"),
    ("Matahari : terbit :: bulan : ?", "muncul"),
    ("Singa : raja hutan :: hiu : ?", "raja laut"),
    ("Pensil : menulis :: gunting : ?", "memotong"),
    ("Buku : membaca :: makanan : ?", "dimakan"),
    ("Mata : melihat :: telinga : ?", "mendengar"),
    ("Hidung : mencium :: lidah : ?", "mengecap"),
    ("Tangan : memegang :: kaki : ?", "berjalan"),
    ("Kucing : meong :: anjing : ?", "guk guk"),
    ("Sapi : moo :: kambing : ?", "mbek mbek"),
    ("Burung : terbang :: ikan : ?", "berenang"),
    ("Mobil : jalan :: kapal : ?", "berlayar"),
    ("Pensil : kayu :: pulpen : ?", "plastik"),
    ("Nasi : karbohidrat :: daging : ?", "protein"),
    ("Apel : buah :: mawar : ?", "bunga"),
    ("Singa : hewan :: pohon : ?", "tumbuhan"),
    ("Jakarta : Indonesia :: Tokyo : ?", "Jepang"),
    ("Rupiah : Indonesia :: yen : ?", "Jepang"),
    ("Senin : hari :: Januari : ?", "bulan"),
    ("Merah : warna :: satu : ?", "angka"),
    ("Besar : kecil :: tinggi : ?", "pendek"),
    ("Cepat : lambat :: panas : ?", "dingin"),
    ("Senang : sedih :: cinta : ?", "benci"),
    ("Siang : malam :: musim panas : ?", "musim dingin"),
    ("Utara : selatan :: timur : ?", "barat"),
    ("Bumi : planet :: matahari : ?", "bintang"),
    ("Bulan : satelit :: bumi : ?", "planet"),
    ("Air : cair :: es : ?", "padat"),
    ("Air : mengalir :: angin : ?", "berhembus"),
    ("Tanaman : tumbuh :: hewan : ?", "berkembang"),
    ("Sekolah : belajar :: rumah sakit : ?", "sembuh"),
    ("Guru : mengajar :: dokter : ?", "menyembuhkan"),
    ("Polisi : menangkap :: hakim : ?", "menghakimi"),
    ("Petani : sawah :: nelayan : ?", "laut"),
    ("Koki : dapur :: guru : ?", "kelas"),
    ("Pilot : pesawat :: masinis : ?", "kereta"),
    ("Pemain bola : sepak bola :: pemain basket : ?", "basket"),
    ("Penyanyi : lagu :: penulis : ?", "buku"),
    ("Lukis : cat :: tulis : ?", "tinta"),
    ("Musim hujan : basah :: musim kemarau : ?", "kering"),
    ("Pegunungan : tinggi :: lembah : ?", "rendah"),
    ("Laut : dalam :: dangkal : ?", "pendek"),
    ("Jalan raya : aspal :: jalan setapak : ?", "tanah"),
    ("Kota : ramai :: desa : ?", "tenang"),
    ("Komputer : data :: otak : ?", "pikiran"),
    ("Internet : jaringan :: buku : ?", "informasi"),
    ("Helm : kepala :: sabuk pengaman : ?", "tubuh"),
    ("Pintu : rumah :: gerbang : ?", "halaman"),
    ("Kunci : gembok :: password : ?", "akun"),
]
analogy_items = []
for analogy, answer in ANALOGIES:
    analogy_items.append({
        'input': analogy,
        'output': answer,
        'bidang': 'logika',
        'judul': 'Analogi',
        'subjudul': 'Perbandingan',
    })
all_items.extend(analogy_items)
print(f"  Analogies: +{len(analogy_items)}")

# ============================================================
# NEW DATA — More Math (mixed operations, time, money, measurement)
# ============================================================
print("Generating more math...")
more_math = []

# Mixed operations (PEMDAS)
for _ in range(150):
    a, b, c = random.randint(1,20), random.randint(1,20), random.randint(1,10)
    ops = random.choice(['+', '-', '*'])
    if ops == '+':
        result = a + b + c
        more_math.append({'input': f'{a} + {b} + {c} = ?', 'output': f'{result}', 'bidang':'matematika','judul':'Operasi Campuran','subjudul':'PEMDAS'})
    elif ops == '-':
        result = a - b + c
        more_math.append({'input': f'{a} - {b} + {c} = ?', 'output': f'{result}', 'bidang':'matematika','judul':'Operasi Campuran','subjudul':'PEMDAS'})
    else:
        result = a * b + c
        more_math.append({'input': f'{a} x {b} + {c} = ?', 'output': f'{result}', 'bidang':'matematika','judul':'Operasi Campuran','subjudul':'PEMDAS'})

# Time problems
for h1, m1, h2, m2 in [(2,30,1,45),(3,15,2,40),(1,50,2,30),(4,10,1,20),(2,45,3,15),(5,30,2,45)]:
    total_h = h1 + h2 + (m1 + m2) // 60
    total_m = (m1 + m2) % 60
    more_math.append({'input': f'{h1} jam {m1} menit + {h2} jam {m2} menit = ?', 'output': f'{total_h} jam {total_m} menit', 'bidang':'matematika','judul':'Waktu','subjudul':'Penjumlahan'})

# Money problems (rupiah)
for amt, spend in [(50000,23500),(100000,45000),(20000,8750),(75000,33500),(150000,89500)]:
    change = amt - spend
    more_math.append({'input': f'Belanja Rp {spend:,} dengan uang Rp {amt:,}. Berapa kembalian?', 'output': f'Kembalian Rp {change:,}', 'bidang':'matematika','judul':'Uang','subjudul':'Kembalian'})

# Measurement
for val, unit1, factor, unit2 in [(2,'m',100,'cm'),(150,'cm',0.01,'m'),(3,'km',1000,'m'),(500,'g',0.001,'kg'),(2,'L',1000,'mL'),(1,'kg',1000,'g')]:
    result = val * factor
    more_math.append({'input': f'{val} {unit1} = ? {unit2}', 'output': f'{result} {unit2}', 'bidang':'matematika','judul':'Pengukuran','subjudul':'Konversi Satuan'})

# Square numbers
for n in range(1, 21):
    more_math.append({'input': f'{n} kuadrat = ?', 'output': f'{n*n}', 'bidang':'matematika','judul':'Pangkat','subjudul':'Kuadrat'})

# Square roots
for n in [1,4,9,16,25,36,49,64,81,100]:
    more_math.append({'input': f'Akar kuadrat dari {n} = ?', 'output': f'{int(n**0.5)}', 'bidang':'matematika','judul':'Pangkat','subjudul':'Akar Kuadrat'})

all_items.extend(more_math)
print(f"  More math: +{len(more_math)}")

# ============================================================
# NEW DATA — More Science (expanded)
# ============================================================
print("Generating more science...")
more_science = [
    {'input':'Apa hukum kekekalan energi?','output':'Energi tidak bisa diciptakan atau dimusnahkan, hanya berubah wujud.','bidang':'sains','judul':'Fisika','subjudul':'Hukum Kekekalan'},
    {'input':'Apa itu kecepatan terminal?','output':'Kecepatan maksimum benda yang jatuh saat gaya gravitasi sama dengan hambatan udara.','bidang':'sains','judul':'Fisika','subjudul':'Mekanika'},
    {'input':'Apa itu tekanan atmosfer?','output':'Tekanan atmosfer adalah tekanan udara di permukaan bumi.','bidang':'sains','judul':'Fisika','subjudul':'Atmosfer'},
    {'input':'Apa itu refleksi cahaya?','output':'Refleksi cahaya adalah pemantulan cahaya pada permukaan.','bidang':'sains','judul':'Fisika','subjudul':'Optik'},
    {'input':'Apa itu refraksi cahaya?','output':'Refraksi cahaya adalah pembelokan cahaya saat melewati medium berbeda.','bidang':'sains','judul':'Fisika','subjudul':'Optik'},
    {'input':'Apa itu daya listrik?','output':'Daya listrik adalah tegangan dikali arus (P = V x I).','bidang':'sains','judul':'Fisika','subjudul':'Listrik'},
    {'input':'Apa itu arus searah (DC)?','output':'DC adalah arus listrik yang mengalir satu arah.','bidang':'sains','judul':'Fisika','subjudul':'Listrik'},
    {'input':'Apa itu arus bolak-balik (AC)?','output':'AC adalah arus listrik yang arahnya berubah-ubah.','bidang':'sains','judul':'Fisika','subjudul':'Listrik'},
    {'input':'Apa itu katalis?','output':'Katalis adalah zat yang mempercepat reaksi kimia tanpa ikut bereaksi.','bidang':'sains','judul':'Kimia','subjudul':'Reaksi'},
    {'input':'Apa itu larutan asam?','output':'Larutan asam memiliki pH di bawah 7 dan rasanya masam.','bidang':'sains','judul':'Kimia','subjudul':'Asam Basa'},
    {'input':'Apa itu larutan basa?','output':'Larutan basa memiliki pH di atas 7 dan rasanya pahit.','bidang':'sains','judul':'Kimia','subjudul':'Asam Basa'},
    {'input':'Apa itu larutan netral?','output':'Larutan netral memiliki pH sama dengan 7.','bidang':'sains','judul':'Kimia','subjudul':'Asam Basa'},
    {'input':'Apa itu pH universal indicator?','output':'Indikator universal adalah kertas yang berubah warna sesuai pH.','bidang':'sains','judul':'Kimia','subjudul':'Asam Basa'},
    {'input':'Apa itu reaksi eksoterm?','output':'Reaksi eksoterm melepaskan panas ke lingkungan.','bidang':'sains','judul':'Kimia','subjudul':'Reaksi'},
    {'input':'Apa itu reaksi endoterm?','output':'Reaksi endoterm menyerap panas dari lingkungan.','bidang':'sains','judul':'Kimia','subjudul':'Reaksi'},
    {'input':'Apa itu alel dominan?','output':'Alel dominan adalah gen yang selalu muncul jika ada.','bidang':'sains','judul':'Biologi','subjudul':'Genetika'},
    {'input':'Apa itu alel resesif?','output':'Alel resesif adalah gen yang muncul hanya jika pasangannya sama.','bidang':'sains','judul':'Biologi','subjudul':'Genetika'},
    {'input':'Apa itu mutasi gen?','output':'Mutasi gen adalah perubahan DNA yang bisa diwariskan.','bidang':'sains','judul':'Biologi','subjudul':'Genetika'},
    {'input':'Apa itu adaptasi morfologi?','output':'Adaptasi morfologi adalah perubahan bentuk tubuh untuk bertahan.','bidang':'sains','judul':'Biologi','subjudul':'Adaptasi'},
    {'input':'Apa itu adaptasi fisiologi?','output':'Adaptasi fisiologi adalah perubahan fungsi organ untuk bertahan.','bidang':'sains','judul':'Biologi','subjudul':'Adaptasi'},
    {'input':'Apa itu adaptasi tingkah laku?','output':'Adaptasi tingkah laku adalah perubahan perilaku untuk bertahan.','bidang':'sains','judul':'Biologi','subjudul':'Adaptasi'},
    {'input':'Apa itu jaring-jaring makanan?','output':'Jaring makanan adalah hubungan makan-dimakan yang kompleks di ekosistem.','bidang':'sains','judul':'Biologi','subjudul':'Ekosistem'},
    {'input':'Apa itu simbiosis?','output':'Simbiosis adalah hubungan interaksi antar makhluk hidup yang berbeda spesies.','bidang':'sains','judul':'Biologi','subjudul':'Ekosistem'},
    {'input':'Apa itu daur air?','output':'Daur air: evaporasi, kondensasi, presipitasi, infiltrasi, limpasan.','bidang':'sains','judul':'Ilmu Bumi','subjudul':'Siklus'},
    {'input':'Apa itu perbedaan iklim dan cuaca?','output':'Cuaca adalah kondisi atmosfer harian, iklim adalah rata-rata cuaca jangka panjang.','bidang':'sains','judul':'Ilmu Bumi','subjudul':'Cuaca'},
    {'input':'Apa itu efek rumah kaca?','output':'Efek rumah kaca adalah gas di atmosfer menangkap panas matahari.','bidang':'sains','judul':'Ilmu Bumi','subjudul':'Lingkungan'},
    {'input':'Apa itu lapisan ozon?','output':'Lapisan ozon melindungi bumi dari radiasi UV matahari.','bidang':'sains','judul':'Ilmu Bumi','subjudul':'Atmosfer'},
    {'input':'Apa itu percepatan gravitasi bumi?','output':'Percepatan gravitasi bumi sekitar sepuluh meter per detik kuadrat.','bidang':'sains','judul':'Fisika','subjudul':'Gravitasi'},
    {'input':'Apa itu massa jenis?','output':'Massa jenis adalah massa per satuan volume (rho = m/V).','bidang':'sains','judul':'Fisika','subjudul':'Materi'},
    {'input':'Apa itu usaha dalam fisika?','output':'Usaha adalah gaya dikali perpindahan (W = F x s).','bidang':'sains','judul':'Fisika','subjudul':'Mekanika'},
    {'input':'Apa itu tekanan?','output':'Tekanan adalah gaya per satuan luas (P = F/A).','bidang':'sains','judul':'Fisika','subjudul':'Mekanika'},
    {'input':'Apa itu momentum?','output':'Momentum adalah massa dikali kecepatan (p = m x v).','bidang':'sains','judul':'Fisika','subjudul':'Mekanika'},
    {'input':'Apa itu tuas?','output':'Tuas adalah alat sederhana yang menggunakan titik tumpu.','bidang':'sains','judul':'Fisika','subjudul':'Pesawat Sederhana'},
    {'input':'Apa itu bidang miring?','output':'Bidang miring adalah alat sederhana untuk memudahkan mengangkat beban.','bidang':'sains','judul':'Fisika','subjudul':'Pesawat Sederhana'},
    {'input':'Apa itu katrol?','output':'Katrol adalah alat sederhana untuk mengangkat beban dengan tali.','bidang':'sains','judul':'Fisika','subjudul':'Pesawat Sederhana'},
    {'input':'Apa itu roda gigi?','output':'Roda gigi adalah alas untuk mengubah arah dan kecepatan putaran.','bidang':'sains','judul':'Fisika','subjudul':'Pesawat Sederhana'},
    {'input':'Apa itu gelombang transversal?','output':'Gelombang transversal arah getar tegak lurus arah rambat, contoh cahaya.','bidang':'sains','judul':'Fisika','subjudul':'Gelombang'},
    {'input':'Apa itu gelombang longitudinal?','output':'Gelombang longitudinal arah getar sejajar arah rambat, contoh bunyi.','bidang':'sains','judul':'Fisika','subjudul':'Gelombang'},
    {'input':'Apa itu resonansi?','output':'Resonansi adalah ikut bergetarnya benda karena getaran lain dengan frekuensi sama.','bidang':'sains','judul':'Fisika','subjudul':'Gelombang'},
    {'input':'Apa itu interferensi?','output':'Interferensi adalah gabungan dua gelombang atau lebih.','bidang':'sains','judul':'Fisika','subjudul':'Gelombang'},
    {'input':'Apa itu difraksi?','output':'Difraksi adalah pembelokan gelombang saat melewati celah.','bidang':'sains','judul':'Fisika','subjudul':'Gelombang'},
]
all_items.extend(more_science)
print(f"  More science: +{len(more_science)}")

# ============================================================
# NEW DATA — More History (expanded)
# ============================================================
print("Generating more history...")
more_history = [
    {'input':'Siapa penemu mesin uap?','output':'James Watt menemukan mesin uap modern.','bidang':'sejarah','judul':'Penemu','subjudul':'Revolusi Industri'},
    {'input':'Siapa penemu telegraf?','output':'Samuel Morse menemukan telegraf.','bidang':'sejarah','judul':'Penemu','subjudul':'Komunikasi'},
    {'input':'Siapa penemu radio?','output':'Guglielmo Marconi menemukan radio.','bidang':'sejarah','judul':'Penemu','subjudul':'Komunikasi'},
    {'input':'Siapa penemu televisi?','output':'John Logie Baird menemukan televisi.','bidang':'sejarah','judul':'Penemu','subjudul':'Komunikasi'},
    {'input':'Siapa penemu komputer pertama?','output':'Charles Babbage disebut bapak komputer.','bidang':'sejarah','judul':'Penemu','subjudul':'Komputer'},
    {'input':'Siapa penemu telepon seluler?','output':'Martin Cooper menemukan ponsel pertama.','bidang':'sejarah','judul':'Penemu','subjudul':'Komunikasi'},
    {'input':'Siapa penemu World Wide Web?','output':'Tim Berners-Lee menemukan WWW.','bidang':'sejarah','judul':'Penemu','subjudul':'Internet'},
    {'input':'Siapa penemu vaksin COVID?','output':'Berbagai ilmuwan mengembangkan vaksin COVID-19.','bidang':'sejarah','judul':'Penemu','subjudul':'Kedokteran'},
    {'input':'Siapa penemu antibiotik penisilin?','output':'Alexander Fleming menemukan penisilin.','bidang':'sejarah','judul':'Penemu','subjudul':'Kedokteran'},
    {'input':'Siapa penemu roket?','output':'Robert Goddard disebut bapak roket modern.','bidang':'sejarah','judul':'Penemu','subjudul':'Antariksa'},
    {'input':'Siapa astronot pertama?','output':'Yuri Gagarin adalah astronot pertama di luar angkasa.','bidang':'sejarah','judul':'Antariksa','subjudul':'Pertama'},
    {'input':'Siapa astronot perempuan pertama?','output':'Valentina Tereshkova adalah astronot perempuan pertama.','bidang':'sejarah','judul':'Antariksa','subjudul':'Pertama'},
    {'input':'Kapan manusia pertama kali mendarat di bulan?','output':'Manusia pertama mendarat di bulan tahun seribu sembilan ratus enam puluh sembilan.','bidang':'sejarah','judul':'Antariksa','subjudul':'Apollo'},
    {'input':'Siapa pemimpin India yang merdeka tanpa kekerasan?','output':'Mahatma Gandhi memimpin India merdeka tanpa kekerasan.','bidang':'sejarah','judul':'Tokoh Dunia','subjudul':'Damai'},
    {'input':'Siapa presiden Afrika Selatan pertama setelah apartheid?','output':'Nelson Mandela adalah presiden pertama setelah apartheid.','bidang':'sejarah','judul':'Tokoh Dunia','subjudul':'Damai'},
    {'input':'Apa yang dimaksud Renaisans?','output':'Renaisans adalah masa kebangkitan kembali seni dan ilmu di Eropa.','bidang':'sejarah','judul':'Era','subjudul':'Eropa'},
    {'input':'Apa yang dimaksud Revolusi Industri?','output':'Revolusi Industri adalah perubahan dari kerja tangan ke mesin.','bidang':'sejarah','judul':'Era','subjudul':'Eropa'},
    {'input':'Apa yang dimaksud Revolusi Digital?','output':'Revolusi Digital adalah perubahan dari analog ke digital.','bidang':'sejarah','judul':'Era','subjudul':'Modern'},
    {'input':'Siapa presiden Amerika pertama?','output':'George Washington adalah presiden Amerika pertama.','bidang':'sejarah','judul':'Tokoh Dunia','subjudul':'Amerika'},
    {'input':'Siapa pemimpin gerakan hak sipil Amerika?','output':'Martin Luther King Jr. memimpin gerakan hak sipil.','bidang':'sejarah','judul':'Tokoh Dunia','subjudul':'Amerika'},
]
all_items.extend(more_history)
print(f"  More history: +{len(more_history)}")

# ============================================================
# NEW DATA — Indonesian Provinces (38)
# ============================================================
print("Generating Indonesian provinces...")
PROVINCES = [
    ('Aceh','Banda Aceh'),('Sumatra Utara','Medan'),('Sumatra Barat','Padang'),
    ('Riau','Pekanbaru'),('Jambi','Jambi'),('Sumatra Selatan','Palembang'),
    ('Bengkulu','Bengkulu'),('Lampung','Bandar Lampung'),('Kepulauan Bangka Belitung','Pangkalpinang'),
    ('Kepulauan Riau','Tanjungpinang'),('DKI Jakarta','Jakarta'),('Jawa Barat','Bandung'),
    ('Jawa Tengah','Semarang'),('DI Yogyakarta','Yogyakarta'),('Jawa Timur','Surabaya'),
    ('Banten','Serang'),('Bali','Denpasar'),('Nusa Tenggara Barat','Mataram'),
    ('Nusa Tenggara Timur','Kupang'),('Kalimantan Barat','Pontianak'),('Kalimantan Tengah','Palangkaraya'),
    ('Kalimantan Selatan','Banjarmasin'),('Kalimantan Timur','Samarinda'),
    ('Kalimantan Utara','Tanjung Selor'),('Sulawesi Utara','Manado'),
    ('Sulawesi Tengah','Palu'),('Sulawesi Selatan','Makassar'),('Sulawesi Tenggara','Kendari'),
    ('Gorontalo','Gorontalo'),('Maluku','Ambon'),('Maluku Utara','Sofifi'),
    ('Papua','Jayapura'),('Papua Barat','Manokwari'),('Papua Selatan','Merauke'),
    ('Papua Tengah','Nabire'),('Papua Pegunungan','Wamena'),('Papua Barat Daya','Sorong'),
    ('Sulawesi Barat','Mamuju'),
]
prov_items = []
for prov, capital in PROVINCES:
    prov_items.append({
        'input': f'Apa ibu kota provinsi {prov}?',
        'output': f'Ibu kota provinsi {prov} adalah {capital}.',
        'bidang': 'geografi', 'judul': 'Provinsi Indonesia', 'subjudul': 'Ibu Kota',
    })
all_items.extend(prov_items)
print(f"  Indonesian provinces: +{len(prov_items)}")

# ============================================================
# NEW DATA — More Logic & Reasoning
# ============================================================
print("Generating more logic...")
more_logic = [
    {'input':'Jika hari ini Senin, besok apa?','output':'Besok adalah Selasa.','bidang':'logika','judul':'Deduktif','subjudul':'Hari'},
    {'input':'Jika sekarang Januari, bulan depan apa?','output':'Bulan depan adalah Februari.','bidang':'logika','judul':'Deduktif','subjudul':'Bulan'},
    {'input':'Jika semua manusia butuh makan, dan Budi manusia, apa yang Budi butuh?','output':'Budi butuh makan.','bidang':'logika','judul':'Silogisme','subjudul':'Deduktif'},
    {'input':'Jika semua ikan hidup di air, dan Nemo ikan, di mana Nemo hidup?','output':'Nemo hidup di air.','bidang':'logika','judul':'Silogisme','subjudul':'Deduktif'},
    {'input':'Jika semua burung bisa terbang, dan Garuda burung, apa yang Garuda bisa?','output':'Garuda bisa terbang.','bidang':'logika','judul':'Silogisme','subjudul':'Deduktif'},
    {'input':'Jika hari hujan tanaman tumbuh, dan hari ini hujan, apa yang terjadi?','output':'Tanaman tumbuh subur.','bidang':'logika','judul':'Sebab-Akibat','subjudul':'Kondisional'},
    {'input':'Jika tidak makan maka lapar, dan Budi tidak makan, apa yang terjadi?','output':'Budi menjadi lapar.','bidang':'logika','judul':'Sebab-Akibat','subjudul':'Kondisional'},
    {'input':'Jika air dipanaskan sampai 100 derajat, apa yang terjadi?','output':'Air akan mendidih.','bidang':'logika','judul':'Sebab-Akibat','subjudul':'Kondisional'},
    {'input':'Jika air didinginkan sampai 0 derajat, apa yang terjadi?','output':'Air akan membeku menjadi es.','bidang':'logika','judul':'Sebab-Akibat','subjudul':'Kondisional'},
    {'input':'Jika lampu merah menyala di jalan, apa yang harus dilakukan?','output':'Harus berhenti.','bidang':'logika','judul':'Sebab-Akibat','subjudul':'Kondisional'},
    {'input':'Jika lampu hijau menyala di jalan, apa yang harus dilakukan?','output':'Boleh jalan.','bidang':'logika','judul':'Sebab-Akibat','subjudul':'Kondisional'},
    {'input':'Jika bel sekolah berbunyi pagi, apa yang harus dilakukan?','output':'Masuk ke kelas.','bidang':'logika','judul':'Sebab-Akibat','subjudul':'Kondisional'},
    {'input':'Jika PR belum selesai, apa yang harus dilakukan?','output':'Mengerjakan PR.','bidang':'logika','judul':'Sebab-Akibat','subjudul':'Kondisional'},
    {'input':'Pola 1,2,4,8,16. Berikutnya?','output':'32 (lipat dua).','bidang':'logika','judul':'Pola','subjudul':'Geometri'},
    {'input':'Pola 1,4,9,16,25. Berikutnya?','output':'36 (kuadrat).','bidang':'logika','judul':'Pola','subjudul':'Kuadrat'},
    {'input':'Pola 2,3,5,7,11. Berikutnya?','output':'13 (bilangan prima).','bidang':'logika','judul':'Pola','subjudul':'Prima'},
    {'input':'Pola 1,1,2,3,5,8. Berikutnya?','output':'13 (Fibonacci).','bidang':'logika','judul':'Pola','subjudul':'Fibonacci'},
    {'input':'Pola 3,6,12,24,48. Berikutnya?','output':'96 (lipat dua).','bidang':'logika','judul':'Pola','subjudul':'Geometri'},
    {'input':'Pola 100,90,80,70. Berikutnya?','output':'60 (turun 10).','bidang':'logika','judul':'Pola','subjudul':'Aritmatika'},
    {'input':'Pola 1,4,9,16. Berikutnya?','output':'25 (kuadrat).','bidang':'logika','judul':'Pola','subjudul':'Kuadrat'},
]
all_items.extend(more_logic)
print(f"  More logic: +{len(more_logic)}")

# ============================================================
# SHUFFLE + SPLIT 80/20
# ============================================================
print(f"\nTotal items: {len(all_items)}")
random.shuffle(all_items)
split = int(len(all_items) * 0.8)
train = all_items[:split]
val = all_items[split:]
print(f"Train: {len(train)} items (80%)")
print(f"Validation: {len(val)} items (20%)")

# ============================================================
# WRITE FILES
# ============================================================
OUTPUT_DIR = Path("/home/z/my-project/download/llm/dataset_final")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Train
with open(OUTPUT_DIR / 'dataset_train.jsonl', 'w', encoding='utf-8') as f:
    for item in train:
        f.write(json.dumps({'input': item['input'], 'output': item['output']}, ensure_ascii=False) + '\n')

# Validation
with open(OUTPUT_DIR / 'dataset_validation.jsonl', 'w', encoding='utf-8') as f:
    for item in val:
        f.write(json.dumps({'input': item['input'], 'output': item['output']}, ensure_ascii=False) + '\n')

# Stats
bidang_counts = {}
for item in all_items:
    b = item.get('bidang', 'unknown')
    bidang_counts[b] = bidang_counts.get(b, 0) + 1

stats = {
    'total_items': len(all_items),
    'train_items': len(train),
    'validation_items': len(val),
    'split_ratio': '80/20',
    'format': 'jsonl',
    'keys': ['input', 'output'],
    'categories': bidang_counts,
    'total_categories': len(bidang_counts),
}

with open(OUTPUT_DIR / 'dataset_stats.json', 'w', encoding='utf-8') as f:
    json.dump(stats, f, indent=2, ensure_ascii=False)

print(f"\nStats: {json.dumps(stats, indent=2)}")

# ============================================================
# ZIP
# ============================================================
zip_path = Path("/home/z/my-project/download/llm/dataset.zip")
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for f in OUTPUT_DIR.iterdir():
        zf.write(f, f.name)

print(f"\n{'='*60}")
print(f"  FINAL DATASET COMPLETE")
print(f"{'='*60}")
print(f"Total: {len(all_items)} Q&A pairs")
print(f"Train: {len(train)} (80%)")
print(f"Validation: {len(val)} (20%)")
print(f"Categories: {len(bidang_counts)}")
print(f"Format: JSONL with input/output keys")
print(f"ZIP: {zip_path} ({zip_path.stat().st_size/1024:.1f} KB)")
