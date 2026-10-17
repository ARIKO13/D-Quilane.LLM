"""
D'QUILANE DATASET v3 — SD KELAS 1-6 LENGKAP
=============================================
Filter: hapus pola bayi/balita yang bikin output random
Tambah: SD1-SD6 full curriculum (matematika, sains, IPS, bahasa)
Target: 8000+ Q&A pairs
"""
import json, os, random
from pathlib import Path

OUTPUT_DIR = Path("/home/z/my-project/download/llm/dataset_v3")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_WORDS = {0:'nol',1:'satu',2:'dua',3:'tiga',4:'empat',5:'lima',6:'enam',7:'tujuh',
    8:'delapan',9:'sembilan',10:'sepuluh',11:'sebelas',12:'dua belas',13:'tiga belas',
    14:'empat belas',15:'lima belas',16:'enam belas',17:'tujuh belas',18:'delapan belas',
    19:'sembilan belas',20:'dua puluh',30:'tiga puluh',40:'empat puluh',50:'lima puluh',
    60:'enam puluh',70:'tujuh puluh',80:'delapan puluh',90:'sembilan puluh',100:'seratus',
    200:'dua ratus',300:'tiga ratus',400:'empat ratus',500:'lima ratus',600:'enam ratus',
    700:'tujuh ratus',800:'delapan ratus',900:'sembilan ratus',1000:'seribu'}

def n2w(n):
    if n in NUM_WORDS: return NUM_WORDS[n]
    if n < 20: return NUM_WORDS.get(n, str(n))
    if n <= 99:
        t, o = (n//10)*10, n%10
        return NUM_WORDS[t] if o==0 else f"{NUM_WORDS[t]} {NUM_WORDS[o]}"
    if n <= 999:
        h, r = n//100, n%100
        p = 'seratus' if h==1 else f"{NUM_WORDS[h]} ratus"
        return p if r==0 else f"{p} {n2w(r)}"
    if n == 1000: return 'seribu'
    if n <= 9999:
        th, r = n//1000, n%1000
        p = 'seribu' if th==1 else f"{NUM_WORDS[th]} ribu"
        return p if r==0 else f"{p} {n2w(r)}"
    return str(n)

def w(path, items):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for i in items: f.write(json.dumps(i, ensure_ascii=False) + '\n')
    return len(items)

total = 0

# ============================================================
# SD1 — KELAS 1 (Usia 6-7)
# ============================================================
print("SD1...")
items = []
# Math: addition 1-20
for a in range(1,21):
    for b in range(1,21):
        if random.random() < 0.25:
            items.append({'bidang':'SD1','judul':'Matematika','subjudul':'Penjumlahan',
                'prompt':f'{n2w(a)} tambah {n2w(b)}?',
                'response':f'{n2w(a)} tambah {n2w(b)} sama dengan {n2w(a+b)}.'})
# Subtraction 1-20
for a in range(5,21):
    for b in range(1,a):
        if random.random() < 0.2:
            items.append({'bidang':'SD1','judul':'Matematika','subjudul':'Pengurangan',
                'prompt':f'{n2w(a)} dikurangi {n2w(b)}?',
                'response':f'{n2w(a)} dikurangi {n2w(b)} sama dengan {n2w(a-b)}.'})
# Counting 1-100
for i in range(1,101):
    items.append({'bidang':'SD1','judul':'Berhitung','subjudul':'Bilangan 1-100',
        'prompt':f'Angka {i} dalam kata?',
        'response':f'Angka {i} adalah {n2w(i)}.'})
# Reading
for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
    items.append({'bidang':'SD1','judul':'Membaca','subjudul':'Huruf',
        'prompt':f'Huruf apa setelah {chr(ord(letter)-1) if letter!="A" else "Z"}?',
        'response':f'Setelah {chr(ord(letter)-1) if letter!="A" else "Z"} adalah {letter}.'})
items.append({'bidang':'SD1','judul':'Membaca','subjudul':'Alfabet',
    'prompt':'Sebutkan huruf A sampai Z?',
    'response':'A B C D E F G H I J K L M N O P Q R S T U V W X Y Z.'})
# Knowledge
for q,a in [
    ('Apa warna langit?','Warna langit adalah biru.'),
    ('Apa warna rumput?','Warna rumput adalah hijau.'),
    ('Apa warna darah?','Warna darah adalah merah.'),
    ('Apa warna matahari?','Warna matahari adalah kuning.'),
    ('Apa warna awan?','Warna awan adalah putih.'),
    ('Apa warna bendera Indonesia?','Bendera Indonesia adalah merah putih.'),
    ('Apa suara kucing?','Kucing bersuara meong.'),
    ('Apa suara anjing?','Anjing bersuara guk guk.'),
    ('Apa suara sapi?','Sapi bersuara moo.'),
    ('Apa suara kambing?','Kambing bersuara mbek mbek.'),
    ('Apa suara ayam jantan?','Ayam jantan bersuara kukuruyuk.'),
    ('Apa suara bebek?','Bebek bersuara kwek kwek.'),
    ('Apa fungsi mata?','Mata adalah alat untuk melihat.'),
    ('Apa fungsi hidung?','Hidung adalah alat untuk mencium bau.'),
    ('Apa fungsi mulut?','Mulut adalah alat untuk makan dan bicara.'),
    ('Apa fungsi telinga?','Telinga adalah alat untuk mendengar.'),
    ('Apa fungsi tangan?','Tangan adalah alat untuk memegang.'),
    ('Apa fungsi kaki?','Kaki adalah alat untuk berjalan.'),
    ('Berapa jari di tangan?','Ada lima jari di setiap tangan.'),
    ('Berapa hari dalam seminggu?','Ada tujuh hari dalam seminggu.'),
    ('Apa tujuh hari?','Senin Selasa Rabu Kamis Jumat Sabtu Minggu.'),
    ('Berapa bulan dalam setahun?','Ada dua belas bulan dalam setahun.'),
    ('Siapa ayah dan ibu?','Ayah dan ibu adalah orang tua.'),
    ('Siapa kakak?','Kakak adalah saudara yang lebih tua.'),
    ('Siapa adik?','Adik adalah saudara yang lebih muda.'),
    ('Siapa kakek?','Kakek adalah ayah dari orang tua.'),
    ('Siapa nenek?','Nenek adalah ibu dari orang tua.'),
    ('Apa itu sekolah?','Sekolah adalah tempat belajar.'),
    ('Apa itu PR?','PR adalah pekerjaan rumah dari sekolah.'),
    ('Apa itu mandi?','Mandi adalah membersihkan tubuh dengan air.'),
    ('Apa itu sarapan?','Sarapan adalah makan pagi sebelum aktivitas.'),
    ('Setelah Senin adalah?','Setelah Senin adalah Selasa.'),
    ('Setelah Selasa adalah?','Setelah Selasa adalah Rabu.'),
    ('Setelah Rabu adalah?','Setelah Rabu adalah Kamis.'),
    ('Setelah Kamis adalah?','Setelah Kamis adalah Jumat.'),
    ('Setelah Jumat adalah?','Setelah Jumat adalah Sabtu.'),
    ('Setelah Sabtu adalah?','Setelah Sabtu adalah Minggu.'),
]:
    items.append({'bidang':'SD1','judul':'Pengetahuan Umum','subjudul':'Dasar','prompt':q,'response':a})
total += w(OUTPUT_DIR/'SD1'/'sd1_lengkap.jsonl', items)

# ============================================================
# SD2 — KELAS 2 (Usia 7-8)
# ============================================================
print("SD2...")
items = []
# Math: addition/subtraction to 100
for _ in range(200):
    a, b = random.randint(10,99), random.randint(1,50)
    items.append({'bidang':'SD2','judul':'Matematika','subjudul':'Penjumlahan',
        'prompt':f'{n2w(a)} tambah {n2w(b)}?',
        'response':f'{n2w(a)} tambah {n2w(b)} sama dengan {n2w(a+b)}.'})
for _ in range(200):
    a, b = random.randint(10,99), random.randint(1,a)
    items.append({'bidang':'SD2','judul':'Matematika','subjudul':'Pengurangan',
        'prompt':f'{n2w(a)} dikurangi {n2w(b)}?',
        'response':f'{n2w(a)} dikurangi {n2w(b)} sama dengan {n2w(a-b)}.'})
# Multiplication 1-10
for a in range(1,11):
    for b in range(1,11):
        items.append({'bidang':'SD2','judul':'Matematika','subjudul':'Perkalian',
            'prompt':f'{n2w(a)} dikali {n2w(b)}?',
            'response':f'{n2w(a)} dikali {n2w(b)} sama dengan {n2w(a*b)}.'})
# Division
for a in range(2,101):
    for b in range(2,11):
        if a%b==0:
            items.append({'bidang':'SD2','judul':'Matematika','subjudul':'Pembagian',
                'prompt':f'{n2w(a)} dibagi {n2w(b)}?',
                'response':f'{n2w(a)} dibagi {n2w(b)} sama dengan {n2w(a//b)}.'})
            if len([i for i in items if i['subjudul']=='Pembagian'])>=150: break
    if len([i for i in items if i['subjudul']=='Pembagian'])>=150: break
# Word problems
for q,a in [
    ('Andi punya tiga apel. Ibu memberi dua apel. Berapa total?','Total apel Andi adalah lima.'),
    ('Sari punya sepuluh permen. Dia makan empat. Berapa sisa?','Sisa permennya enam.'),
    ('Budi punya lima kantong dengan empat kelereng tiap kantong. Berapa total?','Total kelereng Budi adalah dua puluh.'),
    ('Rina punya dua belas pensil. Dia bagi ke empat teman. Berapa per teman?','Setiap teman dapat tiga pensil.'),
    ('Toko buka jam tujuh pagi sampai sembilan malam. Berapa jam buka?','Toko buka selama empat belas jam.'),
    ('Budi punya lima puluh ribu. Belanja dua puluh ribu. Berapa sisa?','Sisa uang Budi tiga puluh ribu.'),
    ('Ada tiga kotak dengan lima bola tiap kotak. Dua bola hilang. Berapa sisa?','Sisa bola adalah tiga belas.'),
    ('Mobil jalan enam puluh km/jam selama dua jam. Berapa jarak?','Jarak tempuh seratus dua puluh kilometer.'),
]:
    items.append({'bidang':'SD2','judul':'Matematika','subjudul':'Soal Cerita','prompt':q,'response':a})
# Science
for q,a in [
    ('Apa itu air?','Air adalah benda cair yang kita minum.'),
    ('Apa itu es?','Es adalah air yang membeku karena dingin.'),
    ('Apa itu uap?','Uap adalah air berbentuk gas.'),
    ('Air bisa berbentuk apa saja?','Air bisa cair, padat (es), dan gas (uap).'),
    ('Apa fungsi akar?','Akar menyerap air dari tanah.'),
    ('Apa fungsi batang?','Batang menjaga tanaman tetap tegak.'),
    ('Apa fungsi daun?','Daun membuat makanan untuk tanaman.'),
    ('Apa fungsi bunga?','Bunga berfungsi untuk reproduksi tanaman.'),
    ('Apa yang dibutuhkan tanaman?','Tanaman butuh air, cahaya matahari, dan udara.'),
    ('Apa itu fotosintesis?','Fotosintesis adalah proses tumbuhan membuat makanan dari cahaya.'),
    ('Apa hewan terbesar di darat?','Gajah adalah hewan terbesar di darat.'),
    ('Siapa raja hutan?','Singa adalah raja hutan.'),
    ('Hewan apa yang bisa terbang?','Burung bisa terbang.'),
    ('Hewan apa yang hidup di air?','Ikan hidup di air.'),
    ('Apa yang dihasilkan sapi?','Sapi memberi susu.'),
    ('Apa yang dihasilkan lebah?','Lebah membuat madu.'),
    ('Dari mana kupu-kupu berasal?','Kupu-kupu berasal dari ulat.'),
    ('Apa itu kompas?','Kompas adalah alat penunjuk arah.'),
    ('Matahari terbit di?','Matahari terbit di timur.'),
    ('Matahari terbenam di?','Matahari terbenam di barat.'),
]:
    items.append({'bidang':'SD2','judul':'Sains','subjudul':'Dasar','prompt':q,'response':a})
# Social
for q,a in [
    ('Apa ibu kota Indonesia?','Ibu kota Indonesia adalah Jakarta.'),
    ('Apa bahasa resmi Indonesia?','Bahasa resmi Indonesia adalah bahasa Indonesia.'),
    ('Apa mata uang Indonesia?','Mata uang Indonesia adalah rupiah.'),
    ('Apa bendera Indonesia?','Bendera Indonesia adalah merah putih.'),
    ('Apa lagu kebangsaan Indonesia?','Lagu kebangsaan Indonesia adalah Indonesia Raya.'),
    ('Kapan Indonesia merdeka?','Indonesia merdeka pada tujuh belas agustus seribu sembilan ratus empat puluh lima.'),
    ('Siapa proklamator Indonesia?','Soekarno dan Hatta adalah proklamator Indonesia.'),
    ('Siapa presiden pertama Indonesia?','Presiden pertama Indonesia adalah Soekarno.'),
    ('Apa itu antre?','Antre adalah menunggu giliran dengan tertib.'),
    ('Apa itu berbagi?','Berbagi adalah memberi sebagian ke orang lain.'),
    ('Apa itu tolong?','Tolong adalah kata sopan minta bantuan.'),
    ('Apa balasan terima kasih?','Sama-sama adalah balasan untuk terima kasih.'),
]:
    items.append({'bidang':'SD2','judul':'IPS','subjudul':'Dasar','prompt':q,'response':a})
total += w(OUTPUT_DIR/'SD2'/'sd2_lengkap.jsonl', items)

# ============================================================
# SD3 — KELAS 3 (Usia 8-9)
# ============================================================
print("SD3...")
items = []
# Math: 2-digit multiplication
for _ in range(200):
    a, b = random.randint(11,50), random.randint(2,12)
    items.append({'bidang':'SD3','judul':'Matematika','subjudul':'Perkalian 2 Angka',
        'prompt':f'{n2w(a)} dikali {n2w(b)}?',
        'response':f'{n2w(a)} dikali {n2w(b)} sama dengan {n2w(a*b)}.'})
# Fractions
for num, den, word in [(1,2,'setengah'),(1,3,'sepertiga'),(1,4,'seperempat'),(1,5,'seperlima'),
    (2,3,'dua per tiga'),(3,4,'tiga per empat'),(2,5,'dua per lima'),(3,5,'tiga per lima'),
    (4,5,'empat per lima'),(1,6,'seperenam'),(5,6,'lima per enam'),(1,8,'seperdelapan'),
    (3,8,'tiga per delapan'),(5,8,'lima per delapan'),(7,8,'tujuh per delapan'),
    (1,10,'sepersepuluh'),(3,10,'tiga per sepuluh'),(7,10,'tujuh per sepuluh')]:
    items.append({'bidang':'SD3','judul':'Matematika','subjudul':'Pecahan',
        'prompt':f'Apa {num}/{den} dalam kata?',
        'response':f'{num}/{den} adalah {word}.'})
    items.append({'bidang':'SD3','judul':'Matematika','subjudul':'Pecahan Desimal',
        'prompt':f'Berapa {num}/{den} dalam desimal?',
        'response':f'{num}/{den} sama dengan {num/den:.4f}.'})
# Decimals
for d in [0.1,0.2,0.25,0.3,0.4,0.5,0.6,0.75,0.8,0.9]:
    items.append({'bidang':'SD3','judul':'Matematika','subjudul':'Desimal',
        'prompt':f'{d} dalam pecahan?','response':f'{d} sama dengan desimal.'})
# Science
for q,a in [
    ('Apa tiga wujud benda?','Tiga wujud benda: padat, cair, dan gas.'),
    ('Apa contoh benda padat?','Batu, kayu, dan es adalah benda padat.'),
    ('Apa contoh benda cair?','Air, minyak, dan susu adalah benda cair.'),
    ('Apa contoh benda gas?','Udara, uap, dan oksigen adalah benda gas.'),
    ('Apa itu perubahan wujud?','Perubahan wujud adalah berubahnya benda dari satu wujud ke wujud lain.'),
    ('Apa itu mencair?','Mencair adalah perubahan dari padat ke cair.'),
    ('Apa itu membeku?','Membeku adalah perubahan dari cair ke padat.'),
    ('Apa itu menguap?','Menguap adalah perubahan dari cair ke gas.'),
    ('Apa itu mengembun?','Mengembun adalah perubahan dari gas ke cair.'),
    ('Apa itu energi?','Energi adalah kemampuan untuk melakukan kerja.'),
    ('Apa itu gaya?','Gaya adalah dorongan atau tarikan pada benda.'),
    ('Apa itu gravitasi?','Gravitasi adalah gaya tarik bumi ke benda.'),
    ('Apa itu gesekan?','Gesekan adalah gaya yang melawan gerak benda.'),
    ('Apa itu magnet?','Magnet adalah benda yang dapat menarik besi.'),
    ('Apa kutub magnet?','Magnet memiliki kutub utara dan selatan.'),
    ('Apa itu rangka tubuh?','Rangka adalah susunan tulang yang menopang tubuh.'),
    ('Berapa tulang manusia?','Manusia memiliki dua ratus enam tulang.'),
    ('Apa fungsi rangka?','Rangka menopang tubuh dan melindungi organ.'),
    ('Apa itu otot?','Otot adalah jaringan yang menggerakkan tubuh.'),
    ('Apa fungsi jantung?','Jantung memompa darah ke seluruh tubuh.'),
    ('Apa fungsi paru-paru?','Paru-paru menukar oksigen dan karbon dioksida.'),
    ('Apa fungsi otak?','Otak mengendalikan seluruh fungsi tubuh.'),
    ('Apa fungsi ginjal?','Ginjal menyaring darah dan menghasilkan urine.'),
    ('Apa fungsi hati?','Hati menghasilkan empedu dan menyaring racun.'),
    ('Apa fungsi lambung?','Lambung mencerna makanan.'),
    ('Apa fungsi usus?','Usus menyerap nutrisi dari makanan.'),
    ('Apa itu ekosistem?','Ekosistem adalah hubungan antara makhluk hidup dan lingkungannya.'),
    ('Apa itu rantai makanan?','Rantai makanan adalah urutan makan dan dimakan.'),
    ('Apa produsen di ekosistem?','Tumbuhan adalah produsen karena bisa membuat makanan sendiri.'),
    ('Apa konsumen di ekosistem?','Hewan adalah konsumen karena memakan tumbuhan atau hewan lain.'),
    ('Apa pengurai di ekosistem?','Bakteri dan jamur adalah pengurai yang mengurai sisa makhluk hidup.'),
]:
    items.append({'bidang':'SD3','judul':'Sains','subjudul':'Materi dan Energi','prompt':q,'response':a})
# IPS
for q,a in [
    ('Apa pulau besar di Indonesia?','Sumatra, Jawa, Kalimantan, Sulawesi, Papua.'),
    ('Apa benua di dunia?','Asia, Afrika, Eropa, Amerika Utara, Amerika Selatan, Australia, Antartika.'),
    ('Berapa benua di dunia?','Ada tujuh benua di dunia.'),
    ('Apa samudra terbesar?','Samudra Pasifik adalah samudra terbesar.'),
    ('Berapa provinsi di Indonesia?','Indonesia memiliki tiga puluh delapan provinsi.'),
    ('Apa ibu kota Jepang?','Ibu kota Jepang adalah Tokyo.'),
    ('Apa ibu kota Prancis?','Ibu kota Prancis adalah Paris.'),
    ('Apa ibu kota Inggris?','Ibu kota Inggris adalah London.'),
    ('Apa ibu kota Amerika?','Ibu kota Amerika adalah Washington DC.'),
    ('Apa ibu kota Tiongkok?','Ibu kota Tiongkok adalah Beijing.'),
]:
    items.append({'bidang':'SD3','judul':'IPS','subjudul':'Geografi','prompt':q,'response':a})
total += w(OUTPUT_DIR/'SD3'/'sd3_lengkap.jsonl', items)

# ============================================================
# SD4 — KELAS 4 (Usia 9-10)
# ============================================================
print("SD4...")
items = []
# Math: long division, mixed fractions, percentages, angles
for _ in range(150):
    a, b = random.randint(100,999), random.randint(2,12)
    if a%b==0:
        items.append({'bidang':'SD4','judul':'Matematika','subjudul':'Pembagian Panjang',
            'prompt':f'{n2w(a)} dibagi {n2w(b)}?',
            'response':f'{n2w(a)} dibagi {n2w(b)} sama dengan {n2w(a//b)}.'})
# Percentages
for pct in [10,20,25,30,40,50,60,70,75,80,90,100]:
    for base in [100,200,500,1000]:
        items.append({'bidang':'SD4','judul':'Matematika','subjudul':'Persentase',
            'prompt':f'Berapa {pct} persen dari {n2w(base)}?',
            'response':f'{pct} persen dari {n2w(base)} adalah {n2w(base*pct//100)}.'})
# Angles
for q,a in [
    ('Apa itu sudut?','Sudut adalah besaran antara dua garis yang bertemu.'),
    ('Berapa derajat sudut siku-siku?','Sudut siku-siku adalah sembilan puluh derajat.'),
    ('Berapa derajat sudut lurus?','Sudut lurus adalah seratus delapan puluh derajat.'),
    ('Berapa derajat sudut penuh?','Sudut penuh adalah tiga ratus enam puluh derajat.'),
    ('Apa sudut lancip?','Sudut lancip adalah sudut kurang dari sembilan puluh derajat.'),
    ('Apa sudut tumpul?','Sudut tumpul adalah sudut lebih dari sembilan puluh derajat.'),
    ('Apa itu segitiga?','Segitiga adalah bentuk dengan tiga sisi.'),
    ('Apa persegi?','Persegi adalah bentuk dengan empat sisi sama panjang.'),
    ('Apa persegi panjang?','Persegi panjang memiliki dua pasang sisi sama panjang.'),
    ('Apa lingkaran?','Lingkaran adalah bentuk dengan semua titik berjarak sama dari pusat.'),
    ('Apa rumus luas persegi?','Luas persegi = sisi kali sisi.'),
    ('Apa rumus luas persegi panjang?','Luas persegi panjang = panjang kali lebar.'),
    ('Apa rumus keliling persegi?','Keliling persegi = empat kali sisi.'),
    ('Apa rumus keliling lingkaran?','Keliling lingkaran = dua kali pi kali jari-jari.'),
    ('Berapa nilai pi?','Pi kira-kira tiga koma empat belas.'),
]:
    items.append({'bidang':'SD4','judul':'Matematika','subjudul':'Geometri','prompt':q,'response':a})
# Science
for q,a in [
    ('Apa siklus air?','Siklus air: menguap, mengembun, turun hujan, mengalir ke laut.'),
    ('Apa itu evaporasi?','Evaporasi adalah proses air berubah menjadi uap.'),
    ('Apa itu kondensasi?','Kondensasi adalah uap berubah menjadi air.'),
    ('Apa itu presipitasi?','Presipitasi adalah hujan atau turunnya air dari langit.'),
    ('Apa sumber air tawar?','Sungai, danau, dan mata air adalah sumber air tawar.'),
    ('Apa itu listrik?','Listrik adalah aliran elektron melalui konduktor.'),
    ('Apa konduktor listrik?','Konduktor adalah benda yang menghantarkan listrik.'),
    ('Apa isolator listrik?','Isolator adalah benda yang tidak menghantarkan listrik.'),
    ('Apa sumber listrik?','Listrik berasal dari pembangkit listrik.'),
    ('Apa itu rangkaian seri?','Rangkaian seri adalah rangkaian listrik yang disusun berurutan.'),
    ('Apa itu rangkaian paralel?','Rangkaian paralel adalah rangkaian listrik yang disusun bercabang.'),
    ('Apa itu globalisasi?','Globalisasi adalah hubungan antar negara di dunia.'),
    ('Apa itu demokrasi?','Demokrasi adalah pemerintahan dari rakyat oleh rakyat untuk rakyat.'),
    ('Apa itu pemilu?','Pemilu adalah pemilihan umum untuk memilih pemimpin.'),
    ('Apa Pancasila?','Pancasila adalah dasar negara Indonesia dengan lima sila.'),
]:
    items.append({'bidang':'SD4','judul':'Sains','subjudul':'Lanjut','prompt':q,'response':a})
total += w(OUTPUT_DIR/'SD4'/'sd4_lengkap.jsonl', items)

# ============================================================
# SD5 — KELAS 5 (Usia 10-11)
# ============================================================
print("SD5...")
items = []
# Math: ratios, speed, volume, algebra basics
for _ in range(100):
    a, b = random.randint(1,20), random.randint(1,20)
    items.append({'bidang':'SD5','judul':'Matematika','subjudul':'Perbandingan',
        'prompt':f'Berapa perbandingan {n2w(a)} dengan {n2w(b)}?',
        'response':f'Perbandingan {n2w(a)} dengan {n2w(b)} adalah {a}:{b}.'})
# Speed/distance/time
for speed, time in [(60,2),(60,3),(80,2),(40,5),(100,3),(50,4),(70,2),(90,3)]:
    dist = speed * time
    items.append({'bidang':'SD5','judul':'Matematika','subjudul':'Kecepatan',
        'prompt':f'Mobil jalan {n2w(speed)} km/jam selama {n2w(time)} jam. Berapa jarak?',
        'response':f'Jarak = kecepatan kali waktu = {n2w(speed)} kali {n2w(time)} = {n2w(dist)} kilometer.'})
# Volume
for ln,wd,ht in [(2,3,4),(3,4,5),(2,2,2),(5,5,5),(3,3,4),(4,4,3),(6,3,2),(2,5,3)]:
    vol = ln*wd*ht
    items.append({'bidang':'SD5','judul':'Matematika','subjudul':'Volume',
        'prompt':f'Berapa volume balok panjang {n2w(ln)} lebar {n2w(wd)} tinggi {n2w(ht)}?',
        'response':f'Volume = panjang kali lebar kali tinggi = {n2w(ln)} kali {n2w(wd)} kali {n2w(ht)} = {n2w(vol)} meter kubik.'})
# Algebra basics
for q,a in [
    ('Jika x + 5 = 12, berapa x?','x sama dengan tujuh.'),
    ('Jika x dikali 3 = 15, berapa x?','x sama dengan lima.'),
    ('Jika x - 4 = 8, berapa x?','x sama dengan dua belas.'),
    ('Jika x dibagi 2 = 6, berapa x?','x sama dengan dua belas.'),
    ('Jika 2x = 10, berapa x?','x sama dengan lima.'),
    ('Jika x + x = 8, berapa x?','x sama dengan empat.'),
    ('Jika 3x = 21, berapa x?','x sama dengan tujuh.'),
    ('Jika x + 3 = 10, berapa x?','x sama dengan tujuh.'),
]:
    items.append({'bidang':'SD5','judul':'Matematika','subjudul':'Aljabar Dasar','prompt':q,'response':a})
# Science
for q,a in [
    ('Apa struktur bumi?','Bumi memiliki kerak, mantel, dan inti.'),
    ('Apa kerak bumi?','Kerak bumi adalah lapisan terluar bumi.'),
    ('Apa mantel bumi?','Mantel bumi adalah lapisan tengah bumi.'),
    ('Apa inti bumi?','Inti bumi adalah lapisan terdalam bumi.'),
    ('Apa itu lempeng tektonik?','Lempeng tektonik adalah potongan kerak bumi yang bergerak.'),
    ('Apa penyebab gempa?','Gempa disebabkan pergerakan lempeng tektonik.'),
    ('Apa penyebab gunung berapi?','Gunung berapi terjadi karena magma keluar dari bumi.'),
    ('Berapa planet di tata surya?','Ada delapan planet di tata surya.'),
    ('Apa planet terdekat matahari?','Merkurius adalah planet terdekat matahari.'),
    ('Apa planet terbesar?','Jupiter adalah planet terbesar.'),
    ('Apa planet berkening?','Saturnus adalah planet berkening.'),
    ('Apa planet merah?','Mars adalah planet merah.'),
    ('Apa bintang terdekat bumi?','Matahari adalah bintang terdekat bumi.'),
    ('Apa galaksi kita?','Bima Sakti adalah galaksi kita.'),
    ('Apa itu black hole?','Black hole adalah lubang hitam dengan gravitasi sangat kuat.'),
    ('Apa itu fotosintesis?','Fotosintesis adalah proses tumbuhan membuat makanan dari cahaya matahari.'),
    ('Apa rumus fotosintesis?','Fotosintesis: karbon dioksida plus air dengan cahaya menghasilkan glukosa dan oksigen.'),
    ('Apa itu penyerbukan?','Penyerbukan adalah pemindahan serbuk sari ke putik bunga.'),
    ('Apa itu penyebaran biji?','Penyebaran biji adalah persebaran biji ke tempat tumbuh baru.'),
    ('Apa itu adaptasi?','Adaptasi adalah kemampuan makhluk hidup menyesuaikan diri dengan lingkungan.'),
    ('Apa contoh adaptasi?','Kaktus memiliki daun berduri untuk hidup di gurun.'),
    ('Apa itu seleksi alam?','Seleksi alam adalah makhluk hidup yang paling sesuai lingkungan akan bertahan.'),
]:
    items.append({'bidang':'SD5','judul':'Sains','subjudul':'Bumi dan Antariksa','prompt':q,'response':a})
# History
for q,a in [
    ('Siapa presiden pertama Indonesia?','Presiden pertama Indonesia adalah Soekarno.'),
    ('Siapa presiden kedua Indonesia?','Presiden kedua Indonesia adalah Soeharto.'),
    ('Siapa presiden ketiga Indonesia?','Presiden ketiga Indonesia adalah Habibie.'),
    ('Siapa presiden keempat Indonesia?','Presiden keempat Indonesia adalah Abdurrahman Wahid.'),
    ('Siapa presiden kelima Indonesia?','Presiden kelima Indonesia adalah Megawati.'),
    ('Siapa presiden keenam Indonesia?','Presiden keenam Indonesia adalah SBY.'),
    ('Siapa presiden ketujuh Indonesia?','Presiden ketujuh Indonesia adalah Jokowi.'),
    ('Apa kerajaan Hindu pertama di Indonesia?','Kutai adalah kerajaan Hindu pertama di Indonesia.'),
    ('Apa kerajaan maritim terbesar?','Majapahit adalah kerajaan maritim terbesar.'),
    ('Apa kerajaan Buddha terbesar?','Sriwijaya adalah kerajaan Buddha terbesar.'),
    ('Siapa pencetus Pancasila?','Soekarno adalah pencetus Pancasila.'),
    ('Apa lima sila Pancasila?','Ketuhanan, kemanusiaan, persatuan, kerakyatan, keadilan sosial.'),
    ('Siapa pahlawan dari Aceh?','Cut Nyak Dien adalah pahlawan dari Aceh.'),
    ('Siapa pahlawan emansipasi wanita?','R.A. Kartini adalah pahlawan emansipasi wanita.'),
    ('Siapa Bapak Pendidikan Indonesia?','Ki Hajar Dewantara adalah Bapak Pendidikan Indonesia.'),
    ('Siapa Bapak Koperasi Indonesia?','Mohammad Hatta adalah Bapak Koperasi Indonesia.'),
    ('Kapan Proklamasi Kemerdekaan?','Proklamasi kemerdekaan dibacakan tujuh belas agustus seribu sembilan ratus empat puluh lima.'),
    ('Apa teks proklamasi?','Teks proklamasi adalah pernyataan kemerdekaan Indonesia.'),
    ('Siapa bapak proklamasi?','Soekarno dan Hatta adalah bapak proklamasi.'),
    ('Apa sumpah pemuda?','Sumpah pemuda diikrarkan tanggal dua puluh delapan oktober seribu sembilan ratus dua puluh delapan.'),
]:
    items.append({'bidang':'SD5','judul':'Sejarah','subjudul':'Indonesia','prompt':q,'response':a})
total += w(OUTPUT_DIR/'SD5'/'sd5_lengkap.jsonl', items)

# ============================================================
# SD6 — KELAS 6 (Usia 11-12)
# ============================================================
print("SD6...")
items = []
# Math: statistics, probability, coordinates
for q,a in [
    ('Apa itu rata-rata?','Rata-rata adalah jumlah semua nilai dibagi banyaknya nilai.'),
    ('Apa itu median?','Median adalah nilai tengah setelah data diurutkan.'),
    ('Apa itu modus?','Modus adalah nilai yang paling sering muncul.'),
    ('Berapa rata-rata dari 10, 20, 30?','Rata-rata dari 10, 20, 30 adalah dua puluh.'),
    ('Berapa rata-rata dari 5, 10, 15, 20?','Rata-rata adalah dua belas koma lima.'),
    ('Apa itu probabilitas?','Probabilitas adalah peluang suatu peristiwa terjadi.'),
    ('Berapa peluang melempar koin?','Peluang muncul gambar atau angka adalah lima puluh persen.'),
    ('Berapa peluang dadu angka enam?','Peluang muncul angka enam pada dadu adalah satu per enam.'),
    ('Apa itu koordinat?','Koordinat adalah pasangan angka yang menunjukkan posisi titik.'),
    ('Apa sumbu X?','Sumbu X adalah garis horizontal pada koordinat.'),
    ('Apa sumbu Y?','Sumbu Y adalah garis vertikal pada koordinat.'),
    ('Apa titik origin?','Origin adalah titik nol koma nol pada koordinat.'),
    ('Apa rumus luas segitiga?','Luas segitiga = setengah kali alas kali tinggi.'),
    ('Apa rumus luas lingkaran?','Luas lingkaran = pi kali jari-jari kali jari-jari.'),
    ('Apa rumus volume kubus?','Volume kubus = sisi kali sisi kali sisi.'),
    ('Apa rumus volume balok?','Volume balok = panjang kali lebar kali tinggi.'),
    ('Apa itu persen?','Persen adalah per seratus.'),
    ('Berapa 50 persen dari 200?','50 persen dari 200 adalah 100.'),
    ('Berapa 25 persen dari 80?','25 persen dari 80 adalah 20.'),
    ('Apa itu ratio?','Ratio adalah perbandingan dua nilai.'),
]:
    items.append({'bidang':'SD6','judul':'Matematika','subjudul':'Statistik dan Peluang','prompt':q,'response':a})
# Science
for q,a in [
    ('Apa itu ekosistem?','Ekosistem adalah hubungan makhluk hidup dan lingkungan.'),
    ('Apa komponen biotik?','Biotik adalah komponen hidup (tumbuhan, hewan, manusia).'),
    ('Apa komponen abiotik?','Abiotik adalah komponen tak hidup (air, tanah, udara, cahaya).'),
    ('Apa produsen?','Produsen adalah tumbuhan yang membuat makanan sendiri.'),
    ('Apa konsumen?','Konsumen adalah hewan yang memakan tumbuhan atau hewan lain.'),
    ('Apa pengurai?','Pengurai adalah bakteri dan jamur yang mengurai sisa makhluk hidup.'),
    ('Apa rantai makanan?','Rantai makanan adalah urutan makan dan dimakan.'),
    ('Apa jaring makanan?','Jaring makanan adalah hubungan makan dan dimakan yang kompleks.'),
    ('Apa simbiosis mutualisme?','Mutualisme adalah hubungan yang saling menguntungkan.'),
    ('Apa simbiosis parasitisme?','Parasitisme adalah hubungan yang menguntungkan satu pihak dan merugikan lain.'),
    ('Apa simbiosis komensalisme?','Komensalisme adalah hubungan yang menguntungkan satu pihak dan tidak merugikan lain.'),
    ('Apa itu energi terbarukan?','Energi terbarukan adalah energi yang bisa diperbarui (surya, angin, air).'),
    ('Apa energi tak terbarukan?','Energi tak terbarukan adalah energi yang tidak bisa diperbarui (minyak, batubara).'),
    ('Apa itu pemanasan global?','Pemanasan global adalah naiknya suhu bumi karena polusi.'),
    ('Apa penyebab pemanasan global?','Pemanasan global disebabkan gas rumah kaca seperti karbon dioksida.'),
    ('Apa itu Recycling?','Recycling adalah daur ulang sampah menjadi barang baru.'),
    ('Apa 3R?','3R adalah Reduce (kurangi), Reuse (pakai lagi), Recycle (daur ulang).'),
    ('Apa itu teknologi?','Teknologi adalah alat atau metode buatan manusia untuk memudahkan hidup.'),
    ('Apa itu internet?','Internet adalah jaringan yang menghubungkan komputer di seluruh dunia.'),
    ('Apa itu AI?','AI adalah kecerdasan buatan, mesin yang meniru kecerdasan manusia.'),
    ('Apa itu LLM?','LLM adalah Large Language Model, model AI bahasa besar.'),
    ('Apa itu coding?','Coding adalah menulis instruksi untuk komputer.'),
    ('Apa itu Python?','Python adalah bahasa pemrograman yang populer.'),
    ('Apa itu JavaScript?','JavaScript adalah bahasa pemrograman untuk website.'),
]:
    items.append({'bidang':'SD6','judul':'Sains','subjudul':'Ekosistem dan Teknologi','prompt':q,'response':a})
# Social
for q,a in [
    ('Apa itu globalisasi?','Globalisasi adalah hubungan antar negara di dunia yang semakin dekat.'),
    ('Apa efek globalisasi?','Globalisasi memudahkan komunikasi, perdagangan, dan pertukaran budaya.'),
    ('Apa itu ASEAN?','ASEAN adalah perkumpulan negara-negara Asia Tenggara.'),
    ('Berapa anggota ASEAN?','ASEAN memiliki sepuluh negara anggota.'),
    ('Apa negara pendiri ASEAN?','Indonesia, Malaysia, Singapura, Thailand, Filipina adalah pendiri ASEAN.'),
    ('Apa itu demokrasi?','Demokrasi adalah pemerintahan dari rakyat oleh rakyat untuk rakyat.'),
    ('Apa itu pemilu?','Pemilu adalah pemilihan umum untuk memilih wakil rakyat.'),
    ('Apa hak warga negara?','Hak warga negara meliputi pendidikan, kesehatan, dan pekerjaan.'),
    ('Apa kewajiban warga negara?','Kewajiban warga negara meliputi membayar pajak dan taat hukum.'),
    ('Apa itu UUD?','UUD adalah Undang-Undang Dasar, hukum tertinggi negara.'),
    ('Apa UUD Indonesia?','UUD negara Indonesia adalah UUD 1945.'),
    ('Apa lambang negara Indonesia?','Lambang negara Indonesia adalah Garuda Pancasila.'),
    ('Apa semboanan Indonesia?','Semboanan Indonesia adalah Bhinneka Tunggal Ika.'),
    ('Apa arti Bhinneka Tunggal Ika?','Bhinneka Tunggal Ika berarti berbeda-beda tetapi tetap satu.'),
    ('Apa bahasa persatuan Indonesia?','Bahasa persatuan Indonesia adalah bahasa Indonesia.'),
]:
    items.append({'bidang':'SD6','judul':'IPS','subjudul':'Globalisasi dan Negara','prompt':q,'response':a})
total += w(OUTPUT_DIR/'SD6'/'sd6_lengkap.jsonl', items)

# ============================================================
# MULTI-BAHASA (keep, no filter)
# ============================================================
print("Multi-Bahasa...")
LANGS = [
    ('Indonesian','Halo','Terima kasih','Ya','Tidak','Selamat pagi','Selamat malam','Goodbye','Apa kabar','Sama-sama'),
    ('English','Hello','Thank you','Yes','No','Good morning','Good night','Goodbye','How are you','You are welcome'),
    ('Spanish','Hola','Gracias','Si','No','Buenos dias','Buenas noches','Adios','Como estas','De nada'),
    ('French','Bonjour','Merci','Oui','Non','Bonjour','Bonsoir','Au revoir','Comment allez-vous','De rien'),
    ('German','Hallo','Danke','Ja','Nein','Guten Morgen','Guten Abend','Auf Wiedersehen','Wie geht es dir','Bitte'),
    ('Italian','Ciao','Grazie','Si','No','Buongiorno','Buonanotte','Arrivederci','Come stai','Prego'),
    ('Portuguese','Ola','Obrigado','Sim','Nao','Bom dia','Boa noite','Tchau','Como vai','De nada'),
    ('Russian','Privet','Spasibo','Da','Nyet','Dobroe utro','Dobry vecher','Do svidaniya','Kak dela','Pozhaluysta'),
    ('Japanese','Konnichiwa','Arigatou','Hai','Iie','Ohayou','Oyasumi','Sayonara','Ogenki desu ka','Dou itashimashite'),
    ('Korean','Annyeong','Gomapda','Ne','Anio','Annyeong haseyo','Annyeong hi jumuseyo','Annyeong','Annyeonghi gyeseyo','Cheonmaneyo'),
    ('Chinese','Ni hao','Xiexie','Shi','Bu','Zao an','Wan an','Zai jian','Ni hao ma','Bu ke qi'),
    ('Arabic','Marhaba','Shukran','Naam','La','Sabah al khair','Tasbih ala khair','Ma as salama','Kayfa haluk','Afwan'),
    ('Hindi','Namaste','Dhanyavad','Haan','Nahi','Suprabhat','Shubh ratri','Alvida','Kaise ho','Swagat hai'),
    ('Dutch','Hallo','Dank je','Ja','Nee','Goedemorgen','Goedenavond','Dag','Hoe gaat het','Graag gedaan'),
    ('Swedish','Hej','Tack','Ja','Nej','God morgon','God kvall','Hej da','Hur mår du','Varsagod'),
    ('Thai','Sawasdee','Khob khun','Chai','Mai','Sawasdee ton chao','Sawasdee ton yen','La gon','Sabai dee mai','Mai pen rai'),
    ('Turkish','Merhaba','Tesekkur','Evet','Hayir','Gunaydin','Iyi geceler','Hoscakal','Nasilsin','Rica ederim'),
    ('Greek','Yassou','Efharisto','Nai','Ochi','Kalimera','Kalinyhta','Antio','Ti kaneis','Parakalo'),
    ('Hebrew','Shalom','Toda','Ken','Lo','Boker tov','Laila tov','Lehitraot','Ma nishma','Bevakasha'),
    ('Polish','Czesc','Dziekuje','Tak','Nie','Dzien dobry','Dobranoc','Do widzenia','Jak sie masz','Prosze'),
    ('Vietnamese','Xin chao','Cam on','Co','Khong','Chao buoi sang','Chuc ngu ngon','Tam biet','Ban khoe khong','Khong co gi'),
    ('Swahili','Jambo','Asante','Ndio','Hapana','Habari ya asubuhi','Habari ya jioni','Kwaheri','Habari yako','Karibu'),
    ('Czech','Ahoj','Dekuji','Ano','Ne','Dobre rano','Dobrou noc','Nashledanou','Jak se mas','Prosím'),
    ('Ukrainian','Pryvit','Dyakuyu','Tak','Ni','Dobranku','Dobroiy vechir','Do pobachennya','Yak spravy','Bud laska'),
]
items = []
for lang, hello, thanks, yes, no, morning, night, bye, how, welcome in LANGS:
    for phrase, translation in [('Hello',hello),('Thank you',thanks),('Yes',yes),('No',no),
                                 ('Good morning',morning),('Good night',night),('Goodbye',bye),
                                 ('How are you',how),('You are welcome',welcome)]:
        items.append({'bidang':'multibahasa','judul':lang,'subjudul':'Terjemahan',
            'prompt':f'{phrase} in {lang}?',
            'response':f'{phrase} in {lang} is {translation}.'})
total += w(OUTPUT_DIR/'multibahasa'/'multi_bahasa.jsonl', items)

# Kosa kata 400 (English-Indonesian + reverse)
VOCAB = [
    ('Mother','Ibu'),('Father','Ayah'),('Sister','Saudara perempuan'),('Brother','Saudara laki-laki'),
    ('Son','Anak laki-laki'),('Daughter','Anak perempuan'),('Grandfather','Kakek'),('Grandmother','Nenek'),
    ('Uncle','Paman'),('Aunt','Tante'),('Cousin','Sepupu'),('Nephew','Keponakan laki-laki'),
    ('Niece','Keponakan perempuan'),('Husband','Suami'),('Wife','Istri'),('Parent','Orang tua'),
    ('Child','Anak'),('Baby','Bayi'),('Family','Keluarga'),('Friend','Teman'),
    ('Cat','Kucing'),('Dog','Anjing'),('Bird','Burung'),('Fish','Ikan'),('Horse','Kuda'),('Cow','Sapi'),
    ('Goat','Kambing'),('Sheep','Domba'),('Pig','Babi'),('Chicken','Ayam'),('Duck','Bebek'),
    ('Elephant','Gajah'),('Tiger','Harimau'),('Lion','Singa'),('Bear','Beruang'),('Snake','Ular'),
    ('Butterfly','Kupu-kupu'),('Bee','Lebah'),('Ant','Semut'),('Spider','Laba-laba'),
    ('Water','Air'),('Fire','Api'),('Earth','Bumi'),('Wind','Angin'),('Rain','Hujan'),('Snow','Salju'),
    ('Ice','Es'),('Sun','Matahari'),('Moon','Bulan'),('Star','Bintang'),('Sky','Langit'),('Cloud','Awan'),
    ('Lightning','Petir'),('Thunder','Guntur'),('Rainbow','Pelangi'),('Storm','Badai'),
    ('Tree','Pohon'),('Flower','Bunga'),('Grass','Rumput'),('Leaf','Daun'),('Root','Akar'),('Branch','Dahan'),
    ('Seed','Biji'),('Fruit','Buah'),('Vegetable','Sayur'),('Rice','Beras'),('Bread','Roti'),('Meat','Daging'),
    ('Egg','Telur'),('Milk','Susu'),('Cheese','Keju'),('Sugar','Gula'),('Salt','Garam'),
    ('Coffee','Kopi'),('Tea','Teh'),('Juice','Jus'),('Soup','Sup'),('Noodle','Mie'),('Cake','Kue'),
    ('House','Rumah'),('Door','Pintu'),('Window','Jendela'),('Roof','Atap'),('Wall','Dinding'),('Floor','Lantai'),
    ('Room','Kamar'),('Kitchen','Dapur'),('Bathroom','Kamar mandi'),('Bedroom','Kamar tidur'),
    ('Chair','Kursi'),('Table','Meja'),('Bed','Kasur'),('Cup','Cangkir'),('Plate','Piring'),
    ('Spoon','Sendok'),('Fork','Garpu'),('Knife','Pisau'),('Bowl','Mangkuk'),('Bottle','Botol'),
    ('Book','Buku'),('Pen','Pulpen'),('Pencil','Pensil'),('Paper','Kertas'),('Eraser','Penghapus'),
    ('Ruler','Penggaris'),('Bag','Tas'),('Shoe','Sepatu'),('Sock','Kaus kaki'),('Shirt','Kemeja'),
    ('Pants','Celana'),('Dress','Gaun'),('Hat','Topi'),('Glasses','Kacamata'),('Umbrella','Payung'),
    ('Car','Mobil'),('Bus','Bus'),('Train','Kereta'),('Plane','Pesawat'),('Boat','Perahu'),('Bicycle','Sepeda'),
    ('Motorcycle','Motor'),('Truck','Truk'),('Ship','Kapal'),('Helicopter','Helikopter'),
    ('Red','Merah'),('Blue','Biru'),('Green','Hijau'),('Yellow','Kuning'),('White','Putih'),('Black','Hitam'),
    ('Orange','Oranye'),('Purple','Ungu'),('Pink','Merah muda'),('Brown','Coklat'),('Gray','Abu-abu'),
    ('Big','Besar'),('Small','Kecil'),('Tall','Tinggi'),('Short','Pendek'),('Long','Panjang'),('Wide','Lebar'),
    ('Narrow','Sempit'),('Thick','Tebal'),('Thin','Tipis'),('Heavy','Berat'),('Light','Ringan'),
    ('Fast','Cepat'),('Slow','Lambat'),('Hot','Panas'),('Cold','Dingin'),('Warm','Hangat'),('Cool','Sejuk'),
    ('Old','Tua'),('New','Baru'),('Good','Baik'),('Bad','Buruk'),('Happy','Senang'),('Sad','Sedih'),
    ('Angry','Marah'),('Afraid','Takut'),('Tired','Lelah'),('Hungry','Lapar'),('Thirsty','Haus'),
    ('Smart','Pintar'),('Beautiful','Cantik'),('Ugly','Jelek'),('Clean','Bersih'),('Dirty','Kotor'),
    ('Love','Cinta'),('Hate','Benci'),('Like','Suka'),('Want','Ingin'),('Need','Butuh'),('Have','Punya'),
    ('Give','Memberi'),('Take','Ambil'),('Make','Membuat'),('Do','Melakukan'),('Go','Pergi'),('Come','Datang'),
    ('See','Melihat'),('Hear','Mendengar'),('Speak','Bicara'),('Read','Membaca'),('Write','Menulis'),
    ('Think','Berpikir'),('Understand','Mengerti'),('Remember','Ingat'),('Forget','Lupa'),
    ('Learn','Belajar'),('Teach','Mengajar'),('Buy','Membeli'),('Sell','Menjual'),('Pay','Membayar'),
    ('Open','Membuka'),('Close','Menutup'),('Push','Mendorong'),('Pull','Menarik'),
    ('Run','Berlari'),('Walk','Berjalan'),('Jump','Melompat'),('Sit','Duduk'),('Stand','Berdiri'),
    ('Sleep','Tidur'),('Eat','Makan'),('Drink','Minum'),('Cook','Memasak'),('Wash','Mencuci'),
]
items = []
for eng, ind in VOCAB:
    items.append({'bidang':'bahasa','judul':'Kosa Kata','subjudul':'Eng->Ind',
        'prompt':f'{eng} in Indonesian?', 'response':f'{eng} in Indonesian is {ind}.'})
    items.append({'bidang':'bahasa','judul':'Kosa Kata','subjudul':'Ind->Eng',
        'prompt':f'{ind} in English?', 'response':f'{ind} in English is {eng}.'})
total += w(OUTPUT_DIR/'bahasa'/'kosa_kata.jsonl', items)

# 80 negara: capital + reverse + currency + language + continent
COUNTRIES = [
    ('Indonesia','Jakarta','rupiah','bahasa Indonesia','Asia'),
    ('Jepang','Tokyo','yen','bahasa Jepang','Asia'),
    ('Prancis','Paris','euro','bahasa Prancis','Eropa'),
    ('Inggris','London','pound sterling','bahasa Inggris','Eropa'),
    ('Spanyol','Madrid','euro','bahasa Spanyol','Eropa'),
    ('Jerman','Berlin','euro','bahasa Jerman','Eropa'),
    ('Italia','Roma','euro','bahasa Italia','Eropa'),
    ('Rusia','Moskow','ruble','bahasa Rusia','Eropa'),
    ('Amerika Serikat','Washington DC','dolar','bahasa Inggris','Amerika Utara'),
    ('Tiongkok','Beijing','yuan','bahasa Mandarin','Asia'),
    ('Korea Selatan','Seoul','won','bahasa Korea','Asia'),
    ('India','New Delhi','rupee','bahasa Hindi','Asia'),
    ('Brazil','Brasilia','real','bahasa Portugal','Amerika Selatan'),
    ('Australia','Canberra','dolar Australia','bahasa Inggris','Oseania'),
    ('Kanada','Ottawa','dolar Kanada','bahasa Inggris','Amerika Utara'),
    ('Mesir','Kairo','pound Mesir','bahasa Arab','Afrika'),
    ('Thailand','Bangkok','baht','bahasa Thai','Asia'),
    ('Vietnam','Hanoi','dong','bahasa Vietnam','Asia'),
    ('Filipina','Manila','peso','bahasa Filipino','Asia'),
    ('Malaysia','Kuala Lumpur','ringgit','bahasa Melayu','Asia'),
    ('Singapura','Singapura','dolar Singapura','bahasa Inggris','Asia'),
    ('Brunei','Bandar Seri Begawan','dolar Brunei','bahasa Melayu','Asia'),
    ('Argentina','Buenos Aires','peso','bahasa Spanyol','Amerika Selatan'),
    ('Meksiko','Mexico City','peso','bahasa Spanyol','Amerika Utara'),
    ('Turki','Ankara','lira','bahasa Turki','Eropa'),
    ('Yunani','Athena','euro','bahasa Yunani','Eropa'),
    ('Belanda','Amsterdam','euro','bahasa Belanda','Eropa'),
    ('Belgia','Brussel','euro','bahasa Belanda','Eropa'),
    ('Swiss','Bern','frank Swiss','bahasa Jerman','Eropa'),
    ('Austria','Wina','euro','bahasa Jerman','Eropa'),
    ('Swedia','Stockholm','krona','bahasa Swedia','Eropa'),
    ('Norwegia','Oslo','krone','bahasa Norwegia','Eropa'),
    ('Finlandia','Helsinki','euro','bahasa Finlandia','Eropa'),
    ('Denmark','Kopenhagen','krone','bahasa Denmark','Eropa'),
    ('Portugal','Lisbon','euro','bahasa Portugal','Eropa'),
    ('Polandia','Warsawa','zloty','bahasa Polandia','Eropa'),
    ('Ceko','Praha','koruna','bahasa Ceko','Eropa'),
    ('Ukraina','Kyiv','hryvnia','bahasa Ukraina','Eropa'),
    ('Arab Saudi','Riyadh','riyal','bahasa Arab','Asia'),
    ('Israel','Yerusalem','shekel','bahasa Ibrani','Asia'),
    ('Iran','Teheran','rial','bahasa Persia','Asia'),
    ('Pakistan','Islamabad','rupee','bahasa Urdu','Asia'),
    ('Bangladesh','Dhaka','taka','bahasa Bengali','Asia'),
    ('Myanmar','Naypyidaw','kyat','bahasa Burma','Asia'),
    ('Kamboja','Phnom Penh','riel','bahasa Khmer','Asia'),
    ('Laos','Vientiane','kip','bahasa Lao','Asia'),
    ('Maroko','Rabat','dirham','bahasa Arab','Afrika'),
    ('Afrika Selatan','Pretoria','rand','bahasa Inggris','Afrika'),
    ('Nigeria','Abuja','naira','bahasa Inggris','Afrika'),
    ('Kenya','Nairobi','shilling','bahasa Swahili','Afrika'),
    ('Kuba','Havana','peso','bahasa Spanyol','Amerika Utara'),
    ('Venezuela','Caracas','bolivar','bahasa Spanyol','Amerika Selatan'),
    ('Ekuador','Quito','dolar','bahasa Spanyol','Amerika Selatan'),
    ('Peru','Lima','sol','bahasa Spanyol','Amerika Selatan'),
    ('Chili','Santiago','peso','bahasa Spanyol','Amerika Selatan'),
    ('Kolombia','Bogota','peso','bahasa Spanyol','Amerika Selatan'),
    ('Selandia Baru','Wellington','dolar Selandia Baru','bahasa Inggris','Oseania'),
    ('Irlandia','Dublin','euro','bahasa Inggris','Eropa'),
    ('Hongaria','Budapest','forint','bahasa Hongaria','Eropa'),
    ('Rumania','Bukarest','leu','bahasa Rumania','Eropa'),
    ('Kroasia','Zagreb','euro','bahasa Kroasia','Eropa'),
    ('Serbia','Belgrade','dinar','bahasa Serbia','Eropa'),
    ('Bulgaria','Sofia','lev','bahasa Bulgaria','Eropa'),
    ('Uni Emirat Arab','Abu Dhabi','dirham','bahasa Arab','Asia'),
    ('Irak','Baghdad','dinar','bahasa Arab','Asia'),
    ('Sudan','Khartoum','pound','bahasa Arab','Afrika'),
    ('Ethiopia','Addis Ababa','birr','bahasa Amharic','Afrika'),
    ('Tanzania','Dodoma','shilling','bahasa Swahili','Afrika'),
    ('Ghana','Accra','cedi','bahasa Inggris','Afrika'),
    ('Uganda','Kampala','shilling','bahasa Inggris','Afrika'),
    ('Aljazair','Aljir','dinar','bahasa Arab','Afrika'),
    ('Tunisia','Tunis','dinar','bahasa Arab','Afrika'),
    ('Jamaika','Kingston','dolar','bahasa Inggris','Amerika Utara'),
    ('Bolivia','Sucre','boliviano','bahasa Spanyol','Amerika Selatan'),
    ('Paraguay','Asuncion','guarani','bahasa Spanyol','Amerika Selatan'),
    ('Uruguay','Montevideo','peso','bahasa Spanyol','Amerika Selatan'),
    ('Papua Nugini','Port Moresby','kina','bahasa Inggris','Oseania'),
    ('Sri Lanka','Sri Jayawardenepura Kotte','rupee','bahasa Sinhala','Asia'),
    ('Timor Leste','Dili','dolar AS','bahasa Portugis','Asia'),
    ('Finlandia','Helsinki','euro','bahasa Finlandia','Eropa'),
    ('Islandia','Reykjavik','krona','bahasa Islandia','Eropa'),
    ('Portugal','Lisbon','euro','bahasa Portugal','Eropa'),
    ('Luksemburg','Luksemburg','euro','bahasa Luksemburg','Eropa'),
    ('Monako','Monako','euro','bahasa Prancis','Eropa'),
    ('Malta','Valletta','euro','bahasa Malta','Eropa'),
    ('Siprus','Nicosia','euro','bahasa Yunani','Eropa'),
    ('Montenegro','Podgorica','euro','bahasa Montenegro','Eropa'),
    ('Moldova','Chisinau','leu','bahasa Rumania','Eropa'),
    ('Georgia','Tbilisi','lari','bahasa Georgia','Eropa'),
    ('Armenia','Yerevan','dram','bahasa Armenia','Eropa'),
    ('Azerbaijan','Baku','manat','bahasa Azerbaijan','Asia'),
    ('Kazakhstan','Astana','tenge','bahasa Kazakh','Asia'),
    ('Uzbekistan','Taskent','som','bahasa Uzbek','Asia'),
    ('Turkmenistan','Asykhabad','manat','bahasa Turkmen','Asia'),
]
items = []
for country, capital, curr, lang, continent in COUNTRIES:
    items.append({'bidang':'geografi','judul':'Ibu Kota','subjudul':continent,
        'prompt':f'Apa ibu kota {country}?', 'response':f'Ibu kota {country} adalah {capital}.'})
    items.append({'bidang':'geografi','judul':'Ibu Kota','subjudul':'Reverse',
        'prompt':f'{capital} itu apa?', 'response':f'{capital} adalah ibu kota {country}.'})
    items.append({'bidang':'uang_ekonomi','judul':'Mata Uang','subjudul':continent,
        'prompt':f'Apa mata uang {country}?', 'response':f'Mata uang {country} adalah {curr}.'})
    items.append({'bidang':'bahasa','judul':'Bahasa Resmi','subjudul':continent,
        'prompt':f'Apa bahasa resmi {country}?', 'response':f'Bahasa resmi {country} adalah {lang}.'})
    items.append({'bidang':'geografi','judul':'Benua','subjudul':'Lokasi',
        'prompt':f'{country} di benua apa?', 'response':f'{country} berada di benua {continent}.'})
total += w(OUTPUT_DIR/'geografi'/'negara_lengkap.jsonl', items)

# Konversasi (filtered — no more "X adalah buah" simple patterns)
items = [
    {'bidang':'konversasi','judul':'Sapaan','subjudul':'Dasar','prompt':'Halo','response':'Halo juga! Senang bertemu denganmu.'},
    {'bidang':'konversasi','judul':'Sapaan','subjudul':'Dasar','prompt':'halo','response':'Halo juga! Senang bertemu denganmu.'},
    {'bidang':'konversasi','judul':'Sapaan','subjudul':'Dasar','prompt':'Hai','response':'Hai! Apa kabar?'},
    {'bidang':'konversasi','judul':'Sapaan','subjudul':'Dasar','prompt':'Halo apa kabar?','response':'Halo! Aku baik-baik saja, terima kasih.'},
    {'bidang':'konversasi','judul':'Sapaan','subjudul':'Dasar','prompt':'Apa kabar?','response':'Aku baik-baik saja, terima kasih sudah bertanya!'},
    {'bidang':'konversasi','judul':'Sapaan','subjudul':'Dasar','prompt':'apa kabar','response':'Aku baik-baik saja, terima kasih sudah bertanya!'},
    {'bidang':'konversasi','judul':'Sapaan','subjudul':'Dasar','prompt':'Selamat pagi','response':'Selamat pagi juga! Semoga harimu menyenangkan.'},
    {'bidang':'konversasi','judul':'Sapaan','subjudul':'Dasar','prompt':'selamat pagi','response':'Selamat pagi juga! Semoga harimu menyenangkan.'},
    {'bidang':'konversasi','judul':'Sapaan','subjudul':'Dasar','prompt':'Selamat malam','response':'Selamat malam! Selamat istirahat.'},
    {'bidang':'konversasi','judul':'Identitas','subjudul':'Dasar','prompt':'Siapa kamu?','response':'Aku D\'QUILANE, LLM buatan Indonesia dari nol dengan arsitektur Blueprint v2.1.'},
    {'bidang':'konversasi','judul':'Identitas','subjudul':'Dasar','prompt':'siapa kamu','response':'Aku D\'QUILANE, LLM buatan Indonesia dari nol dengan arsitektur Blueprint v2.1.'},
    {'bidang':'konversasi','judul':'Identitas','subjudul':'Dasar','prompt':'Kamu siapa?','response':'Aku D\'QUILANE, LLM buatan Indonesia dari nol.'},
    {'bidang':'konversasi','judul':'Identitas','subjudul':'Dasar','prompt':'kamu siapa','response':'Aku D\'QUILANE, LLM buatan Indonesia dari nol.'},
    {'bidang':'konversasi','judul':'Identitas','subjudul':'Dasar','prompt':'Nama kamu siapa?','response':'Namaku D\'QUILANE. Aku LLM multibahasa.'},
    {'bidang':'konversasi','judul':'Identitas','subjudul':'Dasar','prompt':'Kamu dari mana?','response':'Aku dari Indonesia.'},
    {'bidang':'konversasi','judul':'Identitas','subjudul':'Dasar','prompt':'Kamu bisa apa?','response':'Aku bisa matematika SD1-SD6, sains, sejarah, geografi 80 negara, terjemahan 24 bahasa, dan banyak lagi.'},
    {'bidang':'konversasi','judul':'Identitas','subjudul':'Dasar','prompt':'kamu bisa apa','response':'Aku bisa matematika SD1-SD6, sains, sejarah, geografi, terjemahan, dan jawab pertanyaan.'},
    {'bidang':'konversasi','judul':'Penutup','subjudul':'Dasar','prompt':'Terima kasih','response':'Sama-sama! Ada yang bisa aku bantu lagi?'},
    {'bidang':'konversasi','judul':'Penutup','subjudul':'Dasar','prompt':'terima kasih','response':'Sama-sama! Ada yang bisa aku bantu lagi?'},
    {'bidang':'konversasi','judul':'Penutup','subjudul':'Dasar','prompt':'Makasih','response':'Sama-sama!'},
    {'bidang':'konversasi','judul':'Penutup','subjudul':'Dasar','prompt':'Goodbye','response':'Sampai jumpa! Semoga harimu menyenangkan.'},
    {'bidang':'konversasi','judul':'Penutup','subjudul':'Dasar','prompt':'Dadah','response':'Dadah! Sampai bertemu lagi.'},
    {'bidang':'konversasi','judul':'Fallback','subjudul':'Honest','prompt':'Siapa presiden Amerika?','response':'Presiden Amerika dipilih melalui pemilu demokratis.'},
    {'bidang':'konversasi','judul':'Fallback','subjudul':'Honest','prompt':'Apa berita hari ini?','response':'Aku tidak tahu berita terbaru, cek sumber berita terpercaya.'},
    {'bidang':'konversasi','judul':'Fallback','subjudul':'Honest','prompt':'Bagaimana cuaca hari ini?','response':'Aku tidak tahu cuaca hari ini, cek aplikasi cuaca.'},
    {'bidang':'konversasi','judul':'Fallback','subjudul':'Honest','prompt':'Jam berapa sekarang','response':'Aku tidak tahu jam berapa sekarang, cek jam.'},
    {'bidang':'konversasi','judul':'Fallback','subjudul':'Honest','prompt':'Ceritakan cerita','response':'Aku bisa jawab pertanyaan tentang banyak topik.'},
    {'bidang':'konversasi','judul':'Fallback','subjudul':'Honest','prompt':'test','response':'Halo! Aku D\'QUILANE. Ada yang bisa aku bantu?'},
    {'bidang':'konversasi','judul':'Fallback','subjudul':'Honest','prompt':'lagi apa','response':'Aku siap membantu. Ada yang bisa aku bantu?'},
    {'bidang':'konversasi','judul':'Fallback','subjudul':'Honest','prompt':'coba ngomong','response':'Halo! Aku D\'QUILANE. Ada yang bisa aku bantu?'},
    {'bidang':'konversasi','judul':'Fallback','subjudul':'Honest','prompt':'halo dunia','response':'Halo! Aku D\'QUILANE, LLM buatan Indonesia.'},
    {'bidang':'konversasi','judul':'Fallback','subjudul':'Honest','prompt':'jepang di mana','response':'Jepang adalah negara di Asia Timur.'},
    {'bidang':'konversasi','judul':'Fallback','subjudul':'Honest','prompt':'kamu pintar?','response':'Aku belajar dari data. Coba tanya aku!'},
    {'bidang':'konversasi','judul':'Fallback','subjudul':'Honest','prompt':'apa itu cinta','response':'Cinta adalah perasaan sayang yang kuat.'},
    {'bidang':'konversasi','judul':'Fallback','subjudul':'Honest','prompt':'apa arti hidup','response':'Hidup adalah kesempatan untuk belajar dan membantu orang lain.'},
]
total += w(OUTPUT_DIR/'konversasi'/'konversasi.jsonl', items)

# ============================================================
print(f"\n{'='*60}")
print(f"  DATASET v3 COMPLETE")
print(f"{'='*60}")
print(f"Total Q&A pairs: {total:,}")
print(f"Categories: SD1-SD6 + multibahasa + bahasa + geografi + konversasi")
