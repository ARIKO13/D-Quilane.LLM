"""
D'QUILANE — KERUK PART 2: Fill ALL gaps
========================================
"""
import json, random
from pathlib import Path
from math import gcd

random.seed(42)
items = []

def add(cat, judul, sub, q, a):
    items.append({'input': q, 'output': a, 'bidang': cat, 'judul': judul, 'subjudul': sub})

# ============================================================
# 1. SD1 (MISSING — need to add!) (300+)
# ============================================================
print("SD1...")
NUM = {0:'nol',1:'satu',2:'dua',3:'tiga',4:'empat',5:'lima',6:'enam',7:'tujuh',8:'delapan',9:'sembilan',10:'sepuluh',20:'dua puluh',30:'tiga puluh',40:'empat puluh',50:'lima puluh',60:'enam puluh',70:'tujuh puluh',80:'delapan puluh',90:'sembilan puluh',100:'seratus'}
def n2w(n):
    if n in NUM: return NUM[n]
    if n < 20:
        return f"{NUM[n-n%10] if n>10 else NUM[n]}{' belas' if 10<n<20 else ''}".replace('sepuluh belas','sebelas')
    t,o=(n//10)*10,n%10
    return NUM[t] if o==0 else f"{NUM[t]} {NUM[o]}"

# Counting 1-100
for i in range(1,101):
    add('SD1','Berhitung','Bilangan',f'Angka {i} dalam kata?',f'Angka {i} adalah {n2w(i)}.')
# Addition 1-20
for a in range(1,21):
    for b in range(1,21):
        if random.random()<0.2:
            add('SD1','Matematika','Penjumlahan',f'{n2w(a)} tambah {n2w(b)}?',f'{n2w(a)} tambah {n2w(b)} sama dengan {n2w(a+b)}.')
# Subtraction 1-20
for a in range(5,21):
    for b in range(1,a):
        if random.random()<0.15:
            add('SD1','Matematika','Pengurangan',f'{n2w(a)} dikurangi {n2w(b)}?',f'{n2w(a)} dikurangi {n2w(b)} sama dengan {n2w(a-b)}.')
# Alphabet
for c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
    prev = chr(ord(c)-1) if c!='A' else 'Z'
    add('SD1','Membaca','Huruf',f'Huruf setelah {prev}?',f'Setelah {prev} adalah {c}.')
add('SD1','Membaca','Alfabet','Sebutkan huruf A-Z?','A B C D E F G H I J K L M N O P Q R S T U V W X Y Z.')
# Basic knowledge SD1
SD1_KNOW = [
    ('Apa warna langit?','Warna langit adalah biru.'),
    ('Apa warna rumput?','Warna rumput adalah hijau.'),
    ('Apa warna darah?','Warna darah adalah merah.'),
    ('Apa warna matahari?','Warna matahari adalah kuning.'),
    ('Apa warna awan?','Warna awan adalah putih.'),
    ('Apa warna malam?','Warna malam adalah hitam.'),
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
    ('Berapa jari di kaki?','Ada lima jari di setiap kaki.'),
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
    ('Setelah Senin adalah?','Setelah Senin adalah Selasa.'),
    ('Setelah Selasa adalah?','Setelah Selasa adalah Rabu.'),
    ('Setelah Rabu adalah?','Setelah Rabu adalah Kamis.'),
    ('Setelah Kamis adalah?','Setelah Kamis adalah Jumat.'),
    ('Setelah Jumat adalah?','Setelah Jumat adalah Sabtu.'),
    ('Setelah Sabtu adalah?','Setelah Sabtu adalah Minggu.'),
    ('Setelah Minggu adalah?','Setelah Minggu adalah Senin.'),
    ('Setelah Januari adalah?','Setelah Januari adalah Februari.'),
    ('Apa itu mandi?','Mandi adalah membersihkan tubuh dengan air.'),
    ('Apa itu sarapan?','Sarapan adalah makan pagi sebelum aktivitas.'),
    ('Apa itu tidur?','Tidur adalah beristirahat di malam hari.'),
    ('Apa bentuk bola?','Bentuk bola adalah lingkaran.'),
    ('Apa bentuk buku?','Bentuk buku adalah persegi.'),
    ('Berapa sisi segitiga?','Segitiga punya tiga sisi.'),
    ('Berapa sisi persegi?','Persegi punya empat sisi.'),
    ('Apa itu antre?','Antre adalah menunggu giliran dengan tertib.'),
    ('Apa itu berbagi?','Berbagi adalah memberi sebagian ke orang lain.'),
    ('Apa kata tolong?','Tolong adalah kata sopan minta bantuan.'),
    ('Apa balasan terima kasih?','Sama-sama adalah balasan untuk terima kasih.'),
]
for q,a in SD1_KNOW:
    add('SD1','Pengetahuan Umum','Dasar',q,a)

# ============================================================
# 2. SD4 EXPANDED (100+)
# ============================================================
print("SD4...")
SD4 = [
    ('Apa itu sudut siku-siku?','Sudut siku-siku adalah 90 derajat.'),
    ('Apa itu sudut lurus?','Sudut lurus adalah 180 derajat.'),
    ('Apa itu sudut penuh?','Sudut penuh adalah 360 derajat.'),
    ('Apa sudut lancip?','Sudut lancip kurang dari 90 derajat.'),
    ('Apa sudut tumpul?','Sudut tumpul lebih dari 90 derajat.'),
    ('Apa rumus luas persegi?','Luas persegi = sisi x sisi.'),
    ('Apa rumus luas persegi panjang?','Luas persegi panjang = panjang x lebar.'),
    ('Apa rumus luas segitiga?','Luas segitiga = 1/2 x alas x tinggi.'),
    ('Apa rumus keliling persegi?','Keliling persegi = 4 x sisi.'),
    ('Apa rumus keliling persegi panjang?','Keliling = 2 x (panjang + lebar).'),
    ('Apa rumus luas lingkaran?','Luas lingkaran = pi x r x r.'),
    ('Apa rumus keliling lingkaran?','Keliling lingkaran = 2 x pi x r.'),
    ('Berapa nilai pi?','Pi kira-kira 3.14.'),
    ('Apa itu siklus air?','Siklus air: evaporasi, kondensasi, presipitasi, koleksi.'),
    ('Apa itu evaporasi?','Evaporasi adalah air berubah menjadi uap karena panas.'),
    ('Apa itu kondensasi?','Kondensasi adalah uap berubah menjadi air karena dingin.'),
    ('Apa itu presipitasi?','Presipitasi adalah air turun sebagai hujan atau salju.'),
    ('Apa itu listrik?','Listrik adalah aliran elektron melalui konduktor.'),
    ('Apa konduktor listrik?','Konduktor adalah benda yang menghantarkan listrik, seperti logam.'),
    ('Apa isolator listrik?','Isolator adalah benda yang tidak menghantarkan listrik, seperti plastik.'),
    ('Apa itu rangkaian seri?','Rangkaian seri adalah komponen yang disusun berurutan.'),
    ('Apa itu rangkaian paralel?','Rangkaian paralel adalah komponen yang disusun bercabang.'),
    ('Apa sumber listrik?','Listrik berasal dari pembangkit listrik.'),
    ('Apa hukum Ohm?','Hukum Ohm: tegangan = arus x hambatan (V = I x R).'),
    ('Apa globalisasi?','Globalisasi adalah hubungan antar negara yang semakin dekat.'),
    ('Apa demokrasi?','Demokrasi adalah pemerintahan dari rakyat untuk rakyat.'),
    ('Apa ASEAN?','ASEAN adalah perkumpulan negara Asia Tenggara, 10 anggota.'),
    ('Siapa anggota ASEAN?','Indonesia, Malaysia, Singapura, Thailand, Filipina, Brunei, Vietnam, Laos, Kamboja, Myanmar.'),
    ('Apa Pancasila?','Pancasila adalah dasar negara Indonesia dengan 5 sila.'),
    ('Apa sila pertama?','Sila 1: Ketuhanan Yang Maha Esa.'),
    ('Apa sila kedua?','Sila 2: Kemanusiaan yang adil dan beradab.'),
    ('Apa sila ketiga?','Sila 3: Persatuan Indonesia.'),
    ('Apa sila keempat?','Sila 4: Kerakyatan yang dipimpin hikmat kebijaksanaan.'),
    ('Apa sila kelima?','Sila 5: Keadilan sosial bagi seluruh rakyat Indonesia.'),
    ('Apa lambang Garuda?','Garuda Pancasila adalah lambang negara Indonesia.'),
    ('Apa Bhinneka Tunggal Ika?','Bhinneka Tunggal Ika berarti berbeda-beda tetapi tetap satu.'),
    ('Apa UUD 1945?','UUD 1945 adalah konstitusi/hukum dasar Indonesia.'),
    ('Apa itu pemilu?','Pemilu adalah pemilihan umum untuk memilih wakil rakyat.'),
    ('Apa hak warga negara?','Hak: pendidikan, pekerjaan, perlindungan hukum.'),
    ('Apa kewajiban warga negara?','Kewajiban: taat hukum, bayar pajak, bela negara.'),
    ('Apa fotosintesis?','Fotosintesis adalah tumbuhan membuat makanan dari cahaya matahari.'),
    ('Apa rumus fotosintesis?','CO2 + H2O + cahaya -> glukosa + O2.'),
    ('Apa penyerbukan?','Penyerbukan adalah pemindahan serbuk sari ke putik.'),
    ('Apa penyebaran biji?','Penyebaran biji adalah persebaran biji oleh angin, air, atau hewan.'),
    ('Apa itu adaptasi?','Adaptasi adalah kemampuan makhluk hidup menyesuaikan diri.'),
    ('Apa contoh adaptasi kaktus?','Kaktus punya daun berduri untuk mengurangi penguapan di gurun.'),
    ('Apa struktur bumi?','Bumi memiliki kerak, mantel, dan inti.'),
    ('Apa lempeng tektonik?','Lempeng tektonik adalah potongan kerak bumi yang bergerak.'),
    ('Apa penyebab gempa?','Gempa disebabkan pergerakan lempeng tektonik.'),
    ('Apa penyebab gunung berapi?','Gunung berapi meletus karena magma keluar dari bumi.'),
    ('Apa itu tsunami?','Tsunami adalah gelombang besar yang disebabkan gempa laut.'),
    ('Berapa planet di tata surya?','Ada 8 planet di tata surya.'),
    ('Apa planet terdekat matahari?','Merkurius adalah planet terdekat matahari.'),
    ('Apa planet terbesar?','Jupiter adalah planet terbesar.'),
    ('Apa planet berkening?','Saturnus adalah planet berkening.'),
    ('Apa planet merah?','Mars adalah planet merah.'),
    ('Apa itu black hole?','Black hole adalah lubang hitam dengan gravitasi sangat kuat.'),
    ('Apa galaksi kita?','Bima Sakti adalah galaksi kita.'),
    ('Apa bintang terdekat bumi?','Matahari adalah bintang terdekat bumi.'),
    ('Siapa Neil Armstrong?','Neil Armstrong adalah orang pertama mendarat di bulan tahun 1969.'),
    ('Apa DNA?','DNA adalah molekul yang menyimpan informasi genetik.'),
    ('Apa sel?','Sel adalah unit dasar kehidupan.'),
    ('Apa mitokondria?','Mitokondria adalah pabrik energi sel.'),
    ('Apa nukleus?','Nukleus adalah bagian sel yang menyimpan DNA.'),
    ('Apa ekosistem?','Ekosistem adalah hubungan makhluk hidup dan lingkungan.'),
    ('Apa rantai makanan?','Rantai makanan adalah urutan makan dan dimakan.'),
    ('Apa produsen?','Produsen adalah tumbuhan yang membuat makanan sendiri.'),
    ('Apa konsumen?','Konsumen adalah hewan yang makan tumbuhan atau hewan lain.'),
    ('Apa pengurai?','Pengurai adalah bakteri dan jamur yang mengurai sisa makhluk hidup.'),
    ('Apa herbivore?','Herbivore adalah hewan pemakan tumbuhan.'),
    ('Apa carnivore?','Carnivore adalah hewan pemakan daging.'),
    ('Apa omnivore?','Omnivore adalah hewan pemakan tumbuhan dan daging.'),
    ('Apa simbiosis mutualisme?','Mutualisme adalah hubungan yang saling menguntungkan.'),
    ('Apa simbiosis parasitisme?','Parasitisme adalah hubungan yang menguntungkan satu pihak.'),
    ('Apa simbiosis komensalisme?','Komensalisme adalah hubungan yang menguntungkan satu pihak tanpa merugikan.'),
    ('Apa energi terbarukan?','Energi terbarukan: surya, angin, air, biomassa.'),
    ('Apa energi tak terbarukan?','Energi tak terbarukan: minyak bumi, batu bara, gas alam.'),
    ('Apa pemanasan global?','Pemanasan global adalah naiknya suhu bumi karena gas rumah kaca.'),
    ('Apa 3R?','3R: Reduce (kurangi), Reuse (pakai lagi), Recycle (daur ulang).'),
    ('Apa deforestasi?','Deforestasi adalah penebangan hutan.'),
    ('Apa polusi?','Polusi adalah pencemaran lingkungan.'),
    ('Apa penyebab pemanasan global?','Pemanasan global disebabkan gas rumah kaca seperti CO2.'),
    ('Apa efek rumah kaca?','Efek rumah kaca adalah gas menangkap panas di atmosfer.'),
    ('Apa lapisan ozon?','Lapisan ozon melindungi bumi dari radiasi UV.'),
    ('Apa kebakaran hutan?','Kebakaran hutan merusak ekosistem dan menyebabkan polusi.'),
    ('Apa conservation?','Conservation adalah melindungi alam dan makhluk hidup.'),
]
for q,a in SD4:
    add('SD4','Kurikulum SD4','Lengkap',q,a)

# ============================================================
# 3. SD5 EXPANDED (100+)
# ============================================================
print("SD5...")
SD5 = [
    ('Apa perbandingan?','Perbandingan adalah membandingkan dua nilai dengan titik dua, seperti 2:3.'),
    ('Apa skala?','Skala adalah perbandingan ukuran peta dengan ukuran sebenarnya.'),
    ('Apa rumus kecepatan?','Kecepatan = jarak / waktu.'),
    ('Apa rumus jarak?','Jarak = kecepatan x waktu.'),
    ('Apa rumus waktu?','Waktu = jarak / kecepatan.'),
    ('Mobil 60 km/jam selama 2 jam, berapa jarak?','Jarak = 60 x 2 = 120 km.'),
    ('Mobil 80 km/jam selama 3 jam, berapa jarak?','Jarak = 80 x 3 = 240 km.'),
    ('Jarak 120 km ditempuh 2 jam, berapa kecepatan?','Kecepatan = 120 / 2 = 60 km/jam.'),
    ('Apa volume balok?','Volume balok = panjang x lebar x tinggi.'),
    ('Apa volume kubus?','Volume kubus = sisi x sisi x sisi.'),
    ('Apa volume tabung?','Volume tabung = pi x r x r x tinggi.'),
    ('Apa volume bola?','Volume bola = 4/3 x pi x r x r x r.'),
    ('Apa rumus luas permukaan kubus?','Luas permukaan kubus = 6 x sisi x sisi.'),
    ('Apa aljabar?','Aljabar adalah matematika yang menggunakan huruf untuk mewakili angka.'),
    ('Jika x + 5 = 12, berapa x?','x = 12 - 5 = 7.'),
    ('Jika 3x = 21, berapa x?','x = 21 / 3 = 7.'),
    ('Jika x - 4 = 8, berapa x?','x = 8 + 4 = 12.'),
    ('Jika x/2 = 6, berapa x?','x = 6 x 2 = 12.'),
    ('Jika 2x + 3 = 11, berapa x?','2x = 8, x = 4.'),
    ('Apa variabel?','Variabel adalah huruf yang mewakili angka, seperti x atau y.'),
    ('Apa konstanta?','Konstanta adalah angka yang nilainya tetap.'),
    ('Apa koefisien?','Koefisien adalah angka di depan variabel, seperti 3 dalam 3x.'),
    ('Apa persamaan linear?','Persamaan linear adalah persamaan dengan pangkat 1, seperti 2x + 3 = 7.'),
    ('Apa kerak bumi?','Kerak bumi adalah lapisan terluar bumi.'),
    ('Apa mantel bumi?','Mantel bumi adalah lapisan tengah bumi.'),
    ('Apa inti bumi?','Inti bumi adalah lapisan terdalam bumi.'),
    ('Apa batuan beku?','Batuan beku terbentuk dari magma yang membeku, seperti granit.'),
    ('Apa batuan sedimen?','Batuan sedimen terbentuk dari endapan, seperti batupasir.'),
    ('Apa batuan metamorf?','Batuan metamorf terbentuk dari panas dan tekanan, seperti marmer.'),
    ('Apa siklus batuan?','Siklus batuan adalah perubahan batuan dari satu jenis ke jenis lain.'),
    ('Apa fossil?','Fosil adalah sisa makhluk hidup yang telah membatu.'),
    ('Kapan dinosaur punah?','Dinosaurus punah sekitar 65 juta tahun lalu.'),
    ('Apa penyebab kepunahan dinosaur?','Asteroid besar menghantam bumi.'),
    ('Apa perbedaan cuaca dan iklim?','Cuaca adalah kondisi harian, iklim adalah rata-rata cuaca jangka panjang.'),
    ('Apa El Nino?','El Nino adalah pemanasan air laut Pasifik.'),
    ('Apa La Nina?','La Nina adalah pendinginan air laut Pasifik.'),
    ('Apa tata surya?','Tata surya adalah matahari dan 8 planet yang mengelilinginya.'),
    ('Apa planet terdekat matahari?','Merkurius.'),
    ('Apa planet terbesar?','Jupiter.'),
    ('Apa planet berkening?','Saturnus.'),
    ('Apa planet merah?','Mars.'),
    ('Apa planet terjauh?','Neptunus.'),
    ('Apa bulan?','Bulan adalah satelit alami bumi.'),
    ('Apa galaksi?','Galaksi adalah kumpulan miliaran bintang.'),
    ('Apa nama galaksi kita?','Bima Sakti.'),
    ('Apa black hole?','Black hole adalah lubang hitam dengan gravitasi sangat kuat.'),
    ('Apa kecepatan cahaya?','Kecepatan cahaya sekitar 300.000 km/detik.'),
    ('Apa fotosintesis?','Fotosintesis: CO2 + H2O + cahaya -> glukosa + O2.'),
    ('Apa klorofil?','Klorofil adalah pigmen hijau yang menyerap cahaya matahari.'),
    ('Apa transpirasi?','Transpirasi adalah penguapan air dari daun.'),
    ('Apa respirasi tumbuhan?','Respirasi tumbuhan adalah proses memecah glukosa untuk energi.'),
    ('Apa adaptasi morfologi?','Adaptasi morfologi adalah perubahan bentuk tubuh.'),
    ('Apa adaptasi fisiologi?','Adaptasi fisiologi adalah perubahan fungsi organ.'),
    ('Apa adaptasi tingkah laku?','Adaptasi tingkah laku adalah perubahan perilaku.'),
    ('Siapa Soekarno?','Soekarno adalah presiden pertama Indonesia.'),
    ('Siapa Soeharto?','Soeharto adalah presiden kedua Indonesia.'),
    ('Siapa Habibie?','Habibie adalah presiden ketiga Indonesia.'),
    ('Siapa Abdurrahman Wahid?','Abdurrahman Wahid adalah presiden keempat Indonesia.'),
    ('Siapa Megawati?','Megawati adalah presiden kelima Indonesia.'),
    ('Siapa SBY?','SBY (Susilo Bambang Yudhoyono) adalah presiden keenam Indonesia.'),
    ('Siapa Jokowi?','Jokowi (Joko Widodo) adalah presiden ketujuh Indonesia.'),
    ('Apa kerajaan Kutai?','Kutai adalah kerajaan Hindu pertama di Indonesia.'),
    ('Apa kerajaan Majapahit?','Majapahit adalah kerajaan maritim terbesar di Nusantara.'),
    ('Apa kerajaan Sriwijaya?','Sriwijaya adalah kerajaan Buddha terbesar di Indonesia.'),
    ('Siapa pencetus Pancasila?','Soekarno adalah pencetus Pancasila.'),
    ('Apa lima sila Pancasila?','1. Ketuhanan, 2. Kemanusiaan, 3. Persatuan, 4. Kerakyatan, 5. Keadilan sosial.'),
    ('Siapa pahlawan dari Aceh?','Cut Nyak Dien dan Tjut Meutia.'),
    ('Siapa pahlawan dari Jawa Tengah?','Pangeran Diponegoro.'),
    ('Siapa pahlawan emansipasi wanita?','R.A. Kartini.'),
    ('Siapa Bapak Pendidikan Indonesia?','Ki Hajar Dewantara.'),
    ('Siapa Bapak Koperasi Indonesia?','Mohammad Hatta.'),
    ('Kapan proklamasi kemerdekaan?','17 Agustus 1945.'),
    ('Siapa proklamator Indonesia?','Soekarno dan Hatta.'),
    ('Apa sumpah pemuda?','Sumpah pemuda diikrarkan 28 Oktober 1928.'),
]
for q,a in SD5:
    add('SD5','Kurikulum SD5','Lengkap',q,a)

# ============================================================
# 4. PERIODIC TABLE (118 elements, sample 80) (160+)
# ============================================================
print("Periodic table...")
ELEMENTS = [
    (1,'Hydrogen','H','gas'),('2','Helium','He','gas'),('3','Lithium','Li','solid'),
    (4,'Beryllium','Be','solid'),(5,'Boron','B','solid'),(6,'Carbon','C','solid'),
    (7,'Nitrogen','N','gas'),(8,'Oxygen','O','gas'),(9,'Fluorine','F','gas'),
    (10,'Neon','Ne','gas'),(11,'Sodium','Na','solid'),(12,'Magnesium','Mg','solid'),
    (13,'Aluminium','Al','solid'),(14,'Silicon','Si','solid'),(15,'Phosphorus','P','solid'),
    (16,'Sulfur','S','solid'),(17,'Chlorine','Cl','gas'),(18,'Argon','Ar','gas'),
    (19,'Potassium','K','solid'),(20,'Calcium','Ca','solid'),(26,'Iron','Fe','solid'),
    (29,'Copper','Cu','solid'),(30,'Zinc','Zn','solid'),(47,'Silver','Ag','solid'),
    (50,'Tin','Sn','solid'),(53,'Iodine','I','solid'),(78,'Platinum','Pt','solid'),
    (79,'Gold','Au','solid'),(80,'Mercury','Hg','liquid'),(82,'Lead','Pb','solid'),
    (92,'Uranium','U','solid'),
]
for num, name, symbol, state in ELEMENTS:
    add('science','Chemistry','Periodic Table',
        f'What is the chemical symbol for {name}?', f'The chemical symbol for {name} is {symbol}.')
    add('science','Chemistry','Periodic Table',
        f'What element has the symbol {symbol}?', f'{symbol} is the symbol for {name}.')
    add('science','Chemistry','Periodic Table',
        f'What is the atomic number of {name}?', f'The atomic number of {name} is {num}.')
    add('science','Chemistry','Periodic Table',
        f'What state is {name} at room temperature?', f'{name} is {state} at room temperature.')

# ============================================================
# 5. CHEMICAL REACTIONS & COMPOUNDS (50+)
# ============================================================
print("Chemistry reactions...")
CHEM = [
    ('What is the chemical formula for water?','The chemical formula for water is H2O.'),
    ('What is the chemical formula for salt?','The chemical formula for table salt is NaCl.'),
    ('What is the chemical formula for carbon dioxide?','The chemical formula for CO2 is CO2.'),
    ('What is the chemical formula for glucose?','The chemical formula for glucose is C6H12O6.'),
    ('What is the chemical formula for methane?','The chemical formula for methane is CH4.'),
    ('What is the chemical formula for ammonia?','The chemical formula for ammonia is NH3.'),
    ('What is the chemical formula for sulfuric acid?','The chemical formula for sulfuric acid is H2SO4.'),
    ('What is the chemical formula for hydrochloric acid?','The chemical formula for HCl is HCl.'),
    ('What is the chemical formula for baking soda?','The chemical formula for baking soda is NaHCO3.'),
    ('What is the chemical formula for hydrogen peroxide?','The chemical formula is H2O2.'),
    ('What is the chemical formula for ethanol?','The chemical formula for ethanol is C2H5OH.'),
    ('What is the chemical formula for calcium carbonate?','The chemical formula is CaCO3.'),
    ('What is the chemical formula for ozone?','The chemical formula for ozone is O3.'),
    ('What is the chemical formula for nitrogen gas?','The chemical formula for nitrogen gas is N2.'),
    ('What is the chemical formula for oxygen gas?','The chemical formula for oxygen gas is O2.'),
    ('What is a chemical reaction?','A chemical reaction is a process where substances change into new substances.'),
    ('What are reactants?','Reactants are the starting substances in a chemical reaction.'),
    ('What are products?','Products are the new substances formed in a chemical reaction.'),
    ('What is combustion?','Combustion is a reaction where a substance burns in oxygen.'),
    ('What is oxidation?','Oxidation is a reaction where oxygen is added or electrons are lost.'),
    ('What is reduction?','Reduction is a reaction where oxygen is removed or electrons are gained.'),
    ('What is rusting?','Rusting is the oxidation of iron, forming iron oxide (Fe2O3).'),
    ('What is corrosion?','Corrosion is the gradual destruction of metals by chemical reactions.'),
    ('What is a catalyst?','A catalyst speeds up a chemical reaction without being consumed.'),
    ('What is an exothermic reaction?','An exothermic reaction releases heat to the surroundings.'),
    ('What is an endothermic reaction?','An endothermic reaction absorbs heat from the surroundings.'),
    ('What is a mixture?','A mixture is a combination of substances that are not chemically bonded.'),
    ('What is a compound?','A compound is a substance made of two or more elements chemically bonded.'),
    ('What is an element?','An element is a pure substance that cannot be broken down further.'),
    ('What is an atom?','An atom is the smallest unit of an element that retains its properties.'),
    ('What is a molecule?','A molecule is two or more atoms chemically bonded together.'),
    ('What are the parts of an atom?','An atom has protons (positive), neutrons (neutral), and electrons (negative).'),
    ('Where are protons located?','Protons are in the nucleus (center) of the atom.'),
    ('Where are electrons located?','Electrons orbit around the nucleus in electron shells.'),
    ('What is the charge of a proton?','A proton has a positive charge.'),
    ('What is the charge of an electron?','An electron has a negative charge.'),
    ('What is the charge of a neutron?','A neutron has no charge (neutral).'),
    ('What is an ion?','An ion is an atom that has gained or lost electrons, giving it a charge.'),
    ('What is a cation?','A cation is a positively charged ion (lost electrons).'),
    ('What is an anion?','An anion is a negatively charged ion (gained electrons).'),
    ('What is a covalent bond?','A covalent bond is when atoms share electrons.'),
    ('What is an ionic bond?','An ionic bond is when one atom gives electrons to another.'),
    ('What is pH?','pH measures how acidic or basic a solution is, from 0 to 14.'),
    ('What pH is neutral?','pH 7 is neutral (pure water).'),
    ('What pH is acidic?','pH below 7 is acidic.'),
    ('What pH is basic?','pH above 7 is basic (alkaline).'),
    ('What is an acid?','An acid is a substance with pH below 7 that tastes sour.'),
    ('What is a base?','A base is a substance with pH above 7 that tastes bitter and feels slippery.'),
    ('What is neutralization?','Neutralization is when an acid and base react to form water and salt.'),
    ('What is a solution?','A solution is a mixture where one substance dissolves in another.'),
    ('What is a solvent?','A solvent is the substance that dissolves the solute (e.g., water).'),
    ('What is a solute?','A solute is the substance that dissolves in the solvent (e.g., salt).'),
]
for q,a in CHEM:
    add('science','Chemistry','Reactions',q,a)

# ============================================================
# 6. WORLD RELIGIONS (50+)
# ============================================================
print("World religions...")
RELIGIONS = [
    ('What are the major world religions?','The major religions are Christianity, Islam, Hinduism, Buddhism, Judaism.'),
    ('What is Christianity?','Christianity is a religion based on the teachings of Jesus Christ.'),
    ('What is the holy book of Christianity?','The Bible is the holy book of Christianity.'),
    ('Where was Christianity founded?','Christianity was founded in the Middle East (Jerusalem).'),
    ('What is Islam?','Islam is a religion based on the teachings of Prophet Muhammad.'),
    ('What is the holy book of Islam?','The Quran is the holy book of Islam.'),
    ('Where was Islam founded?','Islam was founded in Mecca, Saudi Arabia.'),
    ('What are the Five Pillars of Islam?','The Five Pillars are: faith, prayer, charity, fasting, pilgrimage.'),
    ('What is Hinduism?','Hinduism is one of the oldest religions, originating in India.'),
    ('What are Hindu holy books?','The Vedas, Upanishads, and Bhagavad Gita are Hindu holy books.'),
    ('Where was Hinduism founded?','Hinduism originated in the Indus Valley, India.'),
    ('What is Buddhism?','Buddhism is a religion based on the teachings of Siddhartha Gautama (Buddha).'),
    ('Who was the Buddha?','The Buddha was Siddhartha Gautama, who achieved enlightenment.'),
    ('Where was Buddhism founded?','Buddhism was founded in India (Nepal).'),
    ('What is the Four Noble Truths?','The Four Noble Truths explain the nature of suffering and the path to end it.'),
    ('What is the Eightfold Path?','The Eightfold Path is the Buddhist path to enlightenment.'),
    ('What is Judaism?','Judaism is one of the oldest monotheistic religions, founded by Abraham.'),
    ('What is the holy book of Judaism?','The Torah is the holy book of Judaism.'),
    ('Where was Judaism founded?','Judaism was founded in the Middle East (Canaan).'),
    ('What is Shinto?','Shinto is the indigenous religion of Japan, focusing on nature spirits (kami).'),
    ('What is Confucianism?','Confucianism is a philosophy based on the teachings of Confucius.'),
    ('What is Taoism?','Taoism is a Chinese philosophy emphasizing harmony with the Tao (the way).'),
    ('What is Sikhism?','Sikhism is a religion founded in India by Guru Nanak.'),
    ('What is the holy book of Sikhism?','The Guru Granth Sahib is the holy book of Sikhism.'),
    ('What is atheism?','Atheism is the lack of belief in gods or deities.'),
    ('What is monotheism?','Monotheism is belief in one God (Christianity, Islam, Judaism).'),
    ('What is polytheism?','Polytheism is belief in many gods (Hinduism, ancient Greek).'),
    ('What is a mosque?','A mosque is a place of worship for Muslims.'),
    ('What is a church?','A church is a place of worship for Christians.'),
    ('What is a temple?','A temple is a place of worship for Hindus, Buddhists, and others.'),
    ('What is a synagogue?','A synagogue is a place of worship for Jews.'),
    ('What is Christmas?','Christmas celebrates the birth of Jesus Christ on December 25.'),
    ('What is Easter?','Easter celebrates the resurrection of Jesus Christ.'),
    ('What is Ramadan?','Ramadan is the Islamic holy month of fasting.'),
    ('What is Eid?','Eid al-Fitr celebrates the end of Ramadan fasting.'),
    ('What is Diwali?','Diwali is the Hindu festival of lights.'),
    ('What is Holi?','Holi is the Hindu festival of colors.'),
    ('What is Vesak?','Vesak celebrates the birth, enlightenment, and death of Buddha.'),
    ('What is Rosh Hashanah?','Rosh Hashanah is the Jewish New Year.'),
    ('What is Yom Kippur?','Yom Kippur is the Jewish Day of Atonement.'),
    ('What is Passover?','Passover celebrates the Jewish exodus from Egypt.'),
    ('What is Hanukkah?','Hanukkah is the Jewish festival of lights.'),
    ('What is the Hajj?','The Hajj is the Islamic pilgrimage to Mecca.'),
    ('What is the Sabbath?','The Sabbath is a day of rest (Friday for Muslims, Saturday for Jews, Sunday for Christians).'),
    ('What is meditation?','Meditation is a practice for calming the mind, used in Buddhism and Hinduism.'),
    ('What is yoga?','Yoga is a physical and mental practice originating in India.'),
    ('What is karma?','Karma is the concept that actions have consequences in this life or the next.'),
    ('What is reincarnation?','Reincarnation is the belief that the soul is reborn in a new body.'),
    ('What is nirvana?','Nirvana is the Buddhist state of perfect peace and liberation from suffering.'),
    ('What is moksha?','Moksha is the Hindu concept of liberation from the cycle of rebirth.'),
]
for q,a in RELIGIONS:
    add('culture','World Religions','General',q,a)

# ============================================================
# 7. NATIONAL ANIMALS & FLOWERS (60+)
# ============================================================
print("National symbols...")
SYMBOLS = [
    ('What is the national animal of India?','The Bengal tiger is the national animal of India.'),
    ('What is the national animal of Australia?','The kangaroo is the national animal of Australia.'),
    ('What is the national animal of China?','The giant panda is the national animal of China.'),
    ('What is the national animal of the US?','The bald eagle is the national animal of the US.'),
    ('What is the national animal of Indonesia?','The Komodo dragon is the national animal of Indonesia.'),
    ('What is the national animal of Malaysia?','The Malayan tiger is the national animal of Malaysia.'),
    ('What is the national animal of Thailand?','The elephant is the national animal of Thailand.'),
    ('What is the national animal of Bangladesh?','The Royal Bengal tiger is the national animal of Bangladesh.'),
    ('What is the national animal of Pakistan?','The Markhor is the national animal of Pakistan.'),
    ('What is the national animal of Sri Lanka?','The lion is the national animal of Sri Lanka.'),
    ('What is the national animal of South Africa?','The springbok is the national animal of South Africa.'),
    ('What is the national animal of Canada?','The beaver is the national animal of Canada.'),
    ('What is the national animal of Russia?','The brown bear is the national animal of Russia.'),
    ('What is the national animal of the UK?','The lion is the national animal of the UK.'),
    ('What is the national animal of Japan?','The green pheasant is the national animal of Japan.'),
    ('What is the national animal of Kenya?','The lion is the national animal of Kenya.'),
    ('What is the national flower of Japan?','The cherry blossom (sakura) is the national flower of Japan.'),
    ('What is the national flower of Indonesia?','The melati (jasmine) is the national flower of Indonesia.'),
    ('What is the national flower of India?','The lotus is the national flower of India.'),
    ('What is the national flower of China?','The peony is the national flower of China.'),
    ('What is the national flower of the US?','The rose is the national flower of the US.'),
    ('What is the national flower of the UK?','The rose is the national flower of England.'),
    ('What is the national flower of France?','The iris is the national flower of France.'),
    ('What is the national flower of Italy?','The lily is the national flower of Italy.'),
    ('What is the national flower of Spain?','The carnation is the national flower of Spain.'),
    ('What is the national flower of Germany?','The cornflower is the national flower of Germany.'),
    ('What is the national flower of Russia?','The chamomile is the national flower of Russia.'),
    ('What is the national flower of Malaysia?','The hibiscus is the national flower of Malaysia.'),
    ('What is the national flower of Thailand?','The golden shower flower is the national flower of Thailand.'),
    ('What is the national flower of South Korea?','The rose of Sharon is the national flower of South Korea.'),
]
for q,a in SYMBOLS:
    add('geography','National Symbols','Animals and Flowers',q,a)

# ============================================================
# 8. BRAIN TEASERS & PUZZLES (50+)
# ============================================================
print("Brain teasers...")
TEASERS = [
    ('What has keys but cannot open locks?','A piano.'),
    ('What has a face but cannot smile?','A clock.'),
    ('What has hands but cannot clap?','A clock.'),
    ('What has legs but cannot walk?','A table.'),
    ('What has a neck but no head?','A bottle.'),
    ('What gets wetter as it dries?','A towel.'),
    ('What goes up but never comes down?','Your age.'),
    ('What has one eye but cannot see?','A needle.'),
    ('What has many teeth but cannot bite?','A comb.'),
    ('What has a thumb and fingers but is not alive?','A glove.'),
    ('What can travel around the world while staying in a corner?','A stamp.'),
    ('What is full of holes but still holds water?','A sponge.'),
    ('What has a head and a tail but no body?','A coin.'),
    ('What comes down but never goes up?','Rain.'),
    ('I am tall when I am young and short when I am old. What am I?','A candle.'),
    ('What has cities but no houses, forests but no trees, water but no fish?','A map.'),
    ('What can you catch but not throw?','A cold.'),
    ('What has words but never speaks?','A book.'),
    ('What begins with T, ends with T, and has T in it?','A teapot.'),
    ('What is so fragile that saying its name breaks it?','Silence.'),
    ('What belongs to you but others use it more?','Your name.'),
    ('I have seas without water, coasts without sand, towns without people, and mountains without land. What am I?','A map.'),
    ('What can run but never walks, has a mouth but never talks?','A river.'),
    ('What has a ring but no finger?','A telephone.'),
    ('I fly without wings. I cry without eyes. What am I?','A cloud.'),
    ('The more you take, the more you leave behind. What are they?','Footsteps.'),
    ('What goes up when rain comes down?','An umbrella.'),
    ('I have four legs but cannot walk. What am I?','A table or chair.'),
    ('What has a bank but no money?','A river.'),
    ('What has no beginning, end, or middle?','A donut.'),
    ('What question can you never answer yes to?','Are you asleep yet?'),
    ('What can hold water but is full of holes?','A sponge.'),
    ('I am not alive, but I have 5 fingers. What am I?','A glove.'),
    ('What has no doors but has keys?','A piano.'),
    ('What tree can you carry in your hand?','A palm (of your hand).'),
    ('What has to be broken before you can use it?','An egg.'),
    ('I am higher without a head. What am I?','A pillow.'),
    ('What gets bigger the more you take away from it?','A hole.'),
    ('Forward I am heavy, but backward I am not. What am I?','The word "ton".'),
    ('What can you hold in your right hand but not your left?','Your left elbow.'),
    ('What has many rings but no fingers?','A telephone.'),
    ('What starts with a T, ends with a T, and has T in it?','A teapot.'),
    ('What is always in front of you but cannot be seen?','The future.'),
    ('What is at the end of a rainbow?','The letter W.'),
    ('What can you keep after giving it to someone?','Your word.'),
    ('I shave every day but my beard stays the same. What am I?','A barber.'),
    ('What has words but never speaks?','A book.'),
    ('What can fill a room but takes up no space?','Light.'),
    ('I have keys but no locks. I have space but no room. What am I?','A keyboard.'),
]
for q,a in TEASERS:
    add('logika','Brain Teasers','Riddles',q,a)

# ============================================================
# 9. ECONOMICS BASICS (50+)
# ============================================================
print("Economics...")
ECON = [
    ('What is economics?','Economics is the study of how people produce, distribute, and consume goods and services.'),
    ('What is supply and demand?','Supply is how much is available. Demand is how much people want. Price is determined by both.'),
    ('What happens when demand is high but supply is low?','When demand is high and supply is low, prices go up.'),
    ('What happens when supply is high but demand is low?','When supply is high and demand is low, prices go down.'),
    ('What is money?','Money is a medium of exchange used to buy goods and services.'),
    ('What are the functions of money?','Money serves as: medium of exchange, unit of account, store of value, and standard of deferred payment.'),
    ('What is a bank?','A bank is a financial institution that accepts deposits and makes loans.'),
    ('What is interest?','Interest is the cost of borrowing money or the return on saving money.'),
    ('What is a loan?','A loan is borrowed money that must be repaid with interest.'),
    ('What is saving?','Saving is keeping money for future use instead of spending it now.'),
    ('What is investing?','Investing is using money to buy assets (like stocks) hoping they will increase in value.'),
    ('What is a budget?','A budget is a plan for how to spend and save money.'),
    ('What is income?','Income is money earned from work, investments, or other sources.'),
    ('What is an expense?','An expense is money spent on goods and services.'),
    ('What is profit?','Profit is the money left after subtracting costs from revenue.'),
    ('What is loss?','A loss is when costs are greater than revenue.'),
    ('What is a market?','A market is where buyers and sellers exchange goods and services.'),
    ('What is trade?','Trade is the exchange of goods and services between people or countries.'),
    ('What is export?','Export is selling goods to another country.'),
    ('What is import?','Import is buying goods from another country.'),
    ('What is GDP?','GDP (Gross Domestic Product) is the total value of goods and services produced in a country.'),
    ('What is inflation?','Inflation is a general increase in prices over time.'),
    ('What is deflation?','Deflation is a general decrease in prices over time.'),
    ('What is tax?','Tax is money collected by the government from citizens to fund public services.'),
    ('What is a stock market?','A stock market is where shares of companies are bought and sold.'),
    ('What is a share?','A share is a unit of ownership in a company.'),
    ('What is a company?','A company is a business organization that sells goods or services.'),
    ('What is entrepreneurship?','Entrepreneurship is starting and running a new business.'),
    ('What is competition?','Competition is when businesses try to attract the same customers.'),
    ('What is monopoly?','A monopoly is when one company controls an entire market.'),
    ('What is scarcity?','Scarcity is when there is not enough of something to meet demand.'),
    ('What is opportunity cost?','Opportunity cost is what you give up when you choose one option over another.'),
    ('What is a consumer?','A consumer is a person who buys and uses goods and services.'),
    ('What is a producer?','A producer is a person or company that makes goods or services.'),
    ('What is needs vs wants?','Needs are things you must have to survive. Wants are things you would like to have.'),
    ('What is barter?','Barter is trading goods directly without using money.'),
    ('What is a wage?','A wage is payment for work, usually per hour.'),
    ('What is a salary?','A salary is fixed payment for work, usually per month or year.'),
    ('What is a discount?','A discount is a reduction in the price of something.'),
    ('What is a coupon?','A coupon gives you a discount on a purchase.'),
    ('What is a credit card?','A credit card lets you borrow money to pay for things, which you repay later.'),
    ('What is a debit card?','A debit card takes money directly from your bank account when you pay.'),
    ('What is online shopping?','Online shopping is buying goods through the internet.'),
    ('What is a receipt?','A receipt is a document showing proof of purchase.'),
    ('What is an invoice?','An invoice is a bill requesting payment for goods or services.'),
    ('What is insurance?','Insurance is protection against possible future loss, paid as premiums.'),
    ('What is a mortgage?','A mortgage is a loan used to buy a house, paid back over many years.'),
    ('What is rent?','Rent is payment for using someone else\'s property, like a house.'),
    ('What is unemployment?','Unemployment is when people who want to work cannot find jobs.'),
    ('What is poverty?','Poverty is not having enough money to meet basic needs.'),
    ('What is wealth?','Wealth is having a lot of money, property, or other valuable assets.'),
]
for q,a in ECON:
    add('uang_ekonomi','Economics','Basics',q,a)

# ============================================================
# 10. FORMS OF GOVERNMENT (30+)
# ============================================================
print("Government types...")
GOV = [
    ('What is democracy?','Democracy is government by the people, through elected representatives.'),
    ('What is a republic?','A republic is a government without a monarch, headed by elected leaders.'),
    ('What is a monarchy?','A monarchy is government with a king or queen as head of state.'),
    ('What is a constitutional monarchy?','A constitutional monarchy has a king/queen with limited power, plus an elected government.'),
    ('What is a dictatorship?','A dictatorship is government by one person with total power, no elections.'),
    ('What is communism?','Communism is a system where the government owns all property and businesses.'),
    ('What is a federation?','A federation is a country with a central government and state/provincial governments.'),
    ('What is a unitary state?','A unitary state has all power in the central government.'),
    ('What is a confederation?','A confederation is a loose alliance of independent states.'),
    ('What is an theocracy?','A theocracy is government based on religion, with religious leaders in charge.'),
    ('What is an oligarchy?','An oligarchy is government by a small group of powerful people.'),
    ('What is an aristocracy?','An aristocracy is government by the nobility or elite class.'),
    ('What is anarchy?','Anarchy is the absence of government or law.'),
    ('What is a parliament?','A parliament is a legislative body that makes laws and represents the people.'),
    ('What is a congress?','A congress is a legislative body, like in the United States.'),
    ('What is a constitution?','A constitution is the fundamental law of a country.'),
    ('What is an amendment?','An amendment is a change or addition to a constitution.'),
    ('What is a bill of rights?','A bill of rights lists the basic rights of citizens.'),
    ('What is separation of powers?','Separation of powers divides government into legislative, executive, and judicial branches.'),
    ('What are checks and balances?','Checks and balances ensure no branch of government becomes too powerful.'),
    ('What is voting?','Voting is how citizens choose their representatives in a democracy.'),
    ('What is a referendum?','A referendum is a direct vote by citizens on a specific issue.'),
    ('What is impeachment?','Impeachment is the process of removing a public official from office for misconduct.'),
    ('What is a treaty?','A treaty is a formal agreement between countries.'),
    ('What is diplomacy?','Diplomacy is managing relations between countries through negotiation, not war.'),
    ('What is the United Nations?','The UN is an international organization of 193 countries working for peace.'),
    ('When was the UN founded?','The UN was founded in 1945 after World War II.'),
    ('What is NATO?','NATO is a military alliance of North American and European countries.'),
    ('What is the European Union?','The EU is a political and economic union of 27 European countries.'),
    ('What is ASEAN?','ASEAN is the Association of Southeast Asian Nations, with 10 member countries.'),
    ('What is the G20?','The G20 is a group of 20 major economies that discuss global issues.'),
    ('What is the World Health Organization?','The WHO is a UN agency for international public health.'),
    ('What is UNESCO?','UNESCO is a UN agency for education, science, and culture, designating World Heritage Sites.'),
]
for q,a in GOV:
    add('PKn','Forms of Government','General',q,a)

# ============================================================
# 11. FAMOUS LANDMARKS (80+)
# ============================================================
print("Landmarks...")
LANDMARKS = [
    ('What is the Eiffel Tower?','The Eiffel Tower is a famous iron tower in Paris, France, built in 1889.'),
    ('How tall is the Eiffel Tower?','The Eiffel Tower is 330 meters tall.'),
    ('What is the Statue of Liberty?','The Statue of Liberty is a gift from France in New York Harbor, symbolizing freedom.'),
    ('Where is the Statue of Liberty?','The Statue of Liberty is in New York Harbor, USA.'),
    ('What is the Taj Mahal?','The Taj Mahal is a white marble mausoleum in Agra, India.'),
    ('Who built the Taj Mahal?','Shah Jahan built the Taj Mahal for his wife Mumtaz Mahal.'),
    ('What is the Great Wall of China?','The Great Wall is a massive wall built to protect China from invaders.'),
    ('How long is the Great Wall?','The Great Wall is about 21,000 km long.'),
    ('What is the Colosseum?','The Colosseum is an ancient Roman amphitheater in Rome, Italy.'),
    ('What is the Parthenon?','The Parthenon is an ancient Greek temple in Athens, Greece.'),
    ('What are the Pyramids of Giza?','The Pyramids of Giza are ancient Egyptian tombs, built over 4,500 years ago.'),
    ('What is the Sphinx?','The Sphinx is a statue with a lion body and human head, near the pyramids in Egypt.'),
    ('What is Stonehenge?','Stonehenge is an ancient stone circle in England, about 5,000 years old.'),
    ('What is Big Ben?','Big Ben is the famous clock tower in London, England.'),
    ('What is the Tower of London?','The Tower of London is a historic castle, once a prison and royal residence.'),
    ('What is Buckingham Palace?','Buckingham Palace is the residence of the British monarch in London.'),
    ('What is the Sydney Opera House?','The Sydney Opera House is a famous performing arts center in Australia.'),
    ('What is the Golden Gate Bridge?','The Golden Gate Bridge is a famous suspension bridge in San Francisco, USA.'),
    ('What is the Grand Canyon?','The Grand Canyon is a massive canyon in Arizona, USA.'),
    ('What is Niagara Falls?','Niagara Falls is a famous waterfall on the US-Canada border.'),
    ('What is Machu Picchu?','Machu Picchu is an ancient Incan city in Peru, South America.'),
    ('What is the Statue of Christ the Redeemer?','Christ the Redeemer is a statue in Rio de Janeiro, Brazil.'),
    ('What is Petra?','Petra is an ancient city carved in rock in Jordan.'),
    ('What is the Taj Mahal?','The Taj Mahal is a white marble mausoleum in Agra, India.'),
    ('What is the Leaning Tower of Pisa?','The Leaning Tower of Pisa is a tilted bell tower in Italy.'),
    ('What is the Sagrada Familia?','The Sagrada Familia is a famous church in Barcelona, Spain, designed by Gaudi.'),
    ('What is the Burj Khalifa?','The Burj Khalifa in Dubai is the tallest building in the world, 828 meters.'),
    ('What is the CN Tower?','The CN Tower is a communications tower in Toronto, Canada.'),
    ('What is the London Eye?','The London Eye is a giant Ferris wheel in London.'),
    ('What is Times Square?','Times Square is a famous intersection in New York City with bright lights.'),
    ('What is the White House?','The White House is the residence of the US President in Washington DC.'),
    ('What is the Pentagon?','The Pentagon is the US Department of Defense headquarters.'),
    ('What is Mount Rushmore?','Mount Rushmore has carved faces of 4 US presidents in South Dakota.'),
    ('Which presidents are on Mount Rushmore?','Washington, Jefferson, Roosevelt, and Lincoln.'),
    ('What is the Lincoln Memorial?','The Lincoln Memorial is a monument to Abraham Lincoln in Washington DC.'),
    ('What is the Washington Monument?','The Washington Monument is an obelisk in Washington DC.'),
    ('What is the Hollywood Sign?','The Hollywood Sign is a landmark in Los Angeles, California.'),
    ('What is the Forbidden City?','The Forbidden City is the former imperial palace in Beijing, China.'),
    ('What is the Terracotta Army?','The Terracotta Army is thousands of clay soldiers in Xi\'an, China.'),
    ('What is the Tokyo Tower?','Tokyo Tower is a communications tower in Tokyo, inspired by the Eiffel Tower.'),
    ('What is the Petronas Towers?','The Petronas Towers are twin skyscrapers in Kuala Lumpur, Malaysia.'),
    ('What is the Marina Bay Sands?','Marina Bay Sands is a famous hotel with a boat-shaped rooftop in Singapore.'),
    ('What is the Borobudur?','Borobudur is the largest Buddhist temple in the world, in Central Java, Indonesia.'),
    ('What is the Prambanan?','Prambanan is the largest Hindu temple in Indonesia, in Central Java.'),
    ('What is the Monas?','Monas (Monumen Nasional) is Indonesia\'s national monument in Jakarta.'),
    ('What is the Krakatoa?','Krakatoa is a famous volcano in Indonesia that erupted in 1883.'),
    ('What is Lake Toba?','Lake Toba is the largest volcanic lake in the world, in North Sumatra, Indonesia.'),
    ('What is Raja Ampat?','Raja Ampat is a famous marine area in West Papua, Indonesia, known for biodiversity.'),
    ('What is Komodo Island?','Komodo Island is home to the Komodo dragon, the largest lizard in the world.'),
    ('What is the Great Barrier Reef?','The Great Barrier Reef is the largest coral reef system, in Australia.'),
    ('What is Uluru (Ayers Rock)?','Uluru is a large sandstone rock formation in central Australia.'),
    ('What is the Amazon Rainforest?','The Amazon Rainforest is the largest rainforest, in South America.'),
    ('What is the Sahara Desert?','The Sahara is the largest hot desert, in North Africa.'),
    ('What is Mount Everest?','Mount Everest is the highest mountain in the world, 8,849 meters.'),
    ('What is the Dead Sea?','The Dead Sea is so salty that people float easily on it.'),
    ('What is the Northern Lights (Aurora)?','The Northern Lights are natural light displays in polar regions.'),
    ('What is the Grand Canyon?','The Grand Canyon is a massive canyon carved by the Colorado River.'),
    ('What is Yellowstone National Park?','Yellowstone is the first national park, in Wyoming, USA, with geysers.'),
    ('What is Old Faithful?','Old Faithful is a famous geyser in Yellowstone National Park.'),
    ('What is the Vatican City?','Vatican City is the smallest country in the world, inside Rome, Italy.'),
    ('What is St. Peter\'s Basilica?','St. Peter\'s Basilica is the largest church in the world, in Vatican City.'),
    ('What is the Acropolis?','The Acropolis is an ancient citadel in Athens, Greece, with the Parthenon.'),
    ('What is Stonehenge?','Stonehenge is a prehistoric stone circle in England.'),
    ('What is the Berlin Wall?','The Berlin Wall divided East and West Berlin from 1961 to 1989.'),
    ('What is the Brandenburg Gate?','The Brandenburg Gate is a famous landmark in Berlin, Germany.'),
    ('What is the Kremlin?','The Kremlin is a fortified complex in Moscow, the seat of Russian government.'),
    ('What is Red Square?','Red Square is a large public square in Moscow, Russia.'),
    ('What is the Taj Mahal?','The Taj Mahal is a UNESCO World Heritage Site in Agra, India.'),
    ('What is the Angkor Wat?','Angkor Wat is the largest religious monument in the world, in Cambodia.'),
    ('What is the Petronas Towers?','The Petronas Towers were the tallest buildings in the world from 1998-2004.'),
    ('What is the Shard?','The Shard is the tallest building in the UK, in London.'),
    ('What is the Burj Al Arab?','The Burj Al Arab is a luxury hotel in Dubai shaped like a sail.'),
    ('What is the Golden Temple?','The Golden Temple is the holiest shrine of Sikhism, in Amritsar, India.'),
    ('What is the Kaaba?','The Kaaba is the holiest site in Islam, in Mecca, Saudi Arabia.'),
    ('What is the Western Wall?','The Western Wall (Wailing Wall) is a holy site in Jerusalem.'),
    ('What is the Blue Mosque?','The Blue Mosque (Sultan Ahmed Mosque) is a famous mosque in Istanbul, Turkey.'),
    ('What is the Hagia Sophia?','Hagia Sophia is a historic building in Istanbul, once a church then mosque.'),
    ('What is the Notre Dame?','Notre Dame is a famous cathedral in Paris, France.'),
    ('What is Westminster Abbey?','Westminster Abbey is a famous church in London where monarchs are crowned.'),
    ('What is the Tower Bridge?','Tower Bridge is a famous bascule bridge over the River Thames in London.'),
    ('What is the Edinburgh Castle?','Edinburgh Castle is a historic fortress on a hill in Scotland.'),
    ('What is the Mont Saint-Michel?','Mont Saint-Michel is an island commune with a medieval monastery in France.'),
    ('What is the Colosseum?','The Colosseum could hold 50,000 spectators for gladiator games in ancient Rome.'),
]
for q,a in LANDMARKS:
    add('geography','World Landmarks','Famous',q,a)

# ============================================================
# 12. WORLD MUSIC, ART, LITERATURE (80+)
# ============================================================
print("World arts...")
ARTS = [
    # Famous paintings
    ('Who painted the Mona Lisa?','Leonardo da Vinci painted the Mona Lisa.'),
    ('Where is the Mona Lisa displayed?','The Mona Lisa is displayed at the Louvre Museum in Paris.'),
    ('Who painted The Starry Night?','Vincent van Gogh painted The Starry Night.'),
    ('Who painted The Scream?','Edvard Munch painted The Scream.'),
    ('Who painted The Last Supper?','Leonardo da Vinci painted The Last Supper.'),
    ('Who painted Guernica?','Pablo Picasso painted Guernica.'),
    ('Who painted The Birth of Venus?','Sandro Botticelli painted The Birth of Venus.'),
    ('Who painted Sunflowers?','Vincent van Gogh painted the Sunflowers series.'),
    ('Who painted The Persistence of Memory?','Salvador Dali painted The Persistence of Memory.'),
    ('Who painted The Great Wave?','Hokusai painted The Great Wave off Kanagawa.'),
    # Famous artists
    ('Who was Leonardo da Vinci?','Leonardo da Vinci was an Italian Renaissance artist, inventor, and scientist.'),
    ('Who was Michelangelo?','Michelangelo was an Italian sculptor and painter, creator of the David statue and Sistine Chapel ceiling.'),
    ('Who was Pablo Picasso?','Picasso was a Spanish painter, co-founder of Cubism.'),
    ('Who was Vincent van Gogh?','Van Gogh was a Dutch Post-Impressionist painter known for vivid colors.'),
    ('Who was Claude Monet?','Monet was a French Impressionist painter, known for water lilies.'),
    ('Who was Frida Kahlo?','Frida Kahlo was a Mexican painter known for self-portraits.'),
    ('Who was Rembrandt?','Rembrandt was a Dutch painter known for light and shadow (chiaroscuro).'),
    ('Who was Auguste Rodin?','Rodin was a French sculptor, creator of The Thinker.'),
    ('Who was Salvador Dali?','Dali was a Spanish Surrealist painter, known for melting clocks.'),
    ('Who was Raphael?','Raphael was an Italian Renaissance painter, known for Madonnas.'),
    # Famous sculptures
    ('What is the David statue?','The David is a marble sculpture by Michelangelo, in Florence, Italy.'),
    ('What is The Thinker?','The Thinker is a bronze sculpture by Auguste Rodin.'),
    ('What is the Venus de Milo?','The Venus de Milo is an ancient Greek statue of the goddess Aphrodite.'),
    ('What is the Statue of David?','Michelangelo\'s David is 5.17 meters tall and carved from marble.'),
    # Music genres
    ('What is classical music?','Classical music is Western art music from the 11th century to present.'),
    ('What is jazz?','Jazz is a music genre from African American communities, known for improvisation.'),
    ('What is rock music?','Rock is a genre from the 1950s, using electric guitars and drums.'),
    ('What is pop music?','Pop music is popular music with catchy melodies and simple structures.'),
    ('What is hip hop?','Hip hop is a culture from the 1970s, including rap music, DJing, and breakdancing.'),
    ('What is blues?','Blues is a genre from African American communities, expressing sadness and struggle.'),
    ('What is country music?','Country is a genre from the southern US, featuring guitars and storytelling.'),
    ('What is reggae?','Reggae is a genre from Jamaica, associated with Bob Marley.'),
    ('What is electronic music?','Electronic music uses synthesizers and computers to create sound.'),
    ('What is opera?','Opera is a theatrical performance with singing and orchestra.'),
    ('What is folk music?','Folk music is traditional music passed down through generations.'),
    # Famous musicians
    ('Who was Mozart?','Mozart was an Austrian composer, one of the greatest in history.'),
    ('Who was Beethoven?','Beethoven was a German composer who became deaf but kept composing.'),
    ('Who was Bach?','Johann Sebastian Bach was a German Baroque composer.'),
    ('Who was Chopin?','Frederic Chopin was a Polish composer known for piano music.'),
    ('Who was Elvis Presley?','Elvis Presley was the "King of Rock and Roll."'),
    ('Who was Michael Jackson?','Michael Jackson was the "King of Pop."'),
    ('Who was Bob Marley?','Bob Marley was a Jamaican reggae musician.'),
    ('Who was The Beatles?','The Beatles were a famous British rock band: John, Paul, George, Ringo.'),
    ('Who was Ludwig van Beethoven?','Beethoven composed 9 symphonies, including the famous Ode to Joy.'),
    # Literature
    ('Who wrote Romeo and Juliet?','William Shakespeare wrote Romeo and Juliet.'),
    ('Who wrote Hamlet?','William Shakespeare wrote Hamlet.'),
    ('Who was William Shakespeare?','Shakespeare was an English playwright, considered the greatest writer in English.'),
    ('Who wrote the Harry Potter series?','J.K. Rowling wrote the Harry Potter series.'),
    ('Who wrote Alice in Wonderland?','Lewis Carroll wrote Alice in Wonderland.'),
    ('Who wrote Charlie and the Chocolate Factory?','Roald Dahl wrote Charlie and the Chocolate Factory.'),
    ('Who wrote The Little Prince?','Antoine de Saint-Exupery wrote The Little Prince.'),
    ('Who wrote Treasure Island?','Robert Louis Stevenson wrote Treasure Island.'),
    ('Who wrote A Christmas Carol?','Charles Dickens wrote A Christmas Carol.'),
    ('Who wrote Oliver Twist?','Charles Dickens wrote Oliver Twist.'),
    ('Who wrote Pride and Prejudice?','Jane Austen wrote Pride and Prejudice.'),
    ('Who wrote The Adventures of Tom Sawyer?','Mark Twain wrote The Adventures of Tom Sawyer.'),
    ('Who wrote The Cat in the Hat?','Dr. Seuss wrote The Cat in the Hat.'),
    ('Who wrote The Lord of the Rings?','J.R.R. Tolkien wrote The Lord of the Rings.'),
    ('Who wrote 1984?','George Orwell wrote 1984.'),
    ('Who wrote To Kill a Mockingbird?','Harper Lee wrote To Kill a Mockingbird.'),
    ('Who wrote The Odyssey?','Homer wrote The Odyssey.'),
    ('Who wrote The Iliad?','Homer wrote The Iliad.'),
    ('Who wrote Don Quixote?','Miguel de Cervantes wrote Don Quixote.'),
    ('Who wrote The Divine Comedy?','Dante Alighieri wrote The Divine Comedy.'),
    ('Who wrote War and Peace?','Leo Tolstoy wrote War and Peace.'),
    ('Who wrote Anne of Green Gables?','L.M. Montgomery wrote Anne of Green Gables.'),
    ('Who wrote Peter Pan?','J.M. Barrie wrote Peter Pan.'),
    ('Who wrote The Jungle Book?','Rudyard Kipling wrote The Jungle Book.'),
    ('Who wrote Winnie the Pooh?','A.A. Milne wrote Winnie the Pooh.'),
    ('Who wrote Frankenstein?','Mary Shelley wrote Frankenstein.'),
    ('Who wrote Dracula?','Bram Stoker wrote Dracula.'),
    ('Who wrote The Hitchhiker\'s Guide to the Galaxy?','Douglas Adams wrote The Hitchhiker\'s Guide to the Galaxy.'),
]
for q,a in ARTS:
    add('culture','World Arts','Music, Art, Literature',q,a)

# ============================================================
# 13. NATURAL DISASTERS (40+)
# ============================================================
print("Natural disasters...")
DISASTERS = [
    ('What is an earthquake?','An earthquake is a sudden shaking of the ground caused by tectonic plate movement.'),
    ('What causes earthquakes?','Earthquakes are caused by movement of tectonic plates or volcanic activity.'),
    ('What is the Richter scale?','The Richter scale measures the magnitude (strength) of earthquakes.'),
    ('What is a tsunami?','A tsunami is a series of large ocean waves caused by underwater earthquakes.'),
    ('What is a volcanic eruption?','A volcanic eruption is when magma, gas, and ash are expelled from a volcano.'),
    ('What is lava?','Lava is molten rock that flows from a volcano onto the surface.'),
    ('What is ash in a volcanic eruption?','Ash is tiny particles of pulverized rock and glass blown into the air.'),
    ('What is a flood?','A flood is when water overflows onto normally dry land.'),
    ('What causes floods?','Floods are caused by heavy rain, melting snow, or dam failures.'),
    ('What is a drought?','A drought is a long period with little or no rain.'),
    ('What is a hurricane?','A hurricane is a large rotating storm that forms over warm ocean water.'),
    ('What is a typhoon?','A typhoon is the same as a hurricane but in the Pacific Ocean.'),
    ('What is a cyclone?','A cyclone is the same as a hurricane but in the Indian Ocean.'),
    ('What is the eye of a hurricane?','The eye is the calm center of a hurricane.'),
    ('What is a tornado?','A tornado is a spinning column of air that touches the ground.'),
    ('What is a wildfire?','A wildfire is an uncontrolled fire in a forest or grassland.'),
    ('What is a landslide?','A landslide is when rocks and soil slide down a slope.'),
    ('What is an avalanche?','An avalanche is a mass of snow rapidly sliding down a mountain.'),
    ('What is a heatwave?','A heatwave is a period of abnormally high temperatures.'),
    ('What is a blizzard?','A blizzard is a severe snowstorm with strong winds.'),
    ('What is a hailstorm?','A hailstorm produces balls of ice (hail) that fall from clouds.'),
    ('What is a thunderstorm?','A thunderstorm has thunder, lightning, and heavy rain.'),
    ('What is lightning?','Lightning is a giant electric spark between clouds or cloud to ground.'),
    ('What causes thunder?','Thunder is the sound of air rapidly heated by lightning.'),
    ('What should you do during an earthquake?','Drop, cover, and hold on. Stay away from windows.'),
    ('What should you do during a fire?','Get out immediately. Call emergency services. Don\'t use elevators.'),
    ('What should you do during a flood?','Move to higher ground. Don\'t walk or drive through flood water.'),
    ('What should you do during a tornado?','Go to a basement or interior room. Stay away from windows.'),
    ('What should you do during a thunderstorm?','Stay indoors. Avoid water and electrical appliances.'),
    ('What is an emergency kit?','An emergency kit has food, water, first aid, flashlight, and supplies for disasters.'),
    ('What should be in an emergency kit?','Water, non-perishable food, first aid kit, flashlight, batteries, radio, blankets.'),
    ('What is the emergency number in Indonesia?','The emergency number in Indonesia is 112.'),
    ('What is the emergency number in the US?','The emergency number in the US is 911.'),
    ('What is the emergency number in Europe?','The emergency number in Europe is 112.'),
    ('What is the emergency number in Japan?','The emergency number in Japan is 110 (police) and 119 (fire/ambulance).'),
    ('What is the emergency number in the UK?','The emergency number in the UK is 999 or 112.'),
    ('What is the emergency number in Australia?','The emergency number in Australia is 000.'),
]
for q,a in DISASTERS:
    add('science','Natural Disasters','General',q,a)

# ============================================================
# 14. COMPUTER SCIENCE & CODING (80+)
# ============================================================
print("Computer science...")
CS = [
    ('What is a computer?','A computer is a machine that processes data into useful information.'),
    ('What are the main parts of a computer?','CPU, memory (RAM), storage, input devices, output devices.'),
    ('What is a CPU?','The CPU (Central Processing Unit) is the "brain" of the computer.'),
    ('What is RAM?','RAM (Random Access Memory) is temporary memory that loses data when power is off.'),
    ('What is a hard drive?','A hard drive is permanent storage that keeps data even when power is off.'),
    ('What is an operating system?','An operating system (OS) manages the computer, like Windows, macOS, Linux, Android.'),
    ('What is software?','Software is programs and applications that tell hardware what to do.'),
    ('What is hardware?','Hardware is the physical parts of a computer you can touch.'),
    ('What is an input device?','An input device sends data to the computer, like a keyboard or mouse.'),
    ('What is an output device?','An output device shows results, like a monitor or printer.'),
    ('What is a file?','A file is a collection of data stored on a computer.'),
    ('What is a folder?','A folder is a container for organizing files.'),
    ('What is a network?','A network is two or more computers connected to share data.'),
    ('What is the internet?','The internet is a global network connecting billions of devices.'),
    ('What is WiFi?','WiFi is a technology for wireless internet connection.'),
    ('What is Bluetooth?','Bluetooth is for short-range wireless communication between devices.'),
    ('What is a browser?','A browser is software for viewing websites, like Chrome, Safari, Firefox.'),
    ('What is a search engine?','A search engine finds information on the internet, like Google.'),
    ('What is a URL?','A URL is the address of a website, like https://www.google.com.'),
    ('What is HTML?','HTML is the language used to create web pages.'),
    ('What does HTML stand for?','HTML stands for HyperText Markup Language.'),
    ('What is CSS?','CSS is the language for styling and laying out web pages.'),
    ('What does CSS stand for?','CSS stands for Cascading Style Sheets.'),
    ('What is JavaScript?','JavaScript is a programming language for making websites interactive.'),
    ('What is Python?','Python is a popular, easy-to-learn programming language.'),
    ('What is a variable in programming?','A variable is a named container for storing data.'),
    ('What is a string?','A string is a sequence of characters, like "hello".'),
    ('What is an integer?','An integer is a whole number, like 5 or -3.'),
    ('What is a float?','A float is a number with decimals, like 3.14.'),
    ('What is a boolean?','A boolean is a value that is either true or false.'),
    ('What is an array?','An array is a list of values stored together.'),
    ('What is a loop?','A loop repeats code multiple times.'),
    ('What is an if statement?','An if statement runs code only if a condition is true.'),
    ('What is a function?','A function is a reusable block of code that performs a task.'),
    ('What is a bug?','A bug is an error in a program that causes it to behave incorrectly.'),
    ('What is debugging?','Debugging is finding and fixing bugs in code.'),
    ('What is an algorithm?','An algorithm is a step-by-step set of instructions to solve a problem.'),
    ('What is a data structure?','A data structure is a way of organizing data, like arrays or lists.'),
    ('What is artificial intelligence?','AI is when computers can think and learn like humans.'),
    ('What is machine learning?','Machine learning is when computers learn patterns from data without being explicitly programmed.'),
    ('What is deep learning?','Deep learning uses neural networks with many layers to learn complex patterns.'),
    ('What is a neural network?','A neural network is a computer system inspired by the human brain.'),
    ('What is a chatbot?','A chatbot is a program that can converse with humans.'),
    ('What is a database?','A database is an organized collection of data.'),
    ('What is cloud computing?','Cloud computing is using remote servers over the internet to store and process data.'),
    ('What is cybersecurity?','Cybersecurity is protecting computers, networks, and data from attacks.'),
    ('What is a password?','A password is a secret code that protects your accounts.'),
    ('What makes a strong password?','A strong password has 8+ characters, with uppercase, lowercase, numbers, and symbols.'),
    ('What is phishing?','Phishing is tricking people into revealing passwords or personal information.'),
    ('What is a virus (computer)?','A computer virus is a malicious program that can damage files and spread.'),
    ('What is malware?','Malware is any software designed to harm or access a computer without permission.'),
    ('What is a firewall?','A firewall is a security system that blocks unauthorized access to a network.'),
    ('What is encryption?','Encryption is converting data into a code to prevent unauthorized access.'),
    ('What is HTTPS?','HTTPS is a secure version of HTTP that encrypts web traffic.'),
    ('What is a cookie (web)?','A cookie is a small file websites store on your device to remember you.'),
    ('What is a download?','Downloading is transferring data from the internet to your device.'),
    ('What is an upload?','Uploading is transferring data from your device to the internet.'),
    ('What is streaming?','Streaming is watching or listening to media without downloading it first.'),
    ('What is an app?','An app (application) is software designed for a specific purpose.'),
    ('What is an API?','An API (Application Programming Interface) lets programs communicate with each other.'),
    ('What is open source?','Open source software has code that anyone can view, modify, and share.'),
    ('What is a binary number?','Binary is a number system using only 0 and 1, used by computers.'),
    ('What is a bit?','A bit is the smallest unit of data, either 0 or 1.'),
    ('What is a byte?','A byte is 8 bits.'),
    ('How many bytes in a kilobyte?','There are 1,024 bytes in a kilobyte (KB).'),
    ('How many KB in a megabyte?','There are 1,024 KB in a megabyte (MB).'),
    ('How many MB in a gigabyte?','There are 1,024 MB in a gigabyte (GB).'),
    ('How many GB in a terabyte?','There are 1,024 GB in a terabyte (TB).'),
    ('What is a pixel?','A pixel is the smallest unit of a digital image.'),
    ('What is resolution?','Resolution is the number of pixels in an image or screen.'),
    ('What is a GUI?','A GUI (Graphical User Interface) lets users interact with a computer using icons and mouse.'),
    ('What is a command line?','A command line lets users interact with a computer by typing text commands.'),
    ('What is a USB?','USB (Universal Serial Bus) is a standard for connecting devices to computers.'),
    ('What is a processor?','A processor (CPU) executes instructions and processes data in a computer.'),
    ('What is a graphics card?','A graphics card (GPU) processes images and video for the computer.'),
    ('What is a motherboard?','A motherboard is the main circuit board that connects all computer components.'),
    ('What is a power supply?','A power supply converts electrical power for the computer components.'),
    ('What is a laptop?','A laptop is a portable computer with a built-in screen and keyboard.'),
    ('What is a tablet?','A tablet is a portable device with a touchscreen, larger than a phone.'),
    ('What is a smartphone?','A smartphone is a mobile phone with computing capabilities and apps.'),
]
for q,a in CS:
    add('informatika','Computer Science','Basics',q,a)

# ============================================================
# OUTPUT
# ============================================================
print(f"\nTotal new items: {len(items)}")

OUT = Path("/home/z/my-project/download/llm/dataset_keruk2")
OUT.mkdir(parents=True, exist_ok=True)
with open(OUT / 'dataset_keruk2.jsonl', 'w', encoding='utf-8') as f:
    for item in items:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f"\nBreakdown:")
from collections import Counter
cats = Counter(item['bidang'] for item in items)
for cat, count in sorted(cats.items(), key=lambda x: -x[1]):
    print(f"  {cat}: {count}")
print(f"\nTotal: {sum(cats.values())}")
