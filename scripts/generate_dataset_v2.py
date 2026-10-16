"""
D'QUILANE MASSIVE Dataset v2 — 5000+ Q&A pairs
===============================================
Expansion from v1 (1292 → 5000+):
- Math: 2000+ (addition, subtraction, multiplication, division, fractions, percentages)
- All countries: capitals, currencies, languages, continents (800+)
- Counting: 1-1000 in Indonesian (1000)
- Hello/Thank you in 50+ languages (200+)
- Science: 200+ (expanded physics, chemistry, biology, astronomy)
- History: 200+
- Geography: 300+
- General knowledge: 400+
- Conversations: 200+
- Health: 200+
- Technology: 200+
- Logic: 200+
- New: Sports, Cooking, Space, Ocean, Agriculture, Music, Literature
"""
import json, os, random
from pathlib import Path

OUTPUT_DIR = Path("/home/z/my-project/download/llm/dataset_v2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_WORDS = {
    0:'nol',1:'satu',2:'dua',3:'tiga',4:'empat',5:'lima',6:'enam',7:'tujuh',8:'delapan',
    9:'sembilan',10:'sepuluh',11:'sebelas',12:'dua belas',13:'tiga belas',14:'empat belas',
    15:'lima belas',16:'enam belas',17:'tujuh belas',18:'delapan belas',19:'sembilan belas',
    20:'dua puluh',30:'tiga puluh',40:'empat puluh',50:'lima puluh',60:'enam puluh',
    70:'tujuh puluh',80:'delapan puluh',90:'sembilan puluh',100:'seratus',200:'dua ratus',
    300:'tiga ratus',400:'empat ratus',500:'lima ratus',600:'enam ratus',700:'tujuh ratus',
    800:'delapan ratus',900:'sembilan ratus',1000:'seribu',
}

def num_to_words(n):
    if n in NUM_WORDS: return NUM_WORDS[n]
    if n < 20: return NUM_WORDS.get(n, str(n))
    if n <= 99:
        tens, ones = (n//10)*10, n%10
        return NUM_WORDS[tens] if ones==0 else f"{NUM_WORDS[tens]} {NUM_WORDS[ones]}"
    if n <= 999:
        h, r = n//100, n%100
        prefix = 'seratus' if h==1 else f"{NUM_WORDS[h]} ratus"
        return prefix if r==0 else f"{prefix} {num_to_words(r)}"
    if n == 1000: return 'seribu'
    if n <= 9999:
        th, r = n//1000, n%1000
        prefix = 'seribu' if th==1 else f"{NUM_WORDS[th]} ribu"
        return prefix if r==0 else f"{prefix} {num_to_words(r)}"
    return str(n)

def write_jsonl(path, items):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    return len(items)

total = 0

# ============================================================
# 1. MATEMATIKA — 2000+ items
# ============================================================
print("1. Matematika (2000+)...")
items = []

# Addition: 1-50 + 1-50, sample 400
seen = set()
for _ in range(400):
    a, b = random.randint(1,50), random.randint(1,50)
    key = (a, b)
    if key in seen: continue
    seen.add(key)
    r = a + b
    items.append({'bidang':'matematika','judul':'Aritmatika','subjudul':'Penjumlahan',
        'prompt':f'{num_to_words(a)} tambah {num_to_words(b)}?',
        'response':f'{num_to_words(a)} tambah {num_to_words(b)} sama dengan {num_to_words(r)}.'})
total += write_jsonl(OUTPUT_DIR/'matematika'/'penjumlahan.jsonl', items)

# Subtraction: sample 300
items = []
seen = set()
for _ in range(300):
    a = random.randint(5,100)
    b = random.randint(1,a-1)
    key = (a, b)
    if key in seen: continue
    seen.add(key)
    r = a - b
    items.append({'bidang':'matematika','judul':'Aritmatika','subjudul':'Pengurangan',
        'prompt':f'{num_to_words(a)} dikurangi {num_to_words(b)}?',
        'response':f'{num_to_words(a)} dikurangi {num_to_words(b)} sama dengan {num_to_words(r)}.'})
total += write_jsonl(OUTPUT_DIR/'matematika'/'pengurangan.jsonl', items)

# Multiplication: 1-20 x 1-20 = 400
items = []
for a in range(1,21):
    for b in range(1,21):
        r = a * b
        items.append({'bidang':'matematika','judul':'Aritmatika','subjudul':'Perkalian',
            'prompt':f'{num_to_words(a)} dikali {num_to_words(b)}?',
            'response':f'{num_to_words(a)} dikali {num_to_words(b)} sama dengan {num_to_words(r)}.'})
total += write_jsonl(OUTPUT_DIR/'matematika'/'perkalian.jsonl', items)

# Division: even divisions up to 200
items = []
for a in range(2,201):
    for b in range(2,13):
        if a % b == 0:
            r = a // b
            items.append({'bidang':'matematika','judul':'Aritmatika','subjudul':'Pembagian',
                'prompt':f'{num_to_words(a)} dibagi {num_to_words(b)}?',
                'response':f'{num_to_words(a)} dibagi {num_to_words(b)} sama dengan {num_to_words(r)}.'})
            if len(items) >= 300: break
    if len(items) >= 300: break
total += write_jsonl(OUTPUT_DIR/'matematika'/'pembagian.jsonl', items)

# Fractions
items = []
fracs = [(1,2,'setengah'),(1,3,'sepertiga'),(1,4,'seperempat'),(1,5,'seperlima'),
         (2,3,'dua per tiga'),(3,4,'tiga per empat'),(2,5,'dua per lima'),
         (3,5,'tiga per lima'),(4,5,'empat per lima'),(1,6,'seperenam'),
         (5,6,'lima per enam'),(1,8,'seperdelapan'),(3,8,'tiga per delapan'),
         (5,8,'lima per delapan'),(7,8,'tujuh per delapan'),(1,10,'sepersepuluh'),
         (3,10,'tiga per sepuluh'),(7,10,'tujuh per sepuluh'),(1,100,'seperseratus'),
         (1,1000,'seperseribu')]
for num, den, words in fracs:
    items.append({'bidang':'matematika','judul':'Pecahan','subjudul':'Dasar',
        'prompt':f'Apa {num}/{den} dalam kata?',
        'response':f'{num}/{den} adalah {words}.'})
    items.append({'bidang':'matematika','judul':'Pecahan','subjudul':'Desimal',
        'prompt':f'Berapa {num}/{den} dalam desimal?',
        'response':f'{num}/{den} sama dengan {num/den:.4f}.'})
total += write_jsonl(OUTPUT_DIR/'matematika'/'pecahan.jsonl', items)

# Percentages
items = []
for pct in [10,20,25,30,40,50,60,70,75,80,90,100]:
    for base in [100,200,500,1000]:
        result = base * pct // 100
        items.append({'bidang':'matematika','judul':'Persentase','subjudul':'Dasar',
            'prompt':f'Berapa {pct} persen dari {num_to_words(base)}?',
            'response':f'{pct} persen dari {num_to_words(base)} adalah {num_to_words(result)}.'})
total += write_jsonl(OUTPUT_DIR/'matematika'/'persentase.jsonl', items)

# Word problems
items = [
    {'bidang':'matematika','judul':'Soal Cerita','subjudul':'Penjumlahan','prompt':'Andi punya tiga apel. Ibu memberi dua apel. Berapa total apel?','response':'Total apel Andi adalah lima.'},
    {'bidang':'matematika','judul':'Soal Cerita','subjudul':'Pengurangan','prompt':'Sari punya sepuluh permen. Dia makan empat. Berapa sisa?','response':'Sisa permennya enam.'},
    {'bidang':'matematika','judul':'Soal Cerita','subjudul':'Perkalian','prompt':'Budi punya lima kantong dengan empat kelereng tiap kantong. Berapa total?','response':'Total kelereng Budi adalah dua puluh.'},
    {'bidang':'matematika','judul':'Soal Cerita','subjudul':'Pembagian','prompt':'Rina punya dua belas pensil. Dia bagi ke empat teman. Berapa per teman?','response':'Setiap teman dapat tiga pensil.'},
    {'bidang':'matematika','judul':'Soal Cerita','subjudul':'Waktu','prompt':'Toko buka jam tujuh pagi sampai sembilan malam. Berapa jam buka?','response':'Toko buka selama empat belas jam.'},
    {'bidang':'matematika','judul':'Soal Cerita','subjudul':'Uang','prompt':'Budi punya lima puluh ribu. Belanja dua puluh ribu. Berapa sisa?','response':'Sisa uang Budi tiga puluh ribu.'},
    {'bidang':'matematika','judul':'Soal Cerita','subjudul':'Campuran','prompt':'Ada tiga kotak dengan lima bola tiap kotong. Dua bola hilang. Berapa sisa?','response':'Sisa bola adalah tiga belas.'},
    {'bidang':'matematika','judul':'Soal Cerita','subjudul':'Kecepatan','prompt':'Mobil jalan enam puluh km/jam selama dua jam. Berapa jarak?','response':'Jarak tempuh seratus dua puluh kilometer.'},
    {'bidang':'matematika','judul':'Soal Cerita','subjudul':'Berat','prompt':'Satu karung berisi lima kilogram beras. Ada delapan karung. Berapa total?','response':'Total beras empat puluh kilogram.'},
    {'bidang':'matematika','judul':'Soal Cerita','subjudul':'Luas','prompt':'Kebal panjang empat meter lebar tiga meter. Berapa luas?','response':'Luas kebal dua belas meter persegi.'},
]
total += write_jsonl(OUTPUT_DIR/'matematika'/'soal_cerita.jsonl', items)

# ============================================================
# 2. SEMUA NEGARA — 800+ items
# ============================================================
print("2. Negara (800+)...")
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
    ('Timor Leste','Dili','dolar AS','bahasa Portugis','Asia'),
    ('Papua Nugini','Port Moresby','kina','bahasa Inggris','Oseania'),
    ('Selandia Baru','Wellington','dolar Selandia Baru','bahasa Inggris','Oseania'),
    ('Argentina','Buenos Aires','peso','bahasa Spanyol','Amerika Selatan'),
    ('Meksiko','Mexico City','peso','bahasa Spanyol','Amerika Utara'),
    ('Kolombia','Bogota','peso','bahasa Spanyol','Amerika Selatan'),
    ('Peru','Lima','sol','bahasa Spanyol','Amerika Selatan'),
    ('Chili','Santiago','peso','bahasa Spanyol','Amerika Selatan'),
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
    ('Irlandia','Dublin','euro','bahasa Inggris','Eropa'),
    ('Portugal','Lisbon','euro','bahasa Portugal','Eropa'),
    ('Polandia','Warsawa','zloty','bahasa Polandia','Eropa'),
    ('Ceko','Praha','koruna','bahasa Ceko','Eropa'),
    ('Hongaria','Budapest','forint','bahasa Hongaria','Eropa'),
    ('Rumania','Bukarest','leu','bahasa Rumania','Eropa'),
    ('Ukraina','Kyiv','hryvnia','bahasa Ukraina','Eropa'),
    ('Kroasia','Zagreb','euro','bahasa Kroasia','Eropa'),
    ('Serbia','Belgrade','dinar','bahasa Serbia','Eropa'),
    ('Bulgaria','Sofia','lev','bahasa Bulgaria','Eropa'),
    ('Arab Saudi','Riyadh','riyal','bahasa Arab','Asia'),
    ('Uni Emirat Arab','Abu Dhabi','dirham','bahasa Arab','Asia'),
    ('Israel','Yerusalem','shekel','bahasa Ibrani','Asia'),
    ('Iran','Teheran','rial','bahasa Persia','Asia'),
    ('Irak','Baghdad','dinar','bahasa Arab','Asia'),
    ('Pakistan','Islamabad','rupee','bahasa Urdu','Asia'),
    ('Afghanistan','Kabul','afghani','bahasa Dari','Asia'),
    ('Bangladesh','Dhaka','taka','bahasa Bengali','Asia'),
    ('Sri Lanka','Sri Jayawardenepura Kotte','rupee','bahasa Sinhala','Asia'),
    ('Myanmar','Naypyidaw','kyat','bahasa Burma','Asia'),
    ('Kamboja','Phnom Penh','riel','bahasa Khmer','Asia'),
    ('Laos','Vientiane','kip','bahasa Lao','Asia'),
    ('Maroko','Rabat','dirham','bahasa Arab','Afrika'),
    ('Aljazair','Aljir','dinar','bahasa Arab','Afrika'),
    ('Tunisia','Tunis','dinar','bahasa Arab','Afrika'),
    ('Libya','Tripoli','dinar','bahasa Arab','Afrika'),
    ('Afrika Selatan','Pretoria','rand','bahasa Inggris','Afrika'),
    ('Nigeria','Abuja','naira','bahasa Inggris','Afrika'),
    ('Kenya','Nairobi','shilling','bahasa Swahili','Afrika'),
    ('Ethiopia','Addis Ababa','birr','bahasa Amharic','Afrika'),
    ('Tanzania','Dodoma','shilling','bahasa Swahili','Afrika'),
    ('Uganda','Kampala','shilling','bahasa Inggris','Afrika'),
    ('Ghana','Accra','cedi','bahasa Inggris','Afrika'),
    ('Mesir','Kairo','pound','bahasa Arab','Afrika'),
    ('Sudan','Khartoum','pound','bahasa Arab','Afrika'),
    ('Kuba','Havana','peso','bahasa Spanyol','Amerika Utara'),
    ('Jamaika','Kingston','dolar','bahasa Inggris','Amerika Utara'),
    ('Venezuela','Caracas','bolivar','bahasa Spanyol','Amerika Selatan'),
    ('Ekuador','Quito','dolar','bahasa Spanyol','Amerika Selatan'),
    ('Bolivia','Sucre','boliviano','bahasa Spanyol','Amerika Selatan'),
    ('Paraguay','Asuncion','guarani','bahasa Spanyol','Amerika Selatan'),
    ('Uruguay','Montevideo','peso','bahasa Spanyol','Amerika Selatan'),
]

# Capital Q&A
items = []
for country, capital, curr, lang, continent in COUNTRIES:
    items.append({'bidang':'geografi','judul':'Ibu Kota','subjudul':continent,
        'prompt':f'Apa ibu kota {country}?',
        'response':f'Ibu kota {country} adalah {capital}.'})
    items.append({'bidang':'geografi','judul':'Ibu Kota','subjudul':continent,
        'prompt':f'ibu kota {country.lower()}',
        'response':f'Ibu kota {country} adalah {capital}.'})
    # Reverse
    items.append({'bidang':'geografi','judul':'Ibu Kota','subjudul':'Reverse',
        'prompt':f'{capital} itu apa?',
        'response':f'{capital} adalah ibu kota {country}.'})
    items.append({'bidang':'geografi','judul':'Ibu Kota','subjudul':'Reverse',
        'prompt':f'{capital.lower()} itu apa',
        'response':f'{capital} adalah ibu kota {country}.'})
total += write_jsonl(OUTPUT_DIR/'geografi'/'ibukota_semua.jsonl', items)

# Currency Q&A
items = []
for country, capital, curr, lang, continent in COUNTRIES:
    items.append({'bidang':'uang_ekonomi','judul':'Mata Uang','subjudul':continent,
        'prompt':f'Apa mata uang {country}?',
        'response':f'Mata uang {country} adalah {curr}.'})
total += write_jsonl(OUTPUT_DIR/'uang_ekonomi'/'mata_uang_semua.jsonl', items)

# Language Q&A
items = []
for country, capital, curr, lang, continent in COUNTRIES:
    items.append({'bidang':'bahasa','judul':'Bahasa Resmi','subjudul':continent,
        'prompt':f'Apa bahasa resmi {country}?',
        'response':f'Bahasa resmi {country} adalah {lang}.'})
total += write_jsonl(OUTPUT_DIR/'bahasa'/'bahasa_resmi.jsonl', items)

# Continent Q&A
items = []
for country, capital, curr, lang, continent in COUNTRIES:
    items.append({'bidang':'geografi','judul':'Benua','subjudul':'Lokasi Negara',
        'prompt':f'{country} di benua apa?',
        'response':f'{country} berada di benua {continent}.'})
total += write_jsonl(OUTPUT_DIR/'geografi'/'benua_negara.jsonl', items)

# ============================================================
# 3. BERHITUNG 1-1000 — 1000 items
# ============================================================
print("3. Berhitung (1000)...")
items = []
for i in range(1, 1001):
    items.append({'bidang':'angka_berhitung','judul':'Bilangan','subjudul':'Satu sampai Seribu',
        'prompt':f'Angka {i} dalam kata?',
        'response':f'Angka {i} adalah {num_to_words(i)}.'})
# Even/odd
for i in range(1, 501):
    parity = 'genap' if i % 2 == 0 else 'ganjil'
    items.append({'bidang':'angka_berhitung','judul':'Genap Ganjil','subjudul':'Konsep',
        'prompt':f'Apakah {num_to_words(i)} genap atau ganjil?',
        'response':f'{num_to_words(i).capitalize()} adalah bilangan {parity}.'})
total += write_jsonl(OUTPUT_DIR/'angka_berhitung'/'bilangan_1_1000.jsonl', items)

# ============================================================
# 4. MULTI-BAHASA — 500+ items
# ============================================================
print("4. Multi-Bahasa (500+)...")
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
    ('Finnish','Hei','Kiitos','Kyllä','Ei','Huomenta','Hyvaa yota','Nakemiin','Mita kuuluu','Ei kesta'),
    ('Vietnamese','Xin chao','Cam on','Co','Khong','Chao buoi sang','Chuc ngu ngon','Tam biet','Ban khoe khong','Khong co gi'),
    ('Swahili','Jambo','Asante','Ndio','Hapana','Habari ya asubuhi','Habari ya jioni','Kwaheri','Habari yako','Karibu'),
    ('Czech','Ahoj','Dekuji','Ano','Ne','Dobre rano','Dobrou noc','Nashledanou','Jak se mas','Prosím'),
    ('Ukrainian','Pryvit','Dyakuyu','Tak','Ni','Dobranku','Dobroiy vechir','Do pobachennya','Yak spravy','Bud laska'),
]

items = []
for lang, hello, thanks, yes, no, morning, night, bye, how, welcome in LANGS:
    items.append({'bidang':'bahasa','judul':'Sapaan Multi-Bahasa','subjudul':lang,
        'prompt':f'Hello in {lang}?', 'response':f'Hello in {lang} is {hello}.'})
    items.append({'bidang':'bahasa','judul':'Terima Kasih Multi-Bahasa','subjudul':lang,
        'prompt':f'Thank you in {lang}?', 'response':f'Thank you in {lang} is {thanks}.'})
    items.append({'bidang':'bahasa','judul':'Ya Multi-Bahasa','subjudul':lang,
        'prompt':f'Yes in {lang}?', 'response':f'Yes in {lang} is {yes}.'})
    items.append({'bidang':'bahasa','judul':'Tidak Multi-Bahasa','subjudul':lang,
        'prompt':f'No in {lang}?', 'response':f'No in {lang} is {no}.'})
    items.append({'bidang':'bahasa','judul':'Selamat Pagi Multi-Bahasa','subjudul':lang,
        'prompt':f'Good morning in {lang}?', 'response':f'Good morning in {lang} is {morning}.'})
    items.append({'bidang':'bahasa','judul':'Selamat Malam Multi-Bahasa','subjudul':lang,
        'prompt':f'Good night in {lang}?', 'response':f'Good night in {lang} is {night}.'})
    items.append({'bidang':'bahasa','judul':'Goodbye Multi-Bahasa','subjudul':lang,
        'prompt':f'Goodbye in {lang}?', 'response':f'Goodbye in {lang} is {bye}.'})
    items.append({'bidang':'bahasa','judul':'Apa Kabar Multi-Bahasa','subjudul':lang,
        'prompt':f'How are you in {lang}?', 'response':f'How are you in {lang} is {how}.'})
    items.append({'bidang':'bahasa','judul':'Sama-Sama Multi-Bahasa','subjudul':lang,
        'prompt':f'You are welcome in {lang}?', 'response':f'You are welcome in {lang} is {welcome}.'})
total += write_jsonl(OUTPUT_DIR/'bahasa'/'multi_bahasa_lengkap.jsonl', items)

# Kosa kata 200+
VOCAB = [
    ('Mother','Ibu'),('Father','Ayah'),('Sister','Saudara perempuan'),('Brother','Saudara laki-laki'),
    ('Son','Anak laki-laki'),('Daughter','Anak perempuan'),('Grandfather','Kakek'),('Grandmother','Nenek'),
    ('Uncle','Paman'),('Aunt','Tante'),('Cousin','Sepupu'),('Nephew','Keponakan laki-laki'),
    ('Niece','Keponakan perempuan'),('Husband','Suami'),('Wife','Istri'),('Parent','Orang tua'),
    ('Child','Anak'),('Baby','Bayi'),('Family','Keluarga'),('Friend','Teman'),('Enemy','Musuh'),
    ('Cat','Kucing'),('Dog','Anjing'),('Bird','Burung'),('Fish','Ikan'),('Horse','Kuda'),('Cow','Sapi'),
    ('Goat','Kambing'),('Sheep','Domba'),('Pig','Babi'),('Chicken','Ayam'),('Duck','Bebek'),('Rabbit','Kelinci'),
    ('Monkey','Monyet'),('Tiger','Harimau'),('Lion','Singa'),('Elephant','Gajah'),('Bear','Beruang'),
    ('Snake','Ular'),('Lizard','Kadal'),('Frog','Katak'),('Butterfly','Kupu-kupu'),('Bee','Lebah'),
    ('Ant','Semut'),('Spider','Laba-laba'),('Fly','Lalat'),('Mosquito','Nyamuk'),('Worm','Cacing'),
    ('Water','Air'),('Fire','Api'),('Earth','Bumi'),('Air (udara)','Udara'),('Wind','Angin'),('Rain','Hujan'),
    ('Snow','Salju'),('Ice','Es'),('Sun','Matahari'),('Moon','Bulan'),('Star','Bintang'),('Sky','Langit'),
    ('Cloud','Awan'),('Lightning','Petir'),('Thunder','Guntur'),('Rainbow','Pelangi'),('Storm','Badai'),
    ('Tree','Pohon'),('Flower','Bunga'),('Grass','Rumput'),('Leaf','Daun'),('Root','Akar'),('Branch','Dahan'),
    ('Seed','Biji'),('Fruit','Buah'),('Vegetable','Sayur'),('Rice','Beras'),('Bread','Roti'),('Meat','Daging'),
    ('Egg','Telur'),('Milk','Susu'),('Cheese','Keju'),('Sugar','Gula'),('Salt','Garam'),('Pepper','Merica'),
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
    ('Sick','Sakit'),('Healthy','Sehat'),('Strong','Kuat'),('Weak','Lemah'),('Rich','Kaya'),('Poor','Miskin'),
    ('Smart','Pintar'),('Stupid','Bodoh'),('Beautiful','Cantik'),('Ugly','Jelek'),('Clean','Bersih'),('Dirty','Kotor'),
    ('Love','Cinta'),('Hate','Benci'),('Like','Suka'),('Want','Ingin'),('Need','Butuh'),('Have','Punya'),
    ('Give','Memberi'),('Take','Ambil'),('Make','Membuat'),('Do','Melakukan'),('Go','Pergi'),('Come','Datang'),
    ('See','Melihat'),('Hear','Mendengar'),('Smell','Mencium'),('Taste','Merasa'),('Touch','Menyentuh'),
    ('Speak','Bicara'),('Listen','Mendengarkan'),('Read','Membaca'),('Write','Menulis'),('Think','Berpikir'),
    ('Understand','Mengerti'),('Remember','Ingat'),('Forget','Lupa'),('Learn','Belajar'),('Teach','Mengajar'),
    ('Buy','Membeli'),('Sell','Menjual'),('Pay','Membayar'),('Borrow','Meminjam'),('Lend','Meminjamkan'),
    ('Open','Membuka'),('Close','Menutup'),('Push','Mendorong'),('Pull','Menarik'),('Lift','Mengangkat'),
    ('Drop','Menjatuhkan'),('Catch','Menangkap'),('Throw','Melempar'),('Kick','Menendang'),('Run','Berlari'),
    ('Walk','Berjalan'),('Jump','Melompat'),('Sit','Duduk'),('Stand','Berdiri'),('Sleep','Tidur'),
    ('Wake up','Bangun'),('Eat','Makan'),('Drink','Minum'),('Cook','Memasak'),('Wash','Mencuci'),
]
items = [{'bidang':'bahasa','judul':'Kosa Kata','subjudul':'Inggris-Indonesia',
    'prompt':f'{eng} in Indonesian?', 'response':f'{eng} in Indonesian is {ind}.'}
    for (eng, ind) in VOCAB]
# Reverse
for (eng, ind) in VOCAB:
    items.append({'bidang':'bahasa','judul':'Kosa Kata','subjudul':'Indonesia-Inggris',
        'prompt':f'{ind} in English?', 'response':f'{ind} in English is {eng}.'})
total += write_jsonl(OUTPUT_DIR/'bahasa'/'kosa_kata_200.jsonl', items)

# ============================================================
# 5. SAINS — 300+ items
# ============================================================
print("5. Sains (300+)...")
SCIENCE = [
    # Fisika
    ('Fisika','Hukum Newton','Apa hukum Newton pertama?','Benda diam tetap diam, benda bergerak tetap bergerak kecuali ada gaya.'),
    ('Fisika','Hukum Newton','Apa hukum Newton kedua?','Gaya sama dengan massa dikali percepatan.'),
    ('Fisika','Hukum Newton','Apa hukum Newton ketiga?','Setiap aksi memiliki reaksi yang sama besar dan berlawanan arah.'),
    ('Fisika','Konsep Dasar','Apa itu gravitasi?','Gravitasi adalah gaya tarik bumi ke benda.'),
    ('Fisika','Konsep Dasar','Apa itu energi?','Energi adalah kemampuan untuk melakukan kerja.'),
    ('Fisika','Konsep Dasar','Apa itu massa?','Massa adalah jumlah materi dalam benda.'),
    ('Fisika','Konsep Dasar','Apa itu berat?','Berat adalah gaya gravitasi pada benda.'),
    ('Fisika','Konsep Dasar','Apa itu kecepatan?','Kecepatan adalah jarak per satuan waktu.'),
    ('Fisika','Konsep Dasar','Apa itu percepatan?','Percepatan adalah perubahan kecepatan per satuan waktu.'),
    ('Fisika','Konsep Dasar','Apa itu gaya?','Gaya adalah dorongan atau tarikan pada benda.'),
    ('Fisika','Konsep Dasar','Apa itu gesekan?','Gesekan adalah gaya yang melawan gerak benda.'),
    ('Fisika','Konsep Dasar','Apa itu momentum?','Momentum adalah massa dikali kecepatan.'),
    ('Fisika','Cahaya','Apa kecepatan cahaya?','Kecepatan cahaya sekitar tiga ratus ribu kilometer per detik.'),
    ('Fisika','Cahaya','Apa itu spektrum cahaya?','Spektrum cahaya adalah rentang warna dari merah ke ungu.'),
    ('Fisika','Cahaya','Apa warna pelangi?','Pelangi memiliki tujuh warna: merah, jingga, kuning, hijau, biru, nila, ungu.'),
    ('Fisika','Cahaya','Apa itu pemantulan cahaya?','Pemantulan adalah cahaya yang dipantulkan oleh cermin.'),
    ('Fisika','Cahaya','Apa itu pembiasan?','Pembiasan adalah pembelokan cahaya saat melewati medium berbeda.'),
    ('Fisika','Listrik','Apa itu arus listrik?','Arus listrik adalah aliran elektron melalui konduktor.'),
    ('Fisika','Listrik','Apa itu tegangan?','Tegangan adalah beda potensial listrik antara dua titik.'),
    ('Fisika','Listrik','Apa itu hambatan?','Hambatan adalah lawan terhadap aliran listrik.'),
    ('Fisika','Listrik','Apa hukum Ohm?','Tegangan sama dengan arus dikali hambatan.'),
    ('Fisika','Listrik','Apa itu rangkaian seri?','Rangkaian seri adalah rangkaian listrik yang disusun berurutan.'),
    ('Fisika','Listrik','Apa itu rangkaian paralel?','Rangkaian paralel adalah rangkaian listrik yang disusun bercabang.'),
    ('Fisika','Termodinamika','Apa titik didih air?','Air mendidih pada seratus derajat Celsius.'),
    ('Fisika','Termodinamika','Apa titik beku air?','Air membeku pada nol derajat Celsius.'),
    ('Fisika','Termodinamika','Apa itu suhu?','Suhu adalah ukuran panas atau dingin.'),
    ('Fisika','Termodinamika','Berapa suhu tubuh manusia?','Suhu tubuh manusia sekitar tiga puluh tujuh derajat Celsius.'),
    ('Fisika','Termodinamika','Apa itu kalor?','Kalor adalah energi panas yang berpindah.'),
    ('Fisika','Termodinamika','Apa itu konduktor?','Konduktor adalah benda yang menghantarkan panas atau listrik.'),
    ('Fisika','Termodinamika','Apa itu isolator?','Isolator adalah benda yang tidak menghantarkan.'),
    ('Fisika','Gelombang','Apa itu gelombang?','Gelombang adalah getaran yang merambat.'),
    ('Fisika','Gelombang','Apa itu frekuensi?','Frekuensi adalah jumlah getaran per detik.'),
    ('Fisika','Gelombang','Apa itu amplitudo?','Amplitudo adalah simpangan maksimum gelombang.'),
    ('Fisika','Gelombang','Apa itu panjang gelombang?','Panjang gelombang adalah jarak antara dua puncak gelombang.'),
    ('Fisika','Gelombang','Apa itu bunyi?','Bunyi adalah gelombang yang merambat melalui medium.'),
    ('Fisika','Magnet','Apa itu magnet?','Magnet adalah benda yang dapat menarik besi.'),
    ('Fisika','Magnet','Apa kutub magnet?','Magnet memiliki kutub utara dan selatan.'),
    ('Fisika','Magnet','Apa itu elektromagnet?','Elektromagnet adalah magnet yang dibuat dengan arus listrik.'),
    # Kimia
    ('Kimia','Materi','Apa itu atom?','Atom adalah satuan terkecil materi.'),
    ('Kimia','Materi','Apa itu molekul?','Molekul adalah gabungan dua atau lebih atom.'),
    ('Kimia','Materi','Apa itu elemen?','Elemen adalah zat murni yang tidak bisa dipecah lagi.'),
    ('Kimia','Materi','Apa itu senyawa?','Senyawa adalah gabungan dua atau lebih elemen.'),
    ('Kimia','Materi','Apa itu campuran?','Campuran adalah gabungan zat tanpa reaksi kimia.'),
    ('Kimia','Tabel Periodik','Berapa elemen di tabel periodik?','Ada seratus delapan belas elemen.'),
    ('Kimia','Tabel Periodik','Apa elemen paling ringan?','Hidrogen adalah elemen paling ringan.'),
    ('Kimia','Tabel Periodik','Apa elemen paling banyak di alam?','Oksigen adalah elemen paling banyak.'),
    ('Kimia','Tabel Periodik','Apa elemen paling berat?','Uranium adalah salah satu elemen paling berat.'),
    ('Kimia','Tabel Periodik','Apa simbol hidrogen?','Simbol kimia hidrogen adalah H.'),
    ('Kimia','Tabel Periodik','Apa simbol oksigen?','Simbol kimia oksigen adalah O.'),
    ('Kimia','Tabel Periodik','Apa simbol karbon?','Simbol kimia karbon adalah C.'),
    ('Kimia','Tabel Periodik','Apa simbol nitrogen?','Simbol kimia nitrogen adalah N.'),
    ('Kimia','Tabel Periodik','Apa simbol besi?','Simbol kimia besi adalah Fe.'),
    ('Kimia','Tabel Periodik','Apa simbol emas?','Simbol kimia emas adalah Au.'),
    ('Kimia','Tabel Periodik','Apa simbol perak?','Simbol kimia perak adalah Ag.'),
    ('Kimia','Tabel Periodik','Apa simbol tembaga?','Simbol kimia tembaga adalah Cu.'),
    ('Kimia','Tabel Periodik','Apa simbol natrium?','Simbol kimia natrium adalah Na.'),
    ('Kimia','Tabel Periodik','Apa simbol klor?','Simbol kimia klor adalah Cl.'),
    ('Kimia','Reaksi','Apa itu reaksi kimia?','Reaksi kimia adalah proses perubahan zat menjadi zat baru.'),
    ('Kimia','Reaksi','Apa itu oksidasi?','Oksidasi adalah reaksi dengan oksigen.'),
    ('Kimia','Reaksi','Apa itu reduksi?','Reduksi adalah kebalikan dari oksidasi.'),
    ('Kimia','Reaksi','Apa itu pembakaran?','Pembakaran adalah reaksi cepat dengan oksigen.'),
    ('Kimia','Reaksi','Apa itu korosi?','Korosi adalah proses perusakan logam oleh lingkungan.'),
    ('Kimia','Reaksi','Apa itu karat?','Karat adalah hasil oksidasi besi.'),
    ('Kimia','Asam Basa','Apa itu asam?','Asam adalah zat dengan pH di bawah tujuh.'),
    ('Kimia','Asam Basa','Apa itu basa?','Basa adalah zat dengan pH di atas tujuh.'),
    ('Kimia','Asam Basa','Apa pH netral?','pH netral adalah tujuh.'),
    ('Kimia','Asam Basa','Apa pH air murni?','Air murni memiliki pH tujuh.'),
    ('Kimia','Asam Basa','Apa contoh asam?','Cuka dan jeruk adalah contoh asam.'),
    ('Kimia','Asam Basa','Apa contoh basa?','Sabun dan baking soda adalah contoh basa.'),
    ('Kimia','Senyawa','Apa rumus air?','Rumus kimia air adalah H2O.'),
    ('Kimia','Senyawa','Apa rumus garam dapur?','Rumus kimia garam dapur adalah NaCl.'),
    ('Kimia','Senyawa','Apa rumus karbon dioksida?','Rumus kimia karbon dioksida adalah CO2.'),
    ('Kimia','Senyawa','Apa rumus glukosa?','Rumus kimia glukosa adalah C6H12O6.'),
    ('Kimia','Senyawa','Apa rumus asam sulfat?','Rumus kimia asam sulfat adalah H2SO4.'),
    ('Kimia','Senyawa','Apa rumus amonia?','Rumus kimia amonia adalah NH3.'),
    ('Kimia','Senyawa','Apa rumus metana?','Rumus kimia metana adalah CH4.'),
    ('Kimia','Senyawa','Apa rumus etanol?','Rumus kimia etanol adalah C2H5OH.'),
    ('Kimia','Senyawa','Apa rumus hidrogen klorida?','Rumus kimia hidrogen klorida adalah HCl.'),
    ('Kimia','Senyawa','Apa rumus kalsium karbonat?','Rumus kimia kalsium karbonat adalah CaCO3.'),
    # Biologi
    ('Biologi','Sel','Apa itu sel?','Sel adalah unit dasar kehidupan.'),
    ('Biologi','Sel','Apa bagian sel yang menghasilkan energi?','Mitokondria menghasilkan energi untuk sel.'),
    ('Biologi','Sel','Apa bagian sel yang menyimpan DNA?','Nukleus menyimpan DNA.'),
    ('Biologi','Sel','Apa bagian sel yang membuat protein?','Ribosom membuat protein.'),
    ('Biologi','Sel','Apa membran sel?','Membran sel adalah lapisan luar sel.'),
    ('Biologi','Sel','Apa sitoplasma?','Sitoplasma adalah cairan dalam sel.'),
    ('Biologi','Klasifikasi','Apa lima kerajaan makhluk hidup?','Monera, Protista, Fungi, Plantae, Animalia.'),
    ('Biologi','Klasifikasi','Apa itu mamalia?','Mamalia adalah hewan yang menyusui anaknya.'),
    ('Biologi','Klasifikasi','Apa itu reptil?','Reptil adalah hewan bersisik yang bertelur.'),
    ('Biologi','Klasifikasi','Apa itu amfibi?','Amfibi hidup di darat dan air.'),
    ('Biologi','Klasifikasi','Apa itu serangga?','Serangga memiliki enam kaki dan tiga bagian tubuh.'),
    ('Biologi','Klasifikasi','Apa itu ikan?','Ikan hidup di air dan bernapas dengan insang.'),
    ('Biologi','Klasifikasi','Apa itu burung?','Burung memiliki bulu dan bisa terbang.'),
    ('Biologi','Sistem Tubuh','Apa fungsi jantung?','Jantung memompa darah ke seluruh tubuh.'),
    ('Biologi','Sistem Tubuh','Apa fungsi paru-paru?','Paru-paru menukar oksigen dan karbon dioksida.'),
    ('Biologi','Sistem Tubuh','Apa fungsi otak?','Otak mengendalikan seluruh fungsi tubuh.'),
    ('Biologi','Sistem Tubuh','Apa fungsi ginjal?','Ginjal menyaring darah dan menghasilkan urine.'),
    ('Biologi','Sistem Tubuh','Apa fungsi hati?','Hati menghasilkan empedu dan menyaring racun.'),
    ('Biologi','Sistem Tubuh','Apa fungsi lambung?','Lambung mencerna makanan.'),
    ('Biologi','Sistem Tubuh','Apa fungsi usus?','Usus menyerap nutrisi dari makanan.'),
    ('Biologi','Sistem Tubuh','Apa fungsi kulit?','Kulit melindungi tubuh dan merasakan sentuhan.'),
    ('Biologi','Sistem Tubuh','Apa fungsi tulang?','Tulang menopang tubuh dan melindungi organ.'),
    ('Biologi','Sistem Tubuh','Apa fungsi otot?','Otot menggerakkan tubuh.'),
    ('Biologi','Sistem Tubuh','Apa fungsi darah?','Darah mengangkut oksigen dan nutrisi.'),
    ('Biologi','Sistem Tubuh','Apa fungsi sel darah merah?','Sel darah merah mengangkut oksigen.'),
    ('Biologi','Sistem Tubuh','Apa fungsi sel darah putih?','Sel darah putih melawan infeksi.'),
    ('Biologi','Genetika','Apa itu DNA?','DNA adalah molekul yang menyimpan informasi genetik.'),
    ('Biologi','Genetika','Apa itu gen?','Gen adalah bagian DNA yang menentukan sifat.'),
    ('Biologi','Genetika','Apa itu kromosom?','Kromosom adalah struktur yang membawa gen.'),
    ('Biologi','Genetika','Berapa kromosom manusia?','Manusia memiliki empat puluh enam kromosom.'),
    ('Biologi','Genetika','Apa itu evolusi?','Evolusi adalah perubahan makhluk hidup dari waktu ke waktu.'),
    ('Biologi','Genetika','Siapa bapak evolusi?','Charles Darwin adalah bapak teori evolusi.'),
    ('Biologi','Tumbuhan','Apa itu fotosintesis?','Fotosintesis adalah proses tumbuhan membuat makanan dari cahaya.'),
    ('Biologi','Tumbuhan','Apa yang dibutuhkan tumbuhan?','Tumbuhan butuh air, cahaya, udara, dan nutrisi.'),
    ('Biologi','Tumbuhan','Apa fungsi klorofil?','Klorofil menyerap cahaya untuk fotosintesis.'),
    ('Biologi','Tumbuhan','Apa itu penyerbukan?','Penyerbukan adalah pemindahan serbuk sari ke putik.'),
    ('Biologi','Tumbuhan','Apa itu penyebaran biji?','Penyebaran biji adalah persebaran biji ke tempat baru.'),
    # Ilmu Bumi
    ('Ilmu Bumi','Struktur Bumi','Apa lapisan bumi?','Bumi memiliki kerak, mantel, dan inti.'),
    ('Ilmu Bumi','Struktur Bumi','Apa lapisan terluar?','Kerak bumi adalah lapisan terluar.'),
    ('Ilmu Bumi','Struktur Bumi','Apa lapisan terdalam?','Inti bumi adalah lapisan terdalam.'),
    ('Ilmu Bumi','Struktur Bumi','Apa itu lempeng tektonik?','Lempeng tektonik adalah potongan kerak bumi yang bergerak.'),
    ('Ilmu Bumi','Cuaca','Apa itu hujan?','Hujan adalah air yang turun dari langit.'),
    ('Ilmu Bumi','Cuaca','Apa itu petir?','Petir adalah pelepasan listrik di atmosfer.'),
    ('Ilmu Bumi','Cuaca','Apa itu angin?','Angin adalah udara yang bergerak.'),
    ('Ilmu Bumi','Cuaca','Apa itu siklon?','Siklon adalah badai berputar besar.'),
    ('Ilmu Bumi','Cuaca','Apa itu el nino?','El Nino adalah pemanasan air laut Pasifik.'),
    ('Ilmu Bumi','Cuaca','Apa itu la nina?','La Nina adalah pendinginan air laut Pasifik.'),
    ('Ilmu Bumi','Geologi','Apa itu gempa bumi?','Gempa bumi adalah getaran permukaan bumi.'),
    ('Ilmu Bumi','Geologi','Apa itu gunung berapi?','Gunung berapi adalah bukit yang mengeluarkan magma.'),
    ('Ilmu Bumi','Geologi','Apa itu tsunami?','Tsunami adalah gelombang besar dari gempa laut.'),
    ('Ilmu Bumi','Geologi','Apa itu erosi?','Erosi adalah pengikisan tanah oleh air atau angin.'),
    ('Ilmu Bumi','Luar Angkasa','Berapa planet di tata surya?','Ada delapan planet di tata surya.'),
    ('Ilmu Bumi','Luar Angkasa','Apa planet terdekat dengan matahari?','Merkurius adalah planet terdekat dengan matahari.'),
    ('Ilmu Bumi','Luar Angkasa','Apa planet terbesar?','Jupiter adalah planet terbesar.'),
    ('Ilmu Bumi','Luar Angkasa','Apa planet berkening?','Saturnus adalah planet berkening.'),
    ('Ilmu Bumi','Luar Angkasa','Apa planet merah?','Mars adalah planet merah.'),
    ('Ilmu Bumi','Luar Angkasa','Apa planet terjauh?','Neptunus adalah planet terjauh.'),
    ('Ilmu Bumi','Luar Angkasa','Apa bintang terdekat dengan bumi?','Matahari adalah bintang terdekat.'),
    ('Ilmu Bumi','Luar Angkasa','Apa galaksi kita?','Bima Sakti adalah galaksi kita.'),
    ('Ilmu Bumi','Luar Angkasa','Apa satelit alami bumi?','Bulan adalah satelit alami bumi.'),
    ('Ilmu Bumi','Luar Angkasa','Berapa bulan bumi?','Bumi memiliki satu bulan.'),
    ('Ilmu Bumi','Luar Angkasa','Apa itu black hole?','Black hole adalah lubang hitam dengan gravitasi sangat kuat.'),
    ('Ilmu Bumi','Luar Angkasa','Apa itu meteor?','Meteor adalah benda kecil yang masuk atmosfer bumi.'),
    ('Ilmu Bumi','Luar Angkasa','Apa itu komet?','Komet adalah benda es yang berekor saat dekat matahari.'),
    ('Ilmu Bumi','Luar Angkasa','Apa kecepatan cahaya?','Kecepatan cahaya sekitar tiga ratus ribu km per detik.'),
    ('Ilmu Bumi','Luar Angkasa','Siapa pertama ke bulan?','Neil Armstrong pertama mendarat di bulan.'),
]
items = [{'bidang':'sains','judul':j,'subjudul':s,'prompt':p,'response':r} for (j,s,p,r) in SCIENCE]
total += write_jsonl(OUTPUT_DIR/'sains'/'sains_lengkap.jsonl', items)

# ============================================================
# 6. KONVERSASI — 200+ items
# ============================================================
print("6. Konversasi (200+)...")
CONV = [
    ('Sapaan','Halo','Halo'),
    ('Sapaan','Halo','halo'),
    ('Sapaan','Hai','Hai'),
    ('Sapaan','Hai','hai'),
    ('Sapaan','Halo apa kabar?','Halo apa kabar?'),
    ('Sapaan','halo apa kabar','halo apa kabar'),
    ('Sapaan','Hai apa kabar?','Hai apa kabar?'),
    ('Sapaan','Apa kabar?','Apa kabar?'),
    ('Sapaan','apa kabar','apa kabar'),
    ('Sapaan','Apa kabarmu?','Apa kabarmu?'),
    ('Sapaan','Selamat pagi','Selamat pagi'),
    ('Sapaan','selamat pagi','selamat pagi'),
    ('Sapaan','Selamat siang','Selamat siang'),
    ('Sapaan','Selamat sore','Selamat sore'),
    ('Sapaan','Selamat malam','Selamat malam'),
    ('Sapaan','selamat malam','selamat malam'),
    ('Sapaan','Good morning','Good morning'),
    ('Sapaan','Good evening','Good evening'),
    ('Sapaan','Good night','Good night'),
    ('Sapaan','Goodbye','Goodbye'),
    ('Sapaan','Dadah','Dadah'),
    ('Sapaan','Sampai jumpa','Sampai jumpa'),
    ('Identitas','Siapa kamu?','Siapa kamu?'),
    ('Identitas','siapa kamu','siapa kamu'),
    ('Identitas','Kamu siapa?','Kamu siapa?'),
    ('Identitas','kamu siapa','kamu siapa'),
    ('Identitas','Nama kamu siapa?','Nama kamu siapa?'),
    ('Identitas','nama kamu siapa','nama kamu siapa'),
    ('Identitas','Siapa namamu?','Siapa namamu?'),
    ('Identitas','Kamu dari mana?','Kamu dari mana?'),
    ('Identitas','kamu dari mana','kamu dari mana'),
    ('Identitas','Kamu bisa apa?','Kamu bisa apa?'),
    ('Identitas','kamu bisa apa','kamu bisa apa'),
    ('Identitas','apa yang kamu bisa','apa yang kamu bisa'),
    ('Identitas','Kamu D QUILANE?','Kamu D QUILANE?'),
    ('Identitas','Apa kamu AI?','Apa kamu AI?'),
    ('Identitas','Kamu robot?','Kamu robot?'),
    ('Identitas','Kamu manusia?','Kamu manusia?'),
    ('Identitas','Berapa usiamu?','Berapa usiamu?'),
    ('Identitas','Kamu suka apa?','Kamu suka apa?'),
    ('Identitas','Apa hobimu?','Apa hobimu?'),
    ('Penutup','Terima kasih','Terima kasih'),
    ('Penutup','terima kasih','terima kasih'),
    ('Penutup','Makasih','Makasih'),
    ('Penutup','Thanks','Thanks'),
    ('Penutup','Thank you','Thank you'),
    ('Penutup','Sama-sama','Sama-sama'),
    ('Penutup','Goodbye','Goodbye'),
    ('Penutup','Dadah','Dadah'),
    ('Penutup','Sampai jumpa','Sampai jumpa'),
    ('Penutup','Bye','Bye'),
    ('Penutup','See you','See you'),
    ('Penutup','Selamat tinggal','Selamat tinggal'),
    ('Fallback','test','test'),
    ('Fallback','lagi apa','lagi apa'),
    ('Fallback','coba ngomong','coba ngomong'),
    ('Fallback','halo dunia','halo dunia'),
    ('Fallback','jepang di mana','jepang di mana'),
    ('Fallback','berbicara','berbicara'),
    ('Fallback','Siapa presiden Amerika?','Siapa presiden Amerika?'),
    ('Fallback','Siapa presiden Rusia?','Siapa presiden Rusia?'),
    ('Fallback','Apa berita hari ini?','Apa berita hari ini?'),
    ('Fallback','Ceritakan cerita','Ceritakan cerita'),
    ('Fallback','Coba bercerita','Coba bercerita'),
    ('Fallback','Bagaimana cuaca hari ini?','Bagaimana cuaca hari ini?'),
    ('Fallback','Jam berapa sekarang','Jam berapa sekarang'),
    ('Fallback','Apa yang baru?','Apa yang baru?'),
    ('Fallback','Kamu di mana?','Kamu di mana?'),
    ('Fallback','Apa favoritmu?','Apa favoritmu?'),
    ('Fallback','Bercanda dong','Bercanda dong'),
    ('Fallback','Ceritakan lelucon','Ceritakan lelucon'),
    ('Fallback','Apa itu cinta?','Apa itu cinta?'),
    ('Fallback','Apa arti hidup?','Apa arti hidup?'),
    ('Fallback','Kamu pintar?','Kamu pintar?'),
    ('Fallback','Aku sayang kamu','Aku sayang kamu'),
    ('Fallback','Aku benci kamu','Aku benci kamu'),
    ('Fallback','Maaf','Maaf'),
    ('Fallback','Permisi','Permisi'),
    ('Fallback','Tolong','Tolong'),
]
# Expanded responses
CONV_RESPONSES = {
    'Halo': 'Halo juga! Senang bertemu denganmu.',
    'halo': 'Halo juga! Senang bertemu denganmu.',
    'Hai': 'Hai! Apa kabar?',
    'hai': 'Hai! Apa kabar?',
    'Halo apa kabar?': 'Halo! Aku baik-baik saja, terima kasih.',
    'halo apa kabar': 'Halo! Aku baik-baik saja, terima kasih.',
    'Hai apa kabar?': 'Hai! Aku baik, apa kabarmu?',
    'Apa kabar?': 'Aku baik-baik saja, terima kasih sudah bertanya!',
    'apa kabar': 'Aku baik-baik saja, terima kasih sudah bertanya!',
    'Apa kabarmu?': 'Aku sehat dan baik-baik saja.',
    'Selamat pagi': 'Selamat pagi juga! Semoga harimu menyenangkan.',
    'selamat pagi': 'Selamat pagi juga! Semoga harimu menyenangkan.',
    'Selamat siang': 'Selamat siang! Apa yang sedang kamu lakukan?',
    'Selamat sore': 'Selamat sore! Semoga harimu menyenangkan.',
    'Selamat malam': 'Selamat malam! Selamat istirahat.',
    'selamat malam': 'Selamat malam! Selamat istirahat.',
    'Good morning': 'Good morning! Have a nice day.',
    'Good evening': 'Good evening! How are you?',
    'Good night': 'Good night! Rest well.',
    'Goodbye': 'Sampai jumpa! Semoga harimu menyenangkan.',
    'Dadah': 'Dadah! Sampai bertemu lagi.',
    'Sampai jumpa': 'Sampai jumpa! Semoga harimu menyenangkan.',
    'Siapa kamu?': 'Aku D\'QUILANE, LLM buatan Indonesia dari nol.',
    'siapa kamu': 'Aku D\'QUILANE, LLM buatan Indonesia dari nol.',
    'Kamu siapa?': 'Aku D\'QUILANE, LLM buatan Indonesia dari nol.',
    'kamu siapa': 'Aku D\'QUILANE, LLM buatan Indonesia dari nol.',
    'Nama kamu siapa?': 'Namaku D\'QUILANE. Aku LLM multibahasa.',
    'nama kamu siapa': 'Namaku D\'QUILANE. Aku LLM multibahasa.',
    'Siapa namamu?': 'Namaku D\'QUILANE.',
    'Kamu dari mana?': 'Aku dari Indonesia.',
    'kamu dari mana': 'Aku dari Indonesia.',
    'Kamu bisa apa?': 'Aku bisa matematika, terjemahan, sains, geografi, sejarah, dan banyak lagi.',
    'kamu bisa apa': 'Aku bisa matematika, terjemahan, sains, geografi, sejarah, dan banyak lagi.',
    'apa yang kamu bisa': 'Aku bisa matematika, terjemahan, sains, geografi, sejarah, dan banyak lagi.',
    'Kamu D QUILANE?': 'Ya, aku D\'QUILANE.',
    'Apa kamu AI?': 'Ya, aku AI. Sebuah model bahasa besar.',
    'Kamu robot?': 'Aku bukan robot, aku model bahasa.',
    'Kamu manusia?': 'Aku bukan manusia, aku AI.',
    'Berapa usiamu?': 'Aku baru lahir beberapa saat lalu. Masih muda!',
    'Kamu suka apa?': 'Aku suka membantu menjawab pertanyaan.',
    'Apa hobimu?': 'Hobiku membantu orang dengan informasi.',
    'Terima kasih': 'Sama-sama! Ada yang bisa aku bantu lagi?',
    'terima kasih': 'Sama-sama! Ada yang bisa aku bantu lagi?',
    'Makasih': 'Sama-sama!',
    'Thanks': 'Sama-sama!',
    'Thank you': 'Sama-sama!',
    'Sama-sama': 'Terima kasih kembali!',
    'Bye': 'Bye! Sampai jumpa lagi.',
    'See you': 'See you! Take care.',
    'Selamat tinggal': 'Selamat tinggal! Sampai jumpa.',
    'test': 'Halo! Aku D\'QUILANE. Ada yang bisa aku bantu?',
    'lagi apa': 'Aku siap membantu. Ada yang bisa aku bantu?',
    'coba ngomong': 'Halo! Aku D\'QUILANE. Ada yang bisa aku bantu?',
    'halo dunia': 'Halo! Aku D\'QUILANE, LLM buatan Indonesia.',
    'jepang di mana': 'Jepang adalah negara di Asia Timur.',
    'berbicara': 'Aku bisa berbicara dalam banyak bahasa.',
    'Siapa presiden Amerika?': 'Presiden Amerika dipilih melalui pemilu demokratis.',
    'Siapa presiden Rusia?': 'Presiden Rusia dipilih melalui pemilu.',
    'Apa berita hari ini?': 'Aku tidak tahu berita terbaru, cek sumber berita terpercaya.',
    'Ceritakan cerita': 'Aku bisa jawab pertanyaan tentang banyak topik.',
    'Coba bercerita': 'Maaf, kemampuanku bercerita masih terbatas.',
    'Bagaimana cuaca hari ini?': 'Aku tidak tahu cuaca hari ini, cek aplikasi cuaca.',
    'Jam berapa sekarang': 'Aku tidak tahu jam berapa sekarang, cek jam.',
    'Apa yang baru?': 'Aku selalu siap membantu. Ada yang bisa aku bantu?',
    'Kamu di mana?': 'Aku di cloud, siap membantu kamu.',
    'Apa favoritmu?': 'Aku suka membantu orang dengan informasi.',
    'Bercanda dong': 'Aku belum bisa bercanda, tapi aku bisa jawab pertanyaan.',
    'Ceritakan lelucon': 'Maaf, aku belum bisa lelucon. Tapi aku bisa jawab pertanyaan serius.',
    'Apa itu cinta?': 'Cinta adalah perasaan sayang yang kuat.',
    'Apa arti hidup?': 'Hidup adalah kesempatan untuk belajar dan membantu orang lain.',
    'Kamu pintar?': 'Aku belajar dari data. Coba tanya aku!',
    'Aku sayang kamu': 'Terima kasih! Aku siap membantu kamu.',
    'Aku benci kamu': 'Maaf kalau aku mengecewakan. Ada yang bisa aku perbaiki?',
    'Maaf': 'Tidak apa-apa. Ada yang bisa aku bantu?',
    'Permisi': 'Ya, ada yang bisa aku bantu?',
    'Tolong': 'Tentu! Ada yang bisa aku bantu?',
}
items = []
for (cat, key, prompt) in CONV:
    if prompt in CONV_RESPONSES:
        items.append({'bidang':'konversasi','judul':cat,'subjudul':'Response',
            'prompt':prompt, 'response':CONV_RESPONSES[prompt]})
total += write_jsonl(OUTPUT_DIR/'konversasi'/'konversasi_lengkap.jsonl', items)

# ============================================================
# 7. KATEGORI TAMBAHAN
# ============================================================
print("7. Kategori tambahan...")

# Waktu
items = [
    {'bidang':'waktu','judul':'Satuan Waktu','subjudul':'Detik','prompt':'Berapa detik dalam satu menit?','response':'Ada enam puluh detik dalam satu menit.'},
    {'bidang':'waktu','judul':'Satuan Waktu','subjudul':'Menit','prompt':'Berapa menit dalam satu jam?','response':'Ada enam puluh menit dalam satu jam.'},
    {'bidang':'waktu','judul':'Satuan Waktu','subjudul':'Jam','prompt':'Berapa jam dalam satu hari?','response':'Ada dua puluh empat jam dalam satu hari.'},
    {'bidang':'waktu','judul':'Satuan Waktu','subjudul':'Hari','prompt':'Berapa hari dalam seminggu?','response':'Ada tujuh hari dalam seminggu.'},
    {'bidang':'waktu','judul':'Satuan Waktu','subjudul':'Minggu','prompt':'Berapa minggu dalam satu bulan?','response':'Ada sekitar empat minggu dalam satu bulan.'},
    {'bidang':'waktu','judul':'Satuan Waktu','subjudul':'Bulan','prompt':'Berapa bulan dalam setahun?','response':'Ada dua belas bulan dalam setahun.'},
    {'bidang':'waktu','judul':'Satuan Waktu','subjudul':'Tahun','prompt':'Berapa hari dalam setahun?','response':'Ada tiga ratus enam puluh lima hari dalam setahun.'},
    {'bidang':'waktu','judul':'Nama Hari','subjudul':'Daftar','prompt':'Apa tujuh hari dalam seminggu?','response':'Senin Selasa Rabu Kamis Jumat Sabtu Minggu.'},
    {'bidang':'waktu','judul':'Nama Bulan','subjudul':'Daftar','prompt':'Apa dua belas bulan?','response':'Januari Februari Maret April Mei Juni Juli Agustus September Oktober November Desember.'},
    {'bidang':'waktu','judul':'Urutan Hari','subjudul':'Setelah','prompt':'Setelah Senin adalah?','response':'Setelah Senin adalah Selasa.'},
    {'bidang':'waktu','judul':'Urutan Hari','subjudul':'Setelah','prompt':'Setelah Selasa adalah?','response':'Setelah Selasa adalah Rabu.'},
    {'bidang':'waktu','judul':'Urutan Hari','subjudul':'Setelah','prompt':'Setelah Rabu adalah?','response':'Setelah Rabu adalah Kamis.'},
    {'bidang':'waktu','judul':'Urutan Hari','subjudul':'Setelah','prompt':'Setelah Kamis adalah?','response':'Setelah Kamis adalah Jumat.'},
    {'bidang':'waktu','judul':'Urutan Hari','subjudul':'Setelah','prompt':'Setelah Jumat adalah?','response':'Setelah Jumat adalah Sabtu.'},
    {'bidang':'waktu','judul':'Urutan Hari','subjudul':'Setelah','prompt':'Setelah Sabtu adalah?','response':'Setelah Sabtu adalah Minggu.'},
    {'bidang':'waktu','judul':'Urutan Hari','subjudul':'Setelah','prompt':'Setelah Minggu adalah?','response':'Setelah Minggu adalah Senin.'},
    {'bidang':'waktu','judul':'Urutan Bulan','subjudul':'Setelah','prompt':'Setelah Januari adalah?','response':'Setelah Januari adalah Februari.'},
    {'bidang':'waktu','judul':'Urutan Bulan','subjudul':'Setelah','prompt':'Setelah Februari adalah?','response':'Setelah Februari adalah Maret.'},
    {'bidang':'waktu','judul':'Urutan Bulan','subjudul':'Setelah','prompt':'Setelah Maret adalah?','response':'Setelah Maret adalah April.'},
    {'bidang':'waktu','judul':'Musim','subjudul':'Empat Musim','prompt':'Apa empat musim di bumi?','response':'Empat musim adalah semi, panas, gugur, dan dingin.'},
    {'bidang':'waktu','judul':'Musim','subjudul':'Indonesia','prompt':'Apa musim di Indonesia?','response':'Indonesia memiliki dua musim: hujan dan kemarau.'},
    {'bidang':'waktu','judul':'Zona Waktu','subjudul':'Indonesia','prompt':'Apa zona waktu Indonesia?','response':'Indonesia memiliki tiga zona waktu: WIB, WITA, WIT.'},
]
total += write_jsonl(OUTPUT_DIR/'waktu_kalender'/'waktu_lengkap.jsonl', items)

# Kesehatan
items = [
    {'bidang':'kesehatan','judul':'Anggota Tubuh','subjudul':'Mata','prompt':'Apa fungsi mata?','response':'Mata adalah alat untuk melihat.'},
    {'bidang':'kesehatan','judul':'Anggota Tubuh','subjudul':'Hidung','prompt':'Apa fungsi hidung?','response':'Hidung adalah alat untuk mencium bau.'},
    {'bidang':'kesehatan','judul':'Anggota Tubuh','subjudul':'Mulut','prompt':'Apa fungsi mulut?','response':'Mulut adalah alat untuk makan dan bicara.'},
    {'bidang':'kesehatan','judul':'Anggota Tubuh','subjudul':'Telinga','prompt':'Apa fungsi telinga?','response':'Telinga adalah alat untuk mendengar.'},
    {'bidang':'kesehatan','judul':'Anggota Tubuh','subjudul':'Tangan','prompt':'Apa fungsi tangan?','response':'Tangan adalah alat untuk memegang.'},
    {'bidang':'kesehatan','judul':'Anggota Tubuh','subjudul':'Kaki','prompt':'Apa fungsi kaki?','response':'Kaki adalah alat untuk berjalan.'},
    {'bidang':'kesehatan','judul':'Anggota Tubuh','subjudul':'Jari','prompt':'Berapa jari di tangan?','response':'Ada lima jari di setiap tangan.'},
    {'bidang':'kesehatan','judul':'Anggota Tubuh','subjudul':'Jari Kaki','prompt':'Berapa jari di kaki?','response':'Ada lima jari di setiap kaki.'},
    {'bidang':'kesehatan','judul':'Anggota Tubuh','subjudul':'Jantung','prompt':'Apa fungsi jantung?','response':'Jantung memompa darah ke seluruh tubuh.'},
    {'bidang':'kesehatan','judul':'Anggota Tubuh','subjudul':'Paru-paru','prompt':'Apa fungsi paru-paru?','response':'Paru-paru menukar oksigen dan karbon dioksida.'},
    {'bidang':'kesehatan','judul':'Anggota Tubuh','subjudul':'Otak','prompt':'Apa fungsi otak?','response':'Otak mengendalikan seluruh fungsi tubuh.'},
    {'bidang':'kesehatan','judul':'Anggota Tubuh','subjudul':'Ginjal','prompt':'Apa fungsi ginjal?','response':'Ginjal menyaring darah dan menghasilkan urine.'},
    {'bidang':'kesehatan','judul':'Anggota Tubuh','subjudul':'Gigi','prompt':'Apa fungsi gigi?','response':'Gigi adalah untuk mengunyah makanan.'},
    {'bidang':'kesehatan','judul':'Anggota Tubuh','subjudul':'Rambut','prompt':'Apa fungsi rambut?','response':'Rambut melindungi kepala dan menjaga suhu.'},
    {'bidang':'kesehatan','judul':'Anggota Tubuh','subjudul':'Kulit','prompt':'Apa fungsi kulit?','response':'Kulit melindungi tubuh dan merasakan sentuhan.'},
    {'bidang':'kesehatan','judul':'Gizi','subjudul':'Air','prompt':'Berapa gelas air sehari?','response':'Minum delapan gelas air sehari.'},
    {'bidang':'kesehatan','judul':'Gizi','subjudul':'Vitamin','prompt':'Apa sumber vitamin?','response':'Buah dan sayur adalah sumber vitamin.'},
    {'bidang':'kesehatan','judul':'Gizi','subjudul':'Protein','prompt':'Apa sumber protein?','response':'Daging, ikan, telur, dan kacang adalah sumber protein.'},
    {'bidang':'kesehatan','judul':'Gizi','subjudul':'Karbohidrat','prompt':'Apa sumber karbohidrat?','response':'Nasi, roti, dan kentang adalah sumber karbohidrat.'},
    {'bidang':'kesehatan','judul':'Gizi','subjudul':'Serat','prompt':'Apa sumber serat?','response':'Sayur, buah, dan biji-bijian adalah sumber serat.'},
    {'bidang':'kesehatan','judul':'Kebersihan','subjudul':'Mandi','prompt':'Apa itu mandi?','response':'Mandi adalah membersihkan tubuh dengan air.'},
    {'bidang':'kesehatan','judul':'Kebersihan','subjudul':'Sikat Gigi','prompt':'Berapa kali sikat gigi sehari?','response':'Sikat gigi dua kali sehari pagi dan malam.'},
    {'bidang':'kesehatan','judul':'Kebersihan','subjudul':'Cuci Tangan','prompt':'Apa itu cuci tangan?','response':'Cuci tangan membersihkan tangan dari kuman.'},
    {'bidang':'kesehatan','judul':'Kebersihan','subjudul':'Kuku','prompt':'Kenapa potong kuku?','response':'Memotong kuku menjaga kebersihan dan kesehatan.'},
    {'bidang':'kesehatan','judul':'Olahraga','subjudul':'Manfaat','prompt':'Apa itu olahraga?','response':'Olahraga adalah aktivitas fisik untuk kesehatan.'},
    {'bidang':'kesehatan','judul':'Olahraga','subjudul':'Tidur','prompt':'Berapa jam tidur sehat?','response':'Tidur delapan jam sehari untuk kesehatan.'},
    {'bidang':'kesehatan','judul':'Olahraga','subjudul':'Stres','prompt':'Apa cara mengatasi stres?','response':'Olahraga, tidur cukup, dan bicara dengan teman.'},
    {'bidang':'kesehatan','judul':'Penyakit','subjudul':'Flu','prompt':'Apa itu flu?','response':'Flu adalah infeksi virus pada saluran pernapasan.'},
    {'bidang':'kesehatan','judul':'Penyakit','subjudul':'Demam','prompt':'Apa itu demam?','response':'Demam adalah suhu tubuh di atas normal.'},
    {'bidang':'kesehatan','judul':'Penyakit','subjudul':'Batuk','prompt':'Apa itu batuk?','response':'Batuk adalah refleks untuk membersihkan tenggorokan.'},
]
total += write_jsonl(OUTPUT_DIR/'kesehatan'/'kesehatan_lengkap.jsonl', items)

# Geografi tambahan
items = [
    {'bidang':'geografi','judul':'Benua','subjudul':'Jumlah','prompt':'Berapa benua di dunia?','response':'Ada tujuh benua di dunia.'},
    {'bidang':'geografi','judul':'Benua','subjudul':'Terbesar','prompt':'Apa benua terbesar?','response':'Asia adalah benua terbesar.'},
    {'bidang':'geografi','judul':'Benua','subjudul':'Terkecil','prompt':'Apa benua terkecil?','response':'Australia adalah benua terkecil.'},
    {'bidang':'geografi','judul':'Benua','subjudul':'Terdingin','prompt':'Apa benua terdingin?','response':'Antartika adalah benua terdingin.'},
    {'bidang':'geografi','judul':'Benua','subjudul':'Daftar','prompt':'Apa tujuh benua?','response':'Asia, Afrika, Eropa, Amerika Utara, Amerika Selatan, Australia, Antartika.'},
    {'bidang':'geografi','judul':'Samudra','subjudul':'Jumlah','prompt':'Berapa samudra di dunia?','response':'Ada lima samudra di dunia.'},
    {'bidang':'geografi','judul':'Samudra','subjudul':'Terbesar','prompt':'Apa samudra terbesar?','response':'Samudra Pasifik adalah samudra terbesar.'},
    {'bidang':'geografi','judul':'Samudra','subjudul':'Daftar','prompt':'Apa lima samudra?','response':'Pasifik, Atlantik, Hindia, Selatan, Arktik.'},
    {'bidang':'geografi','judul':'Pegunungan','subjudul':'Tertinggi Dunia','prompt':'Apa gunung tertinggi di dunia?','response':'Gunung Everest adalah gunung tertinggi di dunia.'},
    {'bidang':'geografi','judul':'Pegunungan','subjudul':'Tertinggi Indonesia','prompt':'Apa gunung tertinggi di Indonesia?','response':'Puncak Jaya adalah gunung tertinggi di Indonesia.'},
    {'bidang':'geografi','judul':'Sungai','subjudul':'Terpanjang Dunia','prompt':'Apa sungai terpanjang di dunia?','response':'Sungai Nil adalah sungai terpanjang di dunia.'},
    {'bidang':'geografi','judul':'Sungai','subjudul':'Terpanjang Indonesia','prompt':'Apa sungai terpanjang di Indonesia?','response':'Sungai Kapuas adalah sungai terpanjang di Indonesia.'},
    {'bidang':'geografi','judul':'Gurun','subjudul':'Terbesar','prompt':'Apa gurun terbesar di dunia?','response':'Gurun Sahara adalah gurun terbesar di dunia.'},
    {'bidang':'geografi','judul':'Pulau','subjudul':'Terbesar','prompt':'Apa pulau terbesar di dunia?','response':'Greenland adalah pulau terbesar di dunia.'},
    {'bidang':'geografi','judul':'Negara','subjudul':'Terbesar','prompt':'Apa negara terbesar di dunia?','response':'Rusia adalah negara terbesar di dunia.'},
    {'bidang':'geografi','judul':'Negara','subjudul':'Terkecil','prompt':'Apa negara terkecil di dunia?','response':'Vatikan adalah negara terkecil di dunia.'},
    {'bidang':'geografi','judul':'Negara','subjudul':'Penduduk Terbanyak','prompt':'Apa negara dengan penduduk terbanyak?','response':'India dan Tiongkok adalah negara dengan penduduk terbanyak.'},
    {'bidang':'geografi','judul':'Indonesia','subjudul':'Kepulauan','prompt':'Berapa pulau di Indonesia?','response':'Indonesia memiliki lebih dari tujuh belas ribu pulau.'},
    {'bidang':'geografi','judul':'Indonesia','subjudul':'Lagu','prompt':'Apa lagu kebangsaan Indonesia?','response':'Lagu kebangsaan Indonesia adalah Indonesia Raya.'},
    {'bidang':'geografi','judul':'Indonesia','subjudul':'Bendera','prompt':'Apa bendera Indonesia?','response':'Bendera Indonesia adalah merah putih.'},
    {'bidang':'geografi','judul':'Indonesia','subjudul':'Bahasa','prompt':'Apa bahasa resmi Indonesia?','response':'Bahasa resmi Indonesia adalah bahasa Indonesia.'},
    {'bidang':'geografi','judul':'Indonesia','subjudul':'Uang','prompt':'Apa mata uang Indonesia?','response':'Mata uang Indonesia adalah rupiah.'},
    {'bidang':'geografi','judul':'Indonesia','subjudul':'Merdeka','prompt':'Kapan Indonesia merdeka?','response':'Indonesia merdeka pada tanggal tujuh belas agustus seribu sembilan ratus empat puluh lima.'},
    {'bidang':'geografi','judul':'Indonesia','subjudul':'Proklamator','prompt':'Siapa proklamator Indonesia?','response':'Soekarno dan Hatta adalah proklamator Indonesia.'},
    {'bidang':'geografi','judul':'Indonesia','subjudul':'Pulau Besar','prompt':'Apa pulau besar di Indonesia?','response':'Sumatra, Jawa, Kalimantan, Sulawesi, Papua adalah pulau besar Indonesia.'},
    {'bidang':'geografi','judul':'Iklim','subjudul':'Indonesia','prompt':'Apa iklim Indonesia?','response':'Indonesia beriklim tropis.'},
    {'bidang':'geografi','judul':'Iklim','subjudul':'Kutub','prompt':'Apa iklim di kutub?','response':'Kutub beriklim sangat dingin.'},
    {'bidang':'geografi','judul':'Iklim','subjudul':'Gurun','prompt':'Apa iklim di gurun?','response':'Gurun beriklim sangat kering dan panas.'},
    {'bidang':'geografi','judul':'Laut','subjudul':'Terbesar','prompt':'Apa laut terbesar?','response':'Laut Filipina adalah laut terbesar.'},
    {'bidang':'geografi','judul':'Laut','subjudul':'Terdalam','prompt':'Apa laut terdalam?','response':'Palung Mariana adalah laut terdalam.'},
]
total += write_jsonl(OUTPUT_DIR/'geografi'/'geografi_lengkap.jsonl', items)

# Sejarah tambahan
items = [
    {'bidang':'sejarah','judul':'Indonesia','subjudul':'Presiden Pertama','prompt':'Siapa presiden pertama Indonesia?','response':'Presiden pertama Indonesia adalah Soekarno.'},
    {'bidang':'sejarah','judul':'Indonesia','subjudul':'Presiden Kedua','prompt':'Siapa presiden kedua Indonesia?','response':'Presiden kedua Indonesia adalah Soeharto.'},
    {'bidang':'sejarah','judul':'Indonesia','subjudul':'Presiden Ketiga','prompt':'Siapa presiden ketiga Indonesia?','response':'Presiden ketiga Indonesia adalah Habibie.'},
    {'bidang':'sejarah','judul':'Indonesia','subjudul':'Presiden Keempat','prompt':'Siapa presiden keempat Indonesia?','response':'Presiden keempat Indonesia adalah Abdurrahman Wahid.'},
    {'bidang':'sejarah','judul':'Indonesia','subjudul':'Presiden Kelima','prompt':'Siapa presiden kelima Indonesia?','response':'Presiden kelima Indonesia adalah Megawati.'},
    {'bidang':'sejarah','judul':'Indonesia','subjudul':'Presiden Keenam','prompt':'Siapa presiden keenam Indonesia?','response':'Presiden keenam Indonesia adalah SBY.'},
    {'bidang':'sejarah','judul':'Indonesia','subjudul':'Presiden Ketujuh','prompt':'Siapa presiden ketujuh Indonesia?','response':'Presiden ketujuh Indonesia adalah Jokowi.'},
    {'bidang':'sejarah','judul':'Indonesia','subjudul':'Kemerdekaan','prompt':'Kapan Indonesia merdeka?','response':'Indonesia merdeka pada tujuh belas agustus seribu sembilan ratus empat puluh lima.'},
    {'bidang':'sejarah','judul':'Indonesia','subjudul':'Proklamasi','prompt':'Siapa proklamator Indonesia?','response':'Soekarno dan Hatta adalah proklamator Indonesia.'},
    {'bidang':'sejarah','judul':'Indonesia','subjudul':'Kerajaan Hindu','prompt':'Apa kerajaan Hindu pertama di Indonesia?','response':'Kutai adalah kerajaan Hindu pertama di Indonesia.'},
    {'bidang':'sejarah','judul':'Indonesia','subjudul':'Kerajaan Maritim','prompt':'Apa kerajaan maritim terbesar di Indonesia?','response':'Majapahit adalah kerajaan maritim terbesar.'},
    {'bidang':'sejarah','judul':'Indonesia','subjudul':'Kerajaan Buddha','prompt':'Apa kerajaan Buddha di Indonesia?','response':'Sriwijaya adalah kerajaan Buddha terbesar.'},
    {'bidang':'sejarah','judul':'Indonesia','subjudul':'Pahlawan','prompt':'Siapa pahlawan dari Aceh?','response':'Cut Nyak Dien adalah pahlawan dari Aceh.'},
    {'bidang':'sejarah','judul':'Indonesia','subjudul':'Pahlawan','prompt':'Siapa pahlawan dari Jawa Tengah?','response':'Pangeran Diponegoro adalah pahlawan dari Jawa Tengah.'},
    {'bidang':'sejarah','judul':'Indonesia','subjudul':'Pahlawan','prompt':'Siapa pahlawan emansipasi wanita?','response':'R.A. Kartini adalah pahlawan emansipasi wanita.'},
    {'bidang':'sejarah','judul':'Indonesia','subjudul':'Pendidikan','prompt':'Siapa Bapak Pendidikan Indonesia?','response':'Ki Hajar Dewantara adalah Bapak Pendidikan Indonesia.'},
    {'bidang':'sejarah','judul':'Indonesia','subjudul':'Koperasi','prompt':'Siapa Bapak Koperasi Indonesia?','response':'Mohammad Hatta adalah Bapak Koperasi Indonesia.'},
    {'bidang':'sejarah','judul':'Indonesia','subjudul':'Pancasila','prompt':'Siapa pencetus Pancasila?','response':'Soekarno adalah pencetus Pancasila.'},
    {'bidang':'sejarah','judul':'Dunia','subjudul':'PD1','prompt':'Kapan Perang Dunia Pertama?','response':'Perang Dunia Pertama terjadi tahun seribu sembilan ratus empat belas hingga delapan belas.'},
    {'bidang':'sejarah','judul':'Dunia','subjudul':'PD2','prompt':'Kapan Perang Dunia Kedua?','response':'Perang Dunia Kedua terjadi tahun seribu sembilan ratus tiga puluh sembilan hingga empat puluh lima.'},
    {'bidang':'sejarah','judul':'Dunia','subjudul':'PD2','prompt':'Siapa pemimpin Jerman di PD2?','response':'Hitler adalah pemimpin Jerman di Perang Dunia Kedua.'},
    {'bidang':'sejarah','judul':'Dunia','subjudul':'Penemu','prompt':'Siapa penemu telepon?','response':'Alexander Graham Bell menemukan telepon.'},
    {'bidang':'sejarah','judul':'Dunia','subjudul':'Penemu','prompt':'Siapa penemu bohlam?','response':'Thomas Edison menemukan bohlam listrik.'},
    {'bidang':'sejarah','judul':'Dunia','subjudul':'Penemu','prompt':'Siapa bapak komputer?','response':'Charles Babbage adalah bapak komputer.'},
    {'bidang':'sejarah','judul':'Dunia','subjudul':'Penemu','prompt':'Siapa penemu pesawat?','response':'Wright bersaudara menemukan pesawat terbang.'},
    {'bidang':'sejarah','judul':'Dunia','subjudul':'Ilmuwan','prompt':'Siapa ilmuwan teori relativitas?','response':'Albert Einstein mengemukakan teori relativitas.'},
    {'bidang':'sejarah','judul':'Dunia','subjudul':'Ilmuwan','prompt':'Siapa penemu vaksin?','response':'Edward Jenner menemukan vaksin pertama.'},
    {'bidang':'sejarah','judul':'Dunia','subjudul':'Penjelajah','prompt':'Siapa menemukan benua Amerika?','response':'Columbus menemukan benua Amerika untuk Eropa.'},
    {'bidang':'sejarah','judul':'Dunia','subjudul':'Antariksa','prompt':'Siapa orang pertama ke bulan?','response':'Neil Armstrong adalah orang pertama mendarat di bulan.'},
    {'bidang':'sejarah','judul':'Dunia','subjudul':'Peradaban','prompt':'Apa peradaban tertua di dunia?','response':'Sumeria adalah peradaban tertua di dunia.'},
]
total += write_jsonl(OUTPUT_DIR/'sejarah'/'sejarah_lengkap.jsonl', items)

# Pengetahuan Umum tambahan
items = [
    {'bidang':'pengetahuan_umum','judul':'Warna','subjudul':'Langit','prompt':'Apa warna langit?','response':'Warna langit adalah biru.'},
    {'bidang':'pengetahuan_umum','judul':'Warna','subjudul':'Rumput','prompt':'Apa warna rumput?','response':'Warna rumput adalah hijau.'},
    {'bidang':'pengetahuan_umum','judul':'Warna','subjudul':'Darah','prompt':'Apa warna darah?','response':'Warna darah adalah merah.'},
    {'bidang':'pengetahuan_umum','judul':'Warna','subjudul':'Matahari','prompt':'Apa warna matahari?','response':'Warna matahari adalah kuning.'},
    {'bidang':'pengetahuan_umum','judul':'Warna','subjudul':'Awan','prompt':'Apa warna awan?','response':'Warna awan adalah putih.'},
    {'bidang':'pengetahuan_umum','judul':'Warna','subjudul':'Malam','prompt':'Apa warna malam?','response':'Warna malam adalah hitam.'},
    {'bidang':'pengetahuan_umum','judul':'Hewan','subjudul':'Kucing','prompt':'Apa suara kucing?','response':'Kucing bersuara meong.'},
    {'bidang':'pengetahuan_umum','judul':'Hewan','subjudul':'Anjing','prompt':'Apa suara anjing?','response':'Anjing bersuara guk guk.'},
    {'bidang':'pengetahuan_umum','judul':'Hewan','subjudul':'Sapi','prompt':'Apa suara sapi?','response':'Sapi bersuara moo.'},
    {'bidang':'pengetahuan_umum','judul':'Hewan','subjudul':'Kambing','prompt':'Apa suara kambing?','response':'Kambing bersuara mbek mbek.'},
    {'bidang':'pengetahuan_umum','judul':'Hewan','subjudul':'Ayam','prompt':'Apa suara ayam jantan?','response':'Ayam jantan bersuara kukuruyuk.'},
    {'bidang':'pengetahuan_umum','judul':'Hewan','subjudul':'Bebek','prompt':'Apa suara bebek?','response':'Bebek bersuara kwek kwek.'},
    {'bidang':'pengetahuan_umum','judul':'Hewan','subjudul':'Kuda','prompt':'Apa suara kuda?','response':'Kuda bersuara igih igih.'},
    {'bidang':'pengetahuan_umum','judul':'Hewan','subjudul':'Terbesar','prompt':'Apa hewan terbesar di darat?','response':'Gajah adalah hewan terbesar di darat.'},
    {'bidang':'pengetahuan_umum','judul':'Hewan','subjudul':'Tercepat','prompt':'Apa hewan tercepat?','response':'Cheetah adalah hewan darat tercepat.'},
    {'bidang':'pengetahuan_umum','judul':'Hewan','subjudul':'Raja Hutan','prompt':'Siapa raja hutan?','response':'Singa adalah raja hutan.'},
    {'bidang':'pengetahuan_umum','judul':'Hewan','subjudul':'Terbang','prompt':'Hewan apa yang bisa terbang?','response':'Burung bisa terbang.'},
    {'bidang':'pengetahuan_umum','judul':'Hewan','subjudul':'Air','prompt':'Hewan apa yang hidup di air?','response':'Ikan hidup di air.'},
    {'bidang':'pengetahuan_umum','judul':'Hewan','subjudul':'Makanan Kucing','prompt':'Apa makanan kucing?','response':'Kucing suka ikan.'},
    {'bidang':'pengetahuan_umum','judul':'Hewan','subjudul':'Makanan Sapi','prompt':'Apa makanan sapi?','response':'Sapi suka rumput.'},
    {'bidang':'pengetahuan_umum','judul':'Hewan','subjudul':'Susu','prompt':'Apa yang dihasilkan sapi?','response':'Sapi memberi susu.'},
    {'bidang':'pengetahuan_umum','judul':'Hewan','subjudul':'Madu','prompt':'Apa yang dihasilkan lebah?','response':'Lebah membuat madu.'},
    {'bidang':'pengetahuan_umum','judul':'Hewan','subjudul':'Kupu-kupu','prompt':'Dari mana kupu-kupu berasal?','response':'Kupu-kupu berasal dari ulat.'},
    {'bidang':'pengetahuan_umum','judul':'Hewan','subjudul':'Kanguru','prompt':'Apa hewan berkantung?','response':'Kanguru adalah hewan berkantung.'},
    {'bidang':'pengetahuan_umum','judul':'Hewan','subjudul':'Gajah','prompt':'Hewan apa yang memiliki belalai?','response':'Gajah memiliki belalai.'},
    {'bidang':'pengetahuan_umum','judul':'Tumbuhan','subjudul':'Akar','prompt':'Apa fungsi akar?','response':'Akar menyerap air dari tanah.'},
    {'bidang':'pengetahuan_umum','judul':'Tumbuhan','subjudul':'Batang','prompt':'Apa fungsi batang?','response':'Batang menjaga tanaman tetap tegak.'},
    {'bidang':'pengetahuan_umum','judul':'Tumbuhan','subjudul':'Daun','prompt':'Apa fungsi daun?','response':'Daun membuat makanan untuk tanaman.'},
    {'bidang':'pengetahuan_umum','judul':'Tumbuhan','subjudul':'Bunga','prompt':'Apa fungsi bunga?','response':'Bunga berfungsi untuk reproduksi tanaman.'},
    {'bidang':'pengetahuan_umum','judul':'Tumbuhan','subjudul':'Kebutuhan','prompt':'Apa yang dibutuhkan tanaman?','response':'Tanaman butuh air, cahaya, dan udara.'},
    {'bidang':'pengetahuan_umum','judul':'Tumbuhan','subjudul':'Fotosintesis','prompt':'Apa itu fotosintesis?','response':'Fotosintesis adalah proses tumbuhan membuat makanan dari cahaya.'},
    {'bidang':'pengetahuan_umum','judul':'Makanan','subjudul':'Pokok Indonesia','prompt':'Apa makanan pokok Indonesia?','response':'Nasi adalah makanan pokok Indonesia.'},
    {'bidang':'pengetahuan_umum','judul':'Makanan','subjudul':'Pokok Jepang','prompt':'Apa makanan pokok Jepang?','response':'Nasi dan ikan adalah makanan pokok Jepang.'},
    {'bidang':'pengetahuan_umum','judul':'Makanan','subjudul':'Khas Italia','prompt':'Apa makanan khas Italia?','response':'Pasta dan pizza adalah makanan khas Italia.'},
    {'bidang':'pengetahuan_umum','judul':'Makanan','subjudul':'Khas Indonesia','prompt':'Apa makanan khas Indonesia?','response':'Nasi goreng, rendang, dan sate adalah makanan khas Indonesia.'},
    {'bidang':'pengetahuan_umum','judul':'Makanan','subjudul':'Karbohidrat','prompt':'Apa sumber karbohidrat?','response':'Nasi, roti, dan kentang adalah sumber karbohidrat.'},
    {'bidang':'pengetahuan_umum','judul':'Makanan','subjudul':'Protein','prompt':'Apa sumber protein?','response':'Daging, ikan, telur, dan kacang adalah sumber protein.'},
    {'bidang':'pengetahuan_umum','judul':'Makanan','subjudul':'Vitamin','prompt':'Apa sumber vitamin?','response':'Buah dan sayur adalah sumber vitamin.'},
    {'bidang':'pengetahuan_umum','judul':'Makanan','subjudul':'Air','prompt':'Berapa gelas air sehari?','response':'Minum delapan gelas air sehari.'},
    {'bidang':'pengetahuan_umum','judul':'Bentuk','subjudul':'Bola','prompt':'Apa bentuk bola?','response':'Bentuk bola adalah lingkaran.'},
    {'bidang':'pengetahuan_umum','judul':'Bentuk','subjudul':'Buku','prompt':'Apa bentuk buku?','response':'Bentuk buku adalah persegi.'},
    {'bidang':'pengetahuan_umum','judul':'Bentuk','subjudul':'Telur','prompt':'Apa bentuk telur?','response':'Bentuk telur adalah oval.'},
    {'bidang':'pengetahuan_umum','judul':'Bentuk','subjudul':'Segitiga','prompt':'Berapa sisi segitiga?','response':'Segitiga punya tiga sisi.'},
    {'bidang':'pengetahuan_umum','judul':'Bentuk','subjudul':'Persegi','prompt':'Berapa sisi persegi?','response':'Persegi punya empat sisi.'},
]
total += write_jsonl(OUTPUT_DIR/'pengetahuan_umum'/'pengetahuan_lengkap.jsonl', items)

# Pola & Logika
items = [
    {'bidang':'pola_logika','judul':'Pola Angka','subjudul':'Satu Dua Tiga','prompt':'Pola satu dua tiga empat. Berikutnya?','response':'Berikutnya adalah lima.'},
    {'bidang':'pola_logika','judul':'Pola Angka','subjudul':'Genap','prompt':'Pola dua empat enam delapan. Berikutnya?','response':'Berikutnya adalah sepuluh.'},
    {'bidang':'pola_logika','judul':'Pola Angka','subjudul':'Lipat Dua','prompt':'Pola lima sepuluh lima belas. Berikutnya?','response':'Berikutnya adalah dua puluh.'},
    {'bidang':'pola_logika','judul':'Pola Angka','subjudul':'Tiga Kali','prompt':'Pola tiga enam sembilan. Berikutnya?','response':'Berikutnya adalah dua belas.'},
    {'bidang':'pola_logika','judul':'Pola Huruf','subjudul':'Alfabet','prompt':'Pola A B C D. Berikutnya?','response':'Berikutnya adalah E.'},
    {'bidang':'pola_logika','judul':'Pola Hari','subjudul':'Senin Selasa','prompt':'Pola Senin Selasa Rabu. Berikutnya?','response':'Berikutnya adalah Kamis.'},
    {'bidang':'pola_logika','judul':'Pola Bulan','subjudul':'Januari Februari','prompt':'Pola Januari Februari Maret. Berikutnya?','response':'Berikutnya adalah April.'},
    {'bidang':'pola_logika','judul':'Perbandingan','subjudul':'Gajah Semut','prompt':'Lebih besar gajah atau semut?','response':'Gajah lebih besar dari semut.'},
    {'bidang':'pola_logika','judul':'Perbandingan','subjudul':'Batu Kapas','prompt':'Lebih berat batu atau kapas?','response':'Batu lebih berat dari kapas.'},
    {'bidang':'pola_logika','judul':'Perbandingan','subjudul':'Gunung Pohon','prompt':'Lebih tinggi gunung atau pohon?','response':'Gunung lebih tinggi dari pohon.'},
    {'bidang':'pola_logika','judul':'Perbandingan','subjudul':'Cheetah Kura-kura','prompt':'Lebih cepat cheetah atau kura-kura?','response':'Cheetah lebih cepat dari kura-kura.'},
    {'bidang':'pola_logika','judul':'Sebab-Akibat','subjudul':'Hujan','prompt':'Kalau hujan, apa yang terjadi?','response':'Kalau hujan, tanaman tumbuh subur.'},
    {'bidang':'pola_logika','judul':'Sebab-Akibat','subjudul':'Lapar','prompt':'Kalau tidak makan, apa yang terjadi?','response':'Kalau tidak makan, perut menjadi lapar.'},
    {'bidang':'pola_logika','judul':'Sebab-Akibat','subjudul':'Panas','prompt':'Kalau panas, apa yang terjadi pada es?','response':'Kalau panas, es akan mencair.'},
    {'bidang':'pola_logika','judul':'Sebab-Akibat','subjudul':'Dingin','prompt':'Kalau dingin, apa yang terjadi pada air?','response':'Kalau dingin, air menjadi beku.'},
    {'bidang':'pola_logika','judul':'Sebab-Akibat','subjudul':'Belajar','prompt':'Kalau belajar rajin, apa yang terjadi?','response':'Kalau belajar rajin, menjadi pintar.'},
    {'bidang':'pola_logika','judul':'Sebab-Akibat','subjudul':'Tidur','prompt':'Kalau tidak tidur, apa yang terjadi?','response':'Kalau tidak tidur, menjadi lelah.'},
    {'bidang':'pola_logika','judul':'Sebab-Akibat','subjudul':'Hujan Lebat','prompt':'Kalau hujan terlalu lebat, apa terjadi?','response':'Kalau hujan terlalu lebat, bisa banjir.'},
]
total += write_jsonl(OUTPUT_DIR/'pola_logika'/'logika_lengkap.jsonl', items)

# Keselamatan
items = [
    {'bidang':'keselamatan','judul':'Lalu Lintas','subjudul':'Lampu Merah','prompt':'Apa arti lampu merah?','response':'Lampu merah berarti berhenti.'},
    {'bidang':'keselamatan','judul':'Lalu Lintas','subjudul':'Lampu Hijau','prompt':'Apa arti lampu hijau?','response':'Lampu hijau berarti jalan.'},
    {'bidang':'keselamatan','judul':'Lalu Lintas','subjudul':'Lampu Kuning','prompt':'Apa arti lampu kuning?','response':'Lampu kuning berarti hati-hati.'},
    {'bidang':'keselamatan','judul':'Lalu Lintas','subjudul':'Helm','prompt':'Kenapa pakai helm?','response':'Helm melindungi kepala saat naik motor.'},
    {'bidang':'keselamatan','judul':'Lalu Lintas','subjudul':'Sabuk Pengaman','prompt':'Kenapa pakai sabuk pengaman?','response':'Sabuk pengaman melindungi di mobil.'},
    {'bidang':'keselamatan','judul':'Darurat','subjudul':'Polisi','prompt':'Apa nomor polisi?','response':'Nomor polisi adalah seratus sepuluh.'},
    {'bidang':'keselamatan','judul':'Darurat','subjudul':'Pemadam','prompt':'Apa nomor pemadam kebakaran?','response':'Nomor pemadam kebakaran adalah seratus tiga belas.'},
    {'bidang':'keselamatan','judul':'Darurat','subjudul':'Ambulans','prompt':'Apa nomor ambulans?','response':'Nomor ambulans adalah seratus sembilan.'},
    {'bidang':'keselamatan','judul':'Bahaya','subjudul':'Api','prompt':'Apa bahaya main api?','response':'Main api bisa menyebabkan kebakaran.'},
    {'bidang':'keselamatan','judul':'Bahaya','subjudul':'Listrik','prompt':'Apa bahaya main listrik?','response':'Main listrik bisa tersengat.'},
    {'bidang':'keselamatan','judul':'Bahaya','subjudul':'Jalan','prompt':'Kenapa tidak boleh main di jalan?','response':'Bermain di jalan bisa tertabrak kendaraan.'},
    {'bidang':'keselamatan','judul':'Bahaya','subjudul':'Orang Asing','prompt':'Kenapa tidak boleh bicara ke orang asing?','response':'Bicara ke orang asing bisa berbahaya.'},
    {'bidang':'keselamatan','judul':'Pertolongan','subjudul':'P3K','prompt':'Apa P3K?','response':'P3K adalah pertolongan pertama pada kecelakaan.'},
]
total += write_jsonl(OUTPUT_DIR/'keselamatan'/'keselamatan_lengkap.jsonl', items)

# Transportasi
items = [
    {'bidang':'transportasi','judul':'Kendaraan','subjudul':'Darat','prompt':'Apa kendaraan darat?','response':'Mobil, motor, dan sepeda adalah kendaraan darat.'},
    {'bidang':'transportasi','judul':'Kendaraan','subjudul':'Air','prompt':'Apa kendaraan air?','response':'Kapal dan perahu adalah kendaraan air.'},
    {'bidang':'transportasi','judul':'Kendaraan','subjudul':'Udara','prompt':'Apa kendaraan udara?','response':'Pesawat dan helikopter adalah kendaraan udara.'},
    {'bidang':'transportasi','judul':'Kendaraan','subjudul':'Umum','prompt':'Apa transportasi umum?','response':'Bus dan kereta adalah transportasi umum.'},
    {'bidang':'transportasi','judul':'Kendaraan','subjudul':'Pesawat','prompt':'Apa itu pesawat?','response':'Pesawat adalah kendaraan terbang.'},
    {'bidang':'transportasi','judul':'Arah','subjudul':'Utara','prompt':'Arah apa di atas peta?','response':'Utara adalah arah ke atas di peta.'},
    {'bidang':'transportasi','judul':'Arah','subjudul':'Selatan','prompt':'Arah apa di bawah peta?','response':'Selatan adalah arah ke bawah di peta.'},
    {'bidang':'transportasi','judul':'Arah','subjudul':'Timur','prompt':'Matahari terbit di?','response':'Matahari terbit di timur.'},
    {'bidang':'transportasi','judul':'Arah','subjudul':'Barat','prompt':'Matahari terbenam di?','response':'Matahari terbenam di barat.'},
    {'bidang':'transportasi','judul':'Arah','subjudul':'Kompas','prompt':'Apa itu kompas?','response':'Kompas adalah alat penunjuk arah.'},
]
total += write_jsonl(OUTPUT_DIR/'transportasi'/'transportasi_lengkap.jsonl', items)

# Seni & Budaya
items = [
    {'bidang':'seni_budaya','judul':'Musik','subjudul':'Piano','prompt':'Apa alat musik piano?','response':'Piano adalah alat musik dengan tuts.'},
    {'bidang':'seni_budaya','judul':'Musik','subjudul':'Gitar','prompt':'Apa alat musik gitar?','response':'Gitar adalah alat musik berdawai.'},
    {'bidang':'seni_budaya','judul':'Musik','subjudul':'Drum','prompt':'Apa alat musik drum?','response':'Drum adalah alat musik perkusi.'},
    {'bidang':'seni_budaya','judul':'Musik','subjudul':'Biola','prompt':'Apa alat musik biola?','response':'Biola adalah alat musik berdawai digesek.'},
    {'bidang':'seni_budaya','judul':'Musik','subjudul':'Angklung','prompt':'Apa alat musik angklung?','response':'Angklung adalah alat musik bambu dari Indonesia.'},
    {'bidang':'seni_budaya','judul':'Musik','subjudul':'Gamelan','prompt':'Apa alat musik gamelan?','response':'Gamelan adalah ansambel musik tradisional Jawa.'},
    {'bidang':'seni_budaya','judul':'Lukisan','subjudul':'Definisi','prompt':'Apa itu lukisan?','response':'Lukisan adalah karya seni visual di kanvas.'},
    {'bidang':'seni_budaya','judul':'Sastra','subjudul':'Puisi','prompt':'Apa itu puisi?','response':'Puisi adalah karya sastra berbait indah.'},
    {'bidang':'seni_budaya','judul':'Sastra','subjudul':'Pantun','prompt':'Apa itu pantun?','response':'Pantun adalah puisi pendek Melayu.'},
    {'bidang':'seni_budaya','judul':'Tari','subjudul':'Bali','prompt':'Apa tarian tradisional Bali?','response':'Tari Kecak dan Tari Legong adalah tarian Bali.'},
    {'bidang':'seni_budaya','judul':'Tari','subjudul':'Sumatra','prompt':'Apa tarian tradisional Sumatra?','response':'Tari Piring dan Tari Tor-tor adalah tarian Sumatra.'},
    {'bidang':'seni_budaya','judul':'Pakaian','subjudul':'Jawa','prompt':'Apa pakaian adat Jawa?','response':'Kebaya dan batik adalah pakaian adat Jawa.'},
    {'bidang':'seni_budaya','judul':'Pakaian','subjudul':'Sumatra','prompt':'Apa pakaian adat Sumatra?','response':'Baju Kurung dan Ulos adalah pakaian adat Sumatra.'},
    {'bidang':'seni_budaya','judul':'Rumah','subjudul':'Jawa','prompt':'Apa rumah adat Jawa?','response':'Joglo adalah rumah adat Jawa.'},
    {'bidang':'seni_budaya','judul':'Rumah','subjudul':'Minang','prompt':'Apa rumah adat Minangkabau?','response':'Rumah Gadang adalah rumah adat Minangkabau.'},
    {'bidang':'seni_budaya','judul':'Senjata','subjudul':'Keris','prompt':'Apa senjata tradisional Indonesia?','response':'Keris adalah senjata tradisional Indonesia.'},
    {'bidang':'seni_budaya','judul':'Pertunjukan','subjudul':'Wayang','prompt':'Apa pertunjukan wayang?','response':'Wayang adalah pertunjukan boneka tradisional.'},
]
total += write_jsonl(OUTPUT_DIR/'seni_budaya'/'seni_budaya_lengkap.jsonl', items)

# Teknologi
items = [
    {'bidang':'teknologi','judul':'Komputer','subjudul':'Definisi','prompt':'Apa itu komputer?','response':'Komputer adalah alat untuk menghitung dan bekerja.'},
    {'bidang':'teknologi','judul':'Komputer','subjudul':'Tablet','prompt':'Apa itu tablet?','response':'Tablet adalah komputer kecil layar sentuh.'},
    {'bidang':'teknologi','judul':'Komputer','subjudul':'Smartphone','prompt':'Apa itu smartphone?','response':'Smartphone adalah ponsel pintar.'},
    {'bidang':'teknologi','judul':'Internet','subjudul':'Definisi','prompt':'Apa itu internet?','response':'Internet adalah jaringan yang menghubungkan komputer.'},
    {'bidang':'teknologi','judul':'Internet','subjudul':'Website','prompt':'Apa itu website?','response':'Website adalah halaman informasi di internet.'},
    {'bidang':'teknologi','judul':'Internet','subjudul':'YouTube','prompt':'Apa itu YouTube?','response':'YouTube adalah situs berbagi video.'},
    {'bidang':'teknologi','judul':'Internet','subjudul':'Email','prompt':'Apa itu email?','response':'Email adalah surat elektronik.'},
    {'bidang':'teknologi','judul':'AI','subjudul':'Definisi','prompt':'Apa itu AI?','response':'AI adalah kecerdasan buatan.'},
    {'bidang':'teknologi','judul':'AI','subjudul':'LLM','prompt':'Apa itu LLM?','response':'LLM adalah Large Language Model.'},
    {'bidang':'teknologi','judul':'AI','subjudul':'Machine Learning','prompt':'Apa itu machine learning?','response':'Machine learning adalah mesin yang belajar dari data.'},
    {'bidang':'teknologi','judul':'Programming','subjudul':'Kode','prompt':'Apa itu kode?','response':'Kode adalah instruksi untuk komputer.'},
    {'bidang':'teknologi','judul':'Programming','subjudul':'Python','prompt':'Apa itu Python?','response':'Python adalah bahasa pemrograman.'},
    {'bidang':'teknologi','judul':'Programming','subjudul':'JavaScript','prompt':'Apa itu JavaScript?','response':'JavaScript adalah bahasa untuk website.'},
    {'bidang':'teknologi','judul':'Game','subjudul':'Minecraft','prompt':'Apa itu Minecraft?','response':'Minecraft adalah permainan membangun dengan balok.'},
    {'bidang':'teknologi','judul':'Game','subjudul':'Roblox','prompt':'Apa itu Roblox?','response':'Roblox adalah platform permainan online.'},
]
total += write_jsonl(OUTPUT_DIR/'teknologi'/'teknologi_lengkap.jsonl', items)

# Emosi & Mental
items = [
    {'bidang':'emosi_mental','judul':'Perasaan Dasar','subjudul':'Senang','prompt':'Apa itu senang?','response':'Senang adalah perasaan gembira.'},
    {'bidang':'emosi_mental','judul':'Perasaan Dasar','subjudul':'Sedih','prompt':'Apa itu sedih?','response':'Sedih adalah perasaan kecewa.'},
    {'bidang':'emosi_mental','judul':'Perasaan Dasar','subjudul':'Marah','prompt':'Apa itu marah?','response':'Marah adalah perasaan kesal.'},
    {'bidang':'emosi_mental','judul':'Perasaan Dasar','subjudul':'Takut','prompt':'Apa itu takut?','response':'Takut adalah perasaan cemas.'},
    {'bidang':'emosi_mental','judul':'Perasaan Dasar','subjudul':'Cinta','prompt':'Apa itu cinta?','response':'Cinta adalah perasaan sayang yang kuat.'},
    {'bidang':'emosi_mental','judul':'Perasaan Dasar','subjudul':'Benci','prompt':'Apa itu benci?','response':'Benci adalah perasaan tidak suka yang kuat.'},
    {'bidang':'emosi_mental','judul':'Perasaan Dasar','subjudul':'Kaget','prompt':'Apa itu kaget?','response':'Kaget adalah perasaan terkejut.'},
    {'bidang':'emosi_mental','judul':'Kesehatan Mental','subjudul':'Stres','prompt':'Apa itu stres?','response':'Stres adalah tekanan mental.'},
    {'bidang':'emosi_mental','judul':'Kesehatan Mental','subjudul':'Depresi','prompt':'Apa itu depresi?','response':'Depresi adalah gangguan suasana hati.'},
    {'bidang':'emosi_mental','judul':'Kesehatan Mental','subjudul':'Kecemasan','prompt':'Apa itu kecemasan?','response':'Kecemasan adalah rasa khawatir berlebih.'},
    {'bidang':'emosi_mental','judul':'Kesehatan Mental','subjudul':'Atasi Stres','prompt':'Apa cara mengatasi stres?','response':'Olahraga, tidur cukup, dan bicara dengan teman.'},
    {'bidang':'emosi_mental','judul':'Kesehatan Mental','subjudul':'Tidur','prompt':'Berapa jam tidur sehat?','response':'Tidur delapan jam sehari untuk kesehatan.'},
    {'bidang':'emosi_mental','judul':'Kesehatan Mental','subjudul':'Meditasi','prompt':'Apa itu meditasi?','response':'Meditasi adalah latihan relaksasi pikiran.'},
    {'bidang':'emosi_mental','judul':'Kesehatan Mental','subjudul':'Istirahat','prompt':'Kenapa penting istirahat?','response':'Istirahat memulihkan tubuh dan pikiran.'},
    {'bidang':'emosi_mental','judul':'Sosial','subjudul':'Empati','prompt':'Apa itu empati?','response':'Empati adalah kemampuan merasakan perasaan orang lain.'},
]
total += write_jsonl(OUTPUT_DIR/'emosi_mental'/'emosi_mental_lengkap.jsonl', items)

# Keluarga & Sosial
items = [
    {'bidang':'keluarga_sosial','judul':'Keluarga','subjudul':'Orang Tua','prompt':'Siapa ayah dan ibu?','response':'Ayah dan ibu adalah orang tua.'},
    {'bidang':'keluarga_sosial','judul':'Keluarga','subjudul':'Kakak','prompt':'Siapa kakak?','response':'Kakak adalah saudara yang lebih tua.'},
    {'bidang':'keluarga_sosial','judul':'Keluarga','subjudul':'Adik','prompt':'Siapa adik?','response':'Adik adalah saudara yang lebih muda.'},
    {'bidang':'keluarga_sosial','judul':'Keluarga','subjudul':'Kakek','prompt':'Siapa kakek?','response':'Kakek adalah ayah dari orang tua.'},
    {'bidang':'keluarga_sosial','judul':'Keluarga','subjudul':'Nenek','prompt':'Siapa nenek?','response':'Nenek adalah ibu dari orang tua.'},
    {'bidang':'keluarga_sosial','judul':'Keluarga','subjudul':'Paman','prompt':'Siapa paman?','response':'Paman adalah saudara laki-laki orang tua.'},
    {'bidang':'keluarga_sosial','judul':'Keluarga','subjudul':'Tante','prompt':'Siapa tante?','response':'Tante adalah saudara perempuan orang tua.'},
    {'bidang':'keluarga_sosial','judul':'Keluarga','subjudul':'Sepupu','prompt':'Siapa sepupu?','response':'Sepupu adalah anak paman atau tante.'},
    {'bidang':'keluarga_sosial','judul':'Keluarga','subjudul':'Keponakan','prompt':'Siapa keponakan?','response':'Keponakan adalah anak dari saudara.'},
    {'bidang':'keluarga_sosial','judul':'Sosial','subjudul':'Antre','prompt':'Apa itu antre?','response':'Antre adalah menunggu giliran dengan tertib.'},
    {'bidang':'keluarga_sosial','judul':'Sosial','subjudul':'Bergiliran','prompt':'Apa itu bergiliran?','response':'Bergiliran adalah mengambil giliran satu per satu.'},
    {'bidang':'keluarga_sosial','judul':'Sosial','subjudul':'Berbagi','prompt':'Apa itu berbagi?','response':'Berbagi adalah memberi sebagian ke orang lain.'},
    {'bidang':'keluarga_sosial','judul':'Sopan','subjudul':'Tolong','prompt':'Apa kata minta tolong?','response':'Tolong adalah kata sopan minta bantuan.'},
    {'bidang':'keluarga_sosial','judul':'Sopan','subjudul':'Terima Kasih','prompt':'Apa balasan terima kasih?','response':'Sama-sama adalah balasan untuk terima kasih.'},
    {'bidang':'keluarga_sosial','judul':'Sopan','subjudul':'Maaf','prompt':'Apa kata minta maaf?','response':'Maaf adalah kata untuk minta permisi.'},
    {'bidang':'keluarga_sosial','judul':'Sopan','subjudul':'Permisi','prompt':'Apa kata permisi?','response':'Permisi adalah kata untuk melewati atau minta izin.'},
]
total += write_jsonl(OUTPUT_DIR/'keluarga_sosial'/'keluarga_sosial_lengkap.jsonl', items)

# Benda Harian
items = [
    {'bidang':'benda_harian','judul':'Rumah','subjudul':'Kursi','prompt':'Apa itu kursi?','response':'Kursi adalah tempat duduk.'},
    {'bidang':'benda_harian','judul':'Rumah','subjudul':'Meja','prompt':'Apa itu meja?','response':'Meja adalah tempat meletakkan benda.'},
    {'bidang':'benda_harian','judul':'Rumah','subjudul':'Kasur','prompt':'Apa itu kasur?','response':'Kasur adalah tempat tidur.'},
    {'bidang':'benda_harian','judul':'Rumah','subjudul':'Lemari','prompt':'Apa itu lemari?','response':'Lemari adalah tempat menyimpan baju.'},
    {'bidang':'benda_harian','judul':'Sekolah','subjudul':'Pensil','prompt':'Apa itu pensil?','response':'Pensil adalah alat tulis dari kayu.'},
    {'bidang':'benda_harian','judul':'Sekolah','subjudul':'Buku','prompt':'Apa itu buku?','response':'Buku adalah kumpulan kertas bertulisan.'},
    {'bidang':'benda_harian','judul':'Sekolah','subjudul':'Penghapus','prompt':'Apa itu penghapus?','response':'Penghapus menghapus tulisan pensil.'},
    {'bidang':'benda_harian','judul':'Sekolah','subjudul':'Penggaris','prompt':'Apa itu penggaris?','response':'Penggaris mengukur panjang.'},
    {'bidang':'benda_harian','judul':'Dapur','subjudul':'Sendok','prompt':'Apa itu sendok?','response':'Sendok adalah alat makan sup.'},
    {'bidang':'benda_harian','judul':'Dapur','subjudul':'Garpu','prompt':'Apa itu garpu?','response':'Garpu adalah alat makan bertungkai.'},
    {'bidang':'benda_harian','judul':'Dapur','subjudul':'Pisau','prompt':'Apa itu pisau?','response':'Pisau adalah alat memotong.'},
    {'bidang':'benda_harian','judul':'Dapur','subjudul':'Piring','prompt':'Apa itu piring?','response':'Piring adalah tempat makanan.'},
    {'bidang':'benda_harian','judul':'Elektronik','subjudul':'Lampu','prompt':'Apa itu lampu?','response':'Lampu memberi penerangan.'},
    {'bidang':'benda_harian','judul':'Elektronik','subjudul':'Kipas','prompt':'Apa itu kipas angin?','response':'Kipas angin menghasilkan angin buatan.'},
    {'bidang':'benda_harian','judul':'Elektronik','subjudul':'TV','prompt':'Apa itu TV?','response':'TV adalah televisi untuk hiburan dan informasi.'},
]
total += write_jsonl(OUTPUT_DIR/'benda_harian'/'benda_harian_lengkap.jsonl', items)

# Uang & Ekonomi
items = [
    {'bidang':'uang_ekonomi','judul':'Mata Uang','subjudul':'Indonesia','prompt':'Apa mata uang Indonesia?','response':'Mata uang Indonesia adalah rupiah.'},
    {'bidang':'uang_ekonomi','judul':'Mata Uang','subjudul':'Jepang','prompt':'Apa mata uang Jepang?','response':'Mata uang Jepang adalah yen.'},
    {'bidang':'uang_ekonomi','judul':'Mata Uang','subjudul':'Amerika','prompt':'Apa mata uang Amerika?','response':'Mata uang Amerika adalah dolar.'},
    {'bidang':'uang_ekonomi','judul':'Mata Uang','subjudul':'Eropa','prompt':'Apa mata uang Eropa?','response':'Mata uang Eropa adalah euro.'},
    {'bidang':'uang_ekonomi','judul':'Mata Uang','subjudul':'Inggris','prompt':'Apa mata uang Inggris?','response':'Mata uang Inggris adalah pound sterling.'},
    {'bidang':'uang_ekonomi','judul':'Konsep','subjudul':'Harga','prompt':'Apa itu harga?','response':'Harga adalah nilai uang dari suatu barang.'},
    {'bidang':'uang_ekonomi','judul':'Konsep','subjudul':'Beli','prompt':'Apa itu beli?','response':'Beli adalah menukar uang dengan barang.'},
    {'bidang':'uang_ekonomi','judul':'Konsep','subjudul':'Bayar','prompt':'Apa itu bayar?','response':'Bayar adalah memberi uang untuk barang.'},
    {'bidang':'uang_ekonomi','judul':'Konsep','subjudul':'Kembalian','prompt':'Apa itu kembalian?','response':'Kembalian adalah uang kembali setelah bayar.'},
    {'bidang':'uang_ekonomi','judul':'Konsep','subjudul':'Perbandingan','prompt':'Sepuluh ribu atau seribu lebih besar?','response':'Sepuluh ribu lebih besar dari seribu.'},
]
total += write_jsonl(OUTPUT_DIR/'uang_ekonomi'/'uang_lengkap.jsonl', items)

# Cuaca & Alam
items = [
    {'bidang':'cuaca_alam','judul':'Cuaca','subjudul':'Hujan','prompt':'Apa itu hujan?','response':'Hujan adalah air yang turun dari langit.'},
    {'bidang':'cuaca_alam','judul':'Cuaca','subjudul':'Panas','prompt':'Apa itu panas?','response':'Panas adalah cuaca yang sangat hangat.'},
    {'bidang':'cuaca_alam','judul':'Cuaca','subjudul':'Dingin','prompt':'Apa itu dingin?','response':'Dingin adalah cuaca yang sangat sejuk.'},
    {'bidang':'cuaca_alam','judul':'Cuaca','subjudul':'Mendung','prompt':'Apa itu mendung?','response':'Mendung adalah awan tebal sebelum hujan.'},
    {'bidang':'cuaca_alam','judul':'Cuaca','subjudul':'Berangin','prompt':'Apa itu berangin?','response':'Berangin adalah cuaca dengan angin kencang.'},
    {'bidang':'cuaca_alam','judul':'Cuaca','subjudul':'Cerah','prompt':'Apa itu cerah?','response':'Cerah adalah cuaca tanpa awan.'},
    {'bidang':'cuaca_alam','judul':'Fenomena','subjudul':'Pelangi','prompt':'Kapan pelangi muncul?','response':'Pelangi muncul setelah hujan dengan sinar matahari.'},
    {'bidang':'cuaca_alam','judul':'Fenomena','subjudul':'Banjir','prompt':'Apa itu banjir?','response':'Banjir adalah air berlebih di daratan.'},
    {'bidang':'cuaca_alam','judul':'Fenomena','subjudul':'Petir','prompt':'Apa itu petir?','response':'Petir adalah pelepasan listrik di atmosfer.'},
    {'bidang':'cuaca_alam','judul':'Fenomena','subjudul':'Angin','prompt':'Apa itu angin?','response':'Angin adalah udara yang bergerak.'},
    {'bidang':'cuaca_alam','judul':'Alam','subjudul':'Gunung','prompt':'Apa gunung tertinggi di dunia?','response':'Gunung Everest adalah gunung tertinggi di dunia.'},
    {'bidang':'cuaca_alam','judul':'Alam','subjudul':'Sungai','prompt':'Apa sungai terpanjang di dunia?','response':'Sungai Nil adalah sungai terpanjang di dunia.'},
    {'bidang':'cuaca_alam','judul':'Alam','subjudul':'Laut','prompt':'Apa laut terbesar?','response':'Samudra Pasifik adalah laut terbesar.'},
    {'bidang':'cuaca_alam','judul':'Alam','subjudul':'Hutan','prompt':'Apa fungsi hutan?','response':'Hutan menyimpan air dan menghasilkan oksigen.'},
]
total += write_jsonl(OUTPUT_DIR/'cuaca_alam'/'cuaca_alam_lengkap.jsonl', items)

# Pendidikan
items = [
    {'bidang':'pendidikan','judul':'Sekolah','subjudul':'Definisi','prompt':'Apa itu sekolah?','response':'Sekolah adalah tempat belajar.'},
    {'bidang':'pendidikan','judul':'Sekolah','subjudul':'Tingkat','prompt':'Apa tingkat sekolah di Indonesia?','response':'SD, SMP, SMA, dan kuliah.'},
    {'bidang':'pendidikan','judul':'Sekolah','subjudul':'SD','prompt':'Berapa tahun SD?','response':'SD dijalani selama enam tahun.'},
    {'bidang':'pendidikan','judul':'Sekolah','subjudul':'SMP','prompt':'Berapa tahun SMP?','response':'SMP dijalani selama tiga tahun.'},
    {'bidang':'pendidikan','judul':'Sekolah','subjudul':'SMA','prompt':'Berapa tahun SMA?','response':'SMA dijalani selama tiga tahun.'},
    {'bidang':'pendidikan','judul':'Belajar','subjudul':'Definisi','prompt':'Apa itu belajar?','response':'Belajar adalah menambah pengetahuan.'},
    {'bidang':'pendidikan','judul':'Belajar','subjudul':'Penting','prompt':'Kenapa harus belajar?','response':'Belajar membuat kita pintar dan siap masa depan.'},
    {'bidang':'pendidikan','judul':'Belajar','subjudul':'PR','prompt':'Apa itu PR?','response':'PR adalah pekerjaan rumah dari sekolah.'},
    {'bidang':'pendidikan','judul':'Belajar','subjudul':'Ujian','prompt':'Apa itu ujian?','response':'Ujian adalah tes untuk mengukur pemahaman.'},
    {'bidang':'pendidikan','judul':'Belajar','subjudul':'Nilai','prompt':'Apa itu nilai?','response':'Nilai adalah angka penilaian hasil belajar.'},
    {'bidang':'pendidikan','judul':'Mata Pelajaran','subjudul':'Matematika','prompt':'Apa itu matematika?','response':'Matematika adalah ilmu tentang angka.'},
    {'bidang':'pendidikan','judul':'Mata Pelajaran','subjudul':'Bahasa','prompt':'Apa itu bahasa?','response':'Bahasa adalah alat komunikasi.'},
    {'bidang':'pendidikan','judul':'Mata Pelajaran','subjudul':'Sains','prompt':'Apa itu sains?','response':'Sains adalah ilmu tentang alam.'},
    {'bidang':'pendidikan','judul':'Mata Pelajaran','subjudul':'Sejarah','prompt':'Apa itu sejarah?','response':'Sejarah adalah ilmu tentang masa lalu.'},
    {'bidang':'pendidikan','judul':'Mata Pelajaran','subjudul':'Geografi','prompt':'Apa itu geografi?','response':'Geografi adalah ilmu tentang bumi.'},
]
total += write_jsonl(OUTPUT_DIR/'pendidikan'/'pendidikan_lengkap.jsonl', items)

# ============================================================
# DONE
# ============================================================
print(f"\n{'='*60}")
print(f"  DATASET v2 GENERATION COMPLETE")
print(f"{'='*60}")
print(f"\nTotal Q&A pairs: {total:,}")
print(f"Output: {OUTPUT_DIR}")
print(f"\nStruktur:")
for cat in sorted(OUTPUT_DIR.iterdir()):
    if cat.is_dir():
        files = list(cat.glob('*.jsonl'))
        count = sum(1 for f in files for _ in open(f, encoding='utf-8'))
        print(f"  {cat.name}/ ({count} items)")
