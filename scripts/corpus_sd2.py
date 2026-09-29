"""
D'QUILANE SD2 (Kelas 2) — Curated & Expanded Dataset
=====================================================
Strategy:
- TRIM redundant SD1 data (keep essentials only)
- ADD SD2-level content (multiplication, science, geography, logic, stories)

Topics (15 areas):
1. Math: addition, subtraction, MULTIPLICATION (NEW), division (NEW)
2. Numbers & Counting (trim)
3. Reading: paragraphs + comprehension (UPGRADED)
4. Instructions: 50+ SFT examples (EXPANDED)
5. Logic: cause-effect, sequences, comparisons (UPGRADED)
6. Science basics: states of matter, plants, animals (NEW)
7. Geography: continents, oceans, mountains (NEW)
8. Time: clock, days, months (UPGRADED)
9. Money/currency (NEW)
10. Patterns: number sequences (NEW)
11. Comparisons: big/bigger/biggest (NEW)
12. Storytelling: short narratives (NEW)
13. Family (trim)
14. Multilingual native script (TRIM: 30->15 langs, 4 sentences each)
15. Translations (UPGRADED)
"""

CORPUS = """
=== MATH: ADDITION & SUBTRACTION ===
Satu tambah satu sama dengan dua.
Dua tambah dua sama dengan empat.
Lima tambah lima sama dengan sepuluh.
Tiga tambah empat sama dengan tujuh.
Sepuluh tambah lima sama dengan lima belas.
Sepuluh dikurangi empat sama dengan enam.
Lima belas dikurangi lima sama dengan sepuluh.
=== MATH: MULTIPLICATION (Perkalian) - NEW ===
Satu dikali satu sama dengan satu.
Satu dikali dua sama dengan dua.
Satu dikali tiga sama dengan tiga.
Dua dikali satu sama dengan dua.
Dua dikali dua sama dengan empat.
Dua dikali tiga sama dengan enam.
Dua dikali empat sama dengan delapan.
Dua dikali lima sama dengan sepuluh.
Tiga dikali satu sama dengan tiga.
Tiga dikali dua sama dengan enam.
Tiga dikali tiga sama dengan sembilan.
Tiga dikali empat sama dengan dua belas.
Tiga dikali lima sama dengan lima belas.
Empat dikali dua sama dengan delapan.
Empat dikali tiga sama dengan dua belas.
Empat dikali empat sama dengan enam belas.
Empat dikali lima sama dengan dua puluh.
Lima dikali dua sama dengan sepuluh.
Lima dikali tiga sama dengan lima belas.
Lima dikali empat sama dengan dua puluh.
Lima dikali lima sama dengan dua puluh lima.
Sepuluh dikali satu sama dengan sepuluh.
Sepuluh dikali dua sama dengan dua puluh.
Sepuluh dikali sepuluh sama dengan seratus.
=== MATH: DIVISION (Pembagian) - NEW ===
Dua dibagi dua sama dengan satu.
Empat dibagi dua sama dengan dua.
Enam dibagi dua sama dengan tiga.
Sepuluh dibagi dua sama dengan lima.
Sepuluh dibagi lima sama dengan dua.
Sembilan dibagi tiga sama dengan tiga.
Dua belas dibagi tiga sama dengan empat.
Dua puluh dibagi empat sama dengan lima.
=== MATH: WORD PROBLEMS (Soal Cerita) - NEW ===
Andi punya tiga apel. Ibu memberi dua apel. Berapa total apel Andi? Total apel Andi adalah lima.
Sari punya sepuluh permen. Dia makan empat permen. Berapa sisa permen? Sisa permennya enam.
Budi punya lima kelereng. Ayah memberi lima kelereng lagi. Berapa total kelereng Budi? Totalnya sepuluh kelereng.
Rina punya dua belas pensil. Dia memberi empat pensil ke teman. Berapa sisa pensil? Sisa delapan pensil.
=== NUMBERS & COUNTING ===
Satu dua tiga empat lima enam tujuh delapan sembilan sepuluh.
Ada tujuh hari dalam seminggu.
Ada dua belas bulan dalam setahun.
Ada sepuluh jari di dua tangan.
Ada seratus sentimeter dalam satu meter.
One two three four five six seven eight nine ten.
One hundred is angka seratus.
=== READING COMPREHENSION (Baca) - UPGRADED ===
Indonesia adalah negara kepulauan di Asia Tenggara. Ibu kotanya Jakarta.
Jakarta adalah kota terbesar di Indonesia. Penduduknya sangat padat.
Bahasa Indonesia adalah bahasa resmi negara Indonesia.
Tokyo is the capital of Japan. Japan is an island country.
Paris is the capital of France. France is in Europe.
London is the capital of England. English is spoken there.
=== INSTRUCTIONS (Perintah) - EXPANDED ===
Apa itu Indonesia? Indonesia adalah negara kepulauan.
Siapa kamu? Aku D'QUILANE LLM buatan Indonesia.
Ibu kota Indonesia? Ibu kota Indonesia adalah Jakarta.
Apa ibu kota Jepang? Ibu kota Jepang adalah Tokyo.
Apa ibu kota Prancis? Ibu kota Prancis adalah Paris.
Apa ibu kota Inggris? Ibu kota Inggris adalah London.
Apa ibu kota Spanyol? Ibu kota Spanyol adalah Madrid.
Apa ibu kota Jerman? Ibu kota Jerman adalah Berlin.
Translate halo to English. Halo in English is Hello.
Translate hello to Indonesian. Hello in Indonesian is Halo.
Translate thank you to Indonesian. Thank you in Indonesian is Terima kasih.
Translate terima kasih to English. Terima kasih in English is Thank you.
Count to five. Satu dua tiga empat lima.
Count to ten. Satu dua tiga empat lima enam tujuh delapan sembilan sepuluh.
Apa warna langit? Warna langit adalah biru.
Apa warna rumput? Warna rumput adalah hijau.
Apa warna matahari? Warna matahari adalah kuning.
=== LOGIC (Logika) - UPGRADED ===
Setelah Senin adalah Selasa.
Setelah Selasa adalah Rabu.
After Monday comes Tuesday.
Setelah Januari adalah Februari.
Lebih besar: gajah atau semut? Gajah lebih besar dari semut.
Lebih berat: batu atau kapas? Batu lebih berat dari kapas.
Lebih tinggi: gunung atau pohon? Gunung lebih tinggi dari pohon.
Bigger: elephant or ant? Elephant is bigger than ant.
Heavier: stone or cotton? Stone is heavier than cotton.
=== CAUSE-EFFECT (Sebab-Akibat) - NEW ===
Kalau hujan turun, tanaman tumbuh subur.
Kalau tidak makan, perut menjadi lapar.
Kalau belajar rajin, menjadi pintar.
Kalau tidak tidur, menjadi lelah.
Kalau panas, es akan mencair.
Kalau dingin, air menjadi beku.
If it rains, plants grow well.
If you do not eat, you become hungry.
=== SCIENCE BASIC (Sains Dasar) - NEW ===
Air bisa menjadi tiga bentuk: cair, padat, dan gas.
Air cair adalah air biasa.
Es adalah air padat.
Uap adalah air berbentuk gas.
Matahari memberi cahaya dan panas.
Tanaman butuh air dan sinar matahari untuk tumbuh.
Daun membuat makanan untuk tanaman.
Akar menyerap air dari tanah.
Hewan dibagi menjadi: mamalia, burung, ikan, reptil, dan serangga.
Kucing adalah mamalia. Burung berkembang biak dengan bertelur.
Ikan hidup di air. Reptil memiliki sisik.
Kupu-kupu berasal dari ulat.
=== GEOGRAPHY (Geografi) - NEW ===
Bumi memiliki tujuh benua.
Benua: Asia, Afrika, Eropa, Amerika Utara, Amerika Selatan, Australia, Antartika.
Indonesia berada di benua Asia.
Jepang berada di benua Asia.
Prancis berada di benua Eropa.
Bumi memiliki lima samudra.
Samudra: Pasifik, Atlantik, Hindia, Selatan, Arktik.
Gunung Everest adalah gunung tertinggi di dunia.
Sungai Nil adalah sungai terpanjang di dunia.
Laut adalah air asin. Danau adalah air tawar.
=== TIME (Waktu) - UPGRADED ===
Satu jam sama dengan enam puluh menit.
Satu menit sama dengan enam puluh detik.
Satu hari sama dengan dua puluh empat jam.
Satu minggu sama dengan tujuh hari.
Satu tahun sama dengan dua belas bulan.
Satu tahun sama dengan tiga ratus enam puluh lima hari.
Senin Selasa Rabu Kamis Jumat Sabtu Minggu.
Januari Februari Maret April Mei Juni Juli Agustus September Oktober November Desember.
=== MONEY (Uang) - NEW ===
Mata uang Indonesia adalah rupiah.
Mata uang Amerika adalah dolar.
Mata uang Eropa adalah euro.
Mata uang Jepang adalah yen.
Mata uang Inggris adalah pound sterling.
Satu rupiah seratus sama dengan seratus rupiah.
Sepuluh ribu rupiah lebih besar dari seribu rupiah.
=== PATTERNS (Pola) - NEW ===
Pola: satu dua tiga empat. Berikutnya lima.
Pola: dua empat enam delapan. Berikutnya sepuluh.
Pola: lima sepuluh lima belas. Berikutnya dua puluh.
Pola: tiga enam sembilan. Berikutnya dua belas.
Pola: A B C D. Berikutnya E.
Pola: Senin Selasa Rabu. Berikutnya Kamis.
Pola: Januari Februari Maret. Berikutnya April.
Sequence: two four six eight. Next is ten.
Sequence: A B C D. Next is E.
=== COMPARISONS (Perbandingan) - NEW ===
Besar lebih besar paling besar.
Kecil lebih kecil paling kecil.
Tinggi lebih tinggi paling tinggi.
Cepat lebih cepat paling cepat.
Rumah besar. Gedung lebih besar. Gunung paling besar.
Semut kecil. Tikus lebih kecil. Kutu paling kecil.
Big bigger biggest.
Small smaller smallest.
=== STORYTELLING (Cerita) - NEW ===
Pagi itu Budi bangun tidur. Dia sarapan roti dan minum susu. Lalu dia berangkat ke sekolah.
Di sekolah Budi belajar matematika. Dia belajar tentang perkalian. Saat istirahat Budi bermain bola dengan teman-temannya.
Setelah pulang sekolah Budi mengerjakan PR. Lalu dia makan malam bersama keluarga. Setelah makan Budi tidur.
Once upon a time there was a small rabbit. The rabbit lived in a forest. The rabbit liked to eat carrots.
The rabbit had many friends. They played together every day. They were very happy.
=== FAMILY (Keluarga) - TRIM ===
Ayah dan ibu adalah orang tua.
Kakak dan adik adalah saudara.
Kakek dan nenek adalah orang tua dari orang tua.
Paman dan tante adalah saudara orang tua.
=== KNOWLEDGE (Pengetahuan) - TRIM ===
Apel adalah buah. Pisang adalah buah.
Singa adalah hewan. Sapi adalah hewan.
Mawar adalah bunga. Melati adalah bunga.
Warna langit adalah biru.
Warna rumput adalah hijau.
Warna matahari adalah kuning.
=== MULTILINGUAL NATIVE SCRIPT - TRIM ===
Indonesia adalah negara kepulauan. Ibu kota Indonesia adalah Jakarta.
English is a global language. The world is connected by technology.
El espanol es un idioma hermoso. La familia es muy importante.
Le francais est une belle langue. La famille est tres importante.
Deutsch ist eine wichtige Sprache. Die Familie ist sehr wichtig.
Il italiano e una bella lingua. La famiglia e molto importante.
Portugues e uma lingua bonita. A familia e muito importante.
Русский язык использует кириллицу. Русским языком говорят многие люди.
中文是世界上使用人数最多的语言。许多人每天说普通话。
日本語には三つの書き方があります。東京は日本の首都です。
한국어는 한글을 사용합니다. 서울은 한국의 수도입니다.
العربية تكتب من اليمين إلى اليسار. الخط العربي جميل جدا.
हिन्दी देवनागरी लिपि का उपयोग करती है. नई दिल्ली राजधानी है.
বাংলা একটি সুন্দর ভাষা. বাংলাদেশ বাংলা ব্যাপকভাবে ব্যবহার করে.
ภาษาไทยมีห้าเสียงวรรณยุกต์. กรุงเทพคือเมืองหลวง.
Türkçe Latin harflerini kullanır. İstanbul ünlü bir şehirdir.
Ελληνικά χρησιμοποιούν το ελληνικό αλφάβητο. Η Αθήνα είναι η πρωτεύουσα.
עברית נכתבת מימין לשמאל. ירושלים היא עיר קדושה.
=== TRANSLATIONS - UPGRADED ===
Hello in Indonesian is Halo.
Hello in Spanish is Hola.
Hello in French is Bonjour.
Hello in German is Hallo.
Hello in Japanese is Konnichiwa.
Hello in Korean is Annyeong.
Thank you in Indonesian is Terima kasih.
Thank you in Spanish is Gracias.
Thank you in French is Merci.
Thank you in German is Danke.
Thank you in Japanese is Arigatou.
Thank you in Korean is Gomapda.
Yes in Indonesian is Ya.
Yes in Spanish is Si.
Yes in French is Oui.
Yes in German is Ja.
No in Indonesian is Tidak.
No in Spanish is No.
No in French is Non.
No in German is Nein.
Mother in Indonesian is Ibu.
Father in Indonesian is Ayah.
Cat in Indonesian is Kucing.
Dog in Indonesian is Anjing.
Water in Indonesian is Air.
Fire in Indonesian is Api.
Book in Indonesian is Buku.
House in Indonesian is Rumah.
Tree in Indonesian is Pohon.
Flower in Indonesian is Bunga.
Sun in Indonesian is Matahari.
Moon in Indonesian is Bulan.
Star in Indonesian is Bintang.
Sky in Indonesian is Langit.
Mountain in Indonesian is Gunung.
River in Indonesian is Sungai.
Sea in Indonesian is Laut.
=== EXTRA SFT INSTRUCTIONS ===
Halo. Halo juga! Senang bertemu denganmu.
Siapa kamu? Aku D'QUILANE LLM buatan Indonesia dari nol.
Satu tambah satu? Satu tambah satu sama dengan dua.
Dua tambah dua? Dua tambah dua sama dengan empat.
Tiga dikali dua? Tiga dikali dua sama dengan enam.
Lima dikali lima? Lima dikali lima sama dengan dua puluh lima.
Sepuluh dibagi dua? Sepuluh dibagi dua sama dengan lima.
Apa ibu kota Indonesia? Ibu kota Indonesia adalah Jakarta.
Apa ibu kota Jepang? Ibu kota Jepang adalah Tokyo.
Apa ibu kota Prancis? Ibu kota Prancis adalah Paris.
Apa warna langit? Warna langit adalah biru.
Apa warna rumput? Warna rumput adalah hijau.
Apel adalah? Apel adalah buah.
Singa adalah? Singa adalah hewan.
Mawar adalah? Mawar adalah bunga.
Berapa hari dalam seminggu? Ada tujuh hari dalam seminggu.
Berapa bulan dalam setahun? Ada dua belas bulan dalam setahun.
Berapa menit dalam satu jam? Ada enam puluh menit dalam satu jam.
Berapa detik dalam satu menit? Ada enam puluh detik dalam satu menit.
Berapa jam dalam satu hari? Ada dua puluh empat jam dalam satu hari.
Berapa benua di dunia? Ada tujuh benua di dunia.
Apa mata uang Indonesia? Mata uang Indonesia adalah rupiah.
Apa mata uang Jepang? Mata uang Jepang adalah yen.
Apa gunung tertinggi di dunia? Gunung Everest adalah gunung tertinggi di dunia.
Setelah Senin adalah? Setelah Senin adalah Selasa.
Setelah Januari adalah? Setelah Januari adalah Februari.
Pola satu dua tiga empat. Berikutnya? Berikutnya lima.
Lebih besar gajah atau semut? Gajah lebih besar dari semut.
Translate halo to English. Halo in English is Hello.
Translate hello to Indonesian. Hello in Indonesian is Halo.
Translate thank you to Indonesian. Thank you in Indonesian is Terima kasih.
Translate terima kasih to English. Terima kasih in English is Thank you.
Terima kasih. Sama-sama! Ada yang bisa aku bantu lagi?
Goodbye. Sampai jumpa! Semoga harimu menyenangkan.
Apa kabar? Aku baik-baik saja, terima kasih sudah bertanya!
Count to five. Satu dua tiga empat lima.
Count to ten. Satu dua tiga empat lima enam tujuh delapan sembilan sepuluh.
Apa itu air? Air adalah benda cair yang kita minum.
Apa itu matahari? Matahari adalah benda langit yang memberi cahaya dan panas.
Apa itu hujan? Hujan adalah air yang turun dari langit.
Apa itu es? Es adalah air yang membeku karena dingin.
Apa itu kupu-kupu? Kupu-kupu adalah serangga yang berasal dari ulat.
Apa benua terbesar di dunia? Benua Asia adalah benua terbesar di dunia.
Apa samudra terbesar di dunia? Samudra Pasifik adalah samudra terbesar di dunia.
Budi punya tiga apel. Ibu memberi dua apel. Berapa total apel Budi? Total apel Andi adalah lima.
Sari punya sepuluh permen. Dia makan empat permen. Berapa sisa permen? Sisa permennya enam.
"""

if __name__ == "__main__":
    from pathlib import Path
    out = Path("/home/z/my-project/download/llm/web/corpus_sd2.txt")
    out.write_text(CORPUS)
    chars = sorted(set(CORPUS))
    print(f"Corpus SD2 size: {len(CORPUS)} chars (vs SD1: 16636)")
    print(f"Unique chars: {len(chars)} (vs SD1: 563)")
    print(f"\nNew chars vs SD1:")
    sd1_chars = set("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!&'()+,-.:?=\n áéíóúñçüäößệéàùîôœæøå")
    # rough comparison
    new = [c for c in chars if c not in sd1_chars][:20]
    print(f"  New chars (sample): {new}")
    
    import re
    topics = re.findall(r'=== (.+?) ===', CORPUS)
    print(f"\nTopics: {len(topics)}")
    for t in topics:
        line_count = len([l for l in CORPUS.split('=== ' + t + ' ===')[1].split('===')[0].strip().split('\n') if l.strip()])
        print(f"  - {t} ({line_count} lines)")
