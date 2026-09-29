"""
D'QUILANE v0.9 — Comprehensive Corpus with EOS markers
========================================================
Strategy:
1. EOS token (PAD reused) inserted after each line in pre-training
   => Model learns: "any text completion should end with EOS"
2. More diverse content (more sentences per topic)
3. Common conversation patterns
4. "I don't know" fallbacks for unseen questions

Goal: pre-training loss < 0.7 (vs current 1.38)
"""

CORPUS = """Halo, apa kabar? Aku baik-baik saja, terima kasih.
Senang bertemu denganmu. Selamat datang di D'QUILANE.
Bagaimana harimu? Hariku menyenangkan, terima kasih sudah bertanya.
Selamat pagi. Selamat siang. Selamat sore. Selamat malam.
Apa kabarmu hari ini? Aku sehat dan baik-baik saja.
Senang bisa membantumu. Ada yang bisa aku bantu?
Tentu saja, aku siap membantu. Silakan tanyakan apa saja.
Maaf, aku tidak mengerti pertanyaanmu. Bisa diulang?
Aku belum bisa menjawab itu. Coba tanyakan hal lain.
Tidak tahu pasti, mungkin cek informasi terbaru.
Aku D'QUILANE, LLM buatan Indonesia dari nol.
Namaku D'QUILANE. Aku multibahasa.
Aku model bahasa SD2. Aku bisa matematika dasar.
Aku bisa menjawab pertanyaan sederhana dalam banyak bahasa.
Aku belum bisa bercerita panjang. Tapi aku bisa jawab pertanyaan.
Indonesia adalah negara kepulauan di Asia Tenggara.
Ibu kota Indonesia adalah Jakarta.
Bahasa resmi Indonesia adalah bahasa Indonesia.
Mata uang Indonesia adalah rupiah.
Bendera Indonesia adalah merah putih.
Lagu kebangsaan Indonesia adalah Indonesia Raya.
Penduduk Indonesia sangat beragam suku dan budaya.
Indonesia kaya akan sumber daya alam.
Indonesia terdiri dari ribuan pulau.
English is a global language spoken worldwide.
The capital of Japan is Tokyo.
Paris is the capital of France.
London is the capital of England.
Madrid is the capital of Spain.
Berlin is the capital of Germany.
Roma adalah ibu kota Italia.
Moskow adalah ibu kota Rusia.
Washington DC adalah ibu kota Amerika.
Beijing adalah ibu kota Tiongkok.
Seoul adalah ibu kota Korea Selatan.
Satu tambah satu sama dengan dua.
Dua tambah dua sama dengan empat.
Tiga tambah tiga sama dengan enam.
Lima tambah lima sama dengan sepuluh.
Sepuluh tambah lima sama dengan lima belas.
Satu dikurangi satu sama dengan nol.
Sepuluh dikurangi lima sama dengan lima.
Dua dikali dua sama dengan empat.
Tiga dikali dua sama dengan enam.
Lima dikali lima sama dengan dua puluh lima.
Tiga dikali tiga sama dengan sembilan.
Empat dikali lima sama dengan dua puluh.
Sepuluh dibagi dua sama dengan lima.
Sembilan dibagi tiga sama dengan tiga.
Dua belas dibagi empat sama dengan tiga.
Ada tujuh hari dalam seminggu.
Ada dua belas bulan dalam setahun.
Ada sepuluh jari di dua tangan.
Ada enam puluh menit dalam satu jam.
Ada enam puluh detik dalam satu menit.
Ada dua puluh empat jam dalam satu hari.
Senin Selasa Rabu Kamis Jumat Sabtu Minggu.
Januari Februari Maret April Mei Juni Juli Agustus September Oktober November Desember.
Pagi siang sore malam adalah empat waktu dalam sehari.
Ada tujuh benua di dunia: Asia, Afrika, Eropa, Amerika Utara, Amerika Selatan, Australia, Antartika.
Ada lima samudra di dunia: Pasifik, Atlantik, Hindia, Selatan, Arktik.
Gunung Everest adalah gunung tertinggi di dunia.
Sungai Nil adalah sungai terpanjang di dunia.
Mata uang Indonesia adalah rupiah.
Mata uang Jepang adalah yen.
Mata uang Amerika adalah dolar.
Mata uang Eropa adalah euro.
Mata uang Inggris adalah pound sterling.
Air bisa menjadi tiga bentuk: cair, padat, dan gas.
Air cair adalah air biasa.
Es adalah air padat.
Uap adalah air berbentuk gas.
Matahari memberi cahaya dan panas.
Tanaman butuh air dan sinar matahari untuk tumbuh.
Daun membuat makanan untuk tanaman.
Akar menyerap air dari tanah.
Kucing adalah mamalia.
Burung berkembang biak dengan bertelur.
Ikan hidup di air.
Reptil memiliki sisik.
Kupu-kupu berasal dari ulat.
Apel adalah buah. Pisang adalah buah.
Jeruk adalah buah. Mangga adalah buah.
Singa adalah hewan. Sapi adalah hewan.
Gajah adalah hewan. Kuda adalah hewan.
Mawar adalah bunga. Melati adalah bunga.
Warna langit adalah biru.
Warna rumput adalah hijau.
Warna matahari adalah kuning.
Warna darah adalah merah.
Ayah dan ibu adalah orang tua.
Kakak dan adik adalah saudara.
Kakek dan nenek adalah orang tua dari orang tua.
Paman dan tante adalah saudara orang tua.
Tolong adalah kata sopan untuk meminta bantuan.
Terima kasih adalah ucapan syukur.
Sama-sama adalah balasan untuk terima kasih.
Maaf adalah kata untuk minta permisi.
Satu dua tiga empat lima enam tujuh delapan sembilan sepuluh.
Hello in Indonesian is Halo.
Hello in Spanish is Hola.
Hello in French is Bonjour.
Hello in German is Hallo.
Hello in Italian is Ciao.
Hello in Russian is Privet.
Hello in Japanese is Konnichiwa.
Hello in Korean is Annyeong.
Thank you in Indonesian is Terima kasih.
Thank you in Spanish is Gracias.
Thank you in French is Merci.
Thank you in German is Danke.
Thank you in Japanese is Arigatou.
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
Setelah Senin adalah Selasa.
Setelah Selasa adalah Rabu.
Setelah Rabu adalah Kamis.
Setelah Kamis adalah Jumat.
Setelah Jumat adalah Sabtu.
Setelah Sabtu adalah Minggu.
Setelah Januari adalah Februari.
Setelah Februari adalah Maret.
Lebih besar gajah daripada semut.
Lebih berat batu daripada kapas.
Lebih tinggi gunung daripada pohon.
Lebih kecil semut daripada gajah.
Pola satu dua tiga empat, berikutnya lima.
Pola dua empat enam delapan, berikutnya sepuluh.
Pola lima sepuluh lima belas, berikutnya dua puluh.
Presiden pertama Indonesia adalah Soekarno.
Presiden kedua Indonesia adalah Soeharto.
Bapak proklamasi Indonesia adalah Soekarno dan Hatta.
Wakil presiden pertama Indonesia adalah Hatta.
Pahlawan Indonesia ada banyak, seperti Soekarno, Hatta, Kartini.
Indonesia merdeka pada tahun seribu sembilan ratus empat puluh lima.
Proklamasi kemerdekaan dibacakan pada tanggal tujuh belas agustus.
Tokyo adalah ibu kota Jepang.
Jakarta adalah ibu kota Indonesia.
Paris adalah ibu kota Prancis.
London adalah ibu kota Inggris.
Madrid adalah ibu kota Spanyol.
Berlin adalah ibu kota Jerman.
Roma adalah ibu kota Italia.
Moskow adalah ibu kota Rusia.
Washington DC adalah ibu kota Amerika.
Beijing adalah ibu kota Tiongkok.
Seoul adalah ibu kota Korea Selatan.
Rupiah adalah mata uang Indonesia.
Yen adalah mata uang Jepang.
Dolar adalah mata uang Amerika.
Euro adalah mata uang Eropa.
Pound adalah mata uang Inggris.
Indonesia adalah negara kepulauan di Asia Tenggara.
Jepang adalah negara di Asia Timur.
Prancis adalah negara di Eropa.
Inggris adalah negara di Eropa.
Spanyol adalah negara di Eropa.
Jerman adalah negara di Eropa.
Italia adalah negara di Eropa.
Rusia adalah negara terbesar di dunia.
Amerika adalah negara di benua Amerika Utara.
Tiongkok adalah negara di Asia.
Korea Selatan adalah negara di Asia Timur.
Bahasa Indonesia adalah bahasa resmi Indonesia.
Bahasa Inggris adalah bahasa internasional.
Bahasa Spanyol adalah bahasa di Spanyol.
Bahasa Prancis adalah bahasa di Prancis.
Bahasa Jerman adalah bahasa di Jerman.
Bahasa Italia adalah bahasa di Italia.
Bahasa Rusia adalah bahasa di Rusia.
Bahasa Jepang adalah bahasa di Jepang.
Bahasa Korea adalah bahasa di Korea.
Bahasa Mandarin adalah bahasa di Tiongkok.
Indonesia adalah negara kepulauan.
Ibu kota Indonesia adalah Jakarta.
Bahasa Indonesia adalah bahasa resmi.
Penduduk Indonesia sangat beragam.
Indonesia kaya akan sumber daya alam.
Anak muda harapan masa depan.
Belajar setiap hari bikin hidup bermakna.
Kerja keras kunci pencapaian tujuan.
English is a global language.
The world is connected by technology.
Computers changed how humans communicate.
Artificial intelligence is a global trend.
Large language models are popular now.
Children learn languages quickly.
Education is the key to the future.
Hard work leads to success.
Knowledge is power and wisdom.
El espanol es un idioma hermoso.
La familia es muy importante.
El mundo es grande y diverso.
La educacion es la clave del futuro.
El trabajo duro trae exito.
Le francais est une belle langue.
La famille est tres importante.
Le monde est grand et diversifie.
L education est la cle du futur.
Deutsch ist eine wichtige Sprache.
Die Familie ist sehr wichtig.
Die Welt ist gross und vielfaeltig.
Il italiano e una bella lingua.
La famiglia e molto importante.
Portugues e uma lingua bonita.
A familia e muito importante.
Русский язык использует кириллицу.
Многие страны используют кириллицу.
中文是世界上使用人数最多的语言。
许多人每天说普通话。
日本語には三つの書き方があります。
ひらがなは日本語の単語に使います。
カタカナは外国語の言葉に使います。
漢字は中国語の文字を表します。
한국어는 한글을 사용합니다.
한글은 과학적인 알파벳입니다.
العربية تكتب من اليمين إلى اليسار.
الخط العربي جميل جدا.
हिन्दी देवनागरी लिपि का उपयोग करती है.
বাংলা একটি সুন্দর ভাষা.
ภาษาไทยมีห้าเสียงวรรณยุกต์.
Türkçe Latin harflerini kullanır.
Ελληνικά χρησιμοποιούν το ελληνικό αλφάβητο.
עברית נכתבת מימין לשמאל.
Kiswahili kinazungumzwa Afrika.
Українська використовує кирилицю.
"""

if __name__ == "__main__":
    from pathlib import Path
    out = Path("/home/z/my-project/download/llm/web/corpus_v09.txt")
    out.write_text(CORPUS)
    chars = sorted(set(CORPUS))
    print(f"Corpus v0.9 size: {len(CORPUS)} chars")
    print(f"Unique chars: {len(chars)}")
    print(f"Lines: {len(CORPUS.splitlines())}")
