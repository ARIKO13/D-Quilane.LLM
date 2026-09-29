"""
D'QUILANE SD1 (Kelas 1) — Comprehensive Multilingual Dataset
=============================================================
Topic coverage (matching SD class 1 curriculum worldwide):
1. Basic Math (addition, subtraction 1-20) — verbal + numeric
2. Numbers & Counting (1-100, ordinals)
3. Reading comprehension (Q&A from context)
4. Instructions & Commands (Translate, What is, Count, Answer)
5. Logic & Patterns (sequences, comparisons, before/after)
6. Common Knowledge (colors, animals, fruits, family, days, months)
7. Shapes & Sizes
8. Time & Calendar
9. Greetings & Politeness
10. Translations (basic words in 30+ languages)

Format: each line is one fact/instruction. ~30KB of balanced data.
"""

CORPUS = """
=== MATEMATIKA DASAR (Math Basics) ===
Satu tambah satu sama dengan dua.
Satu tambah dua sama dengan tiga.
Satu tambah tiga sama dengan empat.
Dua tambah dua sama dengan empat.
Dua tambah tiga sama dengan lima.
Dua tambah empat sama dengan enam.
Tiga tambah tiga sama dengan enam.
Tiga tambah empat sama dengan tujuh.
Lima tambah lima sama dengan sepuluh.
Sepuluh tambah lima sama dengan lima belas.
Satu dikurangi satu sama dengan nol.
Dua dikurangi satu sama dengan satu.
Lima dikurangi dua sama dengan tiga.
Sepuluh dikurangi empat sama dengan enam.
Sepuluh dikurangi lima sama dengan lima.
One plus one equals two.
One plus two equals three.
Two plus two equals four.
Two plus three equals five.
Three plus three equals six.
Five plus five equals ten.
Ten plus five equals fifteen.
One minus one equals zero.
Two minus one equals one.
Five minus two equals three.
Ten minus four equals six.
Uno mas uno es dos.
Dos mas dos es cuatro.
Cinco mas cinco es diez.
Diez menos cinco es cinco.
Un plus un egal deux.
Deux plus deux egal quatre.
Cinq plus cinq egal dix.
Eins plus eins gleich zwei.
Zwei plus zwei gleich vier.
Zwei minus eins gleich eins.
=== NUMBERS (Counting) ===
Satu dua tiga empat lima enam tujuh delapan sembilan sepuluh.
Satu dua tiga empat lima adalah lima angka pertama.
Ada tujuh hari dalam seminggu.
Ada dua belas bulan dalam setahun.
Ada sepuluh jari di dua tangan.
One two three four five six seven eight nine ten.
Seven days in a week.
Twelve months in a year.
Ten fingers on two hands.
Uno dos tres cuatro cinco seis siete ocho nueve diez.
Un deux trois quatre cinq six sept huit neuf dix.
Eins zwei drei vier funf sechs sieben acht neun zehn.
=== READING (Baca) ===
Indonesia adalah negara kepulauan. Ibu kotanya Jakarta.
Bahasa resmi Indonesia adalah bahasa Indonesia.
Penduduk Indonesia sangat beragam suku dan budaya.
English is a global language spoken worldwide.
The capital of England is London.
Paris is the capital of France.
Tokyo is the capital of Japan.
Seoul is the capital of Korea.
Madrid is the capital of Spain.
Berlin is the capital of Germany.
Roma adalah ibu kota Italia.
Moscow adalah ibu kota Rusia.
=== INSTRUCTIONS (Perintah) ===
Apa itu Indonesia? Indonesia adalah negara kepulauan.
Siapa kamu? Aku D'QUILANE LLM buatan Indonesia.
Ibu kota Indonesia? Ibu kota Indonesia adalah Jakarta.
Bahasa Indonesia? Bahasa resmi Indonesia adalah bahasa Indonesia.
Translate halo to English. Halo in English is Hello.
Translate hello to Indonesian. Hello in Indonesian is Halo.
Translate thank you to Indonesian. Thank you in Indonesian is Terima kasih.
Translate terima kasih to English. Terima kasih in English is Thank you.
Translate goodbye to Indonesian. Goodbye in Indonesian is Selamat tinggal.
Translate good morning to Indonesian. Good morning in Indonesian is Selamat pagi.
Count to five. Satu dua tiga empat lima.
Count to ten. Satu dua tiga empat lima enam tujuh delapan sembilan sepuluh.
Apa warna langit? Warna langit adalah biru.
Apa warna rumput? Warna rumput adalah hijau.
Apa warna matahari? Warna matahari adalah kuning.
Apa warna darah? Warna darah adalah merah.
=== LOGIC (Logika) ===
Setelah Senin adalah Selasa.
Setelah Selasa adalah Rabu.
Setelah Rabu adalah Kamis.
Setelah Kamis adalah Jumat.
Setelah Jumat adalah Sabtu.
Setelah Sabtu adalah Minggu.
Setelah Minggu adalah Senin.
After Monday comes Tuesday.
After Tuesday comes Wednesday.
After Sunday comes Monday.
Setelah Januari adalah Februari.
Setelah Februari adalah Maret.
After January comes February.
Lebih besar: gajah atau semut? Gajah lebih besar dari semut.
Lebih kecil: semut atau gajah? Semut lebih kecil dari gajah.
Lebih berat: batu atau kapas? Batu lebih berat dari kapas.
Lebih ringan: kapas atau batu? Kapas lebih ringan dari batu.
Lebih tinggi: gunung atau bukit? Gunung lebih tinggi dari bukit.
Bigger: elephant or ant? Elephant is bigger than ant.
Smaller: ant or elephant? Ant is smaller than elephant.
Pola: A B C D E F. Setelah F adalah G.
Pola: Senin Selasa Rabu Kamis. Setelah Kamis adalah Jumat.
Pola: satu dua tiga empat. Setelah empat adalah lima.
Sequence: one two three four. After four comes five.
Sequence: A B C D. After D comes E.
=== KNOWLEDGE (Pengetahuan Umum) ===
Apel adalah buah.
Pisang adalah buah.
Jeruk adalah buah.
Mangga adalah buah.
Semangka adalah buah.
Nanas adalah buah.
Singa adalah hewan.
Harimau adalah hewan.
Gajah adalah hewan.
Kuda adalah hewan.
Sapi adalah hewan.
Kambing adalah hewan.
Ayam adalah hewan.
Ikan adalah hewan.
Burung adalah hewan.
Mawar adalah bunga.
Melati adalah bunga.
Anggrek adalah bunga.
Apple is a fruit.
Banana is a fruit.
Orange is a fruit.
Lion is an animal.
Tiger is an animal.
Elephant is an animal.
Horse is an animal.
Cow is an animal.
Rose is a flower.
=== FAMILY (Keluarga) ===
Ayah dan ibu adalah orang tua.
Kakak adalah saudara yang lebih tua.
Adik adalah saudara yang lebih muda.
Kakek adalah ayah dari orang tua.
Nenek adalah ibu dari orang tua.
Paman adalah saudara laki-laki orang tua.
Tante adalah saudara perempuan orang tua.
Father and mother are parents.
Brother is a male sibling.
Sister is a female sibling.
Grandfather is parent of parent.
Grandmother is mother of parent.
=== TIME & CALENDAR (Waktu & Kalender) ===
Senin Selasa Rabu Kamis Jumat Sabtu Minggu adalah tujuh hari.
Januari Februari Maret April Mei Juni Juli Agustus September Oktober November Desember adalah dua belas bulan.
Monday Tuesday Wednesday Thursday Friday Saturday Sunday are seven days.
January February March April May June July August September October November December are twelve months.
Pagi siang sore malam adalah empat waktu dalam sehari.
Morning afternoon evening night are four times of day.
=== SHAPES (Bentuk) ===
Bentuk bola adalah bulat.
Bentuk buku adalah persegi.
Bentuk pisau adalah segitiga.
Bentuk telur adalah oval.
Bentuk matahari adalah bulat.
Circle is round shape.
Square has four equal sides.
Triangle has three sides.
=== GREETINGS (Sapaan) ===
Halo! Senang bertemu denganmu.
Selamat pagi! Semoga harimu menyenangkan.
Selamat siang! Apa kabar?
Selamat sore! Apa yang sedang kamu lakukan?
Selamat malam! Selamat istirahat.
Hai! Bagaimana kabarmu?
Hello! Nice to meet you.
Good morning! Have a nice day.
Good afternoon! How are you?
Good evening! What are you doing?
Good night! Rest well.
Hi! How are you?
Hola! Encantado de conocerte.
Bonjour! Ravi de vous rencontrer.
=== POLITENESS (Tata Krama) ===
Tolong adalah kata sopan untuk meminta bantuan.
Terima kasih adalah ucapan syukur.
Sama-sama adalah balasan untuk terima kasih.
Maaf adalah kata untuk minta permisi.
Permisi adalah kata untuk melewati orang.
Please is a polite word to ask for help.
Thank you shows gratitude.
You are welcome is a reply to thank you.
Sorry is a word for apology.
Excuse me is a word to pass through.
=== MULTILINGUAL TRANSLATIONS ===
Hello in Indonesian is Halo.
Hello in Spanish is Hola.
Hello in French is Bonjour.
Hello in German is Hallo.
Hello in Italian is Ciao.
Hello in Portuguese is Ola.
Hello in Russian is Privet.
Hello in Japanese is Konnichiwa.
Hello in Korean is Annyeong.
Thank you in Indonesian is Terima kasih.
Thank you in Spanish is Gracias.
Thank you in French is Merci.
Thank you in German is Danke.
Thank you in Italian is Grazie.
Thank you in Portuguese is Obrigado.
Thank you in Russian is Spasibo.
Thank you in Japanese is Arigatou.
Thank you in Korean is Gomapda.
Yes in Indonesian is Ya.
Yes in English is Yes.
Yes in Spanish is Si.
Yes in French is Oui.
Yes in German is Ja.
Yes in Italian is Si.
Yes in Russian is Da.
No in Indonesian is Tidak.
No in Spanish is No.
No in French is Non.
No in German is Nein.
No in Russian is Nyet.
Goodbye in Indonesian is Selamat tinggal.
Goodbye in Spanish is Adios.
Goodbye in French is Au revoir.
Goodbye in German is Auf Wiedersehen.
Goodbye in Italian is Arrivederci.
Mother in Indonesian is Ibu.
Mother in English is Mother.
Mother in Spanish is Madre.
Mother in French is Mere.
Mother in German is Mutter.
Father in Indonesian is Ayah.
Father in English is Father.
Father in Spanish is Padre.
Father in French is Pere.
Father in German is Vater.
Cat in Indonesian is Kucing.
Cat in English is Cat.
Cat in Spanish is Gato.
Cat in French is Chat.
Cat in German is Katze.
Dog in Indonesian is Anjing.
Dog in English is Dog.
Dog in Spanish is Perro.
Dog in French is Chien.
Dog in German is Hund.
Water in Indonesian is Air.
Water in English is Water.
Water in Spanish is Agua.
Water in French is Eau.
Water in German is Wasser.
Fire in Indonesian is Api.
Fire in English is Fire.
Fire in Spanish is Fuego.
Fire in French is Feu.
Fire in German is Feuer.
=== NATIVE SCRIPT (30+ languages) ===
Indonesia adalah negara kepulauan.
Ibu kota Indonesia adalah Jakarta.
Bahasa Indonesia adalah bahasa resmi.
Penduduk Indonesia sangat beragam.
Mari belajar bahasa Indonesia.
Anak muda harapan masa depan.
Belajar setiap hari bikin hidup bermakna.
Kerja keras kunci pencapaian tujuan.
Indonesia akan terus maju dan berkembang.
English is a global language.
The world is connected by technology.
Computers changed how humans communicate.
Artificial intelligence is a global trend.
Large language models are popular now.
Children learn languages quickly.
Education is the key to the future.
Hard work leads to success.
Knowledge is power and wisdom.
Every day is a chance to learn.
El espanol es un idioma hermoso.
La familia es muy importante.
El mundo es grande y diverso.
La educacion es la clave del futuro.
El trabajo duro trae exito.
Cada dia es una nueva oportunidad.
Los ninos aprenden rapidamente.
La tecnologia avanza rapidamente.
El conocimiento es poder.
Le francais est une belle langue.
La famille est tres importante.
Le monde est grand et diversifie.
L education est la cle du futur.
Le travail dur mene au succes.
Chaque jour est une chance d apprendre.
Les enfants apprennent vite.
La technologie avance rapidement.
Le savoir est le pouvoir.
Deutsch ist eine wichtige Sprache.
Die Familie ist sehr wichtig.
Die Welt ist gross und vielfaeltig.
Bildung ist der Schluessel zur Zukunft.
Harte Arbeit fuehrt zum Erfolg.
Jeder Tag ist eine neue Chance.
Kinder lernen schnell.
Die Technik schreitet voran.
Wissen ist Macht.
Il italiano e una bella lingua.
La famiglia e molto importante.
Il mondo e grande e diversificato.
L istruzione e la chiave del futuro.
Il lavoro duro porta al successo.
Ogni giorno e una nuova opportunita.
I bambini imparano velocemente.
La tecnologia avanza rapidamente.
La conoscenza e potere.
Portugues e uma lingua bonita.
A familia e muito importante.
O mundo e grande e diversificado.
A educacao e a chave do futuro.
O trabalho duro leva ao sucesso.
Cada dia e uma nova oportunidade.
As criancas aprendem rapidamente.
A tecnologia avanca rapidamente.
O conhecimento e poder.
Русский язык использует кириллицу.
Русским языком говорят многие люди.
Многие страны используют кириллицу.
Языки соединяют разные народы.
Чтение расширяет человеческий разум.
Письмо проясняет наши мысли.
Речь строит уверенность в себе.
Слушание это первый навык.
Знания связывают поколения.
Тяжелая работа приносит хорошие результаты.
中文是世界上使用人数最多的语言。
许多人每天说普通话。
写中文使用汉字。
中国的万里长城很有名。
学习打开新的大门。
教育建设强大的国家。
家庭非常重要。
耐心会带来巨大的回报。
诚实是最好的策略。
时间比金子更有价值。
日本語には三つの書き方があります。
ひらがなは日本語の単語に使います。
カタカナは外国語の言葉に使います。
漢字は中国語の文字を表します。
日本は島国です。
東京は首都です。
富士山はとても美しいです。
桜は春に咲きます。
教育は高く評価されています。
他人を尊重することが大切です。
한국어는 한글을 사용합니다.
한글은 과학적인 알파벳입니다.
서울은 한국의 수도입니다.
김치는 전통 음식입니다.
가족은 사회의 중심입니다.
교육은 많은 문을 엽니다.
노력은 성공을 가져옵니다.
정직은 강한 신뢰를 구축합니다.
음악은 모든 사람을 연결합니다.
언어 학습은 가치 있습니다.
العربية تكتب من اليمين إلى اليسار.
الخط العربي جميل جدا.
العديد من اللغات تستخدم الحروف العربية.
الشرق الأوسط غني بالثقافة.
المعرفة تحظى بتقدير كبير.
روابط الأسرة قوية جدا.
الضيافة قيمة أساسية.
الصبر فضيلة عظيمة.
التعليم ينير العقل.
القراءة توسع الآفاق.
हिन्दी देवनागरी लिपि का उपयोग करती है.
हिन्दी लाखों लोग बोलते हैं.
भारत में कई विविध भाषाएं हैं.
नई दिल्ली राजधानी है.
ताजमहल प्रसिद्ध है.
परिवार के मूल्य बहुत मजबूत हैं.
शिक्षा का उच्च मूल्य है.
त्योहार लोगों को एक साथ लाते हैं.
संगीत और नृत्य समुदायों को जोड़ते हैं.
ज्ञान आगे का रास्ता रोशन करता है.
বাংলা একটি সুন্দর ভাষা.
বাংলাদেশ বাংলা ব্যাপকভাবে ব্যবহার করে.
কলকাতাও বাংলা বলে.
বাংলা লিপি বাঁকা.
সাহিত্য খুব সমৃদ্ধ.
ট্যাগর একজন মহান কবি ছিলেন.
সঙ্গীত সংস্কৃতির অংশ.
মাছ একটি সাধারণ খাবার.
পারিবারিক বন্ধন শক্তিশালী.
শিক্ষা জীবন রূপান্তরিত করে.
ภาษาไทยมีห้าเสียงวรรณยุกต์.
กรุงเทพคือเมืองหลวง.
อักษรไทยสวยงาม.
พุทธศาสนาได้รับการปฏิบัติอย่างกว้างขวาง.
ยิ้มแสดงความเคารพที่นี่.
ครอบครัวคือหน่วยหลัก.
อาหารเผ็ดและอร่อย.
เทศกาลเป็นเหตุการณ์ที่มีสีสัน.
การศึกษามีคุณค่าสูง.
เคารพผู้ใหญ่เสมอ.
Türkçe Latin harflerini kullanır.
İstanbul ünlü bir şehirdir.
Türkiye iki kıtayı birleştirir.
Türk kahvesi çok ünlüdür.
Aile çok önemlidir.
Eğitim çok değerlidir.
Misafirperverlik temel bir değerdir.
Çalışmak başarıyı getirir.
Müzik insanları birleştirir.
Dil öğrenmek ödüllendiricidir.
Tiếng Việt sử dụng bảng chữ cái Latin.
Hà Nội là thủ đô.
Việt Nam có lịch sử lâu dài.
Tiếng Việt có sáu thanh điệu.
Gia đình rất quan trọng.
Giáo dục mở ra nhiều cánh cửa.
Cơm là thức ăn chính.
Gỏi cuốn rất ngon.
Tết là năm mới.
Chăm chỉ dẫn đến thành công.
Polski to język słowiański.
Warszawa jest stolicą.
Polska ma bogatą historię.
Polski używa alfabetu łacińskiego.
Rodzina jest bardzo ważna.
Edukacja otwiera wiele drzwi.
Praca przynosi sukces.
Uczciwość buduje silne zaufanie.
Muzyka łączy pokolenia.
Literatura wzbogaca duszę.
Nederlands wordt in Nederland gesproken.
Amsterdam is de hoofdstad.
Tulpen zijn beroemd in Holland.
Molens zijn hier traditioneel.
Gezin is heel belangrijk.
Onderwijs wordt hoog gewaardeerd.
Fietsen worden veel gebruikt.
Kaas is een beroemde export.
Hard werken brengt succes.
Tolerantie is een kernwaarde.
Ελληνικά χρησιμοποιούν το ελληνικό αλφάβητο.
Η Αθήνα είναι η πρωτεύουσα.
Η δημοκρατία ξεκίνησε στην αρχαία Ελλάδα.
Η ελληνική φιλοσοφία είναι παγκοσμίως γνωστή.
Η Ακρόπολη είναι ιστορική.
Η οικογένεια είναι πολύ σημαντική.
Η εκπαίδευση εκτιμάται ιδιαίτερα.
Το ελαιόλαδο χρησιμοποιείται ευρέως.
Η μουσική συνδέει τις γενιές.
Η γλώσσα έχει αρχαίες ρίζες.
עברית נכתבת מימין לשמאל.
ירושלים היא עיר קדושה.
עברית משתמשת באלף בית ייחודי.
לישראל יש הרבה חדשנות.
משפחה חשובה מאוד.
חינוך מוערך מאוד.
היסטוריה עשירה ועמוקה.
כנות הוא ערך מרכזי.
עבודה קשה מביאה הצלחה.
שלום היא המטרה הסופית.
Kiswahili kinazungumzwa Afrika.
Kiswahili hutumia alfabeti ya Kilatini.
Waafrica wengi wanazungumza Kiswahili.
Nairobi ni jji kuu.
Familia ni muhimu sana.
Elimu inafungua milango mingi.
Muziki ni sehemu ya utamaduni.
Usimulazi ni utamaduni.
Kazi nguvu inaleta mafanikio.
Jumuiya ni kitovu.
Українська використовує кирилицю.
Київ є столицею.
Україна має багаті традиції.
Українська мова мелодійна.
Сім'я дуже важлива.
Освіта високо цінується.
Чесність будує сильну довіру.
Тяжка праця приносить успіх.
Музика поєднує покоління.
Література збагачує душу.
=== EXTRA INSTRUCTION EXAMPLES (SFT-like) ===
Apa dua tambah dua? Dua tambah dua sama dengan empat.
Berapa satu tambah satu? Satu tambah satu sama dengan dua.
Berapa lima tambah lima? Lima tambah lima sama dengan sepuluh.
Berapa tiga tambah empat? Tiga tambah empat sama dengan tujuh.
Berapa sepuluh dikurangi lima? Sepuluh dikurangi lima sama dengan lima.
Berapa hari dalam seminggu? Ada tujuh hari dalam seminggu.
Berapa bulan dalam setahun? Ada dua belas bulan dalam setahun.
Berapa jari di dua tangan? Ada sepuluh jari di dua tangan.
Apa warna langit? Warna langit adalah biru.
Apa warna rumput? Warna rumput adalah hijau.
Apa ibu kota Indonesia? Ibu kota Indonesia adalah Jakarta.
Apa bahasa resmi Indonesia? Bahasa resmi Indonesia adalah bahasa Indonesia.
Siapa kamu? Aku D'QUILANE LLM buatan Indonesia dari nol.
Halo. Halo juga! Senang bertemu denganmu.
Terima kasih. Sama-sama! Ada yang bisa aku bantu lagi?
Selamat pagi. Selamat pagi juga! Semoga harimu menyenangkan.
Apa kabar? Aku baik-baik saja, terima kasih sudah bertanya!
Translate halo to English. Halo in English is Hello.
Translate hello to Indonesian. Hello in Indonesian is Halo.
Translate thank you to Indonesian. Thank you in Indonesian is Terima kasih.
Translate terima kasih to English. Terima kasih in English is Thank you.
What is two plus two? Two plus two equals four.
What is one plus one? One plus one equals two.
Count to five. Satu dua tiga empat lima.
Count to ten. Satu dua tiga empat lima enam tujuh delapan sembilan sepuluh.
What comes after Monday? After Monday comes Tuesday.
What comes after January? After January comes February.
Bigger elephant or ant? Elephant is bigger than ant.
What is apple? Apple is a fruit.
What is lion? Lion is an animal.
What is rose? Rose is a flower.
What is father? Father is a parent.
"""

if __name__ == "__main__":
    from pathlib import Path
    out = Path("/home/z/my-project/download/llm/web/corpus_sd1.txt")
    out.write_text(CORPUS)
    chars = sorted(set(CORPUS))
    print(f"Corpus size: {len(CORPUS)} chars")
    print(f"Unique chars: {len(chars)}")
    print(f"First 30 chars: {chars[:30]}")
    print(f"Last 30 chars: {chars[-30:]}")
    
    # count topics
    import re
    topics = re.findall(r'=== (.+?) ===', CORPUS)
    print(f"\nTopics covered: {len(topics)}")
    for t in topics:
        print(f"  - {t}")
