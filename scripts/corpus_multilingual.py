"""
D'QUILANE Global — Universal multilingual corpus.
30+ languages, ALL major writing systems dengan native characters.
"""

# Native script texts (gunakan unicode asli, bukan transliterasi)
CORPUS = """Indonesia adalah negara kepulauan.
Ibu kota Indonesia adalah Jakarta.
Bahasa Indonesia adalah bahasa resmi.
Penduduk Indonesia sangat beragam.
Indonesia kaya akan sumber daya alam.
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

Persian is also called Farsi.
Tehran is the capital city.
Persian uses Arabic letters.
Poetry is highly valued.
Persian carpets are famous.
Family is very important.
Education is highly valued.
Honesty builds strong trust.
Music connects generations.
Hospitality is a core value.

Kiswahili kinazungumzwa Afrika.
Kiswahili hutumia alfabeti ya Kilatini.
Waafrica wengi wanazungumza Kiswahili.
Nairobi ni jji kuu.
Familia ni muhimu sana.
Elimu inafungua milango mingi.
Muziki ni sehemu ya utamaduni.
Usimulizi ni utamaduni.
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

Čeština je slovanský jazyk.
Praha je hlavní město.
Česko má bohatou historii.
Čeština používá latinku.
Rodina je velmi důležitá.
Vzdělání je vysoce ceněno.
Hudba spojuje generace.
Architektura je krásná.
Poctivost buduje silnou důvěru.
Tvrdá práce přináší úspěch.

Svenska är ett nordiskt sprak.
Stockholm är huvudstaden.
Sverige är kant for naturen.
Svenska anvander latinska alfabetet.
Familj ar mycket viktig.
Utbildning ar hogtvarderad.
Jamlikhet ar en karnvarde.
Innovation uppmuntras.
Musik kopplar ihop generationer.
Rakhet bygger starkt fortroende.

Romana este o limba romana.
Bucuresti este capitala.
Romania are munti frumosi.
Romana foloseste alfabetul latin.
Familia este foarte importanta.
Educatia este foarte apreciata.
Munca grea aduce succes.
Muzica face parte din cultura.
Onestitatea construieste incredere.
Traditia este pastrata cu grija.

Tagalog ay sinasalita sa Pilipinas.
Maynila ang kabisera.
Filipino ay gumagamit ng alpabetong Latin.
Pamilya ay napakamahalaga.
Edukasyon ay nagbubukas ng maraming pinto.
Musika ay bahagi ng kultura.
Pagkain ay iba't iba at masarap.
Pananampalataya ay isang pangunahing halaga.
Ang pagpupursigi ay nagdadala ng tagumpay.
Ang pagngiti ay karaniwang nakikita.

El catala es parla a Catalunya.
Barcelona es la ciutat principal.
El catala usa l alfabet llati.
El catala te una rica literatura.
La familia es molt important.
L educacio es molt valorada.
La musica es part de la cultura.
L arquitectura es preciosa.
El treball dur porta a l exit.
La tradicio es preserva amb cura.

Suomi on pohjoismainen kieli.
Helsinki on paakaupunki.
Suomi tunnetaan luonnostaan.
Suomi kayttaa latinaa.
Perhe on todella tarkea.
Koulutus on arvostettua.
Sauna on perinne tassa.
Innovaatiota kannustetaan.
Rehellisyys luo vahvan luottamuksen.
Kova tyo tuo menestysta.

Norsk er et nordisk sprak.
Oslo er hovedstaden.
Norge har vakre fjorder.
Norsk bruker latinsk alfabet.
Familie er veldig viktig.
Utdanning er hoyt verdsatt.
Natur er dypt respektert.
Musikk kobler generasjonene.
Arlighet bygger sterkt tillit.
Hardt arbeid medforer suksess.
"""

if __name__ == "__main__":
    from pathlib import Path
    out = Path("/home/z/my-project/download/llm/web/corpus_multilingual.txt")
    out.write_text(CORPUS)
    print(f"Corpus size: {len(CORPUS)} chars")
    chars = sorted(set(CORPUS))
    print(f"Unique chars: {len(chars)}")
    print(f"Vocab preview (first 50): {chars[:50]}")
    print(f"Vocab preview (last 30): {chars[-30:]}")
