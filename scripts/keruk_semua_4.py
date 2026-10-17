"""
D'QUILANE — KERUK PART 4: ABSOLUTELY EVERYTHING ELSE
"""
import json, random
from pathlib import Path

random.seed(42)
items = []

def add(cat, judul, sub, q, a):
    items.append({'input': q, 'output': a, 'bidang': cat, 'judul': judul, 'subjudul': sub})

# ============================================================
# 1. ART MOVEMENTS (20+)
# ============================================================
ART_MOV = [
    ('What is the Renaissance?','The Renaissance was a period of cultural rebirth in Europe (14th-17th century).'),
    ('What is Impressionism?','Impressionism focuses on light and moment, like Monet and Renoir.'),
    ('What is Cubism?','Cubism breaks objects into geometric shapes, pioneered by Picasso and Braque.'),
    ('What is Surrealism?','Surrealism depicts dream-like scenes, like Dali and Magritte.'),
    ('What is Pop Art?','Pop Art uses popular culture imagery, like Andy Warhol and Roy Lichtenstein.'),
    ('What is Baroque?','Baroque is dramatic, ornate art from the 17th century, like Bernini and Caravaggio.'),
    ('What is Realism?','Realism depicts everyday life accurately, like Courbet and Millet.'),
    ('What is Abstract Art?','Abstract art uses shapes and colors instead of realistic images, like Kandinsky.'),
    ('What is Romanticism?','Romanticism emphasizes emotion and nature, like Turner and Delacroix.'),
    ('What is Expressionism?','Expressionism shows inner feelings rather than reality, like Munch and Kirchner.'),
    ('What is Minimalism?','Minimalism uses simple forms and few elements.'),
    ('What is Pointillism?','Pointillism uses small dots of color, like Seurat.'),
    ('What is Art Nouveau?','Art Nouveau uses flowing, organic lines and decorative patterns.'),
    ('What is Gothic art?','Gothic art features pointed arches and tall windows, common in cathedrals.'),
    ('What is Neoclassicism?','Neoclassicism revives ancient Greek and Roman styles.'),
    ('What is Fauvism?','Fauvism uses wild, unnatural colors, like Matisse.'),
    ('What is Dadaism?','Dadaism is anti-art, rejecting traditional artistic values.'),
    ('What is Conceptual Art?','Conceptual Art values the idea over the visual artwork.'),
    ('What is Street Art?','Street Art is art in public spaces, like Banksy and graffiti.'),
    ('What is Photography as art?','Photography captures real moments as artistic images.'),
]
for q,a in ART_MOV:
    add('culture','Art Movements','General',q,a)

# ============================================================
# 2. FAMOUS SONGS & COMPOSITIONS (30+)
# ============================================================
SONGS = [
    ('What is Beethoven\'s 5th Symphony?','Beethoven\'s 5th Symphony opens with the famous "da-da-da-dum" motif.'),
    ('What is Beethoven\'s 9th Symphony?','Beethoven\'s 9th includes "Ode to Joy," now the EU anthem.'),
    ('What is Mozart\'s Eine kleine Nachtmusik?','It is a famous serenade by Mozart.'),
    ('What is Vivaldi\'s Four Seasons?','Four Seasons is a set of violin concertos for spring, summer, autumn, winter.'),
    ('What is Tchaikovsky\'s Swan Lake?','Swan Lake is a famous ballet by Tchaikovsky.'),
    ('What is Tchaikovsky\'s Nutcracker?','The Nutcracker is a famous Christmas ballet by Tchaikovsky.'),
    ('What is The Beatles\' Hey Jude?','Hey Jude is one of The Beatles\' most famous songs.'),
    ('What is Queen\'s Bohemian Rhapsody?','Bohemian Rhapsody is a complex rock opera by Queen.'),
    ('What is John Lennon\'s Imagine?','Imagine is a famous song about world peace.'),
    ('What is Michael Jackson\'s Thriller?','Thriller is the best-selling album of all time.'),
    ('What is Bob Marley\'s Redemption Song?','Redemption Song is a famous reggae song about freedom.'),
    ('What is Elvis\'s Jailhouse Rock?','Jailhouse Rock is a famous rock and roll song by Elvis Presley.'),
    ('What is The Star-Spangled Banner?','The Star-Spangled Banner is the US national anthem.'),
    ('What is O Canada?','O Canada is the Canadian national anthem.'),
    ('What is God Save the King?','God Save the King is the UK national anthem.'),
    ('What is La Marseillaise?','La Marseillaise is the French national anthem.'),
    ('What is Advance Australia Fair?','Advance Australia Fair is the Australian national anthem.'),
    ('What is Kimigayo?','Kimigayo is the Japanese national anthem.'),
    ('What is Arirang?','Arirang is a traditional Korean folk song.'),
    ('What is Jana Gana Mana?','Jana Gana Mana is the Indian national anthem.'),
    ('What is Garuda Pancasila?','Garuda Pancasila is a Indonesian patriotic song about the national symbol.'),
    ('What is Indonesia Raya?','Indonesia Raya is the Indonesian national anthem, by W.R. Supratman.'),
    ('What is Auld Lang Syne?','Auld Lang Syne is a Scottish song sung at New Year.'),
    ('What is Happy Birthday?','Happy Birthday is the most recognized song in English.'),
    ('What is Twinkle Twinkle Little Star?','Twinkle Twinkle is a children\'s nursery rhyme.'),
    ('What is Row Row Row Your Boat?','Row Row Row Your Boat is a children\'s song that can be sung in rounds.'),
    ('What is The Alphabet Song?','The Alphabet Song teaches ABCs to the tune of Twinkle Twinkle.'),
    ('What is Old MacDonald Had a Farm?','Old MacDonald is a children\'s song about farm animals.'),
    ('What is London Bridge is Falling Down?','London Bridge is a traditional nursery rhyme.'),
    ('What is Frere Jacques?','Frere Jacques is a French nursery rhyme about a sleeping monk.'),
    ('What is Baby Shark?','Baby Shark is a viral children\'s song from Pinkfong.'),
]
for q,a in SONGS:
    add('culture','Famous Songs','World',q,a)

# ============================================================
# 3. BAHASA DAERAH INDONESIA (50+)
# ============================================================
DAERAH = [
    # Jawa
    ('Apa itu sopo ing basa Jawa?','Sopo artinya siapa dalam bahasa Jawa.'),
    ('Apa iki ing basa Jawa?','Iki artinya ini dalam bahasa Jawa.'),
    ('Apa iku ing basa Jawa?','Iku artinya itu dalam bahasa Jawa.'),
    ('Apa nggih ing basa Jawa?','Nggih artinya ya dalam bahasa Jawa (halus).'),
    ('Apa mboten ing basa Jawa?','Mboten artinya tidak dalam bahasa Jawa (halus).'),
    ('Apa nopo ing basa Jawa?','Nopo artinya apa dalam bahasa Jawa.'),
    ('Apa nyuwun pangapunten ing basa Jawa?','Nyuwun pangapunten artinya mohon maaf dalam bahasa Jawa (halus).'),
    ('Apa matur nuwun ing basa Jawa?','Matur nuwun artinya terima kasih dalam bahasa Jawa (halus).'),
    # Sunda
    ('Naon arti naon dina basa Sunda?','Naon artinya apa dalam bahasa Sunda.'),
    ('Naon arti ieu dina basa Sunda?','Ieu artinya ini dalam bahasa Sunda.'),
    ('Naon arti eta dina basa Sunda?','Eta artinya itu dalam bahasa Sunda.'),
    ('Naon arti hoyong dina basa Sunda?','Hoyong artinya mau/ingin dalam bahasa Sunda.'),
    ('Naon arti teu dina basa Sunda?','Teu artinya tidak dalam bahasa Sunda.'),
    ('Naon arti mangga dina basa Sunda?','Mangga artinya silakan dalam bahasa Sunda.'),
    ('Naon arti hatur nuhun dina basa Sunda?','Hatur nuhun artinya terima kasih dalam bahasa Sunda.'),
    ('Naon arti punten dina basa Sunda?','Punten artinya maaf dalam bahasa Sunda.'),
    # Batak
    ('Apa artihoras dalam bahasa Batak?','Horas artinya selamat/sehat dalam bahasa Batak.'),
    ('Apa arti inda dalam bahasa Batak?','Inda artinya tidak dalam bahasa Batak.'),
    ('Apa arti au dalam bahasa Batak?','Au artinya saya dalam bahasa Batak.'),
    ('Apa arti ho dalam bahasa Batak?','Ho artinya kamu dalam bahasa Batak.'),
    # Minang
    ('Apa artiak dalam bahasa Minang?','Amak artinya ibu dalam bahasa Minang.'),
    ('Apa arti apak dalam bahasa Minang?','Apak artinya ayah dalam bahasa Minang.'),
    ('Apa arti den dalam bahasa Minang?','Den artinya saya (laki-laki) dalam bahasa Minang.'),
    ('Apa arti undo dalam bahasa Minang?','Undo artinya saya (perempuan) dalam bahasa Minang.'),
    # Bali
    ('Apa arti om swastiastu dalam bahasa Bali?','Om swastiastu adalah sapaan dalam bahasa Bali.'),
    ('Apa arti suksma dalam bahasa Bali?','Suksma artinya terima kasih dalam bahasa Bali.'),
    ('Apa arti ten within bahasa Bali?','Ten artinya tidak dalam bahasa Bali.'),
    # Bugis/Makassar
    ('Apa arti salamatu dalam bahasa Bugis?','Salamatu artinya selamat dalam bahasa Bugis.'),
    ('Apa arti TENRITENRITEN within bahasa Bugis?','Tenri artinya tidak dalam bahasa Bugis.'),
    ('Apa arti assalamualaikum dalam bahasa Makassar?','Assalamualaikum adalah sapaan dalam bahasa Makassar.'),
    # Lainnya
    ('Apa bahasa daerah Aceh?','Bahasa Aceh adalah bahasa daerah di Aceh.'),
    ('Apa bahasa daerah Padang?','Bahasa Minang adalah bahasa daerah di Padang.'),
    ('Apa bahasa daerah Bali?','Bahasa Bali adalah bahasa daerah di Bali.'),
    ('Apa bahasa daerah Sunda?','Bahasa Sunda adalah bahasa daerah di Jawa Barat.'),
    ('Apa bahasa daerah Jawa?','Bahasa Jawa adalah bahasa daerah di Jawa Tengah dan Jawa Timur.'),
    ('Apa bahasa daerah Batak?','Bahasa Batak adalah bahasa daerah di Sumatra Utara.'),
    ('Apa bahasa daerah Banjar?','Bahasa Banjar adalah bahasa daerah di Kalimantan Selatan.'),
    ('Apa bahasa daerah Bugis?','Bahasa Bugis adalah bahasa daerah di Sulawesi Selatan.'),
    ('Apa bahasa daerah Makassar?','Bahasa Makassar adalah bahasa daerah di Sulawesi Selatan.'),
    ('Apa bahasa daerah Sasak?','Bahasa Sasak adalah bahasa daerah di Lombok.'),
    ('Apa bahasa daerah Lampung?','Bahasa Lampung adalah bahasa daerah di Lampung.'),
    ('Apa bahasa daerah Rejang?','Bahasa Rejang adalah bahasa daerah di Bengkulu.'),
    ('Apa bahasa daerah Toraja?','Bahasa Toraja adalah bahasa daerah di Sulawesi Selatan.'),
    ('Apa bahasa daerah Manado?','Bahasa Manado adalah bahasa daerah di Sulawesi Utara.'),
    ('Apa bahasa daerah Ambon?','Bahasa Ambon adalah bahasa daerah di Maluku.'),
    ('Apa bahasa daerah Papua?','Bahasa Papua memiliki ratusan bahasa daerah.'),
    ('Apa bahasa daerah Nias?','Bahasa Nias adalah bahasa daerah di Sumatra Utara.'),
    ('Apa bahasa daerah Madura?','Bahasa Madura adalah bahasa daerah di Madura, Jawa Timur.'),
    ('Apa bahasa daerah Lombok?','Bahasa Sasak adalah bahasa daerah di Lombok.'),
    ('Apa bahasa daerah Timor?','Bahasa Uab Meto adalah bahasa daerah di NTT.'),
]
for q,a in DAERAH:
    add('bahasa','Bahasa Daerah','Indonesia',q,a)

# ============================================================
# 4. MULTITURN DIALOGUES (30+)
# ============================================================
DIALOG = [
    ('What should I do if I\'m lost?','Stay calm, find a safe place, ask a trusted adult or police for help, or call your family.'),
    ('What should I do if there\'s a fire?','Get out immediately, stay low to avoid smoke, call 911 or 112, don\'t use elevators.'),
    ('What should I do if I\'m being bullied?','Tell a trusted adult (parent, teacher, counselor). You are not alone and it\'s not your fault.'),
    ('What should I do if I feel sad?','Talk to someone you trust, do something you enjoy, get enough sleep, and exercise.'),
    ('What should I do if I find something online that makes me uncomfortable?','Tell a trusted adult immediately. Don\'t share personal information with strangers online.'),
    ('What should I do before crossing the street?','Stop, look left, look right, look left again, then cross when it\'s safe.'),
    ('What should I do if I\'m home alone?','Keep doors locked, don\'t open to strangers, have emergency numbers ready.'),
    ('What should I do if I get a bad grade?','Don\'t give up. Ask the teacher for help, study more, and try again next time.'),
    ('What should I do if I\'m stressed about a test?','Study in advance, get enough sleep, eat breakfast, take deep breaths, and do your best.'),
    ('What should I do if I disagree with a friend?','Talk calmly, listen to their side, find a compromise, and respect their opinion.'),
    ('What should I do if I see someone cheating?','It\'s not fair. Tell the teacher privately. Cheating hurts everyone.'),
    ('What should I do if I make a mistake?','Admit it, apologize, learn from it, and try to do better next time.'),
    ('What should I do if I\'m hungry at school?','Eat a snack during break. If you don\'t have food, tell a teacher.'),
    ('What should I do if I lose something?','Retrace your steps, ask people if they\'ve seen it, check lost and found.'),
    ('What should I do if I feel sick at school?','Tell the teacher immediately and go to the school nurse or office.'),
    ('What should I do if someone is mean to me online?','Don\'t respond, take a screenshot, block them, and tell a trusted adult.'),
    ('What should I do if I\'m worried about something?','Talk to someone you trust (parent, teacher, friend). Sharing helps.'),
    ('What should I do if I want to learn a new skill?','Start small, practice every day, be patient, and don\'t give up.'),
    ('What should I do if I\'m bored?','Read a book, play a game, go outside, help someone, or learn something new.'),
    ('What should I do if I\'m angry?','Take deep breaths, count to 10, walk away, talk to someone, or exercise.'),
    ('What should I do if I see a stranger following me?','Go to a safe place with people, tell an adult, don\'t go home if they\'re following.'),
    ('What should I do if I find a lost pet?','Check for a collar tag, ask neighbors, take it to a vet or animal shelter.'),
    ('What should I do if I want to help the environment?','Reduce, reuse, recycle, save water, turn off lights, plant trees.'),
    ('What should I do if I have too much homework?','Make a schedule, do the hardest first, take breaks, and ask for help if needed.'),
    ('What should I do if I\'m being cyberbullied?','Save evidence, block the bully, tell an adult, and report to the platform.'),
    ('What should I do if I feel lonely?','Reach out to a friend, join a club, talk to family, or volunteer.'),
    ('What should I do if I want to make new friends?','Be friendly, join activities, share interests, and be yourself.'),
    ('What should I do if I\'m scared of the dark?','Use a night light, keep a flashlight nearby, and remember you\'re safe.'),
    ('What should I do if I want to save money?','Set a goal, spend less than you earn, avoid impulse buying, and save regularly.'),
    ('What should I do if I want to be healthy?','Eat balanced meals, exercise daily, sleep 8 hours, drink water, and be happy.'),
]
for q,a in DIALOG:
    add('conversations','Life Advice','Situations',q,a)

# ============================================================
# 5. WORLD CAPITALS - REMAINING COUNTRIES (30+)
# ============================================================
MORE_CAPS = [
    ('Antigua and Barbuda','Saint John\'s'),('Bahamas','Nassau'),
    ('Barbados','Bridgetown'),('Dominica','Roseau'),
    ('Grenada','St. George\'s'),('Saint Kitts and Nevis','Basseterre'),
    ('Saint Lucia','Castries'),('Saint Vincent','Kingstown'),
    ('Trinidad and Tobago','Port of Spain'),('Suriname','Paramaribo'),
    ('Guyana','Georgetown'),('Belize','Belmopan'),
    ('Vanuatu','Port Vila'),('Solomon Islands','Honiara'),
    ('Samoa','Apia'),('Tonga','Nuku\'alofa'),
    ('Kiribati','South Tarawa'),('Tuvalu','Funafuti'),
    ('Palau','Ngerulmud'),('Marshall Islands','Majuro'),
    ('Micronesia','Palikir'),('Nauru','Yaren'),
    ('Andorra','Andorra la Vella'),('Monaco','Monaco'),
    ('Liechtenstein','Vaduz'),('San Marino','San Marino'),
    ('Vatican City','Vatican City'),('Malta','Valletta'),
    ('Montenegro','Podgorica'),('Kosovo','Pristina'),
]
for country, capital in MORE_CAPS:
    add('geography','World Capitals','Remaining',
        f'What is the capital of {country}?', f'The capital of {country} is {capital}.')

# ============================================================
# 6. NATIONAL BIRDS & SPORTS (30+)
# ============================================================
NAT_BIRD = [
    ('What is the national bird of the US?','The bald eagle is the national bird of the US.'),
    ('What is the national bird of India?','The peacock is the national bird of India.'),
    ('What is the national bird of Japan?','The green pheasant is the national bird of Japan.'),
    ('What is the national bird of the UK?','The European robin is the national bird of the UK.'),
    ('What is the national bird of Indonesia?','The Javan hawk-eagle is considered the national bird of Indonesia.'),
    ('What is the national sport of India?','Hockey is the national sport of India.'),
    ('What is the national sport of the US?','Baseball is considered the national sport of the US.'),
    ('What is the national sport of Japan?','Sumo is the national sport of Japan.'),
    ('What is the national sport of China?','Table tennis is the national sport of China.'),
    ('What is the national sport of Bangladesh?','Cricket is the national sport of Bangladesh.'),
    ('What is the national sport of Thailand?','Muay Thai (kickboxing) is the national sport of Thailand.'),
    ('What is the national sport of South Korea?','Taekwondo is the national sport of South Korea.'),
    ('What is the national sport of Canada?','Lacrosse and ice hockey are national sports of Canada.'),
    ('What is the national sport of Australia?','Cricket is the national sport of Australia.'),
    ('What is the national sport of Spain?','Football (soccer) is the national sport of Spain.'),
    ('What is the national sport of Brazil?','Football (soccer) is the national sport of Brazil.'),
    ('What is the national sport of Pakistan?','Field hockey is the national sport of Pakistan.'),
    ('What is the national sport of Sri Lanka?','Volleyball is the national sport of Sri Lanka.'),
]
for q,a in NAT_BIRD:
    add('geography','National Symbols','Birds and Sports',q,a)

# ============================================================
# 7. FAMOUS ATHLETES (30+)
# ============================================================
ATHLETES = [
    ('Who is Usain Bolt?','Usain Bolt is the fastest man in the world, holding the 100m record.'),
    ('Who is Michael Jordan?','Michael Jordan is considered the greatest basketball player of all time.'),
    ('Who is Lionel Messi?','Messi is an Argentine soccer player, one of the greatest of all time.'),
    ('Who is Cristiano Ronaldo?','Ronaldo is a Portuguese soccer player, one of the greatest of all time.'),
    ('Who is Pelé?','Pelé is a Brazilian soccer legend, the only player to win 3 World Cups.'),
    ('Who is Muhammad Ali?','Muhammad Ali was a boxing legend and civil rights activist.'),
    ('Who is Serena Williams?','Serena Williams is one of the greatest tennis players of all time.'),
    ('Who is Roger Federer?','Federer is a Swiss tennis player with 20 Grand Slam titles.'),
    ('Who is Rafael Nadal?','Nadal is a Spanish tennis player known as the King of Clay.'),
    ('Who is Novak Djokovic?','Djokovic is a Serbian tennis player with the most Grand Slam titles.'),
    ('Who is Michael Phelps?','Phelps is the most decorated Olympian with 28 medals.'),
    ('Who is Simone Biles?','Simone Biles is an American gymnast with 7 Olympic medals.'),
    ('Who is Sachin Tendulkar?','Tendulkar is an Indian cricket legend, the highest run-scorer.'),
    ('Who is Tiger Woods?','Tiger Woods is one of the greatest golfers of all time.'),
    ('Who is LeBron James?','LeBron James is an American basketball superstar.'),
    ('Who is Kobe Bryant?','Kobe Bryant was an American basketball legend who played for the Lakers.'),
    ('Who is Tom Brady?','Tom Brady is considered the greatest NFL quarterback.'),
    ('Who is Neymar?','Neymar is a Brazilian soccer star.'),
    ('Who is Mbappé?','Mbappé is a French soccer star who won the World Cup in 2018.'),
    ('Who is Steph Curry?','Stephen Curry is an American basketball player known for three-pointers.'),
    ('Who is Manny Pacquiao?','Pacquiao is a Filipino boxing champion and politician.'),
    ('Who is Virat Kohli?','Kohli is an Indian cricket captain and one of the best batters.'),
    ('Who is Lewis Hamilton?','Hamilton is a British Formula 1 driver with 7 world championships.'),
    ('Who is Max Verstappen?','Verstappen is a Dutch Formula 1 world champion.'),
    ('Who is Naomi Osaka?','Osaka is a Japanese tennis player and Grand Slam winner.'),
    ('Who is Sun Yang?','Sun Yang is a Chinese Olympic swimmer.'),
    ('Who is Eliud Kipchoge?','Kipchoge is a Kenyan marathon runner, the first to run under 2 hours.'),
    ('Who is Simone Biles?','Biles is an American gymnast, considered the greatest of all time.'),
    ('Who is Caeleb Dressel?','Dressel is an American Olympic swimmer with multiple gold medals.'),
    ('Who is Kylian Mbappé?','Mbappé is a French soccer star, World Cup winner in 2018.'),
]
for q,a in ATHLETES:
    add('PJOK','Famous Athletes','World',q,a)

# ============================================================
# 8. MORE FAIRY TALES & FABLES (40+)
# ============================================================
FAIRY = [
    ('What is the story of Goldilocks?','Goldilocks enters a house of three bears and tries their things.'),
    ('What is the story of Jack and the Beanstalk?','Jack climbs a magic beanstalk and finds a giant\'s castle.'),
    ('What is the story of Beauty and the Beast?','A girl falls in love with a prince cursed to look like a beast.'),
    ('What is the story of Aladdin?','Aladdin finds a magic lamp with a genie who grants wishes.'),
    ('What is the story of Pinocchio?','Pinocchio is a wooden puppet whose nose grows when he lies.'),
    ('What is the story of Rumpelstiltskin?','A magical creature helps a girl spin straw into gold.'),
    ('What is the story of Rapunzel?','Rapunzel is locked in a tower and lets down her long hair.'),
    ('What is the story of The Ugly Duckling?','An ugly duckling grows up to be a beautiful swan.'),
    ('What is the story of The Emperor\'s New Clothes?','An emperor is tricked into wearing invisible clothes.'),
    ('What is the story of The Princess and the Pea?','A princess feels a pea under many mattresses.'),
    ('What is the story of Puss in Boots?','A clever cat helps his poor master become a prince.'),
    ('What is the story of The Frog Prince?','A frog becomes a prince after a princess kisses him.'),
    ('What is the story of The Little Mermaid?','A mermaid gives up her voice for a human prince.'),
    ('What is the story of Peter Pan?','Peter Pan is a boy who never grows up in Neverland.'),
    ('What is the story of Robin Hood?','Robin Hood steals from the rich to give to the poor.'),
    # Aesop fables
    ('What is the moral of The Tortoise and the Hare?','Slow and steady wins the race.'),
    ('What is the moral of The Boy Who Cried Wolf?','Liars are not believed even when they tell the truth.'),
    ('What is the moral of The Ant and the Grasshopper?','Prepare for the future while you can.'),
    ('What is the moral of The Lion and the Mouse?','No act of kindness is ever wasted.'),
    ('What is the moral of The Fox and the Grapes?','It is easy to despise what you cannot have.'),
    ('What is the moral of The Crow and the Pitcher?','Necessity is the mother of invention.'),
    ('What is the moral of The Goose That Laid Golden Eggs?','Greed destroys the source of good fortune.'),
    ('What is the moral of The Wolf in Sheep\'s Clothing?','Appearances can be deceiving.'),
    ('What is the moral of The Frog and the Ox?','Don\'t try to be something you\'re not.'),
    ('What is the moral of The Wind and the Sun?','Gentleness is stronger than force.'),
    ('What is the moral of The Crab and His Mother?','Example is better than precept.'),
    ('What is the moral of The Dog and His Shadow?','Be content with what you have.'),
    ('What is the moral of The Stag at the Pool?','We often despise what is most useful to us.'),
    ('What is the moral of Belling the Cat?','It is easier said than done.'),
    ('What is the moral of The Fox and the Crow?\'','Don\'t trust flatterers.'),
    ('What is the moral of The Town Mouse and Country Mouse?','Better simple and safe than luxurious and dangerous.'),
    ('What is the moral of The Wolf and the Crane?','You get nothing for helping the wicked.'),
    ('What is the moral of The Farmer and the Snake?','Kindness to evil is wasted.'),
    ('What is the moral of The Mouse and the Weasel?','Don\'t be too greedy.'),
    ('What is the moral of The Two Pots?','In adversity, the weak suffer more.'),
    ('What is the moral of The Wolf and the Lamb?','The tyrant will find a pretext for tyranny.'),
    ('What is the moral of The Oak and the Reed?','It is better to bend than to break.'),
    ('What is the moral of The Mice in Council?','Talk is cheap without action.'),
    ('What is the moral of The Fox Without a Tail?','Don\'t trust advice from those who benefit from it.'),
    ('What is the moral of The Eagle and the Arrow?','We often give our enemies the means for our own destruction.'),
    ('What is the moral of The Thief and His Mother?','A small fault uncorrected becomes a great crime.'),
]
for q,a in FAIRY:
    add('culture','Fairy Tales and Fables','World',q,a)

# ============================================================
# 9. CLIMATE ZONES & WEATHER INSTRUMENTS (30+)
# ============================================================
CLIMATE = [
    ('What are the major climate zones?','Tropical, temperate, and polar are the major climate zones.'),
    ('What is a tropical climate?','A tropical climate is hot and humid all year, near the equator.'),
    ('What is a temperate climate?','A temperate climate has four seasons: spring, summer, autumn, winter.'),
    ('What is a polar climate?','A polar climate is extremely cold all year, near the poles.'),
    ('What is a desert climate?','A desert climate is very dry with little rain.'),
    ('What is a Mediterranean climate?','A Mediterranean climate has hot dry summers and mild wet winters.'),
    ('What is a monsoon climate?','A monsoon climate has seasonal winds that bring heavy rain.'),
    ('What is a tundra climate?','A tundra climate is cold with permafrost (permanently frozen ground).'),
    ('What is a taiga climate?','A taiga climate has long cold winters and short cool summers.'),
    ('What is a rainforest climate?','A rainforest climate is hot and wet all year.'),
    ('What is a savanna climate?','A savanna climate has wet and dry seasons with grasslands.'),
    ('What is the equator?','The equator is an imaginary line at 0 degrees latitude, dividing Earth.'),
    ('What is the Tropic of Cancer?','The Tropic of Cancer is at 23.5 degrees north latitude.'),
    ('What is the Tropic of Capricorn?','The Tropic of Capricorn is at 23.5 degrees south latitude.'),
    ('What is the Arctic Circle?','The Arctic Circle is at 66.5 degrees north latitude.'),
    ('What is the Antarctic Circle?','The Antarctic Circle is at 66.5 degrees south latitude.'),
    # Weather instruments
    ('What is a thermometer?','A thermometer measures temperature.'),
    ('What is a barometer?','A barometer measures air pressure.'),
    ('What is a hygrometer?','A hygrometer measures humidity (moisture in air).'),
    ('What is an anemometer?','An anemometer measures wind speed.'),
    ('What is a rain gauge?','A rain gauge measures rainfall.'),
    ('What is a wind vane?','A wind vane shows wind direction.'),
    ('What is a weather satellite?','A weather satellite takes images of clouds and weather from space.'),
    ('What is a weather balloon?','A weather balloon measures temperature, pressure, and humidity high in the atmosphere.'),
    ('What is Doppler radar?','Doppler radar detects rain and wind direction for weather forecasting.'),
    ('What is a meteorologist?','A meteorologist is a scientist who studies and forecasts the weather.'),
    ('What is a weather forecast?','A weather forecast predicts future weather conditions.'),
    ('What is a hurricane warning?','A hurricane warning means a hurricane is expected within 36 hours.'),
    ('What is a tornado watch?','A tornado watch means conditions are favorable for tornadoes.'),
    ('What is a tornado warning?','A tornado warning means a tornado has been spotted or detected.'),
]
for q,a in CLIMATE:
    add('science','Climate and Weather','Zones and Instruments',q,a)

# ============================================================
# 10. COOKING BASICS (30+)
# ============================================================
COOKING = [
    ('What is boiling?','Boiling is cooking in liquid at 100 degrees Celsius (water).'),
    ('What is frying?','Frying is cooking in hot oil.'),
    ('What is baking?','Baking is cooking in an oven using dry heat.'),
    ('What is grilling?','Grilling is cooking over direct heat, usually fire or coals.'),
    ('What is steaming?','Steaming is cooking food over boiling water using steam.'),
    ('What is roasting?','Roasting is cooking in an oven, usually meat or vegetables.'),
    ('What is sautéing?','Sautéing is cooking quickly in a little oil over high heat.'),
    ('What is simmering?','Simmering is cooking in liquid just below boiling point.'),
    ('What is blending?','Blending is mixing ingredients until smooth, using a blender.'),
    ('What is kneading?','Kneading is working dough with hands to develop gluten.'),
    ('What is whisking?','Whisking is beating ingredients with a whisk to incorporate air.'),
    ('What is marinating?','Marinating is soaking food in seasoned liquid to add flavor.'),
    ('What is a recipe?','A recipe is a set of instructions for preparing a dish.'),
    ('What are ingredients?','Ingredients are the foods and items needed to make a dish.'),
    ('What is a balanced meal?','A balanced meal includes protein, carbohydrates, vegetables, and fruits.'),
    ('How do you wash fruits?','Rinse fruits under running water before eating.'),
    ('How do you wash vegetables?','Rinse vegetables under running water before cooking.'),
    ('What is a spatula used for?','A spatula is used for flipping or scraping food.'),
    ('What is a whisk used for?','A whisk is used for beating eggs or mixing liquids.'),
    ('What is a colander used for?','A colander is used for draining water from pasta or vegetables.'),
    ('What is a cutting board?','A cutting board is a flat surface for cutting food safely.'),
    ('What is a measuring cup?','A measuring cup measures liquid or dry ingredients accurately.'),
    ('What is a rolling pin?','A rolling pin is used to flatten dough.'),
    ('What is a peeler?','A peeler removes the skin from fruits and vegetables.'),
    ('What is a grater?','A grater shreds food like cheese or carrots into small pieces.'),
    ('What is a mixing bowl?','A mixing bowl is a bowl for combining ingredients.'),
    ('What is a frying pan?','A frying pan is a flat pan for cooking food in oil.'),
    ('What is a saucepan?','A saucepan is a deep pan with a handle for cooking liquids.'),
    ('What is an oven?','An oven is a kitchen appliance for baking and roasting.'),
    ('What is a microwave?','A microwave is a kitchen appliance that heats food using microwaves.'),
]
for q,a in COOKING:
    add('culture','Cooking','Basics',q,a)

# ============================================================
# 11. GARDENING & AGRICULTURE (30+)
# ============================================================
GARDEN = [
    ('What is photosynthesis?','Photosynthesis is how plants make food from sunlight, water, and CO2.'),
    ('What do plants need to grow?','Plants need sunlight, water, air (CO2), nutrients, and soil.'),
    ('What is composting?','Composting is turning food and plant waste into nutrient-rich soil.'),
    ('What is organic farming?','Organic farming grows food without synthetic pesticides or fertilizers.'),
    ('What is irrigation?','Irrigation is watering crops artificially using pipes or channels.'),
    ('What is a seed?','A seed contains a tiny plant and food, ready to grow when conditions are right.'),
    ('What is germination?','Germination is when a seed starts to grow into a plant.'),
    ('What does a seed need to germinate?','A seed needs water, warmth, and oxygen to germinate.'),
    ('What is transplanting?','Transplanting is moving a plant from one place to another.'),
    ('What is pruning?','Pruning is cutting off dead or extra branches to help a plant grow better.'),
    ('What is weeding?','Weeding is removing unwanted plants that compete with crops.'),
    ('What is harvesting?','Harvesting is gathering ripe crops from the field.'),
    ('What is crop rotation?','Crop rotation is planting different crops in the same field each season to keep soil healthy.'),
    ('What is a greenhouse?','A greenhouse is a glass building where plants grow in controlled temperature.'),
    ('What is hydroponics?','Hydroponics is growing plants in water with nutrients, without soil.'),
    ('What is a fertilizer?','A fertilizer is a substance added to soil to help plants grow.'),
    ('What is a pesticide?','A pesticide is a chemical that kills insects that harm plants.'),
    ('What is pollination?','Pollination is the transfer of pollen from one flower to another.'),
    ('What are pollinators?','Pollinators include bees, butterflies, birds, and bats.'),
    ('What is a deciduous tree?','A deciduous tree loses its leaves in winter.'),
    ('What is an evergreen tree?','An evergreen tree keeps its leaves year-round.'),
    ('What is photosynthesis formula?','CO2 + H2O + sunlight -> glucose (C6H12O6) + O2.'),
    ('What is chlorophyll?','Chlorophyll is the green pigment that captures sunlight for photosynthesis.'),
    ('What is transpiration?','Transpiration is water evaporating from plant leaves.'),
    ('What is xylem?','Xylem transports water from roots to leaves.'),
    ('What is phloem?','Phloem transports food (sugar) from leaves to the rest of the plant.'),
    ('What is the growing season?','The growing season is the time of year when plants can grow outdoors.'),
    ('What is mulch?','Mulch is a layer of material on soil to retain moisture and prevent weeds.'),
    ('What is vermiculture?','Vermiculture is using worms to compost organic waste.'),
    ('What is aquaponics?','Aquaponics combines fish farming with hydroponic plant growing.'),
]
for q,a in GARDEN:
    add('science','Gardening','Agriculture',q,a)

# ============================================================
# 12. MAP READING (20+)
# ============================================================
MAP = [
    ('What is a map?','A map is a visual representation of an area on a flat surface.'),
    ('What is a compass rose?','A compass rose shows directions (N, S, E, W) on a map.'),
    ('What is a map legend?','A map legend explains the symbols and colors used on the map.'),
    ('What is a scale on a map?','A scale shows the relationship between map distance and real distance.'),
    ('What is latitude?','Latitude measures north-south position, from equator (0) to poles (90).'),
    ('What is longitude?','Longitude measures east-west position, from prime meridian (0) to 180.'),
    ('What is the equator?','The equator is 0 degrees latitude, dividing Earth into north and south.'),
    ('What is the prime meridian?','The prime meridian is 0 degrees longitude, passing through Greenwich, London.'),
    ('What are coordinates?','Coordinates are pairs of numbers (latitude, longitude) showing a position.'),
    ('What is a political map?','A political map shows boundaries, cities, and countries.'),
    ('What is a physical map?','A physical map shows natural features like mountains, rivers, and lakes.'),
    ('What is a topographic map?','A topographic map shows elevation using contour lines.'),
    ('What is a road map?','A road map shows highways, roads, and streets for navigation.'),
    ('What is a climate map?','A climate map shows different climate zones with colors.'),
    ('What is a population map?','A population map shows where people live, with density shown by colors.'),
    ('What is a globe?','A globe is a 3D model of Earth.'),
    ('What is the advantage of a globe over a map?','A globe shows the true shape and size of Earth without distortion.'),
    ('What is the disadvantage of a globe?','A globe is bulky and cannot show small areas in detail.'),
    ('What is GPS?','GPS (Global Positioning System) uses satellites to show your exact location on Earth.'),
    ('What is a contour line?','A contour line connects points of equal elevation on a topographic map.'),
]
for q,a in MAP:
    add('geography','Map Reading','Skills',q,a)

# ============================================================
# 13. JAVASCRIPT CODE SNIPPETS (20+)
# ============================================================
JS = [
    ('How do you print to console in JavaScript?','console.log("Hello, World!");'),
    ('How do you create a variable in JavaScript?','let x = 10; or const y = "hello";'),
    ('How do you create a function in JavaScript?','function myFunc() { return "Hello"; }'),
    ('How do you write an if statement in JavaScript?','if (x > 5) { console.log("Big"); }'),
    ('How do you write a for loop in JavaScript?','for (let i = 0; i < 5; i++) { console.log(i); }'),
    ('How do you create an array in JavaScript?','let arr = [1, 2, 3];'),
    ('How do you add to an array in JavaScript?','arr.push(4);'),
    ('How do you create an object in JavaScript?','let obj = { name: "John", age: 30 };'),
    ('How do you access object properties?','obj.name or obj["name"]'),
    ('How do you write an arrow function?','const add = (a, b) => a + b;'),
    ('How do you concatenate strings in JavaScript?','Use + : "Hello" + " " + "World" = "Hello World".'),
    ('How do you convert string to number?','Number("123") or parseInt("123")'),
    ('How do you get element by ID in JavaScript?','document.getElementById("myId")'),
    ('How do you add event listener in JavaScript?','element.addEventListener("click", function() { ... })'),
    ('How do you write a comment in JavaScript?','// single line or /* multi-line */'),
    ('How do you use template literals in JavaScript?','`Hello ${name}!` uses backticks and ${}.'),
    ('How do you check type in JavaScript?','typeof x returns the type like "number" or "string".'),
    ('How do you round a number in JavaScript?','Math.round(3.7) returns 4.'),
    ('How do you generate random in JavaScript?','Math.random() returns a number between 0 and 1.'),
    ('How do you convert to uppercase?','"hello".toUpperCase() returns "HELLO".'),
]
for q,a in JS:
    add('informatika','JavaScript','Code Snippets',q,a)

# ============================================================
# 14. MORE MATH - GRAPHS, PROBABILITY, RATIO (50+)
# ============================================================
MATH2 = [
    # Data and graphs
    ('What is a bar graph?','A bar graph uses bars of different heights to show data.'),
    ('What is a line graph?','A line graph uses points connected by lines to show trends over time.'),
    ('What is a pie chart?','A pie chart shows parts of a whole using slices of a circle.'),
    ('What is a pictograph?','A pictograph uses pictures or symbols to represent data.'),
    ('What is the x-axis?','The x-axis is the horizontal line on a graph.'),
    ('What is the y-axis?','The y-axis is the vertical line on a graph.'),
    ('What is the origin?','The origin is where the x-axis and y-axis meet (0, 0).'),
    ('What is the mean?','The mean is the average: sum of all values divided by count.'),
    ('What is the median?','The median is the middle value when data is sorted.'),
    ('What is the mode?','The mode is the value that appears most often.'),
    ('What is the range?','The range is the difference between highest and lowest values.'),
    # Probability
    ('What is probability?','Probability is how likely something is to happen, from 0 (impossible) to 1 (certain).'),
    ('What is the probability of flipping heads on a coin?','The probability of heads is 1/2 or 50%.'),
    ('What is the probability of rolling a 6 on a die?','The probability is 1/6.'),
    ('What is the probability of rolling an even number on a die?','The probability is 3/6 = 1/2.'),
    ('What is a certain event?','A certain event has probability 1 (100%).'),
    ('What is an impossible event?','An impossible event has probability 0 (0%).'),
    ('What is a fair coin?','A fair coin has equal probability of heads and tails.'),
    ('What is a standard die?','A standard die has 6 faces numbered 1-6, each equally likely.'),
    ('What is sample space?','The sample space is all possible outcomes of an experiment.'),
    # Ratios and proportions
    ('What is a ratio?','A ratio compares two quantities, like 2:3.'),
    ('What is a proportion?','A proportion is two equal ratios, like 1/2 = 2/4.'),
    ('If 3 apples cost $6, how much is 1 apple?','1 apple costs $6 / 3 = $2.'),
    ('If 5 pens cost $10, how much is 1 pen?','1 pen costs $10 / 5 = $2.'),
    ('If a recipe needs 2 cups of flour for 4 servings, how much for 8 servings?','4 cups of flour for 8 servings.'),
    ('If a car travels 60 km in 1 hour, how far in 3 hours?','180 km in 3 hours.'),
    ('If 1 cm on a map = 10 km real, what is 5 cm on the map?','50 km real distance.'),
    ('What is a unit rate?','A unit rate is a rate per 1 unit, like 60 km per hour.'),
    ('If 8 workers build a wall in 4 days, how long for 2 workers?','16 days (inverse proportion).'),
    ('What is a percentage?','A percentage is a fraction out of 100, like 50% = 50/100 = 0.5.'),
    # Metric conversion
    ('How many millimeters in a centimeter?','10 mm = 1 cm.'),
    ('How many centimeters in a meter?','100 cm = 1 m.'),
    ('How many meters in a kilometer?','1000 m = 1 km.'),
    ('How many grams in a kilogram?','1000 g = 1 kg.'),
    ('How many milligrams in a gram?','1000 mg = 1 g.'),
    ('How many milliliters in a liter?','1000 mL = 1 L.'),
    ('How many liters in a cubic meter?','1000 L = 1 m³.'),
    ('How to convert Celsius to Fahrenheit?','F = (C x 9/5) + 32.'),
    ('How to convert Fahrenheit to Celsius?','C = (F - 32) x 5/9.'),
    ('What is 0 degrees Celsius in Fahrenheit?','0°C = 32°F (freezing point of water).'),
    ('What is 100 degrees Celsius in Fahrenheit?','100°C = 212°F (boiling point of water).'),
    ('What is 37 degrees Celsius in Fahrenheit?','37°C = 98.6°F (body temperature).'),
    # More word problems
    ('A store sells 3 pencils for $1. How much for 12 pencils?','12 pencils = $4 (4 sets of 3).'),
    ('A train travels 120 km in 2 hours. What is its speed?','Speed = 120 / 2 = 60 km/h.'),
    ('A rectangle has length 8 and width 5. What is the perimeter?','Perimeter = 2 x (8 + 5) = 26.'),
    ('A rectangle has length 8 and width 5. What is the area?','Area = 8 x 5 = 40 square units.'),
    ('A triangle has base 6 and height 4. What is the area?','Area = 1/2 x 6 x 4 = 12 square units.'),
    ('A circle has radius 7. What is the circumference? (pi = 22/7)','Circumference = 2 x 22/7 x 7 = 44 units.'),
    ('A cube has side 3. What is the volume?','Volume = 3 x 3 x 3 = 27 cubic units.'),
    ('A box is 4 x 3 x 2. What is the volume?','Volume = 4 x 3 x 2 = 24 cubic units.'),
    ('If 20% of a number is 10, what is the number?','The number is 50 (10 / 0.20 = 50).'),
    ('If 25% of students are boys and there are 200 students, how many boys?','50 boys (200 x 0.25 = 50).'),
]
for q,a in MATH2:
    add('matematika','Advanced','Data and Probability',q,a)

# ============================================================
# OUTPUT
# ============================================================
print(f"\nTotal new items: {len(items)}")

OUT = Path("/home/z/my-project/download/llm/dataset_keruk4")
OUT.mkdir(parents=True, exist_ok=True)
with open(OUT / 'dataset_keruk4.jsonl', 'w', encoding='utf-8') as f:
    for item in items:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f"\nBreakdown:")
from collections import Counter
cats = Counter(item['bidang'] for item in items)
for cat, count in sorted(cats.items(), key=lambda x: -x[1]):
    print(f"  {cat}: {count}")
print(f"\nTotal: {sum(cats.values())}")
