"""
D'QUILANE — KERUK PART 3: Fill ALL remaining gaps
"""
import json, random
from pathlib import Path

random.seed(42)
items = []

def add(cat, judul, sub, q, a):
    items.append({'input': q, 'output': a, 'bidang': cat, 'judul': judul, 'subjudul': sub})

# ============================================================
# 1. LANGUAGE ARTS: Grammar, Spelling, Vocabulary (300+)
# ============================================================
print("Language Arts...")

# Synonyms (50+)
SYN = [
    ('big','large'),('small','tiny'),('fast','quick'),('slow','sluggish'),('happy','joyful'),
    ('sad','sorrowful'),('angry','furious'),('afraid','terrified'),('tired','exhausted'),
    ('smart','intelligent'),('beautiful','gorgeous'),('ugly','hideous'),('clean','spotless'),
    ('dirty','filthy'),('strong','powerful'),('weak','fragile'),('rich','wealthy'),('poor','needy'),
    ('old','ancient'),('new','fresh'),('hot','scorching'),('cold','freezing'),('loud','deafening'),
    ('quiet','silent'),('easy','simple'),('hard','difficult'),('good','excellent'),('bad','terrible'),
    ('begin','start'),('end','finish'),('buy','purchase'),('sell','vend'),('make','create'),
    ('give','donate'),('take','seize'),('run','sprint'),('walk','stroll'),('eat','consume'),
    ('drink','sip'),('sleep','slumber'),('wake','arise'),('think','ponder'),('know','understand'),
    ('see','observe'),('hear','listen'),('speak','utter'),('read','peruse'),('write','scribe'),
]
for word, syn in SYN:
    add('bahasa','Synonyms','Word Pairs',f'What is a synonym for {word}?',f'A synonym for {word} is {syn}.')
    add('bahasa','Synonyms','Word Pairs',f'What is a synonym for {syn}?',f'A synonym for {syn} is {word}.')

# Antonyms (50+)
ANT = [
    ('big','small'),('fast','slow'),('hot','cold'),('up','down'),('good','bad'),('new','old'),
    ('happy','sad'),('light','dark'),('long','short'),('tall','short'),('wide','narrow'),
    ('thick','thin'),('heavy','light'),('loud','quiet'),('strong','weak'),('rich','poor'),
    ('easy','hard'),('clean','dirty'),('full','empty'),('open','close'),('begin','end'),
    ('buy','sell'),('push','pull'),('win','lose'),('yes','no'),('day','night'),('sun','moon'),
    ('summer','winter'),('black','white'),('right','wrong'),('true','false'),('live','die'),
    ('love','hate'),('peace','war'),('first','last'),('top','bottom'),('left','right'),
    ('front','back'),('inside','outside'),('enter','exit'),('arrive','depart'),('rise','fall'),
    ('increase','decrease'),('expand','contract'),('assemble','disassemble'),('connect','disconnect'),
    ('construct','destroy'),('include','exclude'),('accept','reject'),('advance','retreat'),
]
for word, ant in ANT:
    add('bahasa','Antonyms','Word Pairs',f'What is the opposite of {word}?',f'The opposite of {word} is {ant}.')

# Homophones (40+)
HOMO = [
    ('to','too','two'),('there','their','they\'re'),('your','you\'re'),('its','it\'s'),
    ('their','there'),('weather','whether'),('peace','piece'),('plain','plane'),
    ('sea','see'),('write','right'),('hear','here'),('flower','flour'),('hair','hare'),
    ('board','bored'),('bear','bare'),('blue','blew'),('cell','sell'),('cent','scent'),('flue','flew'),
    ('groan','grown'),('hole','whole'),('knew','new'),('knows','nose'),('knead','need'),
    ('main','mane'),('meet','meat'),('pair','pear'),('plain','plane'),('poor','pour'),
    ('red','read'),('ring','wring'),('rode','road'),('sail','sale'),('sea','see'),
    ('steal','steel'),('tail','tale'),('weak','week'),('wear','where'),('wood','would'),
]
for w1, w2, *rest in HOMO:
    add('bahasa','Homophones','Word Pairs',f'What is a homophone for {w1}?',f'A homophone for {w1} is {w2} (and possibly {", ".join(rest)}).')

# Compound words (30+)
COMP = [
    ('sun','flower','sunflower'),('rain','bow','rainbow'),('birth','day','birthday'),
    ('cup','cake','cupcake'),('snow','man','snowman'),('pan','cake','pancake'),
    ('star','fish','starfish'),('fire','fly','firefly'),('moon','light','moonlight'),
    ('book','case','bookcase'),('class','room','classroom'),('bed','room','bedroom'),
    ('bath','room','bathroom'),('play','ground','playground'),('school','house','schoolhouse'),
    ('tree','house','treehouse'),('wind','mill','windmill'),('water','fall','waterfall'),
    ('butter','fly','butterfly'),('grass','hopper','grasshopper'),('straw','berry','strawberry'),
    ('blue','berry','blueberry'),('black','board','blackboard'),('key','board','keyboard'),
    ('note','book','notebook'),('text','book','textbook'),('hand','bag','handbag'),
    ('sun','glasses','sunglasses'),('tooth','brush','toothbrush'),('hair','brush','hairbrush'),
]
for w1, w2, compound in COMP:
    add('bahasa','Compound Words','Formation',f'What compound word is made from {w1} + {w2}?',f'{w1} + {w2} = {compound}.')

# Prefixes (30+)
PREF = [
    ('un-','not, opposite of'),('re-','again'),('dis-','not, opposite of'),('pre-','before'),
    ('mis-','wrong, bad'),('sub-','under'),('inter-','between'),('trans-','across'),
    ('super-','above, beyond'),('auto-','self'),('bi-','two'),('tri-','three'),
    ('multi-','many'),('semi-','half'),('over-','too much'),('under-','below, too little'),
    ('non-','not'),('anti-','against'),('pro-','for, in favor of'),('ex-','former, out of'),
    ('fore-','before'),('mid-','middle'),('out-','surpass'),('in-','not, into'),
    ('im-','not (before b, m, p)'),('il-','not (before l)'),('ir-','not (before r)'),
    ('en-','make, put into'),('em-','make (before b, m, p)'),('co-','together'),('tele-','distant'),
]
for prefix, meaning in PREF:
    add('bahasa','Prefixes','Meaning',f'What does the prefix {prefix} mean?',f'The prefix {prefix} means {meaning}.')

# Suffixes (20+)
SUFF = [
    ('-ful','full of'),('-less','without'),('-ness','state of being'),('-able','capable of'),
    ('-ible','capable of'),('-tion','act or process'),('-ment','act or process'),
    ('-ly','in a way that'),('-er','one who does'),('or','one who does'),
    ('-est','most'),('-ing','doing now'),('-ed','past tense'),('-s','plural'),
    ('-ous','full of'),('-al','relating to'),('-ish','like, similar to'),
    ('-y','characterized by'),('-ward','in the direction of'),('-en','made of, become'),
]
for suffix, meaning in SUFF:
    add('bahasa','Suffixes','Meaning',f'What does the suffix {suffix} mean?',f'The suffix {suffix} means {meaning}.')

# Parts of speech (30+)
POS = [
    ('What is a noun?','A noun names a person, place, thing, or idea.'),
    ('What is a verb?','A verb shows action or state of being.'),
    ('What is an adjective?','An adjective describes a noun.'),
    ('What is an adverb?','An adverb describes a verb, adjective, or other adverb.'),
    ('What is a pronoun?','A pronoun replaces a noun (he, she, it, they).'),
    ('What is a preposition?','A preposition shows relationship (in, on, at, under).'),
    ('What is a conjunction?','A conjunction joins words or sentences (and, but, or).'),
    ('What is an interjection?','An interjection expresses emotion (Wow! Ouch! Oh!).'),
    ('What is a common noun?','A common noun names any person, place, or thing (dog, city).'),
    ('What is a proper noun?','A proper noun names a specific person, place, or thing (Rex, Paris).'),
    ('What is a collective noun?','A collective noun names a group (herd, flock, team).'),
    ('What is an abstract noun?','An abstract noun names an idea (love, freedom, happiness).'),
    ('What is a transitive verb?','A transitive verb takes a direct object (She reads books).'),
    ('What is an intransitive verb?','An intransitive verb does not take a direct object (She sleeps).'),
    ('What is a helping verb?','A helping verb helps the main verb (is running, has eaten).'),
    ('What is a linking verb?','A linking verb connects the subject to a description (is, seems).'),
    ('What is a regular verb?','A regular verb forms past tense with -ed (walk-walked).'),
    ('What is an irregular verb?','An irregular verb changes form (go-went, see-saw).'),
    ('What is a comparative adjective?','A comparative compares two things (bigger, faster).'),
    ('What is a superlative adjective?','A superlative compares three or more (biggest, fastest).'),
    ('What is an article?','Articles are a, an, the.'),
    ('When do you use a?','Use a before words starting with consonant sounds (a cat).'),
    ('When do you use an?','Use an before words starting with vowel sounds (an apple).'),
    ('When do you use the?','Use the for specific things (the sun, the book on the table).'),
    ('What is a subject?','The subject is who or what the sentence is about.'),
    ('What is a predicate?','The predicate tells what the subject does or is.'),
    ('What is a simple sentence?','A simple sentence has one independent clause.'),
    ('What is a compound sentence?','A compound sentence joins two independent clauses with and, but, or.'),
    ('What is a complex sentence?','A complex sentence has one independent and one dependent clause.'),
    ('What is a fragment?','A fragment is an incomplete sentence missing subject or predicate.'),
    ('What is a run-on sentence?','A run-on joins two sentences without proper punctuation.'),
]
for q,a in POS:
    add('bahasa','Parts of Speech','Grammar',q,a)

# Tenses (20+)
TENSES = [
    ('What is the past tense?','Past tense describes actions that already happened.'),
    ('What is the present tense?','Present tense describes actions happening now.'),
    ('What is the future tense?','Future tense describes actions that will happen.'),
    ('What is the past tense of go?','The past tense of go is went.'),
    ('What is the past tense of eat?','The past tense of eat is ate.'),
    ('What is the past tense of see?','The past tense of see is saw.'),
    ('What is the past tense of run?','The past tense of run is ran.'),
    ('What is the past tense of write?','The past tense of write is wrote.'),
    ('What is the past tense of swim?','The past tense of swim is swam.'),
    ('What is the past tense of fly?','The past tense of fly is flew.'),
    ('What is the past tense of drink?','The past tense of drink is drank.'),
    ('What is the past tense of sing?','The past tense of sing is sang.'),
    ('What is the past tense of ring?','The past tense of ring is rang.'),
    ('What is the past tense of begin?','The past tense of begin is began.'),
    ('What is the past tense of take?','The past tense of take is took.'),
    ('What is the past tense of give?','The past tense of give is gave.'),
    ('What is the past tense of come?','The past tense of come is came.'),
    ('What is the past tense of buy?','The past tense of buy is bought.'),
    ('What is the past tense of bring?','The past tense of bring is brought.'),
    ('What is the past tense of think?','The past tense of think is thought.'),
    ('What is the past tense of catch?','The past tense of catch is caught.'),
    ('What is the past tense of teach?','The past tense of teach is taught.'),
    ('What is the past tense of fight?','The past tense of fight is fought.'),
    ('What is the past tense of sleep?','The past tense of sleep is slept.'),
    ('What is the past tense of keep?','The past tense of keep is kept.'),
    ('What is the past tense of feel?','The past tense of feel is felt.'),
    ('What is the past tense of leave?','The past tense of leave is left.'),
    ('What is the past tense of build?','The past tense of build is built.'),
]
for q,a in TENSES:
    add('bahasa','Tenses','Irregular Verbs',q,a)

# Punctuation (20+)
PUNCT = [
    ('When do you use a period?','Use a period at the end of a statement sentence.'),
    ('When do you use a question mark?','Use a question mark at the end of a question.'),
    ('When do you use an exclamation mark?','Use an exclamation mark for strong emotion.'),
    ('When do you use a comma?','Use a comma to separate items in a list or clauses.'),
    ('When do you use a colon?','Use a colon to introduce a list or quotation.'),
    ('When do you use a semicolon?','Use a semicolon to join related independent clauses.'),
    ('When do you use quotation marks?','Use quotation marks around direct speech.'),
    ('When do you use an apostrophe?','Use an apostrophe for contractions or possessives.'),
    ('What is a contraction?','A contraction shortens words (don\'t = do not).'),
    ('What is a possessive?','A possessive shows ownership (John\'s book).'),
    ('When do you capitalize?','Capitalize the first word, names, places, and titles.'),
    ('What is a proper noun?','A proper noun is a specific name (Paris, John).'),
    ('What is the Oxford comma?','The Oxford comma is the last comma before "and" in a list.'),
    ('When do you use a hyphen?','Use a hyphen in compound words (well-known).'),
    ('When do you use a dash?','Use a dash for emphasis or interruption.'),
    ('What are parentheses?','Parentheses add extra information.'),
    ('What is a title?','A title is the name of a book, song, or movie.'),
    ('How do you format a title?','Book titles are italicized; article titles use quotes.'),
    ('What is a sentence fragment?','A fragment is an incomplete sentence.'),
    ('What is a run-on sentence?','A run-on improperly joins two sentences.'),
]
for q,a in PUNCT:
    add('bahasa','Punctuation','Rules',q,a)

# Similes & Metaphors (20+)
SIM = [
    ('What is a simile?','A simile compares using like or as (as brave as a lion).'),
    ('What is a metaphor?','A metaphor compares without like or as (Time is money).'),
    ('What is personification?','Personification gives human qualities to objects (The wind whispered).'),
    ('What is onomatopoeia?','Onomatopoeia are sound words (buzz, splash, moo).'),
    ('What is alliteration?','Alliteration repeats starting sounds (Peter Piper picked).'),
    ('What is hyperbole?','Hyperbole is extreme exaggeration (I\'ve told you a million times).'),
    ('What is a simile for fast?','As fast as a cheetah.'),
    ('What is a simile for slow?','As slow as a turtle.'),
    ('What is a simile for brave?','As brave as a lion.'),
    ('What is a simile for light?','As light as a feather.'),
    ('What is a simile for heavy?','As heavy as lead.'),
    ('What is a simile for cold?','As cold as ice.'),
    ('What is a simile for hot?','As hot as fire.'),
    ('What is a simile for sweet?','As sweet as honey.'),
    ('What is a metaphor for time?','Time is money.'),
    ('What is a metaphor for knowledge?','Knowledge is power.'),
    ('What is a metaphor for life?','Life is a journey.'),
    ('What is a metaphor for anger?','Anger is a fire.'),
    ('What is onomatopoeia for a bee?','Buzz.'),
    ('What is onomatopoeia for water?','Splash.'),
    ('What is onomatopoeia for a clock?','Tick tock.'),
    ('What is onomatopoeia for a cow?','Moo.'),
    ('What is onomatopoeia for wind?','Whoosh.'),
    ('What is onomatopoeia for a bell?','Ding dong.'),
]
for q,a in SIM:
    add('bahasa','Figurative Language','Similes and Metaphors',q,a)

# ============================================================
# 2. WORLD CUISINES (60+)
# ============================================================
print("World cuisines...")
CUISINE = [
    ('What is sushi?','Sushi is a Japanese dish of vinegared rice with fish or vegetables.'),
    ('What is ramen?','Ramen is a Japanese noodle soup with broth, noodles, and toppings.'),
    ('What is tempura?','Tempura is a Japanese dish of battered and fried seafood or vegetables.'),
    ('What is sashimi?','Sashimi is raw fish sliced thinly, served without rice.'),
    ('What is miso soup?','Miso soup is a Japanese soup made from fermented soybean paste.'),
    ('What is pizza?','Pizza is an Italian dish of flat dough with tomato, cheese, and toppings.'),
    ('What is pasta?','Pasta is an Italian dish of dough shaped and cooked with sauce.'),
    ('What is lasagna?','Lasagna is an Italian dish of layered pasta, cheese, and meat sauce.'),
    ('What is risotto?','Risotto is an Italian rice dish cooked in broth until creamy.'),
    ('What is gelato?','Gelato is Italian ice cream, denser and warmer than regular ice cream.'),
    ('What is a croissant?','A croissant is a French crescent-shaped buttery pastry.'),
    ('What is baguette?','A baguette is a long, thin French bread.'),
    ('What is crepe?','A crepe is a thin French pancake.'),
    ('What is escargot?','Escargot is a French dish of cooked snails.'),
    ('What is quiche?','Quiche is a French savory pie with eggs and cheese.'),
    ('What is fish and chips?','Fish and chips is a British dish of fried fish and potato fries.'),
    ('What is a full English breakfast?','A full English breakfast includes eggs, bacon, sausage, beans, toast, and tea.'),
    ('What is shepherd\'s pie?','Shepherd\'s pie is a British meat pie with mashed potato topping.'),
    ('What is a hamburger?','A hamburger is an American sandwich with a beef patty in a bun.'),
    ('What is a hot dog?','A hot dog is a sausage in a bun, an American food.'),
    ('What is BBQ?','BBQ (barbecue) is American slow-cooked smoked meat.'),
    ('What is a taco?','A taco is a Mexican dish of a folded tortilla with filling.'),
    ('What is a burrito?','A burrito is a Mexican flour tortilla wrapped around filling.'),
    ('What is guacamole?','Guacamole is a Mexican dip made from mashed avocados.'),
    ('What is salsa?','Salsa is a Mexican sauce made from tomatoes, onions, and chili.'),
    ('What is nachos?','Nachos are tortilla chips with melted cheese and toppings.'),
    ('What is nasi goreng?','Nasi goreng is Indonesian fried rice with spices.'),
    ('What is rendang?','Rendang is an Indonesian spicy meat dish slow-cooked in coconut milk.'),
    ('What is satay?','Satay is Indonesian grilled meat on skewers with peanut sauce.'),
    ('What is gado-gado?','Gado-gado is an Indonesian salad with peanut sauce dressing.'),
    ('What is soto?','Soto is an Indonesian traditional soup.'),
    ('What is bakso?','Bakso is Indonesian meatball soup.'),
    ('What is mie goreng?','Mie goreng is Indonesian fried noodles.'),
    ('What is kimchi?','Kimchi is a Korean fermented vegetable dish, usually spicy.'),
    ('What is bibimbap?','Bibimbap is a Korean rice bowl with vegetables, meat, and egg.'),
    ('What is bulgogi?','Bulgogi is Korean marinated grilled beef.'),
    ('What is pad thai?','Pad Thai is a Thai stir-fried noodle dish.'),
    ('What is tom yum?','Tom yum is a Thai spicy and sour soup.'),
    ('What is curry?','Curry is a spiced sauce dish from India and Southeast Asia.'),
    ('What is biryani?','Biryani is an Indian rice dish with spices and meat.'),
    ('What is naan?','Naan is an Indian flatbread baked in a tandoor oven.'),
    ('What is dosa?','Dosa is a thin South Indian crepe made from fermented batter.'),
    ('What is samosa?','Samosa is an Indian fried pastry with savory filling.'),
    ('What is masala chai?','Masala chai is Indian spiced tea with milk.'),
    ('What is dim sum?','Dim sum is Chinese small dishes served with tea.'),
    ('What is Peking duck?','Peking duck is a Chinese dish of roasted duck with crispy skin.'),
    ('What is fried rice?','Fried rice is a Chinese dish of rice stir-fried with eggs and vegetables.'),
    ('What is chow mein?','Chow mein is Chinese stir-fried noodles.'),
    ('What is spring roll?','A spring roll is a Chinese fried or fresh roll with vegetable filling.'),
    ('What is dumpling?','A dumpling is a small mass of dough with filling, steamed or fried.'),
    ('What is paella?','Paella is a Spanish rice dish with saffron, seafood, and meat.'),
    ('What is gazpacho?','Gazpacho is a Spanish cold tomato soup.'),
    ('What is churros?','Churros are Spanish fried dough sticks dusted with sugar.'),
    ('What is tapas?','Tapas are Spanish small appetizer dishes.'),
    ('What is falafel?','Falafel is a Middle Eastern fried ball made from chickpeas.'),
    ('What is hummus?','Hummus is a Middle Eastern dip made from mashed chickpeas.'),
    ('What is shawarma?','Shawarma is a Middle Eastern dish of meat on a rotating spit.'),
    ('What is kebab?','Kebab is grilled meat on a skewer, popular in the Middle East.'),
    ('What is baklava?','Baklava is a Middle Eastern sweet pastry with nuts and honey.'),
    ('What is poutine?','Poutine is a Canadian dish of fries with cheese curds and gravy.'),
    ('What is ceviche?','Ceviche is a Latin American dish of raw fish cured in citrus.'),
]
for q,a in CUISINE:
    add('culture','World Cuisine','Dishes',q,a)

# ============================================================
# 3. BIOLOGY DETAIL (100+)
# ============================================================
print("Biology detail...")
BIO = [
    # Cell biology
    ('What is a cell?','A cell is the basic unit of life.'),
    ('What are the two main types of cells?','Prokaryotic (no nucleus) and eukaryotic (with nucleus).'),
    ('What is the cell membrane?','The cell membrane controls what enters and leaves the cell.'),
    ('What is cytoplasm?','Cytoplasm is the jelly-like substance inside the cell.'),
    ('What is the nucleus?','The nucleus is the control center of the cell, containing DNA.'),
    ('What is the mitochondria?','Mitochondria are the powerhouse of the cell, producing energy.'),
    ('What are ribosomes?','Ribosomes make proteins.'),
    ('What is the endoplasmic reticulum?','The ER transports materials through the cell.'),
    ('What is the Golgi apparatus?','The Golgi apparatus packages and sends cell products.'),
    ('What are lysosomes?','Lysosomes break down waste materials in the cell.'),
    ('What is the cell wall?','The cell wall is a rigid outer layer in plant cells (not animal cells).'),
    ('What are chloroplasts?','Chloroplasts contain chlorophyll and do photosynthesis in plant cells.'),
    ('What is a vacuole?','A vacuole stores water and nutrients, large in plant cells.'),
    # Genetics
    ('What is DNA?','DNA (Deoxyribonucleic Acid) stores genetic information.'),
    ('What is the shape of DNA?','DNA has a double helix shape, like a twisted ladder.'),
    ('What is a gene?','A gene is a section of DNA that codes for a specific trait.'),
    ('What is a chromosome?','A chromosome is a structure of DNA in the nucleus.'),
    ('How many chromosomes do humans have?','Humans have 46 chromosomes (23 pairs).'),
    ('What is dominant trait?','A dominant trait always shows if the gene is present.'),
    ('What is a recessive trait?','A recessive trait only shows if both genes are recessive.'),
    ('What is heredity?','Heredity is the passing of traits from parents to offspring.'),
    ('Who discovered genetics?','Gregor Mendel is called the father of genetics.'),
    ('What is a Punnett square?','A Punnett square predicts the probability of genetic traits.'),
    ('What is mutation?','A mutation is a change in DNA that can be inherited.'),
    ('What is genetic engineering?','Genetic engineering is modifying DNA to change traits.'),
    # Classification
    ('What is the classification system?','The system is: Kingdom, Phylum, Class, Order, Family, Genus, Species.'),
    ('What are the 5 kingdoms?','Monera, Protista, Fungi, Plantae, Animalia.'),
    ('What is the scientific name?','Scientific name is Genus + Species, like Homo sapiens for humans.'),
    ('What is binomial nomenclature?','Binomial nomenclature is the two-word naming system (Genus species).'),
    ('Who created the classification system?','Carl Linnaeus created the classification system.'),
    ('What kingdom are bacteria in?','Bacteria are in Kingdom Monera.'),
    ('What kingdom are mushrooms in?','Mushrooms are in Kingdom Fungi.'),
    ('What kingdom are plants in?','Plants are in Kingdom Plantae.'),
    ('What kingdom are animals in?','Animals are in Kingdom Animalia.'),
    # Human biology expanded
    ('How many muscles in the human body?','There are over 600 muscles in the human body.'),
    ('How many bones in the adult body?','There are 206 bones in the adult human body.'),
    ('How many teeth do adults have?','Adults have 32 teeth.'),
    ('How many teeth do children have?','Children have 20 baby teeth.'),
    ('What is the largest organ?','The skin is the largest organ.'),
    ('What is the smallest bone?','The stapes in the ear is the smallest bone.'),
    ('What is the largest bone?','The femur (thigh bone) is the largest bone.'),
    ('How many chambers does the heart have?','The heart has 4 chambers.'),
    ('How much blood is in the body?','There are about 5 liters of blood.'),
    ('What is the normal body temperature?','Normal body temperature is about 37 degrees Celsius.'),
    ('How many breaths per minute?','An adult breathes about 12-20 times per minute.'),
    ('How fast does the heart beat?','The heart beats about 60-100 times per minute.'),
    ('What carries oxygen in blood?','Red blood cells carry oxygen using hemoglobin.'),
    ('What fights infection?','White blood cells fight infection.'),
    ('What clots blood?','Platelets help blood clot.'),
    ('What is the largest artery?','The aorta is the largest artery.'),
    ('What is the smallest blood vessel?','Capillaries are the smallest blood vessels.'),
    # Plant biology
    ('What are the parts of a flower?','Petals, sepals, stamen (anther, filament), pistil (stigma, style, ovary).'),
    ('What is the stamen?','The stamen is the male part of the flower (anther + filament).'),
    ('What is the pistil?','The pistil is the female part of the flower (stigma + style + ovary).'),
    ('What is pollen?','Pollen is the male reproductive cell in plants.'),
    ('What is the ovule?','The ovule is the female reproductive cell that becomes a seed.'),
    ('What is fertilization in plants?','Fertilization is when pollen joins with the ovule to form a seed.'),
    ('What is xylem?','Xylem transports water upward from roots to leaves.'),
    ('What is phloem?','Phloem transports food (sugar) from leaves to the rest of the plant.'),
    ('What is transpiration?','Transpiration is water evaporating from plant leaves.'),
    ('What is respiration in plants?','Plant respiration is using oxygen to break down sugar for energy.'),
    ('What is tropism?','Tropism is plant growth toward or away from a stimulus.'),
    ('What is phototropism?','Phototropism is growth toward light.'),
    ('What is gravitropism?','Gravitropism is growth in response to gravity (roots down, shoots up).'),
    ('What is hydrotropism?','Hydrotropism is growth toward water.'),
]
for q,a in BIO:
    add('science','Biology','Cell and Genetics',q,a)

# ============================================================
# 4. PHYSICS FORMULAS & SI UNITS (60+)
# ============================================================
print("Physics formulas...")
PHYS = [
    # SI Units
    ('What is the SI unit of length?','The SI unit of length is the meter (m).'),
    ('What is the SI unit of mass?','The SI unit of mass is the kilogram (kg).'),
    ('What is the SI unit of time?','The SI unit of time is the second (s).'),
    ('What is the SI unit of temperature?','The SI unit of temperature is the Kelvin (K).'),
    ('What is the SI unit of electric current?','The SI unit of electric current is the ampere (A).'),
    ('What is the SI unit of force?','The SI unit of force is the Newton (N).'),
    ('What is the SI unit of energy?','The SI unit of energy is the joule (J).'),
    ('What is the SI unit of power?','The SI unit of power is the watt (W).'),
    ('What is the SI unit of pressure?','The SI unit of pressure is the pascal (Pa).'),
    ('What is the SI unit of frequency?','The SI unit of frequency is the hertz (Hz).'),
    ('What is the SI unit of area?','The SI unit of area is the square meter (m²).'),
    ('What is the SI unit of volume?','The SI unit of volume is the cubic meter (m³).'),
    ('What is the SI unit of speed?','The SI unit of speed is meters per second (m/s).'),
    # Formulas
    ('What is the formula for speed?','Speed = distance / time.'),
    ('What is the formula for acceleration?','Acceleration = change in velocity / time.'),
    ('What is the formula for force?','Force = mass x acceleration (F = ma).'),
    ('What is Newton\'s second law?','Force equals mass times acceleration (F = ma).'),
    ('What is the formula for weight?','Weight = mass x gravity (W = mg).'),
    ('What is the formula for work?','Work = force x distance (W = Fs).'),
    ('What is the formula for power?','Power = work / time (P = W/t).'),
    ('What is the formula for pressure?','Pressure = force / area (P = F/A).'),
    ('What is the formula for density?','Density = mass / volume (D = m/V).'),
    ('What is the formula for momentum?','Momentum = mass x velocity (p = mv).'),
    ('What is the formula for kinetic energy?','KE = 1/2 x mass x velocity².'),
    ('What is the formula for potential energy?','PE = mass x gravity x height.'),
    ('What is the formula for Ohm\'s law?','Voltage = current x resistance (V = IR).'),
    ('What is the formula for electrical power?','Power = voltage x current (P = VI).'),
    ('What is the speed of light?','The speed of light is about 300,000,000 m/s.'),
    ('What is the speed of sound in air?','The speed of sound in air is about 343 m/s.'),
    ('What is the acceleration due to gravity?','Gravity accelerates objects at about 9.8 m/s².'),
    ('What is the formula for the area of a circle?','Area = pi x radius² (A = pi r²).'),
    ('What is the formula for the circumference of a circle?','Circumference = 2 x pi x radius (C = 2 pi r).'),
    ('What is the formula for the volume of a sphere?','Volume = 4/3 x pi x radius³.'),
    ('What is the formula for the volume of a cylinder?','Volume = pi x radius² x height.'),
    # Laws
    ('What is Newton\'s first law?','An object at rest stays at rest unless acted on by a force.'),
    ('What is Newton\'s second law?','Force = mass x acceleration.'),
    ('What is Newton\'s third law?','For every action, there is an equal and opposite reaction.'),
    ('What is the law of conservation of energy?','Energy cannot be created or destroyed, only transformed.'),
    ('What is the law of conservation of mass?','Mass cannot be created or destroyed in a chemical reaction.'),
    ('What is Archimedes\' principle?','The buoyant force equals the weight of displaced fluid.'),
    ('What is Pascal\'s principle?','Pressure in a fluid is transmitted equally in all directions.'),
    ('What is Bernoulli\'s principle?','Faster moving fluids have lower pressure.'),
    ('What is Hooke\'s law?','Force is proportional to extension (F = kx).'),
    # More concepts
    ('What is friction?','Friction is a force that opposes motion between surfaces.'),
    ('What is gravity?','Gravity is the force of attraction between masses.'),
    ('What is inertia?','Inertia is resistance to changes in motion.'),
    ('What is momentum?','Momentum is mass in motion (mass x velocity).'),
    ('What is a vector?','A vector has both magnitude and direction.'),
    ('What is a scalar?','A scalar has only magnitude (no direction).'),
    ('What is velocity?','Velocity is speed in a specific direction.'),
    ('What is the difference between speed and velocity?','Speed is how fast; velocity is speed with direction.'),
    ('What is terminal velocity?','Terminal velocity is the maximum falling speed when air resistance equals gravity.'),
    ('What is a lever?','A lever is a bar that pivots on a fulcrum.'),
    ('What is a pulley?','A pulley is a wheel with a rope, used to lift loads.'),
    ('What is a gear?','A gear is a toothed wheel that transmits rotation.'),
    ('What is static friction?','Static friction prevents an object from starting to move.'),
    ('What is kinetic friction?','Kinetic friction opposes an object that is already moving.'),
    ('What is the electromagnetic spectrum?','The EM spectrum includes radio, microwave, infrared, visible light, UV, X-rays, gamma rays.'),
    ('What are the colors of visible light?','Red, orange, yellow, green, blue, indigo, violet.'),
    ('What is reflection?','Reflection is light bouncing off a surface.'),
    ('What is refraction?','Refraction is light bending when it enters a new medium.'),
    ('What is diffraction?','Diffraction is light spreading when passing through a gap.'),
    ('What is the Doppler effect?','The Doppler effect is the change in frequency as a source moves.'),
]
for q,a in PHYS:
    add('science','Physics','Formulas and Laws',q,a)

# ============================================================
# 5. SCIENCE EXPERIMENTS (30+)
# ============================================================
print("Science experiments...")
EXP = [
    ('How do you make a volcano?','Mix baking soda and vinegar with red food coloring for a foaming eruption.'),
    ('How do you grow crystals?','Dissolve salt or sugar in hot water, let cool, and crystals form.'),
    ('How do you make slime?','Mix glue with borax solution or contact lens solution.'),
    ('How do you test for starch?','Add iodine to food; if it turns blue-black, starch is present.'),
    ('How do you make a rainbow?','Use a prism or spray water in sunlight to split white light.'),
    ('How do you show static electricity?','Rub a balloon on hair, then it sticks to a wall.'),
    ('How do you make a battery from a lemon?','Insert zinc and copper into a lemon; it generates electricity.'),
    ('How do you show photosynthesis?','Put a leaf in water in sunlight; bubbles of oxygen appear.'),
    ('How do you test water purity?','Use a pH strip; pure water should be pH 7.'),
    ('How do you make ice melt faster?','Add salt, which lowers the freezing point.'),
    ('How do you separate salt from water?','Evaporate the water; salt remains.'),
    ('How do you make invisible ink?','Write with lemon juice; heat to reveal the message.'),
    ('How do you make a compass?','Magnetize a needle, float it on water in a cork; it points north.'),
    ('How do you show air pressure?','Place a ruler on a table, cover with paper, hit the ruler; paper holds it down.'),
    ('How do you make a parachute?','Tie strings to a plastic bag and a weight; it slows falling.'),
    ('How do you show condensation?','Put ice in a glass; water droplets form on the outside.'),
    ('How do you show osmosis?','Put a potato in salt water and fresh water; salt water makes it shrink.'),
    ('How do you make a lava lamp?','Mix oil and water, add food coloring, drop in an effervescent tablet.'),
    ('How do you show gravity?','Drop two objects of different weight; they land at the same time.'),
    ('How do you make a simple circuit?','Connect a battery, wire, and bulb in a loop.'),
    ('How do you test conductivity?','Use a circuit with a gap; if the bulb lights, the material conducts.'),
    ('How do you make a periscope?','Use a box with two mirrors at 45-degree angles.'),
    ('How do you show sound waves?','Sprinkle salt on cling film over a bowl; tap the bowl to see vibrations.'),
    ('How do you make a water filter?','Layer sand, gravel, and charcoal in a bottle.'),
    ('How do you show the greenhouse effect?','Cover a jar with plastic; place in sun; temperature rises inside.'),
    ('How do you make paper?','Soak shredded paper, blend into pulp, spread on screen, dry.'),
    ('How do you show buoyancy?','Float an orange in water; peel it, and it sinks.'),
    ('How do you make a tornado in a bottle?','Fill a bottle with water, swirl it, and a vortex forms.'),
    ('How do you show surface tension?','Float a paperclip on water; add soap to break the tension.'),
    ('How do you make rock candy?','Dissolve sugar in hot water, hang a string, wait for crystals.'),
]
for q,a in EXP:
    add('science','Science Experiments','DIY',q,a)

# ============================================================
# 6. OLYMPICS & SPORTS EVENTS (40+)
# ============================================================
print("Olympics...")
OLYMP = [
    ('What are the Olympic Games?','The Olympics are an international sports event held every 4 years.'),
    ('Where did the Olympics originate?','The Olympics originated in Ancient Greece, in Olympia.'),
    ('When were the modern Olympics started?','The modern Olympics started in 1896 in Athens.'),
    ('Who founded the modern Olympics?','Pierre de Coubertin founded the modern Olympic Games.'),
    ('What are the Olympic rings?','Five interlocking rings: blue, yellow, black, green, red.'),
    ('What do the Olympic rings represent?','The five rings represent the five continents.'),
    ('What is the Olympic motto?','The Olympic motto is "Faster, Higher, Stronger."'),
    ('What is the Olympic flame?','The Olympic flame is lit in Olympia and carried to the host city.'),
    ('What are the Summer Olympics?','Summer Olympics include sports like swimming, track, gymnastics.'),
    ('What are the Winter Olympics?','Winter Olympics include skiing, skating, ice hockey.'),
    ('How often are the Summer and Winter Olympics?','They alternate every 2 years.'),
    ('What is the Paralympic Games?','The Paralympics are for athletes with disabilities.'),
    ('What is the FIFA World Cup?','The World Cup is the biggest soccer tournament, every 4 years.'),
    ('Where was the 2022 World Cup?','The 2022 World Cup was in Qatar.'),
    ('What is the UEFA Champions League?','The Champions League is a European club soccer tournament.'),
    ('What is the Super Bowl?','The Super Bowl is the American football championship.'),
    ('What is the Wimbledon?','Wimbledon is the oldest tennis tournament, in London.'),
    ('What is the Tour de France?','The Tour de France is a famous cycling race.'),
    ('What is the NBA?','The NBA is the National Basketball Association, in the US.'),
    ('What is the Premier League?','The Premier League is the top English soccer league.'),
    ('What is the World Series?','The World Series is the baseball championship in the US.'),
    ('What are the Asian Games?','The Asian Games are a multi-sport event for Asian countries.'),
    ('What is the Commonwealth Games?','The Commonwealth Games are for Commonwealth nations.'),
    ('What is the World Athletics Championships?','It is the global championship for track and field.'),
    ('What is Formula 1?','Formula 1 is the highest class of international auto racing.'),
    ('What is the Marathon?','A marathon is a 42.195 km running race.'),
    ('Where did the marathon originate?','The marathon commemorates a run from Marathon to Athens in Greece.'),
    ('What is the World Swimming Championship?','It is the global championship for swimming.'),
    ('What is the Grand Slam in tennis?','Grand Slam tournaments: Australian Open, French Open, Wimbledon, US Open.'),
    ('How many Grand Slam tournaments are there?','There are 4 Grand Slam tournaments.'),
    ('What is a hat-trick in soccer?','A hat-trick is scoring 3 goals in one game.'),
    ('What is a home run in baseball?','A home run is hitting the ball out of the park.'),
    ('What is a knockout in boxing?','A knockout is when a fighter cannot continue.'),
    ('What is a checkmate in chess?','Checkmate is when the king cannot escape and the game ends.'),
    ('What is the Indy 500?','The Indy 500 is a famous 500-mile car race in the US.'),
    ('What is the America\'s Cup?','The America\'s Cup is the oldest international sailing trophy.'),
    ('What is the Rugby World Cup?','The Rugby World Cup is held every 4 years.'),
    ('What is the Cricket World Cup?','The Cricket World Cup is held every 4 years.'),
    ('What is the Stanley Cup?','The Stanley Cup is the NHL ice hockey championship.'),
    ('What is the Ryder Cup?','The Ryder Cup is a golf competition between US and Europe.'),
]
for q,a in OLYMP:
    add('PJOK','Sports Events','World',q,a)

# ============================================================
# 7. FAMOUS PEOPLE EXPANDED (100+)
# ============================================================
print("Famous people...")
FAMOUS = [
    # Scientists
    ('Who was Albert Einstein?','Einstein developed the theory of relativity (E=mc²).'),
    ('Who was Isaac Newton?','Newton discovered gravity and the three laws of motion.'),
    ('Who was Charles Darwin?','Darwin developed the theory of evolution by natural selection.'),
    ('Who was Galileo Galilei?','Galileo improved the telescope and proved Earth orbits the Sun.'),
    ('Who was Nikola Tesla?','Tesla pioneered alternating current (AC) electricity.'),
    ('Who was Marie Curie?','Curie discovered radioactivity and won two Nobel Prizes.'),
    ('Who was Louis Pasteur?','Pasteur invented pasteurization and vaccines.'),
    ('Who was Alexander Fleming?','Fleming discovered penicillin (antibiotics).'),
    ('Who was Gregor Mendel?','Mendel is the father of genetics, studying pea plants.'),
    ('Who was Carl Linnaeus?','Linnaeus created the classification system for living things.'),
    ('Who was Stephen Hawking?','Hawking was a physicist who studied black holes.'),
    ('Who was Charles Babbage?','Babbage is called the father of the computer.'),
    ('Who was Ada Lovelace?','Lovelace was the first computer programmer.'),
    ('Who was Alan Turing?','Turing was a mathematician who broke the Enigma code.'),
    ('Who was Rosalind Franklin?','Franklin contributed to discovering the structure of DNA.'),
    ('Who was James Watson?','Watson co-discovered the structure of DNA.'),
    ('Who was Francis Crick?','Crick co-discovered the structure of DNA.'),
    ('Who was Johannes Kepler?','Kepler discovered the laws of planetary motion.'),
    ('Who was Antonie van Leeuwenhoek?','Leeuwenhoek invented the microscope and discovered bacteria.'),
    ('Who was Michael Faraday?','Faraday discovered electromagnetic induction.'),
    # Inventors
    ('Who invented the telephone?','Alexander Graham Bell invented the telephone.'),
    ('Who invented the light bulb?','Thomas Edison invented the practical light bulb.'),
    ('Who invented the airplane?','The Wright Brothers invented the airplane.'),
    ('Who invented the printing press?','Johannes Gutenberg invented the printing press.'),
    ('Who invented the radio?','Guglielmo Marconi invented the radio.'),
    ('Who invented the television?','John Logie Baird invented television.'),
    ('Who invented the automobile?','Karl Benz invented the modern automobile.'),
    ('Who invented the steam engine?','James Watt improved the steam engine.'),
    ('Who invented the computer?','Charles Babbage is the father of the computer.'),
    ('Who invented the World Wide Web?','Tim Berners-Lee invented the WWW.'),
    ('Who invented the telephone?','Alexander Graham Bell patented the telephone.'),
    ('Who invented penicillin?','Alexander Fleming discovered penicillin.'),
    ('Who invented the vaccine?','Edward Jenner created the first vaccine (smallpox).'),
    ('Who invented the compass?','The compass was invented in ancient China.'),
    ('Who invented paper?','Paper was invented in China by Cai Lun.'),
    ('Who invented gunpowder?','Gunpowder was invented in China.'),
    ('Who invented the telescope?','Galileo improved the telescope for astronomy.'),
    ('Who invented the microscope?','Leeuwenhoek improved the microscope.'),
    ('Who invented the elevator?','Elisha Otis invented the safety elevator.'),
    ('Who invented the refrigerator?','The refrigerator was developed by multiple inventors.'),
    # Leaders
    ('Who was Nelson Mandela?','Mandela ended apartheid in South Africa and became president.'),
    ('Who was Mahatma Gandhi?','Gandhi led India to independence through non-violence.'),
    ('Who was Martin Luther King Jr.?','MLK was a US civil rights leader.'),
    ('Who was Abraham Lincoln?','Lincoln was the US president who ended slavery.'),
    ('Who was George Washington?','Washington was the first US president.'),
    ('Who was Winston Churchill?','Churchill was the UK prime minister during World War II.'),
    ('Who was Napoleon Bonaparte?','Napoleon was a French emperor who conquered much of Europe.'),
    ('Who was Alexander the Great?','Alexander was a Macedonian king who conquered the known world.'),
    ('Who was Julius Caesar?','Caesar was a Roman general and dictator.'),
    ('Who was Cleopatra?','Cleopatra was the last pharaoh of Egypt.'),
    ('Who was Queen Victoria?','Victoria was UK queen for 63 years during the Victorian era.'),
    ('Who was Queen Elizabeth I?','Elizabeth I ruled England during its golden age.'),
    ('Who was Mother Teresa?','Mother Teresa helped the poor in India, winning the Nobel Peace Prize.'),
    ('Who was Dalai Lama?','The Dalai Lama is the spiritual leader of Tibetan Buddhism.'),
    ('Who was Soekarno?','Soekarno was the first president of Indonesia and proclamator.'),
    ('Who was Mohammad Hatta?','Hatta was the first vice president of Indonesia.'),
    ('Who was R.A. Kartini?','Kartini was an Indonesian heroine for women\'s rights.'),
    ('Who was Ki Hajar Dewantara?','Ki Hajar Dewantara is the father of Indonesian education.'),
    # Artists
    ('Who was Leonardo da Vinci?','Da Vinci was an Italian Renaissance painter, inventor, and scientist.'),
    ('Who was Michelangelo?','Michelangelo sculpted David and painted the Sistine Chapel.'),
    ('Who was Pablo Picasso?','Picasso was a Spanish painter, co-founder of Cubism.'),
    ('Who was Vincent van Gogh?','Van Gogh was a Dutch painter known for vivid colors.'),
    ('Who was Claude Monet?','Monet was a French Impressionist painter.'),
    ('Who was Frida Kahlo?','Kahlo was a Mexican painter known for self-portraits.'),
    ('Who was Rembrandt?','Rembrandt was a Dutch master of light and shadow.'),
    ('Who was Salvador Dali?','Dali was a Spanish Surrealist painter.'),
    ('Who was Raphael?','Raphael was an Italian Renaissance painter.'),
    ('Who was Andy Warhol?','Warhol was an American Pop Art artist.'),
    # Writers
    ('Who was William Shakespeare?','Shakespeare was an English playwright, considered the greatest writer.'),
    ('Who wrote Romeo and Juliet?','Shakespeare wrote Romeo and Juliet.'),
    ('Who was Mark Twain?','Mark Twain was an American author who wrote Tom Sawyer.'),
    ('Who was Charles Dickens?','Dickens wrote Oliver Twist and A Christmas Carol.'),
    ('Who was Jane Austen?','Austen wrote Pride and Prejudice.'),
    ('Who was J.K. Rowling?','Rowling wrote the Harry Potter series.'),
    ('Who was Roald Dahl?','Dahl wrote Charlie and the Chocolate Factory.'),
    ('Who was Dr. Seuss?','Dr. Seuss wrote The Cat in the Hat.'),
    ('Who was J.R.R. Tolkien?','Tolkien wrote The Lord of the Rings.'),
    ('Who was Lewis Carroll?','Carroll wrote Alice in Wonderland.'),
    ('Who was George Orwell?','Orwell wrote 1984 and Animal Farm.'),
    ('Who was Ernest Hemingway?','Hemingway was an American novelist.'),
    ('Who was Agatha Christie?','Christie was a famous mystery writer.'),
    ('Who was Rudyard Kipling?','Kipling wrote The Jungle Book.'),
    ('Who was Hans Christian Andersen?','Andersen wrote fairy tales like The Little Mermaid.'),
    ('Who was Brothers Grimm?','The Brothers Grimm collected fairy tales like Cinderella.'),
    # Musicians
    ('Who was Mozart?','Mozart was an Austrian musical prodigy and composer.'),
    ('Who was Beethoven?','Beethoven was a German composer who became deaf.'),
    ('Who was Bach?','Bach was a German Baroque composer.'),
    ('Who was Elvis Presley?','Elvis was called the King of Rock and Roll.'),
    ('Who was Michael Jackson?','Michael Jackson was called the King of Pop.'),
    ('Who was Bob Marley?','Marley was a Jamaican reggae legend.'),
    ('Who was The Beatles?','The Beatles were a British rock band: John, Paul, George, Ringo.'),
    ('Who was Freddie Mercury?','Mercury was the lead singer of the band Queen.'),
    ('Who was Aretha Franklin?','Franklin was called the Queen of Soul.'),
    ('Who was Louis Armstrong?','Armstrong was a jazz trumpet player and singer.'),
]
for q,a in FAMOUS:
    add('world_history','Famous People','World',q,a)

# ============================================================
# 8. READING COMPREHENSION (30+)
# ============================================================
print("Reading comprehension...")
READING = [
    ('Read: "The cat sat on the mat." Where is the cat?','The cat is on the mat.'),
    ('Read: "Birds fly in the sky." Where do birds fly?','Birds fly in the sky.'),
    ('Read: "The sun rises in the east." Where does the sun rise?','The sun rises in the east.'),
    ('Read: "Water boils at 100 degrees Celsius." At what temperature does water boil?','Water boils at 100 degrees Celsius.'),
    ('Read: "Plants need sunlight, water, and air." What do plants need?','Plants need sunlight, water, and air.'),
    ('Read: "The Earth orbits the Sun." What does the Earth orbit?','The Earth orbits the Sun.'),
    ('Read: "Humans have 206 bones." How many bones do humans have?','Humans have 206 bones.'),
    ('Read: "The Nile is the longest river." What is the longest river?','The Nile is the longest river.'),
    ('Read: "Mount Everest is the highest mountain." What is the highest mountain?','Mount Everest.'),
    ('Read: "A year has 12 months." How many months in a year?','A year has 12 months.'),
    ('Read: "The heart pumps blood." What does the heart do?','The heart pumps blood.'),
    ('Read: "Photosynthesis makes food for plants." What does photosynthesis do?','Photosynthesis makes food for plants.'),
    ('Read: "Gravity pulls objects down." What does gravity do?','Gravity pulls objects down.'),
    ('Read: "The Pacific is the largest ocean." What is the largest ocean?','The Pacific Ocean.'),
    ('Read: "Japan is an island country." What kind of country is Japan?','Japan is an island country.'),
    ('Read: "A square has 4 equal sides." How many sides does a square have?','A square has 4 equal sides.'),
    ('Read: "The brain controls the body." What controls the body?','The brain controls the body.'),
    ('Read: "Bees make honey." What do bees make?','Bees make honey.'),
    ('Read: "Ice is frozen water." What is ice?','Ice is frozen water.'),
    ('Read: "A compass shows direction." What does a compass do?','A compass shows direction.'),
    ('Read: "The Olympic Games happen every 4 years." How often are the Olympics?','Every 4 years.'),
    ('Read: "DNA stores genetic information." What does DNA store?','DNA stores genetic information.'),
    ('Read: "The speed of light is about 300,000 km/s." How fast is light?','About 300,000 km/s.'),
    ('Read: "A butterfly comes from a caterpillar." Where does a butterfly come from?','A butterfly comes from a caterpillar.'),
    ('Read: "The Amazon is the largest rainforest." What is the largest rainforest?','The Amazon.'),
    ('Read: "The human body is about 60% water." How much of the body is water?','About 60%.'),
    ('Read: "Sound travels at about 343 m/s in air." How fast is sound?','About 343 m/s in air.'),
    ('Read: "There are 7 continents." How many continents are there?','There are 7 continents.'),
    ('Read: "The Moon causes ocean tides." What causes tides?','The Moon causes ocean tides.'),
    ('Read: "Trees absorb carbon dioxide." What do trees absorb?','Trees absorb carbon dioxide.'),
]
for q,a in READING:
    add('bahasa','Reading Comprehension','Passages',q,a)

# ============================================================
# 9. PYTHON CODE SNIPPETS (30+)
# ============================================================
print("Python code...")
PY = [
    ('How do you print Hello World in Python?','print("Hello, World!")'),
    ('How do you create a variable in Python?','x = 10'),
    ('How do you create a string in Python?','name = "Hello"'),
    ('How do you create a list in Python?','my_list = [1, 2, 3]'),
    ('How do you create a function in Python?','def my_function():\n    return "Hello"'),
    ('How do you write an if statement in Python?','if x > 5:\n    print("Big")'),
    ('How do you write a for loop in Python?','for i in range(5):\n    print(i)'),
    ('How do you write a while loop in Python?','while x < 10:\n    x += 1'),
    ('How do you get user input in Python?','name = input("Enter name: ")'),
    ('How do you convert string to integer in Python?','x = int("123")'),
    ('How do you find the length of a string?','len("hello") returns 5.'),
    ('How do you concatenate strings in Python?','Use + : "Hello" + " " + "World" = "Hello World".'),
    ('How do you add to a list in Python?','my_list.append(4)'),
    ('How do you remove from a list in Python?','my_list.remove(3) or my_list.pop()'),
    ('How do you sort a list in Python?','my_list.sort() or sorted(my_list)'),
    ('How do you import a module in Python?','import math or from math import sqrt'),
    ('How do you create a dictionary in Python?','my_dict = {"key": "value"}'),
    ('How do you get a dictionary value in Python?','my_dict["key"] or my_dict.get("key")'),
    ('How do you write a comment in Python?','Use # for single line or """ for multi-line.'),
    ('How do you generate random numbers in Python?','import random; x = random.randint(1, 10)'),
    ('How do you round a number in Python?','round(3.7) returns 4.'),
    ('How do you convert to uppercase in Python?','"hello".upper() returns "HELLO".'),
    ('How do you convert to lowercase in Python?','"HELLO".lower() returns "hello".'),
    ('How do you split a string in Python?','"a,b,c".split(",") returns ["a", "b", "c"].'),
    ('How do you join a list into a string?','",".join(["a","b"]) returns "a,b".'),
    ('How do you check if a key is in a dictionary?','if "key" in my_dict:'),
    ('How do you create a class in Python?','class Dog:\n    def __init__(self, name):\n        self.name = name'),
    ('How do you try-except in Python?','try:\n    x = 1/0\nexcept ZeroDivisionError:\n    print("Error")'),
    ('How do you open a file in Python?','with open("file.txt", "r") as f:\n    data = f.read()'),
    ('How do you write to a file in Python?','with open("file.txt", "w") as f:\n    f.write("Hello")'),
]
for q,a in PY:
    add('informatika','Python','Code Snippets',q,a)

# ============================================================
# 10. WORLD HOLIDAYS & FESTIVALS (40+)
# ============================================================
print("World holidays...")
HOLIDAYS = [
    ('What is New Year?','New Year celebrates the start of a new year on January 1.'),
    ('What is Chinese New Year?','Chinese New Year follows the lunar calendar, with zodiac animals.'),
    ('What is Christmas?','Christmas celebrates the birth of Jesus Christ on December 25.'),
    ('What is Easter?','Easter celebrates the resurrection of Jesus Christ.'),
    ('What is Diwali?','Diwali is the Hindu festival of lights, celebrating good over evil.'),
    ('What is Holi?','Holi is the Hindu festival of colors, celebrating spring.'),
    ('What is Ramadan?','Ramadan is the Islamic holy month of fasting from dawn to sunset.'),
    ('What is Eid al-Fitr?','Eid al-Fitr celebrates the end of Ramadan fasting.'),
    ('What is Eid al-Adha?','Eid al-Adha commemorates Ibrahim\'s willingness to sacrifice his son.'),
    ('What is Hanukkah?','Hanukkah is the Jewish festival of lights, lasting 8 days.'),
    ('What is Passover?','Passover commemorates the Jewish exodus from Egypt.'),
    ('What is Rosh Hashanah?','Rosh Hashanah is the Jewish New Year.'),
    ('What is Yom Kippur?','Yom Kippur is the Jewish Day of Atonement.'),
    ('What is Thanksgiving?','Thanksgiving is a US holiday to give thanks, on the 4th Thursday of November.'),
    ('What is Halloween?','Halloween is on October 31, when children dress up and trick-or-treat.'),
    ('What is Valentine\'s Day?','Valentine\'s Day on February 14 celebrates love.'),
    ('What is St. Patrick\'s Day?','St. Patrick\'s Day on March 17 celebrates Irish culture.'),
    ('What is the 4th of July?','The 4th of July is US Independence Day.'),
    ('What is Bastille Day?','Bastille Day on July 14 is France\'s national day.'),
    ('What is Canada Day?','Canada Day on July 1 celebrates Canadian confederation.'),
    ('What is Australia Day?','Australia Day on January 26 marks the First Fleet arrival.'),
    ('What is Independence Day of India?','India\'s Independence Day is August 15.'),
    ('What is Indonesia\'s Independence Day?','Indonesia\'s Independence Day is August 17.'),
    ('What is Independence Day of the US?','US Independence Day is July 4, 1776.'),
    ('What is Vesak?','Vesak celebrates the birth, enlightenment, and death of Buddha.'),
    ('What is Nowruz?','Nowruz is the Persian New Year, marking the start of spring.'),
    ('What is Songkran?','Songkran is the Thai New Year, celebrated with water fights.'),
    ('What is Oktoberfest?','Oktoberfest is a beer festival in Munich, Germany.'),
    ('What is Carnival?','Carnival is a festival before Lent, celebrated in Brazil and Europe.'),
    ('What is La Tomatina?','La Tomatina is a tomato-throwing festival in Spain.'),
    ('What is Holi?','Holi is the Hindu festival of colors, celebrating spring.'),
    ('What is Setsubun?','Setsubun is a Japanese festival marking the start of spring.'),
    ('What is Golden Week?','Golden Week is a collection of Japanese holidays in late April/early May.'),
    ('What is Obon?','Obon is a Japanese Buddhist festival honoring ancestors.'),
    ('What is Children\'s Day?','Children\'s Day celebrates children (Nov 20 worldwide, May 5 Japan).'),
    ('What is Mother\'s Day?','Mother\'s Day honors mothers (2nd Sunday of May in many countries).'),
    ('What is Father\'s Day?','Father\'s Day honors fathers (3rd Sunday of June in many countries).'),
    ('What is Earth Day?','Earth Day on April 22 promotes environmental protection.'),
    ('What is World Environment Day?','World Environment Day on June 5 promotes environmental awareness.'),
    ('What is International Women\'s Day?','International Women\'s Day on March 8 celebrates women\'s achievements.'),
    ('What is Labor Day?','Labor Day honors workers (May 1 internationally, September in US).'),
    ('What is Memorial Day?','Memorial Day honors fallen soldiers (last Monday of May in US).'),
    ('What is ANZAC Day?','ANZAC Day on April 25 honors Australian and New Zealand soldiers.'),
    ('What is Burns Night?','Burns Night on January 25 celebrates Scottish poet Robert Burns.'),
    ('What is Diwali?','Diwali is celebrated by Hindus, Sikhs, and Jains worldwide.'),
    ('What is Day of the Dead?','Day of the Dead (Dia de los Muertos) is a Mexican holiday on Nov 1-2.'),
]
for q,a in HOLIDAYS:
    add('culture','World Holidays','Festivals',q,a)

# ============================================================
# OUTPUT
# ============================================================
print(f"\nTotal new items: {len(items)}")

OUT = Path("/home/z/my-project/download/llm/dataset_keruk3")
OUT.mkdir(parents=True, exist_ok=True)
with open(OUT / 'dataset_keruk3.jsonl', 'w', encoding='utf-8') as f:
    for item in items:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f"\nBreakdown:")
from collections import Counter
cats = Counter(item['bidang'] for item in items)
for cat, count in sorted(cats.items(), key=lambda x: -x[1]):
    print(f"  {cat}: {count}")
print(f"\nTotal: {sum(cats.values())}")
