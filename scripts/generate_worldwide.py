"""
D'QUILANE WORLDWIDE SD DATASET
==============================
Covers elementary school curriculum from ALL OVER THE WORLD:
- US (K-5 Common Core)
- UK (Primary Year 1-6)
- Japan (小学校)
- China (小学)
- India
- Europe (France, Germany, Spain, Italy)
- Australia
- Africa
- South America
- World facts, world geography, world history
"""
import json, random
from pathlib import Path

random.seed(42)
items = []

def add(cat, judul, sub, q, a):
    items.append({'input': q, 'output': a, 'bidang': cat, 'judul': judul, 'subjudul': sub})

# ============================================================
# 1. US ELEMENTARY (K-5)
# ============================================================
print("US Elementary...")

# US States & Capitals (50 states)
US_STATES = [
    ('Alabama','Montgomery'),('Alaska','Juneau'),('Arizona','Phoenix'),('Arkansas','Little Rock'),
    ('California','Sacramento'),('Colorado','Denver'),('Connecticut','Hartford'),
    ('Delaware','Dover'),('Florida','Tallahassee'),('Georgia','Atlanta'),
    ('Hawaii','Honolulu'),('Idaho','Boise'),('Illinois','Springfield'),('Indiana','Indianapolis'),
    ('Iowa','Des Moines'),('Kansas','Topeka'),('Kentucky','Frankfort'),('Louisiana','Baton Rouge'),
    ('Maine','Augusta'),('Maryland','Annapolis'),('Massachusetts','Boston'),('Michigan','Lansing'),
    ('Minnesota','Saint Paul'),('Mississippi','Jackson'),('Missouri','Jefferson City'),
    ('Montana','Helena'),('Nebraska','Lincoln'),('Nevada','Carson City'),
    ('New Hampshire','Concord'),('New Jersey','Trenton'),('New Mexico','Santa Fe'),
    ('New York','Albany'),('North Carolina','Raleigh'),('North Dakota','Bismarck'),
    ('Ohio','Columbus'),('Oklahoma','Oklahoma City'),('Oregon','Salem'),
    ('Pennsylvania','Harrisburg'),('Rhode Island','Providence'),('South Carolina','Columbia'),
    ('South Dakota','Pierre'),('Tennessee','Nashville'),('Texas','Austin'),
    ('Utah','Salt Lake City'),('Vermont','Montpelier'),('Virginia','Richmond'),
    ('Washington','Olympia'),('West Virginia','Charleston'),('Wisconsin','Madison'),
    ('Wyoming','Cheyenne'),
]
for state, capital in US_STATES:
    add('US_elementary','US Geography','State Capitals',
        f'What is the capital of {state}?', f'The capital of {state} is {capital}.')
    add('US_elementary','US Geography','State Capitals (Reverse)',
        f'{capital} is the capital of which US state?', f'{capital} is the capital of {state}.')

# US Presidents
US_PRESIDENTS = [
    (1,'George Washington'),(2,'John Adams'),(3,'Thomas Jefferson'),(4,'James Madison'),
    (5,'James Monroe'),(6,'John Quincy Adams'),(7,'Andrew Jackson'),(8,'Martin Van Buren'),
    (9,'William Henry Harrison'),(10,'John Tyler'),(11,'James K. Polk'),(12,'Zachary Taylor'),
    (13,'Millard Fillmore'),(14,'Franklin Pierce'),(15,'James Buchanan'),
    (16,'Abraham Lincoln'),(17,'Andrew Johnson'),(18,'Ulysses S. Grant'),
    (19,'Rutherford B. Hayes'),(20,'James A. Garfield'),(21,'Chester A. Arthur'),
    (22,'Grover Cleveland'),(23,'Benjamin Harrison'),(24,'Grover Cleveland'),
    (25,'William McKinley'),(26,'Theodore Roosevelt'),(27,'William Howard Taft'),
    (28,'Woodrow Wilson'),(29,'Warren G. Harding'),(30,'Calvin Coolidge'),
    (31,'Herbert Hoover'),(32,'Franklin D. Roosevelt'),(33,'Harry S. Truman'),
    (34,'Dwight D. Eisenhower'),(35,'John F. Kennedy'),(36,'Lyndon B. Johnson'),
    (37,'Richard Nixon'),(38,'Gerald Ford'),(39,'Jimmy Carter'),(40,'Ronald Reagan'),
    (41,'George H. W. Bush'),(42,'Bill Clinton'),(43,'George W. Bush'),
    (44,'Barack Obama'),(45,'Donald Trump'),(46,'Joe Biden'),
]
for num, name in US_PRESIDENTS:
    add('US_elementary','US History','Presidents',
        f'Who was the {num}{"st" if num==1 else "nd" if num==2 else "rd" if num==3 else "th"} President of the United States?',
        f'The {num}{"st" if num==1 else "nd" if num==2 else "rd" if num==3 else "th"} President was {name}.')

# US History
US_HIST = [
    ('When did America gain independence?','America declared independence on July 4, 1776.'),
    ('From which country did America gain independence?','America gained independence from Great Britain.'),
    ('Who was the first President of the United States?','George Washington was the first President.'),
    ('What is the Declaration of Independence?','The Declaration of Independence is the document declaring freedom from Britain.'),
    ('What is the US Constitution?','The Constitution is the supreme law of the United States.'),
    ('How many amendments in the Bill of Rights?','There are 10 amendments in the Bill of Rights.'),
    ('What is the First Amendment?','The First Amendment protects freedom of speech, religion, and press.'),
    ('What is the US flag called?','The US flag is called the Stars and Stripes.'),
    ('How many stars on the US flag?','There are 50 stars on the US flag, one for each state.'),
    ('How many stripes on the US flag?','There are 13 stripes, representing the original 13 colonies.'),
    ('What is the national bird of the US?','The bald eagle is the national bird of the United States.'),
    ('What is the US national anthem?','The Star-Spangled Banner is the US national anthem.'),
    ('Who wrote the US national anthem?','Francis Scott Key wrote the Star-Spangled Banner.'),
    ('What is the Statue of Liberty?','The Statue of Liberty is a gift from France, symbolizing freedom.'),
    ('Where is the Statue of Liberty?','The Statue of Liberty is in New York Harbor.'),
    ('What is the White House?','The White House is the official residence of the US President.'),
    ('Where is the White House?','The White House is in Washington, D.C.'),
    ('How many branches of US government?','There are 3 branches: executive, legislative, and judicial.'),
    ('What is the executive branch?','The executive branch is headed by the President.'),
    ('What is the legislative branch?','The legislative branch is Congress (Senate and House of Representatives).'),
    ('What is the judicial branch?','The judicial branch is the Supreme Court.'),
    ('How many senators does each state have?','Each state has 2 senators.'),
    ('How long is a presidential term?','A presidential term is 4 years.'),
    ('What is the maximum number of terms?','A president can serve maximum 2 terms (8 years).'),
    ('What is Thanksgiving?','Thanksgiving is a US holiday on the fourth Thursday of November.'),
    ('When is Independence Day?','Independence Day is July 4th.'),
    ('What is Memorial Day?','Memorial Day honors soldiers who died in service, last Monday of May.'),
    ('What is Labor Day?','Labor Day honors workers, first Monday of September.'),
    ('What is Martin Luther King Jr. Day?','MLK Day honors civil rights leader, third Monday of January.'),
    ('Who was Martin Luther King Jr.?','MLK was a civil rights leader who fought for equality.'),
]
for q, a in US_HIST:
    add('US_elementary','US History','Events', q, a)

# US Measurement
US_MEAS = [
    ('How many inches in a foot?','There are 12 inches in a foot.'),
    ('How many feet in a yard?','There are 3 feet in a yard.'),
    ('How many feet in a mile?','There are 5,280 feet in a mile.'),
    ('How many ounces in a pound?','There are 16 ounces in a pound.'),
    ('How many pounds in a ton?','There are 2,000 pounds in a ton.'),
    ('How many cups in a pint?','There are 2 cups in a pint.'),
    ('How many pints in a quart?','There are 2 pints in a quart.'),
    ('How many quarts in a gallon?','There are 4 quarts in a gallon.'),
    ('How many cups in a gallon?','There are 16 cups in a gallon.'),
    ('How many inches in a yard?','There are 36 inches in a yard.'),
    ('What is freezing in Fahrenheit?','Water freezes at 32 degrees Fahrenheit.'),
    ('What is boiling in Fahrenheit?','Water boils at 212 degrees Fahrenheit.'),
    ('What is body temperature in Fahrenheit?','Body temperature is about 98.6 degrees Fahrenheit.'),
    ('What is room temperature in Fahrenheit?','Room temperature is about 72 degrees Fahrenheit.'),
]
for q, a in US_MEAS:
    add('US_elementary','US Math','Measurement', q, a)

# US Money
add('US_elementary','US Math','Money','What coins are used in the US?','US coins: penny (1 cent), nickel (5 cents), dime (10 cents), quarter (25 cents).')
add('US_elementary','US Math','Money','How many cents in a dollar?','There are 100 cents in one dollar.')
add('US_elementary','US Math','Money','What is a penny worth?','A penny is worth 1 cent.')
add('US_elementary','US Math','Money','What is a nickel worth?','A nickel is worth 5 cents.')
add('US_elementary','US Math','Money','What is a dime worth?','A dime is worth 10 cents.')
add('US_elementary','US Math','Money','What is a quarter worth?','A quarter is worth 25 cents.')

# US Geography
US_GEO = [
    ('What is the largest US state by area?','Alaska is the largest US state by area.'),
    ('What is the smallest US state?','Rhode Island is the smallest US state.'),
    ('What is the most populous US state?','California is the most populous US state.'),
    ('What ocean is east of the US?','The Atlantic Ocean is east of the US.'),
    ('What ocean is west of the US?','The Pacific Ocean is west of the US.'),
    ('What is the longest river in the US?','The Missouri River is the longest river in the US.'),
    ('What is the Grand Canyon?','The Grand Canyon is a massive canyon in Arizona.'),
    ('What is Yellowstone?','Yellowstone is the first national park in the US, in Wyoming.'),
    ('What are the Great Lakes?','The Great Lakes are 5 large lakes on the US-Canada border.'),
    ('What is Mount Rushmore?','Mount Rushmore has carved faces of 4 US presidents in South Dakota.'),
    ('Which presidents are on Mount Rushmore?','Washington, Jefferson, Roosevelt, and Lincoln are on Mount Rushmore.'),
    ('What is the Mississippi River?','The Mississippi is one of the longest rivers in the US.'),
    ('What is Death Valley?','Death Valley is the hottest and lowest point in the US.'),
    ('What is the Rocky Mountains?','The Rocky Mountains are a major mountain range in the western US.'),
    ('What is the Appalachian Mountains?','The Appalachian Mountains are in the eastern US.'),
]
for q, a in US_GEO:
    add('US_elementary','US Geography','Landmarks', q, a)

# US Science
US_SCI = [
    ('What are the three states of matter?','Solid, liquid, and gas are the three states of matter.'),
    ('What is the water cycle?','The water cycle: evaporation, condensation, precipitation, collection.'),
    ('What is photosynthesis?','Photosynthesis is how plants make food from sunlight, water, and CO2.'),
    ('What is the solar system?','The solar system is the Sun and 8 planets that orbit it.'),
    ('What is gravity?','Gravity is the force that pulls objects toward Earth.'),
    ('What is friction?','Friction is a force that slows down moving objects.'),
    ('What is electricity?','Electricity is the flow of electrons through a conductor.'),
    ('What is a magnet?','A magnet is an object that attracts iron and has north and south poles.'),
    ('What is an ecosystem?','An ecosystem is a community of living and non-living things.'),
    ('What is a food chain?','A food chain shows who eats whom in an ecosystem.'),
    ('What are producers in a food chain?','Producers are plants that make their own food.'),
    ('What are consumers in a food chain?','Consumers are animals that eat plants or other animals.'),
    ('What are decomposers?','Decomposers are organisms that break down dead matter.'),
    ('What is the Earth made of?','Earth has layers: crust, mantle, outer core, and inner core.'),
    ('What causes seasons?','Seasons are caused by Earth\'s tilt as it orbits the Sun.'),
    ('What causes day and night?','Day and night are caused by Earth\'s rotation.'),
    ('How long does Earth take to rotate?','Earth rotates once every 24 hours.'),
    ('How long does Earth take to orbit the Sun?','Earth orbits the Sun once every 365 days.'),
    ('What is the Moon?','The Moon is Earth\'s natural satellite.'),
    ('What causes ocean tides?','Ocean tides are caused by the Moon\'s gravity.'),
]
for q, a in US_SCI:
    add('US_elementary','US Science','General', q, a)

# ============================================================
# 2. UK PRIMARY (Year 1-6)
# ============================================================
print("UK Primary...")
UK = [
    ('What is the capital of England?','The capital of England is London.'),
    ('What is the capital of Scotland?','The capital of Scotland is Edinburgh.'),
    ('What is the capital of Wales?','The capital of Wales is Cardiff.'),
    ('What is the capital of Northern Ireland?','The capital of Northern Ireland is Belfast.'),
    ('What countries make up the UK?','The UK consists of England, Scotland, Wales, and Northern Ireland.'),
    ('Who is the British monarch?','The British monarch is the King or Queen of the United Kingdom.'),
    ('What is Big Ben?','Big Ben is the famous clock tower in London.'),
    ('What is Stonehenge?','Stonehenge is an ancient stone circle in England.'),
    ('What is the Tower of London?','The Tower of London is a historic castle in London.'),
    ('What is Buckingham Palace?','Buckingham Palace is the residence of the British monarch.'),
    ('What is the UK Parliament?','The UK Parliament makes laws, with House of Commons and House of Lords.'),
    ('What is the River Thames?','The River Thames is the main river flowing through London.'),
    ('What is the UK currency?','The UK currency is the pound sterling.'),
    ('How many pence in a pound?','There are 100 pence in one pound.'),
    ('What is the UK national anthem?','God Save the King is the UK national anthem.'),
    ('What is the Magna Carta?','The Magna Carta was a document limiting royal power in 1215.'),
    ('Who was Queen Victoria?','Queen Victoria ruled the UK for 63 years during the Victorian era.'),
    ('What is the Industrial Revolution?','The Industrial Revolution was when manufacturing moved from hand to machine.'),
    ('Where did the Industrial Revolution start?','The Industrial Revolution started in Great Britain.'),
    ('What is the NHS?','The NHS is the National Health Service, providing free healthcare.'),
    ('What is the BBC?','The BBC is the British Broadcasting Corporation, a public service broadcaster.'),
    ('What is the London Underground?','The London Underground, or Tube, is the underground railway system.'),
    ('What is Hadrian\'s Wall?','Hadrian\'s Wall was built by Romans to protect their northern border.'),
    ('What is Loch Ness?','Loch Ness is a famous lake in Scotland, home of the Loch Ness Monster.'),
    ('What are the Scottish Highlands?','The Scottish Highlands are a mountainous region in northern Scotland.'),
    ('What is the UK flag called?','The UK flag is called the Union Jack.'),
    ('What is the UK national flower of England?','The rose is the national flower of England.'),
    ('What is the UK national flower of Scotland?','The thistle is the national flower of Scotland.'),
    ('What is the UK national flower of Wales?','The daffodil is the national flower of Wales.'),
    ('What is Burns Night?','Burns Night celebrates Scottish poet Robert Burns on January 25.'),
    ('What is St. Patrick\'s Day?','St. Patrick\'s Day celebrates the patron saint of Ireland on March 17.'),
    ('What is Bonfire Night?','Bonfire Night, or Guy Fawkes Night, is on November 5.'),
    ('What is Remembrance Day?','Remembrance Day honors war dead on November 11.'),
    ('What is Boxing Day?','Boxing Day is the day after Christmas, December 26.'),
    ('What is the Commonwealth?','The Commonwealth is an association of former British colonies.'),
    ('How many countries in the Commonwealth?','There are 56 countries in the Commonwealth.'),
]
for q, a in UK:
    add('UK_primary','UK Studies','General', q, a)

# ============================================================
# 3. JAPAN ELEMENTARY (小学校)
# ============================================================
print("Japan Elementary...")
JAPAN = [
    ('What is the capital of Japan?','The capital of Japan is Tokyo.'),
    ('What are the main islands of Japan?','Japan has four main islands: Honshu, Hokkaido, Kyushu, and Shikoku.'),
    ('What is the highest mountain in Japan?','Mount Fuji is the highest mountain in Japan.'),
    ('How tall is Mount Fuji?','Mount Fuji is 3,776 meters tall.'),
    ('What is the currency of Japan?','The currency of Japan is the Japanese yen.'),
    ('What is the Japanese national anthem?','Kimigayo is the Japanese national anthem.'),
    ('What is the Japanese flag?','The Japanese flag is white with a red circle in the center, called Hinomaru.'),
    ('What is the Japanese writing system?','Japan uses three writing systems: hiragana, katakana, and kanji.'),
    ('What is a kimono?','A kimono is a traditional Japanese garment.'),
    ('What is sushi?','Sushi is a Japanese dish of vinegared rice with fish or vegetables.'),
    ('What is ramen?','Ramen is a Japanese noodle soup dish.'),
    ('What is tempura?','Tempura is a Japanese dish of battered and fried seafood or vegetables.'),
    ('What is origami?','Origami is the Japanese art of paper folding.'),
    ('What is sumo?','Sumo is a traditional Japanese wrestling sport.'),
    ('What is karate?','Karate is a Japanese martial art.'),
    ('What is judo?','Judo is a Japanese martial art focused on throws.'),
    ('What is kendo?','Kendo is a Japanese martial art using bamboo swords.'),
    ('What is the Japanese school year?','The Japanese school year starts in April and ends in March.'),
    ('What is Golden Week in Japan?','Golden Week is a collection of national holidays in late April and early May.'),
    ('What is hanami?','Hanami is the Japanese tradition of viewing cherry blossoms.'),
    ('What is the shinkansen?','The shinkansen is the Japanese bullet train.'),
    ('What is Tokyo Tower?','Tokyo Tower is a communications tower in Tokyo, inspired by the Eiffel Tower.'),
    ('What is the Imperial Palace?','The Imperial Palace is the residence of the Japanese Emperor in Tokyo.'),
    ('Who is the Japanese Emperor?','The Emperor is the symbolic head of state of Japan.'),
    ('What is the Japanese Diet?','The Diet is the national parliament of Japan.'),
    ('What is sake?','Sake is a Japanese rice wine.'),
    ('What is matcha?','Matcha is a powdered green tea used in Japanese tea ceremonies.'),
    ('What is the Japanese tea ceremony?','The tea ceremony is a traditional ritual of preparing and serving matcha.'),
    ('What is manga?','Manga are Japanese comic books or graphic novels.'),
    ('What is anime?','Anime is Japanese animation or cartoons.'),
    ('What is Nintendo?','Nintendo is a Japanese video game company.'),
    ('What is Pokemon?','Pokemon is a Japanese media franchise about catching creatures.'),
    ('What is Shinto?','Shinto is the indigenous religion of Japan.'),
    ('What is Buddhism in Japan?','Buddhism was introduced to Japan from China and Korea.'),
    ('What is a torii gate?','A torii is a traditional Japanese gate marking the entrance to a Shinto shrine.'),
    ('What is a samurai?','A samurai was a warrior class in feudal Japan.'),
    ('What is bushido?','Bushido is the code of the samurai, emphasizing loyalty and honor.'),
    ('What is the Edo period?','The Edo period was from 1603 to 1868, under the Tokugawa shogunate.'),
    ('What is the Meiji Restoration?','The Meiji Restoration modernized Japan starting in 1868.'),
    ('What is Hiroshima?','Hiroshima is a city in Japan, known for being hit by an atomic bomb in 1945.'),
    ('What is Nagasaki?','Nagasaki is a city in Japan, the second city hit by an atomic bomb in 1945.'),
    ('What is the Japanese New Year?','Japanese New Year, or Oshogatsu, is the most important holiday in Japan.'),
    ('What is Obon?','Obon is a Japanese Buddhist festival honoring ancestors in August.'),
    ('What is Setsubun?','Setsubun is a Japanese festival marking the beginning of spring.'),
    ('What is Children\'s Day in Japan?','Children\'s Day is on May 5, celebrating children\'s health and happiness.'),
    ('What is Girls\' Day in Japan?','Girls\' Day, or Hinamatsuri, is on March 3.'),
    ('What is Boys\' Day in Japan?','Boys\' Day is on May 5, now celebrated as Children\'s Day.'),
    ('What is the Japanese currency symbol?','The Japanese yen symbol is a circle with a line, similar to a Y.'),
    ('What is a yen made of?','Yen coins are made of aluminum, bronze, and other metals.'),
    ('What is a ryokan?','A ryokan is a traditional Japanese inn.'),
    ('What is an onsen?','An onsen is a Japanese hot spring.'),
]
for q, a in JAPAN:
    add('Japan_elementary','Japan Studies','General', q, a)

# ============================================================
# 4. CHINA ELEMENTARY (小学)
# ============================================================
print("China Elementary...")
CHINA = [
    ('What is the capital of China?','The capital of China is Beijing.'),
    ('What is the Great Wall of China?','The Great Wall is a massive wall built to protect China from invaders.'),
    ('How long is the Great Wall?','The Great Wall is about 21,000 kilometers long.'),
    ('What is the Forbidden City?','The Forbidden City is the former imperial palace in Beijing.'),
    ('What is the Terracotta Army?','The Terracotta Army is a collection of clay soldiers in Xi\'an.'),
    ('What is the currency of China?','The currency of China is the Chinese yuan, also called renminbi.'),
    ('What is the most spoken language in China?','Mandarin Chinese is the most spoken language in China.'),
    ('What is the Chinese writing system?','Chinese uses characters, with simplified and traditional forms.'),
    ('What is the Chinese New Year?','Chinese New Year is the most important traditional Chinese holiday.'),
    ('What animal represents the Chinese New Year cycle?','The Chinese zodiac has 12 animals, one for each year.'),
    ('What are the 12 Chinese zodiac animals?','Rat, Ox, Tiger, Rabbit, Dragon, Snake, Horse, Goat, Monkey, Rooster, Dog, and Pig.'),
    ('What is the Mid-Autumn Festival?','The Mid-Autumn Festival celebrates the harvest moon.'),
    ('What is the Dragon Boat Festival?','The Dragon Boat Festival commemorates the poet Qu Yuan.'),
    ('What is Confucianism?','Confucianism is a philosophy emphasizing respect, loyalty, and education.'),
    ('Who was Confucius?','Confucius was a Chinese philosopher who lived around 500 BCE.'),
    ('What is Taoism?','Taoism is a Chinese philosophy emphasizing harmony with nature.'),
    ('What is the Yangtze River?','The Yangtze is the longest river in China and Asia.'),
    ('What is the Yellow River?','The Yellow River, or Huang He, is known as the cradle of Chinese civilization.'),
    ('What is Mount Everest called in China?','Mount Everest is called Qomolangma in China.'),
    ('What is the Silk Road?','The Silk Road was a trade route connecting China to Europe.'),
    ('What is kung fu?','Kung fu is a Chinese martial art.'),
    ('What is tai chi?','Tai chi is a Chinese practice of slow movements for health and meditation.'),
    ('What is the Ming Dynasty?','The Ming Dynasty ruled China from 1368 to 1644.'),
    ('What is the Qing Dynasty?','The Qing Dynasty was the last imperial dynasty of China, 1644-1912.'),
    ('What is panda?','The giant panda is native to China and is a national symbol.'),
    ('What is Peking duck?','Peking duck is a famous Chinese dish from Beijing.'),
    ('What is dim sum?','Dim sum is a Chinese cuisine of small bite-sized portions.'),
    ('What is rice?','Rice is the staple food in southern China, while wheat is common in the north.'),
    ('What is calligraphy?','Chinese calligraphy is the art of writing Chinese characters beautifully.'),
    ('What is a pagoda?','A pagoda is a multi-tiered tower common in East Asian architecture.'),
    ('What is chopsticks?','Chopsticks are the traditional eating utensils in China.'),
    ('What is the Lantern Festival?','The Lantern Festival marks the end of Chinese New Year celebrations.'),
    ('What is tea in Chinese culture?','Tea is an important part of Chinese culture, originating in China.'),
    ('What is the Yangtze River length?','The Yangtze River is about 6,300 kilometers long.'),
    ('What is the population of China?','China has about 1.4 billion people, the largest in the world.'),
    ('What is Shanghai?','Shanghai is the largest city in China by population.'),
    ('What is Hong Kong?','Hong Kong is a special administrative region of China.'),
    ('What is Macau?','Macau is a special administrative region of China, known for casinos.'),
    ('What is paper?','Paper was invented in China during the Han Dynasty.'),
    ('What is printing?','Printing was invented in China, with movable type by Bi Sheng.'),
    ('What is gunpowder?','Gunpowder was invented in China during the Tang Dynasty.'),
    ('What is the compass?','The compass was invented in China during the Han Dynasty.'),
    ('What are the Four Great Inventions of China?','The Four Great Inventions are paper, printing, gunpowder, and the compass.'),
    ('What is the Chinese flag?','The Chinese flag is red with five yellow stars, one large and four small.'),
    ('What is Tiananmen Square?','Tiananmen Square is a large public square in Beijing.'),
    ('What is the Temple of Heaven?','The Temple of Heaven is an imperial complex in Beijing for prayers.'),
    ('What is the Summer Palace?','The Summer Palace is a former imperial garden in Beijing.'),
    ('What is acupuncture?','Acupuncture is a traditional Chinese medical practice using thin needles.'),
]
for q, a in CHINA:
    add('China_elementary','China Studies','General', q, a)

# ============================================================
# 5. INDIA ELEMENTARY
# ============================================================
print("India Elementary...")
INDIA = [
    ('What is the capital of India?','The capital of India is New Delhi.'),
    ('What is the currency of India?','The currency of India is the Indian rupee.'),
    ('What is the Taj Mahal?','The Taj Mahal is a famous mausoleum in Agra, India.'),
    ('Who built the Taj Mahal?','The Taj Mahal was built by Emperor Shah Jahan for his wife.'),
    ('What is the Ganges River?','The Ganges is the most sacred river in India.'),
    ('What is the Himalayas?','The Himalayas are the highest mountain range in the world, in northern India.'),
    ('What is Diwali?','Diwali is the festival of lights, the most important Hindu festival.'),
    ('What is Holi?','Holi is the festival of colors, celebrating spring.'),
    ('What is the caste system?','The caste system is a traditional social hierarchy in India.'),
    ('Who was Mahatma Gandhi?','Gandhi led India\'s independence through non-violence.'),
    ('When did India gain independence?','India gained independence from Britain on August 15, 1947.'),
    ('What is the Indian national flag?','The Indian flag has saffron, white, and green stripes with a blue chakra.'),
    ('What is the Indian national anthem?','Jana Gana Mana is the Indian national anthem.'),
    ('What is Bollywood?','Bollywood is the Hindi film industry based in Mumbai.'),
    ('What is yoga?','Yoga is a practice for physical and mental well-being, originating in India.'),
    ('What is Ayurveda?','Ayurveda is a traditional Indian system of medicine.'),
    ('What is cricket?','Cricket is the most popular sport in India.'),
    ('What is the Indian population?','India has about 1.4 billion people.'),
    ('What languages are spoken in India?','India has 22 official languages, including Hindi and English.'),
    ('What is the caste system?','The caste system was a social stratification, now officially abolished.'),
    ('What is the monsoon?','The monsoon is a seasonal wind bringing heavy rain to India.'),
    ('What is curry?','Curry is a dish with spices, common in Indian cuisine.'),
    ('What is naan?','Naan is a type of flatbread popular in India.'),
    ('What is chai?','Chai is Indian spiced tea with milk and spices.'),
    ('What is the lotus?','The lotus is the national flower of India.'),
    ('What is the Bengal tiger?','The Bengal tiger is the national animal of India.'),
    ('What is the peacock?','The peacock is the national bird of India.'),
]
for q, a in INDIA:
    add('India_elementary','India Studies','General', q, a)

# ============================================================
# 6. EUROPE
# ============================================================
print("Europe Elementary...")
EUROPE = [
    ('What is the capital of France?','The capital of France is Paris.'),
    ('What is the Eiffel Tower?','The Eiffel Tower is an iron tower in Paris, built in 1889.'),
    ('What is the Louvre?','The Louvre is a famous art museum in Paris.'),
    ('What is the capital of Germany?','The capital of Germany is Berlin.'),
    ('What is the Brandenburg Gate?','The Brandenburg Gate is a famous landmark in Berlin.'),
    ('What is the capital of Italy?','The capital of Italy is Rome.'),
    ('What is the Colosseum?','The Colosseum is an ancient Roman amphitheater in Rome.'),
    ('What is the capital of Spain?','The capital of Spain is Madrid.'),
    ('What is the capital of Greece?','The capital of Greece is Athens.'),
    ('What is the Parthenon?','The Parthenon is an ancient temple in Athens.'),
    ('What is the Acropolis?','The Acropolis is an ancient citadel in Athens.'),
    ('What is the capital of the Netherlands?','The capital of the Netherlands is Amsterdam.'),
    ('What is the capital of Belgium?','The capital of Belgium is Brussels.'),
    ('What is the capital of Switzerland?','The capital of Switzerland is Bern.'),
    ('What is the capital of Austria?','The capital of Austria is Vienna.'),
    ('What is the capital of Sweden?','The capital of Sweden is Stockholm.'),
    ('What is the capital of Norway?','The capital of Norway is Oslo.'),
    ('What is the capital of Denmark?','The capital of Denmark is Copenhagen.'),
    ('What is the capital of Finland?','The capital of Finland is Helsinki.'),
    ('What is the capital of Portugal?','The capital of Portugal is Lisbon.'),
    ('What is the capital of Poland?','The capital of Poland is Warsaw.'),
    ('What is the capital of Ireland?','The capital of Ireland is Dublin.'),
    ('What is the capital of Russia?','The capital of Russia is Moscow.'),
    ('What is Red Square?','Red Square is a famous square in Moscow.'),
    ('What is the Kremlin?','The Kremlin is a fortified complex in Moscow, the seat of government.'),
    ('What is the capital of Ukraine?','The capital of Ukraine is Kyiv.'),
    ('What is the EU?','The EU is the European Union, a political and economic union of 27 countries.'),
    ('What is the Euro?','The Euro is the common currency of many EU countries.'),
    ('What is the European Union flag?','The EU flag is blue with 12 yellow stars in a circle.'),
    ('What is the Roman Empire?','The Roman Empire was an ancient empire centered in Rome.'),
    ('What is Ancient Greece?','Ancient Greece was the birthplace of democracy and philosophy.'),
    ('What is the Renaissance?','The Renaissance was a period of cultural rebirth in Europe.'),
    ('What is the Industrial Revolution?','The Industrial Revolution started in Europe, shifting from hand to machine.'),
    ('What is World War I?','World War I was from 1914 to 1918, centered in Europe.'),
    ('What is World War II?','World War II was from 1939 to 1945.'),
    ('What is the Berlin Wall?','The Berlin Wall divided East and West Berlin from 1961 to 1989.'),
    ('What is the Vatican City?','Vatican City is the smallest country in the world, inside Rome.'),
    ('What is the Pope?','The Pope is the head of the Catholic Church, based in Vatican City.'),
    ('What is Big Ben?','Big Ben is the clock tower in London.'),
    ('What is the Leaning Tower of Pisa?','The Leaning Tower of Pisa is a tilted bell tower in Italy.'),
    ('What is the Sagrada Familia?','The Sagrada Familia is a famous church in Barcelona, Spain.'),
    ('What is the Nordic region?','The Nordic region includes Sweden, Norway, Denmark, Finland, and Iceland.'),
    ('What is the metric system?','The metric system is the decimal system of measurement, used in Europe.'),
    ('What is the Mediterranean Sea?','The Mediterranean Sea is between Europe, Africa, and Asia.'),
    ('What is the Alps?','The Alps are the highest mountain range in Europe.'),
    ('What is the Danube River?','The Danube is the second longest river in Europe.'),
    ('What is the Seine River?','The Seine is the river that flows through Paris.'),
    ('What is the Thames River?','The Thames is the river that flows through London.'),
    ('What is Oktoberfest?','Oktoberfest is a beer festival in Munich, Germany.'),
    ('What is Carnival?','Carnival is a festival before Lent, celebrated in many European countries.'),
]
for q, a in EUROPE:
    add('Europe_elementary','European Studies','General', q, a)

# ============================================================
# 7. AUSTRALIA
# ============================================================
print("Australia Elementary...")
AUS = [
    ('What is the capital of Australia?','The capital of Australia is Canberra.'),
    ('What is the largest city in Australia?','Sydney is the largest city in Australia.'),
    ('What is the Sydney Opera House?','The Sydney Opera House is a famous building shaped like shells.'),
    ('What is the Great Barrier Reef?','The Great Barrier Reef is the world\'s largest coral reef system.'),
    ('What is the Outback?','The Outback is the remote, arid interior of Australia.'),
    ('What animals are native to Australia?','Kangaroos, koalas, wallabies, wombats, and platypuses are native to Australia.'),
    ('What is a kangaroo?','A kangaroo is a marsupial that hops and carries babies in a pouch.'),
    ('What is a koala?','A koala is a marsupial that lives in trees and eats eucalyptus leaves.'),
    ('What is the currency of Australia?','The currency of Australia is the Australian dollar.'),
    ('What is the Australian flag?','The Australian flag is blue with the Union Jack, Southern Cross, and Commonwealth Star.'),
    ('What are the Aboriginal peoples?','Aboriginal peoples are the indigenous inhabitants of Australia.'),
    ('What is Uluru?','Uluru, or Ayers Rock, is a large sandstone rock formation in central Australia.'),
    ('What is the Australian Outback known for?','The Outback is known for its desert, red soil, and unique wildlife.'),
    ('What is a boomerang?','A boomerang is a throwing stick that returns to the thrower, used by Aboriginal peoples.'),
    ('What is a didgeridoo?','A didgeridoo is a wind instrument used by Aboriginal peoples.'),
    ('What is the Australian summer?','Australian summer is in December, January, and February.'),
    ('What is Australia Day?','Australia Day is on January 26, marking the arrival of the First Fleet.'),
    ('What is ANZAC Day?','ANZAC Day on April 25 honors Australian and New Zealand soldiers.'),
    ('What is the Great Ocean Road?','The Great Ocean Road is a scenic coastal drive in Victoria, Australia.'),
    ('What is the Tasmanian devil?','The Tasmanian devil is a carnivorous marsupial found in Tasmania.'),
    ('What is the platypus?','The platypus is a unique mammal that lays eggs and has a duck bill.'),
    ('What is the emu?','The emu is a large flightless bird native to Australia.'),
    ('What is the kookaburra?','The kookaburra is a bird known for its laughing call.'),
    ('What is the dingo?','The dingo is a wild dog found in Australia.'),
    ('What is the Great Dividing Range?','The Great Dividing Range is the largest mountain range in Australia.'),
]
for q, a in AUS:
    add('Australia_elementary','Australia Studies','General', q, a)

# ============================================================
# 8. WORLD FACTS (universal)
# ============================================================
print("World Facts...")
WORLD = [
    ('What is the largest country in the world?','Russia is the largest country in the world by area.'),
    ('What is the smallest country in the world?','Vatican City is the smallest country in the world.'),
    ('What is the most populous country?','India and China are the most populous countries.'),
    ('What is the largest ocean?','The Pacific Ocean is the largest ocean.'),
    ('What is the largest continent?','Asia is the largest continent.'),
    ('What is the smallest continent?','Australia is the smallest continent.'),
    ('What is the coldest continent?','Antarctica is the coldest continent.'),
    ('What is the highest mountain in the world?','Mount Everest is the highest mountain in the world.'),
    ('How tall is Mount Everest?','Mount Everest is 8,849 meters tall.'),
    ('What is the longest river in the world?','The Nile River is the longest river in the world.'),
    ('How long is the Nile River?','The Nile River is about 6,650 kilometers long.'),
    ('What is the largest desert in the world?','The Sahara is the largest hot desert, but Antarctica is the largest desert overall.'),
    ('What is the largest country by population?','India and China have the largest populations.'),
    ('What is the largest island in the world?','Greenland is the largest island in the world.'),
    ('What is the largest lake in the world?','The Caspian Sea is the largest lake by area.'),
    ('What is the deepest ocean trench?','The Mariana Trench is the deepest ocean trench.'),
    ('How deep is the Mariana Trench?','The Mariana Trench is about 11,000 meters deep.'),
    ('What is the largest rainforest?','The Amazon Rainforest is the largest rainforest.'),
    ('Where is the Amazon Rainforest?','The Amazon Rainforest is in South America, mostly in Brazil.'),
    ('What is the largest country in Africa?','Algeria is the largest country in Africa by area.'),
    ('What is the longest river in Africa?','The Nile River is the longest river in Africa.'),
    ('What is the largest lake in Africa?','Lake Victoria is the largest lake in Africa.'),
    ('What is the Sahara Desert?','The Sahara is the largest hot desert, located in North Africa.'),
    ('What is the capital of Egypt?','The capital of Egypt is Cairo.'),
    ('What are the pyramids of Giza?','The pyramids of Giza are ancient Egyptian tombs.'),
    ('Who built the pyramids?','The pyramids were built by ancient Egyptians as tombs for pharaohs.'),
    ('What is the Sphinx?','The Sphinx is a statue with a lion body and human head in Egypt.'),
    ('What is the largest country in South America?','Brazil is the largest country in South America.'),
    ('What is the Amazon River?','The Amazon River is the largest river by volume in the world.'),
    ('What is the capital of Brazil?','The capital of Brazil is Brasilia.'),
    ('What is the capital of Argentina?','The capital of Argentina is Buenos Aires.'),
    ('What is Machu Picchu?','Machu Picchu is an ancient Incan city in Peru.'),
    ('What is the Galapagos Islands?','The Galapagos Islands are known for unique wildlife, studied by Darwin.'),
    ('What is the Panama Canal?','The Panama Canal connects the Atlantic and Pacific Oceans.'),
    ('What is the capital of Canada?','The capital of Canada is Ottawa.'),
    ('What are the Canadian provinces?','Canada has 10 provinces and 3 territories.'),
    ('What is Niagara Falls?','Niagara Falls is a famous waterfall on the US-Canada border.'),
    ('What is the northernmost country?','Iceland and Norway are among the northernmost countries.'),
    ('What is the southernmost country?','Chile and Argentina are among the southernmost countries.'),
    ('What is the highest waterfall?','Angel Falls in Venezuela is the highest waterfall.'),
    ('What is the largest archipelago?','Indonesia is the largest archipelago in the world.'),
    ('What is the Ring of Fire?','The Ring of Fire is an area around the Pacific with many volcanoes and earthquakes.'),
    ('What is the equator?','The equator is an imaginary line dividing Earth into northern and southern hemispheres.'),
    ('What are the hemispheres?','Earth has northern, southern, eastern, and western hemispheres.'),
    ('What is the tropic of Cancer?','The Tropic of Cancer is at 23.5 degrees north of the equator.'),
    ('What is the Tropic of Capricorn?','The Tropic of Capricorn is at 23.5 degrees south of the equator.'),
    ('What is GMT?','GMT is Greenwich Mean Time, the reference for world time zones.'),
    ('What is UTC?','UTC is Coordinated Universal Time, the modern time standard.'),
    ('What are the polar regions?','The Arctic is at the North Pole, Antarctica is at the South Pole.'),
    ('What is the aurora?','Aurora, or Northern/Southern Lights, are natural light displays in polar regions.'),
    ('What is the International Space Station?','The ISS is a space station orbiting Earth, a collaboration of many countries.'),
    ('What is UNESCO?','UNESCO is a UN agency that designates World Heritage Sites.'),
    ('What is the Olympic Games?','The Olympics are an international sports event held every 4 years.'),
    ('What are the Summer and Winter Olympics?','The Summer and Winter Olympics alternate every 2 years.'),
    ('What is the Paralympic Games?','The Paralympics are for athletes with disabilities.'),
    ('What is the World Cup?','The FIFA World Cup is the biggest soccer tournament, held every 4 years.'),
    ('What is the UN?','The United Nations is an international organization for peace and cooperation.'),
    ('When was the UN founded?','The UN was founded in 1945 after World War II.'),
    ('What are human rights?','Human rights are basic rights every person is entitled to.'),
    ('What is the Universal Declaration of Human Rights?','It is a document listing basic human rights, adopted by the UN in 1948.'),
    ('What is climate change?','Climate change is long-term change in Earth\'s temperature and weather.'),
    ('What is global warming?','Global warming is the increase in Earth\'s average temperature.'),
    ('What causes global warming?','Burning fossil fuels releases greenhouse gases that trap heat.'),
    ('What is recycling?','Recycling is converting waste into reusable materials.'),
    ('What are the 3 Rs?','The 3 Rs are Reduce, Reuse, and Recycle.'),
    ('What is deforestation?','Deforestation is cutting down forests, often for agriculture.'),
    ('What is biodiversity?','Biodiversity is the variety of living things in an ecosystem.'),
    ('What is an endangered species?','An endangered species is at risk of extinction.'),
    ('What is extinction?','Extinction is when a species completely dies out.'),
    ('What is the ozone layer?','The ozone layer protects Earth from harmful UV radiation.'),
    ('What is carbon footprint?','Carbon footprint is the amount of greenhouse gases produced by human activities.'),
    ('What is solar energy?','Solar energy is power from the sun.'),
    ('What is wind energy?','Wind energy is power from wind using turbines.'),
    ('What is hydroelectric power?','Hydroelectric power is energy from flowing water.'),
    ('What is geothermal energy?','Geothermal energy is heat from inside the Earth.'),
    ('What is biomass energy?','Biomass energy is from organic materials like plants and waste.'),
    ('What is nuclear energy?','Nuclear energy is from splitting atoms in a process called fission.'),
    ('What is the greenhouse effect?','The greenhouse effect is when gases trap heat in the atmosphere.'),
    ('What are fossil fuels?','Fossil fuels are coal, oil, and natural gas, formed from ancient organisms.'),
    ('What is renewable energy?','Renewable energy comes from sources that don\'t run out, like sun and wind.'),
    ('What is non-renewable energy?','Non-renewable energy comes from sources that will run out, like fossil fuels.'),
]
for q, a in WORLD:
    add('world_facts','World Studies','General', q, a)

# ============================================================
# WORLD LANGUAGES (greetings in 50+ languages)
# ============================================================
print("World Languages...")
LANG_GREETINGS = [
    ('Afrikaans','Hallo'),('Albanian','Pershendetje'),('Arabic','Marhaba'),('Armenian','Barev'),
    ('Bengali','Nomoskar'),('Bulgarian','Zdravei'),('Burmese','Mingalaba'),
    ('Catalan','Hola'),('Croatian','Bok'),('Czech','Ahoj'),('Danish','Hej'),
    ('Dutch','Hallo'),('English','Hello'),('Estonian','Tere'),('Filipino','Kumusta'),
    ('Finnish','Hei'),('French','Bonjour'),('Georgian','Gamarjoba'),('German','Hallo'),
    ('Greek','Yassou'),('Hawaiian','Aloha'),('Hebrew','Shalom'),('Hindi','Namaste'),
    ('Hungarian','Szia'),('Icelandic','Godoan daginn'),('Indonesian','Halo'),
    ('Irish','Dia dhuit'),('Italian','Ciao'),('Japanese','Konnichiwa'),
    ('Korean','Annyeong'),('Latvian','Sveiks'),('Lithuanian','Labas'),
    ('Malay','Halo'),('Maltese','Bonjour'),('Maori','Kia ora'),('Mongolian','Sain bainuu'),
    ('Nepali','Namaste'),('Norwegian','Hei'),('Persian','Salaam'),('Polish','Czesc'),
    ('Portuguese','Ola'),('Romanian','Salut'),('Russian','Privet'),('Serbian','Cao'),
    ('Slovak','Ahoj'),('Slovenian','Zdravo'),('Somali','Salaan'),('Spanish','Hola'),
    ('Swahili','Jambo'),('Swedish','Hej'),('Tamil','Vanakkam'),('Thai','Sawasdee'),
    ('Turkish','Merhaba'),('Ukrainian','Pryvit'),('Urdu','Assalamualaikum'),
    ('Vietnamese','Xin chao'),('Welsh','Helo'),('Zulu','Sawubona'),
]
for lang, greeting in LANG_GREETINGS:
    add('world_languages','World Languages','Greetings',
        f'How do you say hello in {lang}?', f'Hello in {lang} is {greeting}.')

# Thank you in many languages
LANG_THANKS = [
    ('Afrikaans','Dankie'),('Arabic','Shukran'),('Bengali','Dhanyabad'),
    ('Bulgarian','Blagodarya'),('Chinese','Xiexie'),('Croatian','Hvala'),
    ('Czech','Dekuji'),('Danish','Tak'),('Dutch','Dank je'),('English','Thank you'),
    ('Finnish','Kiitos'),('French','Merci'),('German','Danke'),('Greek','Efharisto'),
    ('Hebrew','Toda'),('Hindi','Dhanyavad'),('Hungarian','Koszonom'),
    ('Indonesian','Terima kasih'),('Italian','Grazie'),('Japanese','Arigatou'),
    ('Korean','Gomapda'),('Malay','Terima kasih'),('Norwegian','Takk'),
    ('Polish','Dziekuje'),('Portuguese','Obrigado'),('Romanian','Multumesc'),
    ('Russian','Spasibo'),('Spanish','Gracias'),('Swahili','Asante'),
    ('Swedish','Tack'),('Thai','Khob khun'),('Turkish','Tesekkur'),
    ('Ukrainian','Dyakuyu'),('Vietnamese','Cam on'),('Welsh','Diolch'),
]
for lang, thanks in LANG_THANKS:
    add('world_languages','World Languages','Thank You',
        f'How do you say thank you in {lang}?', f'Thank you in {lang} is {thanks}.')

# Numbers 1-10 in major languages
LANG_NUMS = [
    ('English',['one','two','three','four','five','six','seven','eight','nine','ten']),
    ('Spanish',['uno','dos','tres','cuatro','cinco','seis','siete','ocho','nueve','diez']),
    ('French',['un','deux','trois','quatre','cinq','six','sept','huit','neuf','dix']),
    ('German',['eins','zwei','drei','vier','funf','sechs','sieben','acht','neun','zehn']),
    ('Italian',['uno','due','tre','quattro','cinque','sei','sette','otto','nove','dieci']),
    ('Portuguese',['um','dois','tres','quatro','cinco','seis','sete','oito','nove','dez']),
    ('Japanese',['ichi','ni','san','shi','go','roku','shichi','hachi','ku','ju']),
    ('Chinese',['yi','er','san','si','wu','liu','qi','ba','jiu','shi']),
    ('Korean',['il','i','sam','sa','o','yuk','chil','pal','gu','sip']),
    ('Hindi',['ek','do','teen','char','panch','chhah','sat','aath','nau','das']),
    ('Arabic',['wahid','ithnan','thalatha','arbaa','khamsa','sitta','saba','thamaniya','tisa','ashara']),
    ('Russian',['odin','dva','tri','chetyre','pyat','shest','sem','vosem','devyat','desyat']),
    ('Swahili',['moja','mbili','tatu','nne','tano','sita','saba','nane','tisa','kumi']),
    ('Dutch',['een','twee','drie','vier','vijf','zes','zeven','acht','negen','tien']),
    ('Swedish',['ett','tva','tre','fyra','fem','sex','sju','atta','nio','tio']),
]
for lang, nums in LANG_NUMS:
    for i, num in enumerate(nums, 1):
        add('world_languages','World Languages',f'Numbers in {lang}',
            f'How do you say {i} in {lang}?', f'{i} in {lang} is {num}.')

# ============================================================
# WORLD HISTORY (universal)
# ============================================================
print("World History...")
WORLD_HIST = [
    ('What was the ancient Egyptian civilization?','Ancient Egypt was a civilization along the Nile River, known for pyramids and pharaohs.'),
    ('Who were the pharaohs?','Pharaohs were the kings of ancient Egypt.'),
    ('Who was Tutankhamun?','Tutankhamun was a boy pharaoh of Egypt, famous for his golden tomb.'),
    ('What was the ancient Greek civilization?','Ancient Greece was known for philosophy, democracy, and the Olympics.'),
    ('Who was Socrates?','Socrates was a Greek philosopher, teacher of Plato.'),
    ('Who was Plato?','Plato was a Greek philosopher who wrote about justice and the ideal state.'),
    ('Who was Aristotle?','Aristotle was a Greek philosopher, student of Plato, teacher of Alexander the Great.'),
    ('Who was Alexander the Great?','Alexander the Great was a Macedonian king who conquered much of the known world.'),
    ('What was the Roman Empire?','The Roman Empire was a powerful ancient empire centered in Rome.'),
    ('Who was Julius Caesar?','Julius Caesar was a Roman general and dictator.'),
    ('What was the Middle Ages?','The Middle Ages, or medieval period, was from about 500 to 1500 CE.'),
    ('What was the Renaissance?','The Renaissance was a period of cultural rebirth in Europe from the 14th to 17th century.'),
    ('Who was Leonardo da Vinci?','Leonardo da Vinci was an Italian Renaissance artist and inventor, painter of the Mona Lisa.'),
    ('Who painted the Mona Lisa?','Leonardo da Vinci painted the Mona Lisa.'),
    ('Who was Michelangelo?','Michelangelo was an Italian Renaissance sculptor and painter, creator of the David statue.'),
    ('What was the Age of Exploration?','The Age of Exploration was when Europeans sailed to discover new lands.'),
    ('Who was Christopher Columbus?','Columbus was an explorer who reached the Americas in 1492.'),
    ('Who was Ferdinand Magellan?','Magellan led the first expedition to circumnavigate the globe.'),
    ('Who was Vasco da Gama?','Vasco da Gama was a Portuguese explorer who found a sea route to India.'),
    ('What was the French Revolution?','The French Revolution was from 1789, ending the monarchy in France.'),
    ('Who was Napoleon Bonaparte?','Napoleon was a French military leader who conquered much of Europe.'),
    ('What was the American Revolution?','The American Revolution was when 13 colonies gained independence from Britain.'),
    ('What was the Industrial Revolution?','The Industrial Revolution shifted manufacturing from hand to machine, starting in Britain.'),
    ('Who was Albert Einstein?','Einstein was a physicist who developed the theory of relativity.'),
    ('Who was Isaac Newton?','Newton was a physicist who discovered gravity and the laws of motion.'),
    ('Who was Charles Darwin?','Darwin was a biologist who developed the theory of evolution by natural selection.'),
    ('Who was Galileo Galilei?','Galileo was an astronomer who supported that Earth orbits the Sun.'),
    ('Who was Nikola Tesla?','Tesla was an inventor who contributed to alternating current electricity.'),
    ('Who was Marie Curie?','Marie Curie was a scientist who discovered radioactivity, winning two Nobel Prizes.'),
    ('Who was Neil Armstrong?','Neil Armstrong was the first person to walk on the Moon in 1969.'),
    ('What was the Cold War?','The Cold War was a period of tension between the US and Soviet Union from 1947 to 1991.'),
    ('What was the Berlin Wall?','The Berlin Wall divided East and West Berlin from 1961 to 1989.'),
    ('What was the Apollo 11 mission?','Apollo 11 was the first manned mission to land on the Moon.'),
    ('What was the Great Wall of China built for?','The Great Wall was built to protect China from invaders from the north.'),
    ('What were the ancient pyramids?','Pyramids were monumental tombs built by ancient Egyptians.'),
    ('Who were the Vikings?','Vikings were Norse seafarers who raided and traded from the 8th to 11th centuries.'),
    ('What was the Inca Empire?','The Inca Empire was in South America, centered in Peru.'),
    ('What was the Aztec Empire?','The Aztec Empire was in central Mexico before the Spanish conquest.'),
    ('What was the Maya civilization?','The Maya were known for their writing, math, and astronomy in Mesoamerica.'),
    ('Who was Mahatma Gandhi?','Gandhi led India to independence through non-violent resistance.'),
    ('Who was Nelson Mandela?','Mandela was the first Black president of South Africa, ending apartheid.'),
    ('Who was Martin Luther King Jr.?','MLK was a US civil rights leader who fought for racial equality.'),
    ('Who was Abraham Lincoln?','Lincoln was the US president who ended slavery.'),
    ('Who was George Washington?','Washington was the first US president.'),
    ('Who was Queen Elizabeth I?','Elizabeth I was Queen of England during the Elizabethan era.'),
    ('Who was Queen Victoria?','Victoria was Queen of the UK during the Victorian era, ruling 63 years.'),
    ('Who was Mozart?','Mozart was an Austrian composer, one of the greatest in history.'),
    ('Who was Beethoven?','Beethoven was a German composer, known for his symphonies.'),
    ('Who was Shakespeare?','Shakespeare was an English playwright, author of Romeo and Juliet and Hamlet.'),
    ('Who was Pablo Picasso?','Picasso was a Spanish painter, co-founder of Cubism.'),
]
for q, a in WORLD_HIST:
    add('world_history','World History','General', q, a)

# ============================================================
# WORLD SCIENCE (universal)
# ============================================================
print("World Science...")
WORLD_SCI = [
    ('What is the scientific method?','The scientific method is a process: observe, hypothesize, experiment, conclude.'),
    ('What is a hypothesis?','A hypothesis is an educated guess that can be tested.'),
    ('What is an experiment?','An experiment is a test to see if a hypothesis is correct.'),
    ('What is a theory in science?','A theory is a well-supported explanation for natural phenomena.'),
    ('What is the periodic table?','The periodic table organizes all chemical elements.'),
    ('How many elements are in the periodic table?','There are 118 elements in the periodic table.'),
    ('What is the most abundant element in the universe?','Hydrogen is the most abundant element in the universe.'),
    ('What is the most abundant element on Earth?','Oxygen is the most abundant element on Earth.'),
    ('What is the symbol for water?','The chemical formula for water is H2O.'),
    ('What is the symbol for oxygen?','The chemical symbol for oxygen is O.'),
    ('What is the symbol for carbon?','The chemical symbol for carbon is C.'),
    ('What is DNA?','DNA is a molecule that carries genetic instructions.'),
    ('What does DNA stand for?','DNA stands for Deoxyribonucleic Acid.'),
    ('What is a cell?','A cell is the basic unit of life.'),
    ('What is the powerhouse of the cell?','Mitochondria are the powerhouse of the cell.'),
    ('What is the brain?','The brain is the organ that controls the body and thinking.'),
    ('What is the heart?','The heart pumps blood throughout the body.'),
    ('What are lungs?','Lungs exchange oxygen and carbon dioxide.'),
    ('What is the largest organ of the body?','The skin is the largest organ of the human body.'),
    ('How many bones in the human body?','There are 206 bones in the adult human body.'),
    ('How many teeth does an adult have?','An adult has 32 teeth.'),
    ('What is the normal body temperature?','Normal body temperature is about 37 degrees Celsius.'),
    ('How many chambers does the heart have?','The heart has 4 chambers: 2 atria and 2 ventricles.'),
    ('What are the five senses?','The five senses are sight, hearing, smell, taste, and touch.'),
    ('What is the speed of light?','The speed of light is about 300,000 kilometers per second.'),
    ('What is the speed of sound?','The speed of sound in air is about 343 meters per second.'),
    ('What is gravity?','Gravity is the force that attracts objects with mass toward each other.'),
    ('What is the law of conservation of energy?','Energy cannot be created or destroyed, only transformed.'),
    ('What is photosynthesis?','Photosynthesis is how plants use sunlight to make food from CO2 and water.'),
    ('What is the food chain?','A food chain shows the flow of energy from producers to consumers.'),
    ('What is an ecosystem?','An ecosystem is a community of organisms and their environment.'),
    ('What is adaptation?','Adaptation is how living things change to survive in their environment.'),
    ('What is evolution?','Evolution is the process of species changing over time through natural selection.'),
    ('Who proposed the theory of evolution?','Charles Darwin proposed the theory of evolution.'),
    ('What is a habitat?','A habitat is the natural home of an organism.'),
    ('What is the water cycle?','The water cycle is evaporation, condensation, precipitation, and collection.'),
    ('What is the carbon cycle?','The carbon cycle is how carbon moves between the atmosphere, oceans, and living things.'),
    ('What is pollution?','Pollution is the introduction of harmful substances into the environment.'),
    ('What is global warming?','Global warming is the long-term increase in Earth\'s average temperature.'),
    ('What is renewable energy?','Renewable energy comes from sources that naturally replenish, like sun and wind.'),
    ('What is the solar system?','The solar system consists of the Sun and everything orbiting it.'),
    ('How many planets in the solar system?','There are 8 planets in the solar system.'),
    ('What is the Sun?','The Sun is a star at the center of our solar system.'),
    ('What is a star?','A star is a luminous sphere of plasma held together by gravity.'),
    ('What is a galaxy?','A galaxy is a system of stars, gas, and dust held together by gravity.'),
    ('What is the Milky Way?','The Milky Way is the galaxy that contains our solar system.'),
    ('What is a black hole?','A black hole is a region of space where gravity is so strong nothing can escape.'),
    ('What is the Moon?','The Moon is Earth\'s natural satellite.'),
    ('What causes eclipses?','Eclipses occur when one celestial body blocks the light to another.'),
    ('What is a solar eclipse?','A solar eclipse is when the Moon blocks the Sun.'),
    ('What is a lunar eclipse?','A lunar eclipse is when Earth blocks sunlight to the Moon.'),
    ('What is a comet?','A comet is an icy body that develops a tail when near the Sun.'),
    ('What is a meteor?','A meteor is a streak of light from a space rock burning in the atmosphere.'),
    ('What is an asteroid?','An asteroid is a rocky body orbiting the Sun, mostly between Mars and Jupiter.'),
    ('What is the International Space Station?','The ISS is a space station orbiting Earth for research.'),
    ('What is gravity on the Moon?','Gravity on the Moon is about one-sixth of Earth\'s gravity.'),
    ('How long does light take to reach Earth from the Sun?','Light takes about 8 minutes to reach Earth from the Sun.'),
    ('What is the closest star to Earth (besides the Sun)?','Proxima Centauri is the closest star to our solar system.'),
    ('What is the largest planet?','Jupiter is the largest planet in the solar system.'),
    ('What is the smallest planet?','Mercury is the smallest planet in the solar system.'),
    ('What is the hottest planet?','Venus is the hottest planet due to its greenhouse effect.'),
    ('What is the red planet?','Mars is called the red planet due to iron oxide on its surface.'),
    ('Which planet has the most rings?','Saturn has the most prominent rings.'),
    ('Which planet is tilted on its side?','Uranus rotates on its side.'),
    ('What is the most distant planet?','Neptune is the most distant planet from the Sun.'),
    ('What is a dwarf planet?','A dwarf planet is like Pluto, round but hasn\'t cleared its orbit.'),
    ('What is the asteroid belt?','The asteroid belt is between Mars and Jupiter.'),
]
for q, a in WORLD_SCI:
    add('world_science','World Science','General', q, a)

# ============================================================
# OUTPUT
# ============================================================
print(f"\nTotal new items: {len(items)}")

OUT = Path("/home/z/my-project/download/llm/dataset_worldwide")
OUT.mkdir(parents=True, exist_ok=True)
with open(OUT / 'dataset_worldwide.jsonl', 'w', encoding='utf-8') as f:
    for item in items:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f"\nBreakdown:")
from collections import Counter
cats = Counter(item['bidang'] for item in items)
for cat, count in sorted(cats.items(), key=lambda x: -x[1]):
    print(f"  {cat}: {count}")
print(f"\nTotal: {sum(cats.values())}")
