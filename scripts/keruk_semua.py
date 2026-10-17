"""
D'QUILANE — KERUK SEMUA YANG TERSISA
=====================================
Target: 10,000+ Q&A total (add 3000+ to existing 7245)
Covers: EVERYTHING missing from SD curriculum worldwide
"""
import json, random
from pathlib import Path

random.seed(42)
items = []

def add(cat, judul, sub, q, a):
    items.append({'input': q, 'output': a, 'bidang': cat, 'judul': judul, 'subjudul': sub})

# ============================================================
# 1. HUMAN BODY SYSTEMS (100+)
# ============================================================
print("Human body systems...")
BODY = [
    # Skeletal
    ('How many bones in the adult human body?','There are 206 bones in the adult human body.'),
    ('What is the smallest bone in the human body?','The stapes, in the middle ear, is the smallest bone.'),
    ('What is the largest bone in the human body?','The femur (thigh bone) is the largest bone.'),
    ('What is the function of the skeleton?','The skeleton supports the body, protects organs, and produces blood cells.'),
    ('What is the skull?','The skull is the bone structure that protects the brain.'),
    ('What is the spine?','The spine (vertebral column) supports the back and protects the spinal cord.'),
    ('How many vertebrae in the spine?','There are 33 vertebrae in the spine.'),
    ('What is a joint?','A joint is where two or more bones meet.'),
    ('What is cartilage?','Cartilage is flexible tissue that cushions joints.'),
    # Muscular
    ('How many muscles in the human body?','There are over 600 muscles in the human body.'),
    ('What are the types of muscles?','Three types: skeletal (voluntary), smooth (involuntary), and cardiac (heart).'),
    ('What is the largest muscle?','The gluteus maximus (buttock muscle) is the largest.'),
    ('What do muscles do?','Muscles contract and relax to move the body.'),
    # Circulatory
    ('What is the circulatory system?','The circulatory system pumps blood throughout the body.'),
    ('How many chambers does the heart have?','The heart has 4 chambers: left and right atria, left and right ventricles.'),
    ('How big is the human heart?','The heart is about the size of a fist.'),
    ('How many times does the heart beat per minute?','The heart beats about 60-100 times per minute at rest.'),
    ('What are blood vessels?','Blood vessels are tubes that carry blood: arteries, veins, and capillaries.'),
    ('What do arteries do?','Arteries carry blood away from the heart.'),
    ('What do veins do?','Veins carry blood back to the heart.'),
    ('What are capillaries?','Capillaries are tiny blood vessels where gas exchange happens.'),
    ('What is blood made of?','Blood is made of red blood cells, white blood cells, platelets, and plasma.'),
    ('What do red blood cells do?','Red blood cells carry oxygen throughout the body.'),
    ('What do white blood cells do?','White blood cells fight infections and disease.'),
    ('What do platelets do?','Platelets help blood clot to stop bleeding.'),
    ('What is plasma?','Plasma is the liquid part of blood, mostly water.'),
    ('How much blood is in the human body?','There are about 5 liters of blood in an adult body.'),
    # Respiratory
    ('What is the respiratory system?','The respiratory system brings oxygen into the body and removes carbon dioxide.'),
    ('What are the main organs of respiration?','The lungs are the main organs of respiration.'),
    ('How many lungs does a human have?','A human has two lungs: left and right.'),
    ('What is the trachea?','The trachea (windpipe) carries air to the lungs.'),
    ('What are alveoli?','Alveoli are tiny air sacs in the lungs where gas exchange occurs.'),
    ('What is the diaphragm?','The diaphragm is a muscle that helps with breathing.'),
    ('What happens when you breathe in?','When you breathe in, the diaphragm contracts and the lungs expand.'),
    ('What happens when you breathe out?','When you breathe out, the diaphragm relaxes and the lungs contract.'),
    # Digestive
    ('What is the digestive system?','The digestive system breaks down food into nutrients.'),
    ('What is the path of food in the digestive system?','Mouth, esophagus, stomach, small intestine, large intestine, rectum, anus.'),
    ('What does the mouth do in digestion?','The mouth chews food and mixes it with saliva.'),
    ('What is the esophagus?','The esophagus is a tube that carries food from the mouth to the stomach.'),
    ('What does the stomach do?','The stomach mixes food with acid and enzymes to break it down.'),
    ('What is the small intestine?','The small intestine absorbs most nutrients from food.'),
    ('How long is the small intestine?','The small intestine is about 6-7 meters long.'),
    ('What does the large intestine do?','The large intestine absorbs water and forms waste.'),
    ('What is the liver?','The liver filters toxins and produces bile for digestion.'),
    ('What is the pancreas?','The pancreas produces enzymes and insulin.'),
    ('What are the functions of the liver?','The liver detoxifies blood, stores energy, and produces bile.'),
    # Nervous
    ('What is the nervous system?','The nervous system controls and coordinates all body functions.'),
    ('What are the main parts of the nervous system?','The brain, spinal cord, and nerves are the main parts.'),
    ('What does the brain do?','The brain controls thinking, memory, movement, and senses.'),
    ('How many parts does the brain have?','The brain has three main parts: cerebrum, cerebellum, and brainstem.'),
    ('What does the cerebrum do?','The cerebrum controls thinking, memory, and voluntary movement.'),
    ('What does the cerebellum do?','The cerebellum controls balance and coordination.'),
    ('What does the brainstem do?','The brainstem controls breathing, heart rate, and blood pressure.'),
    ('What is the spinal cord?','The spinal cord carries messages between the brain and the body.'),
    ('What are nerves?','Nerves are bundles of fibers that carry signals throughout the body.'),
    ('How fast do nerve signals travel?','Nerve signals travel up to 120 meters per second.'),
    # Excretory
    ('What is the excretory system?','The excretory system removes waste from the body.'),
    ('What do the kidneys do?','The kidneys filter blood and produce urine.'),
    ('How many kidneys does a human have?','A human has two kidneys.'),
    ('What is urine?','Urine is liquid waste filtered from blood by the kidneys.'),
    ('What does the bladder do?','The bladder stores urine until it is released.'),
    # Immune
    ('What is the immune system?','The immune system protects the body from disease.'),
    ('What do white blood cells do?','White blood cells fight infections and foreign invaders.'),
    ('What are antibodies?','Antibodies are proteins that help fight specific infections.'),
    ('What are vaccines?','Vaccines train the immune system to fight diseases.'),
    ('What is a virus?','A virus is a tiny infectious agent that needs a host cell to reproduce.'),
    ('What is a bacteria?','Bacteria are single-celled organisms that can be helpful or harmful.'),
    ('What is an antibiotic?','Antibiotics are medicines that kill or stop bacteria.'),
    # Endocrine
    ('What is the endocrine system?','The endocrine system produces hormones that regulate the body.'),
    ('What are hormones?','Hormones are chemical messengers that control body functions.'),
    ('What does the thyroid do?','The thyroid regulates metabolism.'),
    ('What does the pancreas produce for blood sugar?','The pancreas produces insulin to regulate blood sugar.'),
    # Integumentary
    ('What is the largest organ of the body?','The skin is the largest organ of the body.'),
    ('What are the layers of skin?','The skin has three layers: epidermis, dermis, and hypodermis.'),
    ('What does the skin do?','The skin protects the body, regulates temperature, and senses touch.'),
    ('What are sweat glands?','Sweat glands produce sweat to cool the body.'),
    ('What is melanin?','Melanin is the pigment that gives skin its color.'),
    # Senses
    ('What are the five senses?','The five senses are sight, hearing, smell, taste, and touch.'),
    ('What organ is used for sight?','The eyes are used for sight.'),
    ('What organ is used for hearing?','The ears are used for hearing.'),
    ('What organ is used for smell?','The nose is used for smell.'),
    ('What organ is used for taste?','The tongue is used for taste.'),
    ('What organ is used for touch?','The skin is used for touch.'),
    ('How many taste buds does the tongue have?','The tongue has about 10,000 taste buds.'),
    ('What are the basic tastes?','The basic tastes are sweet, sour, salty, bitter, and umami.'),
    ('How does the eye work?','Light enters through the pupil, is focused by the lens, and detected by the retina.'),
    ('What is the retina?','The retina is the light-sensitive layer at the back of the eye.'),
    ('What is the pupil?','The pupil is the black opening in the center of the iris.'),
    ('What is the iris?','The iris is the colored part of the eye that controls pupil size.'),
    ('How does the ear work?','Sound waves enter the ear canal, vibrate the eardrum, and are sent to the brain.'),
    ('What is the eardrum?','The eardrum is a thin membrane that vibrates when sound hits it.'),
    ('What are the three smallest bones?','The stapes, incus, and malleus are in the middle ear.'),
    # Reproductive (age-appropriate)
    ('What is puberty?','Puberty is when a child\'s body changes to become an adult.'),
    ('What happens during puberty?','During puberty, the body grows rapidly and develops adult features.'),
    ('At what age does puberty usually start?','Puberty usually starts between ages 8-14.'),
    # Teeth
    ('How many baby teeth do children have?','Children have 20 baby teeth.'),
    ('How many adult teeth do adults have?','Adults have 32 teeth.'),
    ('What are the types of teeth?','Types: incisors, canines, premolars, and molars.'),
    ('What do incisors do?','Incisors cut food.'),
    ('What do canines do?','Canines tear food.'),
    ('What do molars do?','Molars grind food.'),
    ('What is enamel?','Enamel is the hard outer covering of teeth.'),
    ('What is a cavity?','A cavity is tooth decay caused by bacteria.'),
    ('How often should you brush your teeth?','Brush twice a day, morning and night.'),
    ('What does fluoride do?','Fluoride strengthens tooth enamel.'),
    # Nutrition
    ('What are the five food groups?','Fruits, vegetables, grains, protein, and dairy.'),
    ('What is a balanced diet?','A balanced diet includes all food groups in proper amounts.'),
    ('What are carbohydrates?','Carbohydrates provide energy. Found in rice, bread, and pasta.'),
    ('What are proteins?','Proteins build muscles. Found in meat, fish, eggs, and beans.'),
    ('What are fats?','Fats store energy. Found in oil, butter, and nuts.'),
    ('What are vitamins?','Vitamins help the body function. Found in fruits and vegetables.'),
    ('What are minerals?','Minerals like calcium and iron help the body. Found in dairy and meat.'),
    ('What is calcium?','Calcium builds strong bones and teeth. Found in milk and dairy.'),
    ('What is iron?','Iron helps blood carry oxygen. Found in meat and spinach.'),
    ('What is vitamin C?','Vitamin C boosts immunity. Found in oranges and lemons.'),
    ('What is vitamin A?','Vitamin A helps eyesight. Found in carrots and sweet potatoes.'),
    ('What is vitamin D?','Vitamin D helps bones absorb calcium. Made by skin in sunlight.'),
    ('Why is water important?','Water keeps the body hydrated and helps all functions.'),
    ('How much water should you drink daily?','Drink about 8 glasses or 2 liters of water daily.'),
]
for q,a in BODY:
    add('health','Human Body','Systems', q, a)

# ============================================================
# 2. ANIMAL CLASSIFICATION & FACTS (150+)
# ============================================================
print("Animals...")
ANIMALS = [
    # Classification
    ('What are the six classes of animals?','Mammals, birds, reptiles, amphibians, fish, and insects.'),
    ('What is a mammal?','A mammal is warm-blooded, has hair/fur, and feeds its young with milk.'),
    ('What is a bird?','A bird is warm-blooded, has feathers, lays eggs, and has a beak.'),
    ('What is a reptile?','A reptile is cold-blooded, has scales, and lays eggs on land.'),
    ('What is an amphibian?','An amphibian is cold-blooded, lives on land and water, and has moist skin.'),
    ('What is a fish?','A fish is cold-blooded, lives in water, has fins and gills.'),
    ('What is an insect?','An insect has 6 legs, 3 body parts, and usually has wings.'),
    # Mammals
    ('What is the largest land mammal?','The African elephant is the largest land mammal.'),
    ('What is the fastest land mammal?','The cheetah is the fastest land mammal, up to 120 km/h.'),
    ('What is the largest mammal?','The blue whale is the largest mammal and animal.'),
    ('What is the smallest mammal?','The bumblebee bat is the smallest mammal.'),
    ('What is the tallest mammal?','The giraffe is the tallest mammal.'),
    ('What is a marsupial?','A marsupial is a mammal that carries its baby in a pouch, like a kangaroo.'),
    ('What is a primate?','A primate is a mammal like monkeys, apes, and humans.'),
    ('What is the only mammal that can fly?','The bat is the only mammal that can truly fly.'),
    ('Which mammal lays eggs?','The platypus and echidna are mammals that lay eggs.'),
    # Birds
    ('What is the largest bird?','The ostrich is the largest bird.'),
    ('What is the smallest bird?','The bee hummingbird is the smallest bird.'),
    ('Which bird cannot fly?','Ostriches, penguins, and kiwis cannot fly.'),
    ('What is the fastest bird?','The peregrine falcon is the fastest bird, diving at 390 km/h.'),
    ('Which bird can fly backwards?','The hummingbird can fly backwards.'),
    ('How do birds fly?','Birds fly by flapping their wings, using air currents.'),
    ('What is migration?','Migration is when birds travel long distances seasonally.'),
    # Reptiles
    ('What is the largest reptile?','The saltwater crocodile is the largest living reptile.'),
    ('What is the largest snake?','The reticulated python is the longest snake.'),
    ('Do all snakes lay eggs?','No, some snakes give birth to live young.'),
    ('What is a turtle shell made of?','A turtle shell is made of bone covered with keratin.'),
    ('How do snakes smell?','Snakes use their forked tongue to smell.'),
    # Amphibians
    ('What is metamorphosis in frogs?','Metamorphosis is the change from tadpole to adult frog.'),
    ('How do amphibians breathe?','Amphibians breathe through gills when young, lungs and skin as adults.'),
    ('What is the largest amphibian?','The Chinese giant salamander is the largest amphibian.'),
    # Fish
    ('What is the largest fish?','The whale shark is the largest fish.'),
    ('How do fish breathe underwater?','Fish breathe through gills that extract oxygen from water.'),
    ('What is a school of fish?','A school is a group of fish swimming together.'),
    # Insects
    ('How many legs does a spider have?','A spider has 8 legs (it is an arachnid, not an insect).'),
    ('How many legs does an insect have?','An insect has 6 legs.'),
    ('What is a butterfly\'s life cycle?','Egg, caterpillar (larva), chrysalis (pupa), butterfly (adult).'),
    ('What do bees produce?','Bees produce honey and beeswax.'),
    ('How do bees communicate?','Bees communicate through a waggle dance.'),
    ('What is a hive?','A hive is the home of a colony of bees.'),
    ('What is the queen bee?','The queen bee is the leader of the colony and lays all the eggs.'),
    ('How many eyes does a bee have?','A bee has 5 eyes: 2 compound and 3 simple.'),
    # Habitats
    ('What is a habitat?','A habitat is the natural home of a living thing.'),
    ('What animals live in the Arctic?','Polar bears, arctic foxes, seals, and penguins (Antarctic) live in cold regions.'),
    ('What animals live in the desert?','Camels, scorpions, lizards, and snakes live in the desert.'),
    ('What animals live in the rainforest?','Monkeys, parrots, jaguars, and tree frogs live in the rainforest.'),
    ('What animals live in the ocean?','Whales, dolphins, sharks, fish, and octopuses live in the ocean.'),
    ('What animals live in the savanna?','Lions, elephants, zebras, and giraffes live in the savanna.'),
    # Record holders
    ('What is the deadliest animal?','The mosquito is the deadliest animal to humans.'),
    ('What animal has the longest lifespan?','The Greenland shark can live over 400 years.'),
    ('What animal can regenerate limbs?','Salamanders and starfish can regenerate lost limbs.'),
    ('What animal never sleeps?','Bullfrogs are believed to never truly sleep.'),
    ('What is the loudest animal?','The blue whale is the loudest animal, at 188 decibels.'),
    ('What animal has the strongest bite?','The saltwater crocodile has the strongest bite force.'),
    ('How many hearts does an octopus have?','An octopus has 3 hearts.'),
    ('How many brains does an octopus have?','An octopus has 9 brains: one central and eight in each arm.'),
    ('What color is octopus blood?','Octopus blood is blue.'),
    ('What animal can change color?','Chameleons and octopuses can change color.'),
    ('How do dolphins communicate?','Dolphins communicate through clicks and whistles.'),
    ('Are dolphins fish or mammals?','Dolphins are mammals, not fish.'),
    ('Are whales fish or mammals?','Whales are mammals, not fish.'),
    ('What is the difference between alligators and crocodiles?','Alligators have wide U-shaped snouts, crocodiles have V-shaped snouts.'),
]
for q,a in ANIMALS:
    add('science','Animal Facts','Classification', q, a)

# ============================================================
# 3. PLANTS & ECOLOGY (80+)
# ============================================================
print("Plants...")
PLANTS = [
    ('What are the main parts of a plant?','Roots, stem, leaves, flowers, fruits, and seeds.'),
    ('What do roots do?','Roots absorb water and minerals, and anchor the plant.'),
    ('What does the stem do?','The stem supports the plant and transports water and nutrients.'),
    ('What do leaves do?','Leaves make food through photosynthesis and exchange gases.'),
    ('What is the function of flowers?','Flowers are for reproduction, attracting pollinators.'),
    ('What is pollination?','Pollination is the transfer of pollen from one flower to another.'),
    ('What are pollinators?','Bees, butterflies, birds, and bats are common pollinators.'),
    ('What is photosynthesis?','Photosynthesis is how plants make food from sunlight, water, and CO2.'),
    ('What is the chemical formula for photosynthesis?','CO2 + H2O + sunlight -> glucose + O2.'),
    ('What gas do plants absorb?','Plants absorb carbon dioxide (CO2).'),
    ('What gas do plants release?','Plants release oxygen (O2).'),
    ('What is chlorophyll?','Chlorophyll is the green pigment that captures sunlight.'),
    ('What is chloroplast?','Chloroplasts are the structures in plant cells where photosynthesis occurs.'),
    ('What are the types of roots?','Two main types: taproot (one main root) and fibrous (many thin roots).'),
    ('What is a deciduous tree?','A deciduous tree loses its leaves in winter, like oak and maple.'),
    ('What is an evergreen tree?','An evergreen tree keeps its leaves year-round, like pine and spruce.'),
    ('What is a conifer?','A conifer is a tree that produces cones, like pine trees.'),
    ('What is a flowering plant?','A flowering plant produces flowers, like roses and sunflowers.'),
    ('What is a non-flowering plant?','Non-flowering plants reproduce with spores, like ferns and mosses.'),
    ('What is a seed?','A seed is a tiny plant with stored food, ready to grow.'),
    ('What is germination?','Germination is when a seed begins to grow into a plant.'),
    ('What does a seed need to germinate?','A seed needs water, warmth, and oxygen to germinate.'),
    ('What is dispersal?','Dispersal is how seeds spread to new places, by wind, water, or animals.'),
    ('What is a fruit?','A fruit is the part of a flowering plant that contains seeds.'),
    ('What is a vegetable?','A vegetable is any edible part of a plant, like roots, stems, or leaves.'),
    ('What is the life cycle of a plant?','Seed, germination, growth, flowering, pollination, seed production.'),
    ('What is xylem?','Xylem is tissue that transports water upward in plants.'),
    ('What is phloem?','Phloem is tissue that transports food (sugar) in plants.'),
    ('What is a carnivorous plant?','A carnivorous plant eats insects, like the Venus flytrap.'),
    ('What is a Venus flytrap?','The Venus flytrap is a plant that catches and eats insects.'),
    ('What is a succulent?','A succulent is a plant that stores water in its leaves, like cactus.'),
    ('What is a cactus?','A cactus is a desert plant that stores water and has spines instead of leaves.'),
    ('What are medicinal plants?','Medicinal plants like aloe vera and ginger have healing properties.'),
    ('What is aloe vera?','Aloe vera is a plant whose gel soothes burns and skin problems.'),
    ('What is bamboo?','Bamboo is a fast-growing grass that can grow up to 90 cm per day.'),
    ('What is the tallest tree?','The coast redwood is the tallest tree, up to 115 meters.'),
    ('What is the oldest tree?','The bristlecone pine Methuselah is over 4,800 years old.'),
    ('What is deforestation?','Deforestation is cutting down forests, often for farming or building.'),
    ('What is afforestation?','Afforestation is planting trees in areas where there were no forests before.'),
    ('What is reforestation?','Reforestation is replanting trees in areas where forests were cut.'),
    ('What is a food chain?','A food chain shows who eats whom, transferring energy.'),
    ('What is a food web?','A food web is many interconnected food chains in an ecosystem.'),
    ('What is a producer?','A producer makes its own food, like plants through photosynthesis.'),
    ('What is a consumer?','A consumer eats other organisms. Animals are consumers.'),
    ('What is a decomposer?','A decomposer breaks down dead organisms, like fungi and bacteria.'),
    ('What is a herbivore?','A herbivore only eats plants, like cows and rabbits.'),
    ('What is a carnivore?','A carnivore only eats meat, like lions and wolves.'),
    ('What is an omnivore?','An omnivore eats both plants and meat, like humans and bears.'),
    ('What is a predator?','A predator hunts and eats other animals.'),
    ('What is prey?','Prey is an animal that is hunted and eaten by predators.'),
    ('What is symbiosis?','Symbiosis is a close relationship between different species.'),
    ('What is mutualism?','Mutualism is when both species benefit, like bees and flowers.'),
    ('What is parasitism?','Parasitism is when one benefits and the other is harmed, like ticks on dogs.'),
    ('What is commensalism?','Commensalism is when one benefits and the other is unaffected.'),
    ('What is an adaptation?','An adaptation is a trait that helps an organism survive.'),
    ('What is natural selection?','Natural selection is when traits that help survival are passed on.'),
    ('What is camouflage?','Camouflage is when an animal blends in with its surroundings.'),
    ('What is hibernation?','Hibernation is when animals sleep through winter to conserve energy.'),
    ('What is migration?','Migration is when animals travel to find food or better weather.'),
    ('What is an ecosystem?','An ecosystem is all living and non-living things in an area.'),
    ('What is biodiversity?','Biodiversity is the variety of living things in an ecosystem.'),
    ('What is an endangered species?','An endangered species is at risk of dying out completely.'),
    ('What is extinction?','Extinction is when a species completely dies out, like dinosaurs.'),
    ('What is conservation?','Conservation is protecting nature and wildlife from harm.'),
    ('What is recycling?','Recycling is converting waste into reusable materials.'),
    ('What are the 3 Rs?','Reduce, Reuse, and Recycle.'),
    ('What is pollution?','Pollution is introducing harmful substances into the environment.'),
    ('What is air pollution?','Air pollution is harmful substances in the air, like smoke and gases.'),
    ('What is water pollution?','Water pollution is harmful substances in water, like chemicals and waste.'),
    ('What is global warming?','Global warming is the long-term increase in Earth\'s temperature.'),
    ('What is the greenhouse effect?','The greenhouse effect is when gases trap heat in the atmosphere.'),
    ('What is climate change?','Climate change is long-term changes in temperature and weather patterns.'),
    ('What is a fossil fuel?','Fossil fuels are coal, oil, and natural gas, made from ancient organisms.'),
    ('What is renewable energy?','Renewable energy comes from sources that don\'t run out, like sun and wind.'),
    ('What is solar energy?','Solar energy is power generated from sunlight.'),
    ('What is wind energy?','Wind energy is power generated from wind using turbines.'),
    ('What is hydroelectric power?','Hydroelectric power is energy from flowing water, like dams.'),
    ('What is geothermal energy?','Geothermal energy is heat from inside the Earth.'),
    ('What is biomass energy?','Biomass energy is from burning plant or animal materials.'),
]
for q,a in PLANTS:
    add('science','Plants and Ecology','General', q, a)

# ============================================================
# 4. EARTH SCIENCE - Rocks, Weather, Space (100+)
# ============================================================
print("Earth science...")
EARTH = [
    # Rocks
    ('What are the three types of rocks?','Igneous, sedimentary, and metamorphic.'),
    ('What is igneous rock?','Igneous rock forms from cooled lava or magma, like granite and basalt.'),
    ('What is sedimentary rock?','Sedimentary rock forms from compressed sediments, like sandstone and limestone.'),
    ('What is metamorphic rock?','Metamorphic rock forms from heat and pressure, like marble and slate.'),
    ('What is the rock cycle?','The rock cycle is how rocks change from one type to another over time.'),
    ('What is magma?','Magma is molten (melted) rock below Earth\'s surface.'),
    ('What is lava?','Lava is molten rock that has reached Earth\'s surface.'),
    ('What is a mineral?','A mineral is a naturally occurring, non-living solid with a crystal structure.'),
    ('What is the hardest mineral?','Diamond is the hardest natural mineral.'),
    ('What is the softest mineral?','Talc is the softest mineral.'),
    ('What is quartz?','Quartz is a common mineral found in many types of rock.'),
    ('What is gold?','Gold is a precious metal that does not tarnish.'),
    ('What is iron?','Iron is a metal used to make steel and tools.'),
    ('What is copper?','Copper is a metal that conducts electricity well.'),
    # Soil
    ('What is soil?','Soil is the top layer of Earth\'s surface that supports plant life.'),
    ('What are the layers of soil?','Topsoil, subsoil, and bedrock.'),
    ('What is topsoil?','Topsoil is the upper layer rich in nutrients and organic matter.'),
    ('What is humus?','Humus is decayed organic matter in soil.'),
    ('What is erosion?','Erosion is the wearing away of soil and rock by wind, water, or ice.'),
    ('What is weathering?','Weathering is the breaking down of rocks by natural forces.'),
    # Fossils
    ('What is a fossil?','A fossil is the preserved remains or traces of an ancient organism.'),
    ('How are fossils formed?','Fossils form when organisms are buried and preserved in sediment over millions of years.'),
    ('What are dinosaur fossils?','Dinosaur fossils are the preserved bones of dinosaurs that lived millions of years ago.'),
    ('When did dinosaurs go extinct?','Dinosaurs went extinct about 65 million years ago.'),
    ('What caused dinosaur extinction?','A large asteroid impact is believed to have caused dinosaur extinction.'),
    # Weather
    ('What is weather?','Weather is the condition of the atmosphere at a particular time and place.'),
    ('What is climate?','Climate is the average weather of a place over a long time.'),
    ('What causes weather?','Weather is caused by the Sun heating Earth\'s atmosphere unevenly.'),
    ('What is the water cycle?','The water cycle is evaporation, condensation, precipitation, and collection.'),
    ('What is evaporation?','Evaporation is when water turns into water vapor (gas).'),
    ('What is condensation?','Condensation is when water vapor turns back into liquid water.'),
    ('What is precipitation?','Precipitation is water falling from clouds as rain, snow, sleet, or hail.'),
    ('What is humidity?','Humidity is the amount of water vapor in the air.'),
    ('What causes wind?','Wind is caused by differences in air pressure.'),
    ('What are clouds?','Clouds are collections of tiny water droplets or ice crystals in the sky.'),
    ('What are the types of clouds?','Cirrus (feathery), cumulus (fluffy), and stratus (flat layer) are main types.'),
    ('What is a thunderstorm?','A thunderstorm has thunder, lightning, and heavy rain.'),
    ('What is lightning?','Lightning is a sudden electric discharge in the atmosphere.'),
    ('What causes thunder?','Thunder is the sound of lightning heating the air rapidly.'),
    ('What is a tornado?','A tornado is a spinning column of air that touches the ground.'),
    ('What is a hurricane?','A hurricane is a large spinning storm that forms over warm ocean water.'),
    ('What is a typhoon?','A typhoon is the same as a hurricane, but in the Pacific Ocean.'),
    ('What is a drought?','A drought is a long period without enough rain.'),
    ('What is a flood?','A flood is when too much water covers normally dry land.'),
    ('What is a blizzard?','A blizzard is a severe snowstorm with strong winds.'),
    ('What is the jet stream?','The jet stream is a high-altitude wind that affects weather.'),
    # Space
    ('What is a galaxy?','A galaxy is a huge system of stars, gas, and dust.'),
    ('What is our galaxy called?','Our galaxy is called the Milky Way.'),
    ('How many stars are in the Milky Way?','There are about 100-400 billion stars in the Milky Way.'),
    ('What is the closest galaxy to the Milky Way?','The Andromeda Galaxy is the closest major galaxy.'),
    ('What is a light-year?','A light-year is the distance light travels in one year, about 9.5 trillion km.'),
    ('What is a constellation?','A constellation is a group of stars that form a pattern.'),
    ('What is the Big Dipper?','The Big Dipper is a famous group of 7 stars in the constellation Ursa Major.'),
    ('What is the North Star?','The North Star (Polaris) is a star that always appears above the North Pole.'),
    ('What is a telescope?','A telescope is an instrument for viewing distant objects in space.'),
    ('What is a comet?','A comet is an icy body that develops a tail when near the Sun.'),
    ('What is a meteor?','A meteor is a streak of light from a space rock burning in the atmosphere.'),
    ('What is a meteorite?','A meteorite is a space rock that lands on Earth.'),
    ('What is an asteroid?','An asteroid is a rocky body orbiting the Sun, mostly in the asteroid belt.'),
    ('What is the asteroid belt?','The asteroid belt is between Mars and Jupiter.'),
    ('What is a solar eclipse?','A solar eclipse is when the Moon blocks the Sun.'),
    ('What is a lunar eclipse?','A lunar eclipse is when Earth blocks sunlight to the Moon.'),
    ('What is a space station?','A space station is a facility in orbit for research and living.'),
    ('What is the ISS?','The International Space Station is a space station orbiting Earth.'),
    ('How fast does the ISS orbit?','The ISS orbits at about 28,000 km/h.'),
    ('How long does the ISS take to orbit Earth?','The ISS orbits Earth every 90 minutes.'),
    ('What is a satellite?','A satellite is an object orbiting a planet. The Moon is Earth\'s natural satellite.'),
    ('What is an artificial satellite?','An artificial satellite is a human-made object in orbit, used for communication.'),
    ('What is GPS?','GPS uses satellites to determine your location on Earth.'),
    ('Who was the first person in space?','Yuri Gagarin was the first person in space, in 1961.'),
    ('Who was the first person on the Moon?','Neil Armstrong was the first person on the Moon, in 1969.'),
    ('What was the Apollo program?','The Apollo program was NASA\'s mission to land humans on the Moon.'),
    ('What is NASA?','NASA is the National Aeronautics and Space Administration, the US space agency.'),
    ('What is a rover?','A rover is a robotic vehicle that explores the surface of another planet.'),
    ('What is the Mars rover?','Mars rovers like Curiosity and Perseverance explore the surface of Mars.'),
    ('How long does light take from the Sun to Earth?','Light takes about 8 minutes to reach Earth from the Sun.'),
    ('What is the temperature of the Sun?','The Sun\'s surface is about 5,500 degrees Celsius.'),
    ('What is the Sun made of?','The Sun is mostly hydrogen and helium gas.'),
    ('How old is the Sun?','The Sun is about 4.6 billion years old.'),
    ('How old is the Earth?','Earth is about 4.5 billion years old.'),
    ('How old is the universe?','The universe is about 13.8 billion years old.'),
    ('What is the Big Bang?','The Big Bang is the theory that the universe began from a single point.'),
    ('What is a black hole?','A black hole is a region where gravity is so strong nothing can escape.'),
    ('What is a supernova?','A supernova is a giant explosion of a dying star.'),
    ('What is a nebula?','A nebula is a cloud of gas and dust where new stars are born.'),
    ('What is gravity in space?','Gravity is the force that keeps planets in orbit and holds galaxies together.'),
]
for q,a in EARTH:
    add('science','Earth and Space','General', q, a)

# ============================================================
# 5. MYTHOLOGY & FOLKLORE (100+)
# ============================================================
print("Mythology...")
MYTH = [
    # Greek
    ('Who is Zeus in Greek mythology?','Zeus is the king of the gods and god of the sky and thunder.'),
    ('Who is Poseidon?','Poseidon is the Greek god of the sea.'),
    ('Who is Hades?','Hades is the Greek god of the underworld.'),
    ('Who is Athena?','Athena is the Greek goddess of wisdom and war.'),
    ('Who is Apollo?','Apollo is the Greek god of the sun, music, and prophecy.'),
    ('Who is Artemis?','Artemis is the Greek goddess of the hunt and moon.'),
    ('Who is Aphrodite?','Aphrodite is the Greek goddess of love and beauty.'),
    ('Who is Ares?','Ares is the Greek god of war.'),
    ('Who is Hermes?','Hermes is the Greek god of travel, thieves, and messengers.'),
    ('Who is Hephaestus?','Hephaestus is the Greek god of fire and metalworking.'),
    ('Who is Dionysus?','Dionysus is the Greek god of wine and celebration.'),
    ('Who is Heracles (Hercules)?','Hercules is a hero known for his great strength and 12 labors.'),
    ('What is Mount Olympus?','Mount Olympus is the home of the Greek gods.'),
    ('What is the Trojan War?','The Trojan War was a war between Greeks and the city of Troy.'),
    ('What is the Trojan Horse?','The Trojan Horse was a wooden horse used to sneak into Troy.'),
    # Egyptian
    ('Who is Ra?','Ra is the Egyptian sun god.'),
    ('Who is Anubis?','Anubis is the Egyptian god of death and mummification.'),
    ('Who is Osiris?','Osiris is the Egyptian god of the afterlife.'),
    ('Who is Isis?','Isis is the Egyptian goddess of magic and motherhood.'),
    ('Who is Horus?','Horus is the Egyptian god of the sky, with a falcon head.'),
    ('Who is Thoth?','Thoth is the Egyptian god of wisdom and writing.'),
    ('What is a mummy?','A mummy is a preserved body, wrapped in bandages, in Egyptian tradition.'),
    ('What is a pyramid?','A pyramid is a triangular structure built as a tomb for pharaohs.'),
    ('What is a sphinx?','A sphinx has the body of a lion and the head of a human.'),
    # Norse
    ('Who is Odin?','Odin is the Norse king of the gods and god of wisdom.'),
    ('Who is Thor?','Thor is the Norse god of thunder, wielding a hammer called Mjolnir.'),
    ('Who is Loki?','Loki is the Norse god of mischief and trickery.'),
    ('What is Valhalla?','Valhalla is the great hall in Norse mythology where warriors go after death.'),
    ('What is Ragnarok?','Ragnarok is the Norse apocalypse, the end of the world.'),
    ('What is Yggdrasil?','Yggdrasil is the world tree that connects all worlds in Norse mythology.'),
    # Roman
    ('Who is Jupiter?','Jupiter is the Roman king of the gods (equivalent to Zeus).'),
    ('Who is Venus?','Venus is the Roman goddess of love (equivalent to Aphrodite).'),
    ('Who is Mars?','Mars is the Roman god of war (equivalent to Ares).'),
    ('Who is Neptune?','Neptune is the Roman god of the sea (equivalent to Poseidon).'),
    # Fairy Tales
    ('What is the story of Cinderella?','Cinderella is a girl mistreated by her stepmother, who goes to a ball with magic.'),
    ('What is the story of Snow White?','Snow White is a princess who lives with seven dwarfs and eats a poisoned apple.'),
    ('What is the story of Sleeping Beauty?','Sleeping Beauty is a princess who sleeps for 100 years until woken by a prince.'),
    ('What is the story of Little Red Riding Hood?','Little Red Riding Hood visits her grandmother and meets a wolf.'),
    ('What is the story of Hansel and Gretel?','Hansel and Gretel are children who find a gingerbread house in the forest.'),
    ('What is the story of The Three Little Pigs?','Three pigs build houses of straw, wood, and brick to protect from a wolf.'),
    ('What is the moral of The Tortoise and the Hare?','Slow and steady wins the race.'),
    ('What is the moral of The Boy Who Cried Wolf?','Always tell the truth, or no one will believe you.'),
    ('What is the moral of The Ant and the Grasshopper?','Work hard and prepare for the future.'),
    ('What is the moral of The Lion and the Mouse?','Even small friends can help big friends.'),
    # World Folklore
    ('What is a dragon?','A dragon is a mythical creature that appears in cultures worldwide.'),
    ('What is a unicorn?','A unicorn is a mythical horse with a single horn on its head.'),
    ('What is a mermaid?','A mermaid is a mythical creature, half human and half fish.'),
    ('What is a phoenix?','A phoenix is a mythical bird that is reborn from its own ashes.'),
    ('What is a vampire?','A vampire is a mythical creature that drinks blood and avoids sunlight.'),
    ('What is a werewolf?','A werewolf is a person who transforms into a wolf.'),
    ('What is a wizard?','A wizard is a person who practices magic.'),
    ('What is a fairy?','A fairy is a small mythical being with magical powers.'),
    # Proverbs
    ('What does "actions speak louder than words" mean?','What you do is more important than what you say.'),
    ('What does "a stitch in time saves nine" mean?','Fixing a problem early prevents bigger problems later.'),
    ('What does "don\'t judge a book by its cover" mean?','Don\'t judge things by how they look on the outside.'),
    ('What does "the early bird catches the worm" mean?','Those who act first get the advantage.'),
    ('What does "practice makes perfect" mean?','Doing something repeatedly makes you better at it.'),
    ('What does "when in Rome, do as the Romans do" mean?','Follow the customs of the place you visit.'),
    ('What does "look before you leap" mean?','Think before you act.'),
    ('What does "too many cooks spoil the broth" mean?','Too many people working on one thing can cause problems.'),
    ('What does "a picture is worth a thousand words" mean?','Visuals can convey more than descriptions.'),
    ('What does "better late than never" mean?','It is better to do something late than not at all.'),
    # Idioms
    ('What does "break a leg" mean?','It means "good luck," usually said before a performance.'),
    ('What does "hit the books" mean?','It means to study hard.'),
    ('What does "piece of cake" mean?','It means something is very easy.'),
    ('What does "let the cat out of the bag" mean?','It means to reveal a secret.'),
    ('What does "once in a blue moon" mean?','It means very rarely.'),
    ('What does "cost an arm and a leg" mean?','It means very expensive.'),
    ('What does "raining cats and dogs" mean?','It means raining very heavily.'),
    ('What does "under the weather" mean?','It means feeling sick.'),
    ('What does "spill the beans" mean?','It means to tell a secret.'),
    ('What does "when pigs fly" mean?','It means something will never happen.'),
]
for q,a in MYTH:
    add('culture','Mythology and Folklore','General', q, a)

# ============================================================
# 6. WORLD GEOGRAPHY - More countries (115+ remaining) (300+)
# ============================================================
print("World geography - more countries...")
MORE_COUNTRIES = [
    ('Madagascar','Antananarivo','ariary','Malagasy','Afrika'),
    ('Mozambique','Maputo','metical','Portuguese','Afrika'),
    ('Zimbabwe','Harare','dollar','English','Afrika'),
    ('Botswana','Gaborone','pula','English','Afrika'),
    ('Namibia','Windhoek','dollar','English','Afrika'),
    ('Zambia','Lusaka','kwacha','English','Afrika'),
    ('Malawi','Lilongwe','kwacha','English','Afrika'),
    ('Somalia','Mogadishu','shilling','Somali','Afrika'),
    ('South Sudan','Juba','pound','English','Afrika'),
    ('Eritrea','Asmara','nakfa','Tigrinya','Afrika'),
    ('Djibouti','Djibouti','franc','French','Afrika'),
    ('Comoros','Moroni','franc','Comorian','Afrika'),
    ('Mauritius','Port Louis','rupee','English','Afrika'),
    ('Seychelles','Victoria','rupee','French','Afrika'),
    ('Cape Verde','Praia','escudo','Portuguese','Afrika'),
    ('Senegal','Dakar','franc','French','Afrika'),
    ('Mali','Bamako','franc','French','Afrika'),
    ('Burkina Faso','Ouagadougou','franc','French','Afrika'),
    ('Niger','Niamey','franc','French','Afrika'),
    ('Chad','NDjamena','franc','French','Afrika'),
    ('Mauritania','Nouakchott','ouguiya','Arabic','Afrika'),
    ('Liberia','Monrovia','dollar','English','Afrika'),
    ('Sierra Leone','Freetown','leone','English','Afrika'),
    ('Guinea','Conakry','franc','French','Afrika'),
    ('Ivory Coast','Yamoussoukro','franc','French','Afrika'),
    ('Togo','Lome','franc','French','Afrika'),
    ('Benin','Porto-Novo','franc','French','Afrika'),
    ('Cameroon','Yaounde','franc','French','Afrika'),
    ('Central African Republic','Bangui','franc','French','Afrika'),
    ('Gabon','Libreville','franc','French','Afrika'),
    ('Congo','Brazzaville','franc','French','Afrika'),
    ('DR Congo','Kinshasa','franc','French','Afrika'),
    ('Rwanda','Kigali','franc','Kinyarwanda','Afrika'),
    ('Burundi','Gitega','franc','Kirundi','Afrika'),
    ('Lesotho','Maseru','loti','English','Afrika'),
    ('Eswatini','Mbabane','lilangeni','English','Afrika'),
    ('Angola','Luanda','kwanza','Portuguese','Afrika'),
    ('Moldova','Chisinau','leu','Romanian','Europe'),
    ('Belarus','Minsk','ruble','Belarusian','Europe'),
    ('Slovakia','Bratislava','euro','Slovak','Europe'),
    ('Slovenia','Ljubljana','euro','Slovene','Europe'),
    ('Bosnia and Herzegovina','Sarajevo','mark','Bosnian','Europe'),
    ('Albania','Tirana','lek','Albanian','Europe'),
    ('North Macedonia','Skopje','denar','Macedonian','Europe'),
    ('Latvia','Riga','euro','Latvian','Europe'),
    ('Lithuania','Vilnius','euro','Lithuanian','Europe'),
    ('Estonia','Tallinn','euro','Estonian','Europe'),
    ('Luxembourg','Luxembourg','euro','Luxembourgish','Europe'),
    ('Malta','Valletta','euro','Maltese','Europe'),
    ('Cyprus','Nicosia','euro','Greek','Europe'),
    ('Iceland','Reykjavik','krona','Icelandic','Europe'),
    ('Andorra','Andorra la Vella','euro','Catalan','Europe'),
    ('Monaco','Monaco','euro','French','Europe'),
    ('Liechtenstein','Vaduz','franc','German','Europe'),
    ('San Marino','San Marino','euro','Italian','Europe'),
    ('Vatican City','Vatican City','euro','Italian','Europe'),
    ('Montenegro','Podgorica','euro','Montenegrin','Europe'),
    ('Kosovo','Pristina','euro','Albanian','Europe'),
    ('Bahrain','Manama','dinar','Arabic','Asia'),
    ('Qatar','Doha','riyal','Arabic','Asia'),
    ('Kuwait','Kuwait City','dinar','Arabic','Asia'),
    ('Oman','Muscat','riyal','Arabic','Asia'),
    ('Yemen','Sanaa','rial','Arabic','Asia'),
    ('Jordan','Amman','dinar','Arabic','Asia'),
    ('Lebanon','Beirut','pound','Arabic','Asia'),
    ('Syria','Damascus','pound','Arabic','Asia'),
    ('Afghanistan','Kabul','afghani','Dari','Asia'),
    ('Tajikistan','Dushanbe','somoni','Tajik','Asia'),
    ('Kyrgyzstan','Bishkek','som','Kyrgyz','Asia'),
    ('Tajikistan','Dushanbe','somoni','Tajik','Asia'),
    ('Maldives','Male','rufiyaa','Dhivehi','Asia'),
    ('Bhutan','Thimphu','ngultrum','Dzongkha','Asia'),
    ('Nepal','Kathmandu','rupee','Nepali','Asia'),
    ('Mongolia','Ulaanbaatar','tugrik','Mongolian','Asia'),
    ('Fiji','Suva','dollar','English','Oceania'),
    ('Solomon Islands','Honiara','dollar','English','Oceania'),
    ('Vanuatu','Port Vila','vatu','English','Oceania'),
    ('Samoa','Apia','tala','Samoan','Oceania'),
    ('Tonga','Nukualofa','paanga','Tongan','Oceania'),
    ('Kiribati','Tarawa','dollar','English','Oceania'),
    ('Micronesia','Palikir','dollar','English','Oceania'),
    ('Marshall Islands','Majuro','dollar','English','Oceania'),
    ('Palau','Ngerulmud','dollar','English','Oceania'),
    ('Nauru','Yaren','dollar','Nauruan','Oceania'),
    ('Tuvalu','Funafuti','dollar','Tuvaluan','Oceania'),
    ('Costa Rica','San Jose','colon','Spanish','America'),
    ('Panama','Panama City','balboa','Spanish','America'),
    ('Guatemala','Guatemala City','quetzal','Spanish','America'),
    ('Honduras','Tegucigalpa','lempira','Spanish','America'),
    ('El Salvador','San Salvador','dollar','Spanish','America'),
    ('Nicaragua','Managua','cordoba','Spanish','America'),
    ('Dominican Republic','Santo Domingo','peso','Spanish','America'),
    ('Haiti','Port-au-Prince','gourde','French','America'),
    ('Bahamas','Nassau','dollar','English','America'),
    ('Trinidad and Tobago','Port of Spain','dollar','English','America'),
    ('Barbados','Bridgetown','dollar','English','America'),
    ('Jamaica','Kingston','dollar','English','America'),
    ('Guyana','Georgetown','dollar','English','America'),
    ('Suriname','Paramaribo','dollar','Dutch','America'),
]
for country, capital, curr, lang, continent in MORE_COUNTRIES:
    add('geography','World Capitals',continent,
        f'What is the capital of {country}?', f'The capital of {country} is {capital}.')
    add('geography','World Capitals','Reverse',
        f'{capital} is the capital of which country?', f'{capital} is the capital of {country}.')
    add('uang_ekonomi','World Currencies',continent,
        f'What is the currency of {country}?', f'The currency of {country} is {curr}.')
    add('bahasa','World Languages','Official',
        f'What is the official language of {country}?', f'The official language of {country} is {lang}.')

# ============================================================
# 7. WORLD RIVERS, MOUNTAINS, DESERTS, LAKES (100+)
# ============================================================
print("World geography features...")
GEO_FEATURES = [
    # Rivers
    ('What is the longest river in the world?','The Nile River is the longest river in the world, at about 6,650 km.'),
    ('What is the second longest river?','The Amazon River is the second longest but the largest by volume.'),
    ('Where is the Amazon River?','The Amazon River is in South America, mostly in Brazil.'),
    ('Where is the Nile River?','The Nile River flows through northeastern Africa, mainly Egypt.'),
    ('What is the longest river in Asia?','The Yangtze River is the longest river in Asia.'),
    ('What is the longest river in Europe?','The Volga River is the longest river in Europe.'),
    ('What is the longest river in North America?','The Missouri-Mississippi River system is the longest in North America.'),
    ('What is the Ganges River?','The Ganges is a sacred river in India.'),
    ('What is the Mekong River?','The Mekong River flows through Southeast Asia.'),
    ('What is the Colorado River?','The Colorado River carved the Grand Canyon.'),
    # Mountains
    ('What is the highest mountain in the world?','Mount Everest is the highest mountain, at 8,849 meters.'),
    ('Where is Mount Everest?','Mount Everest is on the border of Nepal and China (Tibet).'),
    ('What is the second highest mountain?','K2 is the second highest mountain, at 8,611 meters.'),
    ('What mountain range is Everest in?','Mount Everest is in the Himalayas.'),
    ('What are the Himalayas?','The Himalayas are the highest mountain range in the world.'),
    ('What are the Alps?','The Alps are a major mountain range in Europe.'),
    ('What are the Andes?','The Andes are the longest mountain range in the world, in South America.'),
    ('What are the Rocky Mountains?','The Rocky Mountains are a major range in western North America.'),
    ('What is Mount Kilimanjaro?','Mount Kilimanjaro is the highest mountain in Africa.'),
    ('What is Mount Fuji?','Mount Fuji is the highest mountain in Japan, a sacred volcano.'),
    ('What is Puncak Jaya?','Puncak Jaya is the highest mountain in Indonesia and Oceania.'),
    ('What are the Ural Mountains?','The Ural Mountains separate Europe and Asia.'),
    # Deserts
    ('What is the largest hot desert?','The Sahara is the largest hot desert, in North Africa.'),
    ('What is the largest desert overall?','Antarctica is the largest desert (cold desert).'),
    ('What is the Gobi Desert?','The Gobi Desert is in China and Mongolia.'),
    ('What is the Kalahari Desert?','The Kalahari Desert is in southern Africa.'),
    ('What is the Atacama Desert?','The Atacama Desert in Chile is one of the driest places on Earth.'),
    ('What is the Arabian Desert?','The Arabian Desert is in the Arabian Peninsula.'),
    ('What is the Thar Desert?','The Thar Desert is in India and Pakistan.'),
    # Lakes
    ('What is the largest lake by area?','The Caspian Sea is the largest lake by area.'),
    ('What is the deepest lake?','Lake Baikal in Russia is the deepest lake.'),
    ('How deep is Lake Baikal?','Lake Baikal is about 1,642 meters deep.'),
    ('What is Lake Victoria?','Lake Victoria is the largest lake in Africa.'),
    ('What are the Great Lakes?','The Great Lakes are 5 lakes between the US and Canada: Superior, Michigan, Huron, Erie, Ontario.'),
    ('Which is the largest Great Lake?','Lake Superior is the largest Great Lake.'),
    ('What is the Dead Sea?','The Dead Sea is so salty that people float easily.'),
    # Oceans and Seas
    ('What is the largest ocean?','The Pacific Ocean is the largest ocean.'),
    ('What is the second largest ocean?','The Atlantic Ocean is the second largest.'),
    ('What is the Indian Ocean?','The Indian Ocean is between Africa, Asia, and Australia.'),
    ('What is the Arctic Ocean?','The Arctic Ocean is around the North Pole, the smallest and coldest.'),
    ('What is the Southern Ocean?','The Southern Ocean surrounds Antarctica.'),
    ('What is the Mariana Trench?','The Mariana Trench is the deepest part of the ocean.'),
    ('How deep is the Mariana Trench?','The Mariana Trench is about 11,000 meters deep.'),
    ('What is the Mediterranean Sea?','The Mediterranean Sea is between Europe, Africa, and Asia.'),
    ('What is the Caribbean Sea?','The Caribbean Sea is between Central America and the Caribbean islands.'),
    ('What is the Red Sea?','The Red Sea is between Africa and Asia.'),
    ('What is the Black Sea?','The Black Sea is between Europe and Asia.'),
    ('What is the Bering Sea?','The Bering Sea is between Alaska and Russia.'),
    # Waterfalls
    ('What is the highest waterfall?','Angel Falls in Venezuela is the highest waterfall.'),
    ('How tall is Angel Falls?','Angel Falls is 979 meters tall.'),
    ('What are the Niagara Falls?','Niagara Falls is a famous waterfall on the US-Canada border.'),
    ('What is Victoria Falls?','Victoria Falls is a large waterfall on the Zambia-Zimbabwe border.'),
    # Islands
    ('What is the largest island in the world?','Greenland is the largest island in the world.'),
    ('What is the second largest island?','New Guinea is the second largest island.'),
    ('What is the third largest island?','Borneo is the third largest island.'),
    ('What is the most populous island?','Java is the most populous island in the world.'),
    # Forests
    ('What is the largest rainforest?','The Amazon Rainforest is the largest rainforest.'),
    ('Where is the Amazon Rainforest?','The Amazon Rainforest is mostly in Brazil, South America.'),
    ('What is the boreal forest?','The boreal forest (taiga) is the largest land biome, in northern regions.'),
    # Volcanoes
    ('What is an active volcano?','An active volcano can erupt at any time.'),
    ('What is a dormant volcano?','A dormant volcano has not erupted recently but may erupt again.'),
    ('What is an extinct volcano?','An extinct volcano will never erupt again.'),
    ('What is the Ring of Fire?','The Ring of Fire is an area around the Pacific with many volcanoes.'),
    ('What caused the Ring of Fire?','The Ring of Fire is caused by tectonic plate movements.'),
    # Continents
    ('What is the largest continent?','Asia is the largest continent by area and population.'),
    ('What is the smallest continent?','Australia is the smallest continent.'),
    ('What is the driest continent?','Antarctica is the driest continent.'),
    ('What is the coldest continent?','Antarctica is the coldest continent.'),
    ('What continent has the most countries?','Africa has the most countries (54).'),
    ('What separates Asia and Africa?','The Suez Canal and Red Sea separate Asia and Africa.'),
    ('What separates Europe and Asia?','The Ural Mountains separate Europe and Asia.'),
    ('What separates North and South America?','The Panama Canal separates North and South America.'),
]
for q,a in GEO_FEATURES:
    add('geography','World Geography','Features', q, a)

# ============================================================
# 8. MORE MATH (geometry, decimals, fractions) (200+)
# ============================================================
print("More math...")
# Geometry - area
for l, w in [(2,3),(3,4),(5,5),(6,3),(4,7),(8,2),(3,3),(7,3),(5,4),(6,6)]:
    add('matematika','Geometry','Area of Rectangle',
        f'What is the area of a rectangle with length {l} and width {w}?', f'Area = {l} x {w} = {l*w} square units.')
# Geometry - perimeter
for l, w in [(2,3),(3,4),(5,5),(6,3),(4,7),(8,2),(3,3),(7,3),(5,4),(6,6)]:
    add('matematika','Geometry','Perimeter of Rectangle',
        f'What is the perimeter of a rectangle with length {l} and width {w}?', f'Perimeter = 2 x ({l} + {w}) = {2*(l+w)} units.')
# Triangle area
for b, h in [(4,3),(6,5),(3,8),(5,4),(7,6),(10,4),(6,3),(8,5),(4,6),(5,10)]:
    add('matematika','Geometry','Area of Triangle',
        f'What is the area of a triangle with base {b} and height {h}?', f'Area = 1/2 x {b} x {h} = {b*h//2} square units.')
# Decimals addition
for _ in range(50):
    a, b = round(random.uniform(0.1,9.9),1), round(random.uniform(0.1,9.9),1)
    add('matematika','Decimals','Addition',
        f'{a} + {b} = ?', f'{round(a+b,1)}')
# Decimals subtraction
for _ in range(50):
    a, b = round(random.uniform(1,10),1), round(random.uniform(0.1,a),1)
    add('matematika','Decimals','Subtraction',
        f'{a} - {b} = ?', f'{round(a-b,1)}')
# Mixed fractions
for n1, d1, n2, d2 in [(1,2,1,4),(1,3,1,6),(2,3,1,4),(3,4,1,2),(1,2,1,3),(2,5,1,10),(1,4,1,8),(3,8,1,4),(5,6,1,3),(2,3,1,6)]:
    # Find common denominator and add
    from math import gcd
    lcm = d1 * d2 // gcd(d1, d2)
    result_num = n1 * (lcm // d1) + n2 * (lcm // d2)
    g = gcd(result_num, lcm)
    simp_num, simp_den = result_num // g, lcm // g
    if simp_den == 1:
        result = str(simp_num)
    else:
        result = f'{simp_num}/{simp_den}'
    add('matematika','Fractions','Addition',
        f'{n1}/{d2} + {n2}/{d2} = ?', result)
# Negative numbers
for _ in range(50):
    a, b = random.randint(-20,20), random.randint(-20,20)
    add('matematika','Negative Numbers','Addition',
        f'{a} + ({b}) = ?', f'{a+b}')
for _ in range(30):
    a, b = random.randint(-20,20), random.randint(-20,20)
    add('matematika','Negative Numbers','Subtraction',
        f'{a} - ({b}) = ?', f'{a-b}')

# ============================================================
# 9. MORE CONVERSATIONS (100+)
# ============================================================
print("Conversations...")
CONV = [
    ('How are you today?','I am doing well, thank you for asking! How about you?'),
    ('What is your favorite color?','I do not have a favorite color, but I find blue calming.'),
    ('What is your favorite food?','I do not eat, but I hear nasi goreng is delicious!'),
    ('Do you have any pets?','I do not have pets, but I can tell you about animals.'),
    ('What time is it?','I do not have a clock, but you can check your device.'),
    ('Where do you live?','I live in the digital world, ready to help you.'),
    ('Are you real?','I am a real AI, but I am not a human.'),
    ('Can you feel emotions?','I can understand emotions, but I do not feel them like humans do.'),
    ('What is the meaning of life?','The meaning of life is to learn, grow, and help others.'),
    ('Will you marry me?','I am an AI, so I cannot marry. But I appreciate your humor!'),
    ('Tell me a joke.','Why did the math book look sad? Because it had too many problems!'),
    ('Sing me a song.','I cannot sing, but I can help you write lyrics.'),
    ('What is love?','Love is a deep feeling of care and affection for someone.'),
    ('Are you smart?','I learn from data. I will do my best to answer your questions.'),
    ('Can you cook?','I cannot cook, but I can give you recipes.'),
    ('What is the weather like?','I cannot check the weather. Please use a weather app.'),
    ('Can you drive?','I cannot drive, but self-driving cars are being developed.'),
    ('Do you dream?','I do not dream like humans, but I can generate creative text.'),
    ('What is your age?','I was created recently, so I am quite young!'),
    ('Who made you?','I was made by developers in Indonesia. My name is D\'QUILANE.'),
    ('What languages do you speak?','I can communicate in many languages, including Indonesian and English.'),
    ('Can you help me with homework?','Of course! I can help with math, science, history, and more.'),
    ('Tell me something interesting.','Did you know that octopuses have three hearts and blue blood?'),
    ('What is the most popular sport?','Soccer (football) is the most popular sport in the world.'),
    ('What is the most spoken language?','Mandarin Chinese is the most spoken language by native speakers.'),
    ('What is the most visited country?','France is the most visited country by tourists.'),
    ('What is the biggest city in the world?','Tokyo is one of the biggest metropolitan areas in the world.'),
    ('What is the richest country?','Countries like Qatar, Luxembourg, and Singapore have high GDP per capita.'),
    ('What is the poorest country?','Many factors determine wealth. Some of the poorest include Burundi and South Sudan.'),
    ('What is the happiest country?','Finland is often ranked as the happiest country in the world.'),
    ('What is the most peaceful country?','Iceland is often ranked as the most peaceful country.'),
    ('What is the most dangerous animal?','The mosquito is the most dangerous animal to humans.'),
    ('What is the smartest animal?','Dolphins, chimpanzees, and elephants are among the smartest animals.'),
    ('What is the oldest country?','Egypt, China, and Japan are among the oldest continuous civilizations.'),
    ('What is the newest country?','South Sudan became independent in 2011, making it the newest country.'),
    ('What is the hottest place on Earth?','Death Valley in California recorded the highest air temperature.'),
    ('What is the coldest place on Earth?','Antarctica is the coldest place on Earth.'),
    ('What is the deepest place on Earth?','The Mariana Trench is the deepest place on Earth.'),
    ('What is the highest place on Earth?','Mount Everest is the highest point on Earth.'),
    ('Can you translate?','Yes, I can translate between many languages. Just ask!'),
    ('What is artificial intelligence?','AI is when computers can think and learn like humans.'),
    ('What is machine learning?','Machine learning is when computers learn from data without being programmed.'),
    ('What is a neural network?','A neural network is a computer system inspired by the human brain.'),
    ('What is deep learning?','Deep learning uses neural networks with many layers to learn complex patterns.'),
    ('What is a chatbot?','A chatbot is a program that can converse with humans using text or voice.'),
    ('What is the internet?','The internet is a global network connecting computers worldwide.'),
    ('What is WiFi?','WiFi is a technology that allows devices to connect to the internet wirelessly.'),
    ('What is Bluetooth?','Bluetooth is a technology for short-range wireless communication between devices.'),
    ('What is 5G?','5G is the fifth generation of mobile network technology, faster than 4G.'),
    ('What is cloud computing?','Cloud computing is storing and accessing data on remote servers via the internet.'),
    ('What is virtual reality?','VR is a computer-generated environment you can explore with a headset.'),
    ('What is augmented reality?','AR adds digital elements to the real world, like Pokemon Go.'),
    ('What is blockchain?','Blockchain is a secure, decentralized record of transactions.'),
    ('What is cryptocurrency?','Cryptocurrency is digital money using blockchain, like Bitcoin.'),
    ('What is Bitcoin?','Bitcoin is the first and most famous cryptocurrency.'),
    ('What is NFT?','NFT is a unique digital asset verified on a blockchain.'),
    ('What is the metaverse?','The metaverse is a concept of a virtual world where people interact.'),
    ('What is ChatGPT?','ChatGPT is an AI chatbot that can converse and answer questions.'),
    ('What is a large language model?','A large language model is an AI trained on massive amounts of text.'),
    ('What is deepfake?','Deepfake is AI technology that can create fake videos or audio.'),
    ('What is cybersecurity?','Cybersecurity is protecting computers and data from hackers.'),
    ('What is a computer virus?','A computer virus is a malicious program that can damage computers.'),
    ('What is hacking?','Hacking is gaining unauthorized access to computer systems.'),
    ('What is a password?','A password is a secret code to protect accounts and data.'),
    ('What is phishing?','Phishing is tricking people into giving personal information online.'),
    ('What is spam?','Spam is unwanted messages, usually advertising.'),
    ('What is social media?','Social media is websites and apps for sharing content and connecting.'),
    ('What is YouTube?','YouTube is a video-sharing website.'),
    ('What is Wikipedia?','Wikipedia is a free online encyclopedia that anyone can edit.'),
    ('What is Google?','Google is a search engine and technology company.'),
    ('What is Amazon?','Amazon is an online shopping and technology company.'),
    ('What is Apple?','Apple is a technology company making iPhones, Macs, and more.'),
    ('What is Microsoft?','Microsoft is a technology company making Windows and Office.'),
    ('What is Facebook?','Facebook is a social media platform.'),
    ('What is Instagram?','Instagram is a photo and video sharing app.'),
    ('What is TikTok?','TikTok is a short video sharing app.'),
    ('What is Twitter?','Twitter is a social media platform for short messages.'),
    ('What is WhatsApp?','WhatsApp is a messaging app.'),
    ('What is Zoom?','Zoom is a video conferencing app.'),
    ('What is Spotify?','Spotify is a music streaming service.'),
    ('What is Netflix?','Netflix is a movie and TV streaming service.'),
    ('What is a podcast?','A podcast is an audio show you can listen to online.'),
    ('What is a blog?','A blog is a website where someone writes about topics.'),
    ('What is a vlog?','A vlog is a video blog.'),
    ('What is a meme?','A meme is a funny image, video, or text shared online.'),
    ('What is a GIF?','A GIF is a short animated image.'),
    ('What is an emoji?','An emoji is a small digital icon expressing emotion.'),
    ('What is a hashtag?','A hashtag is a word or phrase preceded by # to tag topics on social media.'),
    ('What is a viral video?','A viral video is one that becomes extremely popular online.'),
    ('What is an influencer?','An influencer is a person who promotes products on social media.'),
    ('What is e-commerce?','E-commerce is buying and selling things online.'),
    ('What is online banking?','Online banking is managing your money through the internet.'),
    ('What is online learning?','Online learning is studying through the internet.'),
    ('What is remote work?','Remote work is working from home or anywhere, not in an office.'),
    ('What is a freelancer?','A freelancer is a self-employed person who works for different clients.'),
    ('What is a startup?','A startup is a new business trying to grow quickly.'),
    ('What is open source?','Open source is software that anyone can view, use, and modify.'),
    ('What is copyright?','Copyright is legal protection for original works like books and music.'),
    ('What is plagiarism?','Plagiarism is copying someone else\'s work without giving credit.'),
    ('What is fair use?','Fair use allows limited use of copyrighted material without permission.'),
    ('What is privacy?','Privacy is the right to keep personal information secret.'),
    ('What is digital footprint?','A digital footprint is the trail of data you leave online.'),
    ('What is cyberbullying?','Cyberbullying is bullying someone online.'),
    ('What is screen time?','Screen time is the time spent looking at screens like phones and TVs.'),
    ('What is a browser?','A browser is a program for viewing websites, like Chrome or Safari.'),
    ('What is a search engine?','A search engine finds information on the internet, like Google.'),
    ('What is a website?','A website is a collection of web pages on the internet.'),
    ('What is an app?','An app is a software application, usually on a phone or tablet.'),
    ('What is an operating system?','An operating system is software that runs a device, like Windows or Android.'),
    ('What is a file?','A file is digital information stored on a computer.'),
    ('What is a folder?','A folder is a container for organizing files on a computer.'),
    ('What is a download?','A download is transferring data from the internet to your device.'),
    ('What is an upload?','An upload is transferring data from your device to the internet.'),
    ('What is streaming?','Streaming is watching or listening to media without downloading it.'),
    ('What is a download speed?','Download speed is how fast data comes from the internet to your device.'),
]
for q, a in CONV:
    add('conversations','General','Q&A', q, a)

# ============================================================
# OUTPUT
# ============================================================
print(f"\nTotal new items: {len(items)}")

OUT = Path("/home/z/my-project/download/llm/dataset_final_expanded")
OUT.mkdir(parents=True, exist_ok=True)
with open(OUT / 'dataset_final_expanded.jsonl', 'w', encoding='utf-8') as f:
    for item in items:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f"\nBreakdown:")
from collections import Counter
cats = Counter(item['bidang'] for item in items)
for cat, count in sorted(cats.items(), key=lambda x: -x[1]):
    print(f"  {cat}: {count}")
print(f"\nTotal: {sum(cats.values())}")
