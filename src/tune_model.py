from ProductScraper import ProductScraper

WEBSITES = [
    "https://www.valentinska.cz/home",
    "https://trigon-knihy.cz/antikvariat/",
    "https://www.artonpaper.ch/new",
    "http://antik-stafl.cz/",
    "http://antikvariat-bohemia.cz/",
    "http://antikvariat-cypris.cz/novinky.php",
    "http://antikvariat-malyctenar.cz/",
    "http://antikvariat-pce.cz/",
    "http://antikvariat-prelouc.cz/",
    "http://antikvariat-vinohradska.cz/",
    "http://dva-antikvari.cz/nabidka/strana/1",
    "https://antik-bilevrany.cz/",
    "https://antikvariatelement.cz/",
    "https://antikvariat-fryc.cz/",
    "https://obchod.kniharium.eu/142-avantgarda-a-bibliofilie",
    "https://samota.cz/kat/nov/antikvariat-samota-novinky.html",
    "https://spalena53.cz/nove-knihy",
    "https://umeleckafotografie.cz/",
    "https://www.adplus.cz/",
    "https://www.antikalfa.cz/bibliofilie/",
    "https://www.antikalfa.cz/krasna-literatura/",
    "https://www.antikalfa.cz/obalky-ilustrace-vazby-podpisy-1-vydani/",
    "https://www.antikavion.cz/",
    "https://www.antikvariat-benes.cz/eshop/",
    "https://www.antikvariatchrudim.cz/",
    "https://www.antikvariat-delta.cz/",
    "https://www.antikvariat-divis.cz/cze/novinky",
    "https://www.antikvariatik.sk/en/novinky",
    "https://www.antikvariatkrenek.com/",
    "https://www.antikvariat-olomouc.cz/cz-sekce-novinky.html",
    "https://www.antikvariatschodydoneba.sk/obchod",
    "https://www.antikvariatshop.sk/",
    "https://www.antikvariatsteiner.sk/eshop",
    "https://www.antikvariatukostela.cz/cz-sekce-novinky.html",
    "https://www.antikvariat-vltavin.cz/",
    "https://www.artbook.cz/collections/akutalni-nabidka",
    "https://www.leonclifton.cz/novinky?page=0&size=50",
    "https://www.morganbooks.eu",
    "https://www.podzemni-antikvariat.cz/",
    "https://www.shioriantikvariat.cz/search.php?search=novinka",
    "https://www.woodbook.cz/",
    "https://www.woodbook.cz/vzacne-knihy/",
    "http://www.antikbuddha.com/czech/article.php?new=1&test=Y",
    "http://www.antiknadrazismichov.cz/",
    "http://www.antiknarynku.cz/",
    "http://www.antikopava.cz/prodej-knih/novinky",
    "http://www.antikvariat-janos.cz/",
    "http://www.antikvariatkamyk.cz/",
    "http://www.antikvariatkarlin.cz/",
    "http://www.antikvariatpocta.cz/novinky",
    "http://www.antikvariat-susice.cz/index.php?typ=php&kategorie=novinky",
    "http://www.antikvariat-vltavin.cz/",
    "http://www.antikvariaty.cz/",
    "http://www.antikvariat-zlin.cz/",
    "http://www.dantikvariat.cz/nabidka-knihy",
    "http://www.galerie-ilonka.cz/galerie-ilonka/eshop/9-1-Antikvariat",
    "http://www.knizky.com/index.php?Akce=Prove%EF&CenterFrame=hledej.php&LeftFrame=prohlmenu.php&order_id=7&order_dir=1",
    "http://www.levnyantikvariat.cz/czech/",
    "https://www.valentinska.cz/home",
    "http://www.ztichlaklika.cz/antikvariat?page=1",
]


CATEGORIES = ["title", "price", "image"]

try:
    # Pscraper = ProductScraper.load()
    Pscraper = ProductScraper(categories=CATEGORIES)

except FileNotFoundError:
    Pscraper = ProductScraper(categories=CATEGORIES)

Pscraper.load_selectors()

Pscraper.train_model(create_data=True)

Pscraper.save()
Pscraper.add_websites(WEBSITES)
Pscraper.create_training_data()
