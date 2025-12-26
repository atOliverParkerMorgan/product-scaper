"""
Example script for running ProductScraper on a set of websites and categories.
"""

from product_scraper.core import ProductScraper

WEBSITES = [
    "https://www.valentinska.cz/home",
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
    "https://www.antikvariat-delta.cz/",
    "https://www.antikvariat-divis.cz/cze/novinky",
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
    "http://www.antikbuddha.com/czech/article.php?new=1&test=Y",
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
    "http://www.ztichlaklika.cz/antikvariat?page=1",
]

CATEGORIES = ["title", "price", "image"]

if __name__ == "__main__":
    # If the example data directory does not exist, create a new scraper
    product_scraper = ProductScraper(
        categories=CATEGORIES,
        websites_urls=WEBSITES,
        save_dir="./src/example_scraper_data",
    )

    product_scraper.load_selectors()

    # Load the example selectors provided with the package
    # product_scraper = ProductScraper.load(save_dir="./src/example_scraper_data")
    # Run just in case there is missing data
    product_scraper.create_training_data()

    # Train the model
    product_scraper.train_model(show_model_figure=True)

    # Predict on all websites
    results = product_scraper.predict(["https://www.antikvariatchrudim.cz"])

    for url, products in results.items():
        print(f"\n--- Found {len(products)} products on {url} ---")

        for i, product in enumerate(products):
            print(f"Product #{i + 1}")
            for category, data in product.items():
                print(f"{category}: {data}")

    product_scraper.save()
