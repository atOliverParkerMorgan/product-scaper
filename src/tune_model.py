from ProductScraper import ProductScraper

WEBSITE = [
   "https://www.valentinska.cz/home",
   "https://www.antikvariatchrudim.cz/",
   "http://antikvariat-cypris.cz/novinky.php",
   "https://www.valentinska.cz/home",
   "https://trigon-knihy.cz/antikvariat/",
   "https://www.artonpaper.ch/new"
]

CATEGORIES = [
    "title",
    "price",
    "image"
]

Pscraper = ProductScraper(CATEGORIES, WEBSITE)

Pscraper.load_selectors()

Pscraper.train_model(create_data=True)
