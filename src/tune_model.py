from ProductScraper import ProductScraper
from utils.console import log_info

WEBSITES = [
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
try:
    Pscraper = ProductScraper.load()
    log_info("Loaded existing ProductScraper state.")
except FileNotFoundError:
    Pscraper = ProductScraper(categories=CATEGORIES)

Pscraper.load_selectors()

Pscraper.train_model(create_data=True)

Pscraper.save()
print(Pscraper.predict(WEBSITES))
