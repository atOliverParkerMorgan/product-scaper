from create_data import select_data
from train_model.process_data import data_to_csv


WEBSITES = [
    "https://www.artonpaper.ch/new",

]

CATEGORIES = [
    "title",
    "price",
    "image",
]

if __name__ == "__main__":

    for url in WEBSITES:
        # select_data(url, CATEGORIES)
        data_to_csv()
        # model = train_model()