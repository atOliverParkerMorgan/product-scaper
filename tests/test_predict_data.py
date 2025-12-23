from product_scraper.train_model.predict_data import (
    calculate_proximity_score,
    group_prediction_to_products,
)
from product_scraper.utils.features import extract_index, get_xpath_segments


def test_get_xpath_segments():
    xpath = "/html/body/div[2]/span"
    segs = get_xpath_segments(xpath)
    assert segs == ["html", "body", "div[2]", "span"]


def test_extract_index():
    assert extract_index("div[3]") == 3
    assert extract_index("span") == 1


def test_calculate_proximity_score():
    x1 = "/html/body/div[2]/span[1]"
    x2 = "/html/body/div[2]/span[2]"
    score = calculate_proximity_score(x1, x2)
    assert isinstance(score, tuple)
    assert score[0] >= 0 and score[1] >= 0


def test_group_prediction_to_products():
    selectors = {
        "title": [
            {
                "xpath": "/html/body/div/h1",
                "tag": "h1",
                "preview": "Title",
                "class": "",
                "id": "",
                "index": 0,
            }
        ],
        "price": [
            {
                "xpath": "/html/body/div/span",
                "tag": "span",
                "preview": "Price",
                "class": "",
                "id": "",
                "index": 0,
            }
        ],
    }
    categories = ["title", "price"]
    products = group_prediction_to_products(selectors, categories)
    assert isinstance(products, list)
    assert len(products) == 1
    assert "title" in products[0] and "price" in products[0]
