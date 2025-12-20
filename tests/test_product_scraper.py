
import pytest
import pandas as pd
from ProductScraper import ProductScraper

def test_product_scraper_basic_usage():
    categories = ['title', 'price']
    websites = ['http://example.com']
    selectors = {
        'http://example.com': {
            'title': ['//h1'],
            'price': ['//span[@class="price"]']
        }
    }
    scraper = ProductScraper(categories, websites, selectors=selectors)
    assert scraper.categories == categories
    assert scraper.websites_urls == websites
    assert scraper.selectors == selectors
    # Test iteration
    for url, preds in scraper:
        assert url == 'http://example.com'
        assert 'title' in preds and 'price' in preds
    # Test __len__
    assert len(scraper) == 1

def test_product_scraper_setters():
    categories = ['title']
    websites = ['http://test.com']
    scraper = ProductScraper(categories, websites)
    scraper.add_website('http://another.com')
    assert 'http://another.com' in scraper.websites_urls
    scraper.set_website_selectors('http://test.com', {'title': ['//h2']})
    assert scraper.selectors['http://test.com']['title'] == ['//h2']
