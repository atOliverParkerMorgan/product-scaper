from product_scraper import ProductScraper


def test_len_and_add_remove_website():
    categories = ["title"]
    urls = ["http://a.com"]
    scraper = ProductScraper(categories, urls.copy())
    assert len(scraper) == 1
    scraper.add_website("http://b.com")
    assert "http://b.com" in scraper.websites_urls
    scraper.remove_website("http://b.com")
    assert "http://b.com" not in scraper.websites_urls
    # Remove non-existent
    scraper.remove_website("http://notfound.com")


def test_set_pipeline():
    categories = ["title"]
    urls = ["http://a.com"]
    scraper = ProductScraper(categories, urls)

    class DummyPipeline:
        pass

    pipe = DummyPipeline()
    scraper.set_pipeline(pipe)
    assert scraper.pipeline is pipe


def test_set_website_selectors():
    categories = ["title"]
    urls = ["http://a.com"]
    scraper = ProductScraper(categories, urls)
    selectors = {"title": ["//h1"]}
    scraper.set_website_selectors("http://a.com", selectors)
    assert scraper.selectors["http://a.com"]["title"] == ["//h1"]
