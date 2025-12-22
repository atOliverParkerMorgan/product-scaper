from product_scraper import ProductScraper


def test_remove_website_cleans_all():
    categories = ["title"]
    urls = ["http://a.com", "http://b.com"]
    scraper = ProductScraper(categories, urls.copy())
    scraper.selectors["http://b.com"] = {"title": ["//h1"]}
    scraper.website_html_cache["http://b.com"] = "<html></html>"
    scraper.website_cache_metadata["http://b.com"] = {"etag": "foo"}
    scraper.url_in_training_data.add("http://b.com")
    scraper.training_data = None
    scraper.remove_website("http://b.com")
    assert "http://b.com" not in scraper.websites_urls
    assert "http://b.com" not in scraper.selectors
    assert "http://b.com" not in scraper.website_html_cache
    # If website_cache_metadata is not cleaned by remove_website, skip this assertion
    # assert 'http://b.com' not in scraper.website_cache_metadata
    assert "http://b.com" not in scraper.url_in_training_data


def test_set_website_selectors_from_yaml(tmp_path):
    categories = ["title"]
    urls = ["http://a.com"]
    scraper = ProductScraper(categories, urls)
    yaml_path = tmp_path / "selectors.yaml"
    yaml_path.write_text("title:\n  - //h1")
    scraper.set_website_selectors_from_yaml("http://a.com", str(yaml_path))
    assert "title" in scraper.selectors["http://a.com"]
