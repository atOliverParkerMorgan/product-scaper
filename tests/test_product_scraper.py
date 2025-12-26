from product_scraper import ProductScraper


def test_product_scraper_basic_usage():
    categories = ["title", "price"]
    websites = ["http://example.com"]
    selectors = {
        "http://example.com": {"title": ["//h1"], "price": ['//span[@class="price"]']}
    }
    scraper = ProductScraper(categories, websites, selectors=selectors)
    # Compare sorted categories to avoid order issues
    assert sorted(scraper.categories) == sorted(categories)
    assert scraper.websites_urls == websites
    assert scraper.selectors == selectors
    # Test iteration
    for url, preds in scraper:
        assert url == "http://example.com"
        assert "title" in preds and "price" in preds
    # Test __len__
    assert len(scraper) == 1


def test_product_scraper_setters():
    categories = ["title"]
    websites = ["http://test.com"]
    scraper = ProductScraper(categories, websites)
    scraper.add_website("http://another.com")
    assert "http://another.com" in scraper.websites_urls
    scraper.set_website_selectors("http://test.com", {"title": ["//h2"]})
    assert scraper.selectors["http://test.com"]["title"] == ["//h2"]


def test_get_selectors_predict(monkeypatch):
    categories = ["title"]
    websites = ["http://test.com"]
    scraper = ProductScraper(categories, websites)
    scraper.model = object()
    monkeypatch.setattr(scraper, "predict_category", lambda url, cat: ["pred"])
    scraper.selectors = {}
    out = scraper.get_selectors("http://test.com")
    assert out["title"] == ["pred"]


def test_save_model_untrained(tmp_path):
    categories = ["title"]
    websites = ["http://test.com"]
    scraper = ProductScraper(categories, websites)
    scraper.model = None
    try:
        scraper.save_model(str(tmp_path / "model.pkl"))
    except ValueError:
        pass
    else:
        assert False, "Should raise ValueError if model is not trained"


def test_len_and_iter(monkeypatch):
    categories = ["title", "price"]
    websites = ["http://a.com", "http://b.com"]
    scraper = ProductScraper(categories, websites)
    scraper.model = object()
    monkeypatch.setattr(scraper, "predict_category", lambda url, cat: ["dummy"])
    urls = [url for url, _ in scraper]
    assert urls == ["http://a.com", "http://b.com"]


def test_add_category_and_categories(monkeypatch):
    categories = ["title"]
    websites = ["http://test.com"]
    scraper = ProductScraper(categories.copy(), websites)
    scraper.add_category("price")
    assert "price" in scraper.categories
    # Adding duplicate triggers warning
    monkeypatch.setattr("product_scraper.core.log_warning", lambda msg: None)
    scraper.add_category("price")  # Should not add again
    before = set(scraper.categories)
    scraper.add_categories(["image", "title"])
    assert "image" in scraper.categories
    assert set(before).issubset(scraper.categories)


def test_add_websites():
    categories = ["title"]
    websites = ["http://a.com"]
    scraper = ProductScraper(categories, websites)
    scraper.add_websites(["http://b.com", "http://c.com"])
    assert "http://b.com" in scraper.websites_urls
    assert "http://c.com" in scraper.websites_urls


def test_set_website_selectors_from_yaml(tmp_path):
    categories = ["title"]
    websites = ["http://test.com"]
    scraper = ProductScraper(categories, websites)
    yaml_path = tmp_path / "selectors.yaml"
    yaml_content = {"title": ["//h1"]}
    import yaml as _yaml

    with open(yaml_path, "w", encoding="utf-8") as f:
        _yaml.safe_dump(yaml_content, f)
    scraper.set_website_selectors_from_yaml("http://test.com", str(yaml_path))
    assert scraper.selectors["http://test.com"] == yaml_content
    assert "http://test.com" in scraper.websites_urls
