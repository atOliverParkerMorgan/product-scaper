"""Tests for extracting main content from HTML pages."""

import pytest

from train_model import process_data

# -----------------------------
# Fixtures
# -----------------------------

@pytest.fixture
def html_with_main():
    return """
    <html>
        <body>
            <header>This is header</header>
            <menu>
                <a href="#">Home</a>
                <a href="#">Products</a>
                <a href="#">About</a>
                <a href="#">Contact</a>
            </menu>
            <main>
                <div class="product">Product 1 <p>Some log decription of some random product bla bla</p><img></div>
                <div class="product">Product 2 <p>Some log decription of some random product bla bla</p><img></div>
                <div class="product">Product 3 <p>Some log decription of some random product bla bla</p><img></div>
                <div class="product">Product 4<p>Some log decription of some random product bla bla</p><img></div>
                <div class="product">Product 5<p>Some log decription of some random product bla bla bla bla</p><img></div>
                <div class="product">Product 6 <p>Some log decription of some random product bla bla bla bla</p><img></div>
            </main>
            <footer>This is footer</footer>
        </body>
    </html>
    """


@pytest.fixture
def html_without_main():
    return """
    <html>
        <style>body {font-family: Arial;}</style>
        <body></body>
        <iframe src="ads.html"></iframe>
        <script>console.log("Hello World")</script>
    </html>
    """


@pytest.fixture
def load_html():
    """Fixture returning a loader function."""
    def _loader(name):
        with open(f"tests/test_data/{name}.html", encoding="utf-8") as f:
            return f.read()
    return _loader


# -----------------------------
# Tests
# -----------------------------

def test_basic_main_content(html_with_main):
    main_content = process_data.get_main_html_content_tag(html_with_main)

    assert main_content is not None
    assert main_content.tag == "main"

    products = main_content.findall('.//div[@class="product"]')
    assert len(products) == 6

    assert products[0].text_content() == (
        "Product 1 Some log decription of some random product bla bla"
    )
    assert products[1].text_content() == (
        "Product 2 Some log decription of some random product bla bla"
    )


def test_no_main_content(html_without_main):
    main_content = process_data.get_main_html_content_tag(html_without_main)

    assert main_content is not None
    assert main_content.tag == "body"  # fallback behavior


# -----------------------------
# Parametrized page tests
# -----------------------------

@pytest.mark.parametrize(
    "page,expected_tag,expected_class,expected_id",
    [
        ("page_1", "div", "w3-row-padding", None),
        ("page_2", "div", None, "content"),
        ("page_3", "div", None, "content"),
        ("page_4", "div", "items-list", None),
        ("page_5", "div", None, "content"),
        ("page_6", "ul", "products", None),
        ("page_7", "div", None, "content"),
        ("page_8", "div", None, "content"),
        ("page_9", "div", None, "content"),
        ("page_10", "ul", "list-style-none", None),
        ("page_11", "div", None, "content"),
        ("page_12", "ul", None, None),
        ("page_13", "div", None, None),
        ("page_14", "div", None, "incenterpage"),
        ("page_15", "div", "tab-content", None),
    ],
)
def test_main_content_pages(load_html, page, expected_tag, expected_class, expected_id):
    html = load_html(page)
    main_content = process_data.get_main_html_content_tag(html)

    assert main_content is not None
    assert main_content.tag == expected_tag

    if expected_id:
        assert main_content.get("id") == expected_id

    if expected_class:
        assert expected_class in list(main_content.classes)
