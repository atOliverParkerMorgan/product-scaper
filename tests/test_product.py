import pytest
from decimal import Decimal
from product_scraper import Product

def test_product_creation():
    product = Product(
        name="Test Product",
        price=Decimal("99.99"),
        url="https://example.com/product",
        currency="USD"
    )
    
    assert product.name == "Test Product"
    assert product.price == Decimal("99.99")
    assert product.currency == "USD"
    assert product.url == "https://example.com/product"

def test_product_to_dict():
    product = Product(
        name="Test Product",
        price=Decimal("99.99"),
        url="https://example.com/product",
        currency="USD",
        description="A test product",
        specifications={"color": "red"}
    )
    
    product_dict = product.to_dict()
    assert product_dict["name"] == "Test Product"
    assert product_dict["price"] == "99.99"
    assert product_dict["specifications"] == {"color": "red"}
    assert "sku" not in product_dict  # Should not include None values