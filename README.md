# Product Scraper

A Python package for extracting product information from e-commerce websites using machine learning.

## Installation

```bash
pip install product-scraper
```

## Basic Usage

```python
import productscraper
    
ps = productscraper(categories=["title", "price", "image"], websites_urls=["https://www.morganbooks.eu/"])
ps.predict()


```

Tests
```bash
pytest --cov=src/utils/features.py --cov-report=term-missing
```

## Training Data Format

The training data should be a JSON file with the following structure:

```json
{
    "examples": [
        {
            "url": "https://example.com/product/123",
            "selectors": {
                "name": ".product-name",
                "price": "#price",
                "description": ".description"
            },
            "data": {
                "name": "Example Product",
                "price": 99.99,
                "currency": "USD"
            }
        }
    ]
}
```

