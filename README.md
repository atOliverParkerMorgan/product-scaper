# Product Scraper

A Python package for extracting product information from e-commerce websites using machine learning.

## Installation

```bash
pip install product-scraper
```

## Basic Usage

```python
import productscaper as ps

# Train the model with your data
ps.train('training_data.json')

# Extract from a single URL
product = ps.extract('https://example.com/product/123')
print(f"{product.name}: {product.price} {product.currency}")

# Or extract from multiple URLs
products = ps.extractAll('urls.txt')
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

## Features

- Machine learning based extraction
- Support for multiple e-commerce platforms
- Bulk extraction capability
- Customizable through training data
- Clean and simple API

## Project Structure

```
product_scraper/
├── __init__.py         # Package initialization
├── core.py            # Core functionality
├── models/            # Data models
├── training/          # Training functionality
└── extractors/        # Extraction logic
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.