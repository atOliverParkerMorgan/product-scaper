"""
Example usage of the product scraper package
"""
import productscaper as ps

# Train the model with example data
ps.train('training_data.json')

# Extract from a single URL
product = ps.extract('https://example.com/product/123')
print(f"Found product: {product.name} - {product.price} {product.currency}")

# Extract from multiple URLs
urls_file = 'urls.txt'
products = ps.extractAll(urls_file)
for product in products:
    print(f"Found: {product.name} - {product.price} {product.currency}")