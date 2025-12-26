# Product Scraper 🛒🤖

**Product Scraper** is a Python library designed to extract structured product data from e-commerce websites. Instead of relying on fragile CSS selectors or hard-coded XPaths, it **learns how products look** on a page—mimicking how a human identifies a title, price, or image.

By combining machine learning, visual cues, and DOM structure, Product Scraper adapts to different site layouts and remains resilient even when the HTML structure changes.

---

## ✨ What Makes It Different?

Traditional scrapers break the moment a website tweaks its layout. **Product Scraper adapts.**

It analyzes:
* **Visual Rendering:** Font size, font weight, and element position.
* **Text Patterns:** Currencies, keywords (e.g., "add to cart"), and price formats.
* **DOM Structure:** The spatial relationships between elements.

This multi-faceted approach allows the library to generalize across different pages and withstand minor layout updates.

---

## 🚀 Features

* **🧠 Machine Learning Driven:** Uses a **Random Forest classifier** to identify product elements based on visual, textual, and structural features.
* **🖱️ Interactive Training:** Includes a **browser-based UI** (powered by Playwright). Simply open a page, click on the elements you want (titles, prices, images), and the system learns from your input—no complex XPath coding required.
* **🧩 Automatic Product Grouping:** Detected elements are automatically grouped into coherent product objects using **spatial clustering**, ensuring the correct title is associated with the correct price and image.
* **🛡️ Resilient Scraping:** By learning patterns rather than strict paths, the scraper is far less brittle than traditional rule-based solutions.

---

## 📦 Installation

### Requirements
* Python **3.9+**
* pip
* Playwright (Chromium browser)

### Windows (PowerShell)
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1

pip install product-scraper
playwright install chromium
```

### Linux / macOS
```bash
python3 -m venv .venv
source .venv/bin/activate

pip install product-scraper
playwright install chromium
```

## 🛠️ Usage Guide

### 1. Interactive Training
To start, you need to teach the model what your target data looks like. You define the categories you want to extract (e.g., image, title, price) and provide a list of example URLs.

```python
from product_scraper import ProductScraper

# Define the data points you want to extract
CATEGORIES = ["image", "title", "price"]

# List of URLs to train on (more variety = better predictions)
WEBSITES = [
    "https://www.morganbooks.eu/",
    # Add more URLs here...
]

# Initialize the scraper
scraper = ProductScraper(categories=CATEGORIES, websites_urls=WEBSITES)

# Launch the interactive training UI
scraper.create_training_data()
```

### 2. Labeling Data
Running the code above will launch a Chromium browser window with the Product Scraper UI overlay.

![example of selecting xpaths of elements](src/product_scraper/assets/example1.png)

1. Select a Category: Choose a category (e.g., "title") from the UI.

2. Click Elements: Click on the corresponding elements on the webpage to label them.


* Tip: Use the "Class Select" (orange button) to speed up the process. Once you select one element, clicking this will automatically select all other visible elements with the same CSS class.

3) Next Category: Proceed to the next category (e.g., "price") and repeat.

4) Finish: Click the green Finish button when done.

![example of finishing up the selection](src/product_scraper/assets/example2.png)

The scraper will save your labeled data to `product_scraper_data/selectors.yaml` and `product_scraper_data/training_data.csv` in your current directory.


### 3. Selectors File (selectors.yaml)
This file stores the XPaths of the elements you manually selected during training. It serves as the ground truth for the model.

```yaml
https://www.morganbooks.eu/:
  image:
  - /html[1]/body[1]/.../div[1]/img[1]
  - /html[1]/body[1]/.../div[1]/img[1]
  ...
  price:
  - /html[1]/body[1]/.../footer[1]/div[2]/h3[1]
  ...
  title:
  - /html[1]/body[1]/.../article[2]/a[1]/h2[1]
  ...
```

### 4. Load / Save ProductScraper
Use the following syntax
```python
    try:
        product_scraper = ProductScraper(categories=CATEGORIES, websites_urls=WEBSITES, save_dir="./product_scraper_data") #  save_dir="./product_scraper_data" is default
        product_scraper.load_selectors() # load selector from ./product_scraper_data/selectors.yaml
        # or use ProductScraper.load() to load the full object after product_scraper.save()
    except FileNotFoundError:
        product_scraper = ProductScraper(categories=CATEGORIES, websites_urls=WEBSITES)
    ... 
    
    
    product_scraper.save()
```


### 5. Training and Prediction
Once you have collected enough training data (ideally from 50+ diverse product pages for robust generalization), you can train the model and start scraping new pages automatically.
```python
from product_scraper import ProductScraper

CATEGORIES = ["title", "image", "price"] # add more categories if needed
WEBSITES = [
    "https://example-ecommerce-site.com/products",
    "https://another-shop.com/collection",
    # ...
]

# Load existing data or initialize a new session
scraper = ProductScraper(categories=CATEGORIES, websites_urls=WEBSITES)

# 1. Create the training data
scraper.create_training_data()

# 2. Train the model using the collected data
scraper.train_model()

# 3. Predict and extract data from the websites
results = scraper.predict(["website-to-predcit-product-selectors-from.com", ...])

# 4. Save the results and the trained model state
scraper.save() 

# Print extracted data
for url, products in results.items():
    print(f"\n--- Found {len(products)} products on {url} ---")

    for i, product in enumerate(products):
            print(f"Product #{i + 1}")
            for category, data in product.items():
                print(f"{category}: ({data['xpath']})"

# example output
/*

*/

```

### Training Data (training_data.csv)
This CSV file contains the extracted features (visual, text, DOM) for every labeled element. This is the dataset used to train the Random Forest model.

## Developement
Use the src/example.py file for testing and check the example_scraper_data for debugging / inspiration
```python
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 ./src/example.py 
```

## 🧪 Testing
To run the test suite and check code coverage:

```bash
pip install -r requirements.txt
PYTHONPATH=src pytest src --cov=src --cov-report=term-missing
```

## 🤝 Contributing
Contributions are welcome! Please follow these steps:

1. Fork the repository.

2. Create a feature branch (git checkout -b feature/AmazingFeature).

3. Commit your changes (git commit -m 'Add some AmazingFeature').

4. Push to the branch (git push origin feature/AmazingFeature).

5. Open a Pull Request.

## 📄 License
Distributed under the MIT License. See LICENSE for more information.