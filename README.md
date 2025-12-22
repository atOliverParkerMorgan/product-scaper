# Product Scraper

Product Scraper is a friendly but powerful Python library for pulling structured product data out of e-commerce websites.  
Instead of relying on fragile CSS selectors or hard-coded XPaths, it **learns how products look** on a page — the way a human would spot a title, a price, or an image at a glance.

By combining machine learning, visual cues, and DOM structure, Product Scraper adapts to different site layouts and keeps working even when the HTML changes a bit.

---

## ✨ What makes it different?

Most scrapers break the moment a website tweaks its layout. Product Scraper doesn’t.

It watches how elements are rendered (font size, weight, position), reads the text (currency symbols, keywords), and understands where elements sit in the DOM — then learns patterns that generalize across pages.

---

## 🚀 Features

### 🧠 Machine-learning driven
Uses a **Random Forest classifier** to recognize product elements based on:
- Visual hints (font size, boldness)
- Text patterns (prices, currencies, keywords)
- Structural context in the DOM

### 🖱️ Interactive training
Comes with a **browser-based UI** (powered by Playwright).  
You simply open a page, click on prices, titles, or images, and label them — no XPath gymnastics required.

### 🧩 Automatic product grouping
Detected elements are automatically grouped into products using **spatial clustering**, so titles, prices, and images end up together where they belong.

### 🛡️ More resilient scraping
Because it learns patterns instead of exact paths, the scraper survives small layout changes that would break traditional rule-based scrapers.

---

## 📦 Installation

Clone the repository and install the dependencies.  
Playwright is used for rendering pages and interacting with them.

```bash
pip install -r requirements.txt
playwright install chromium
