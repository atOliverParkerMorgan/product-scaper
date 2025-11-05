"""
Product Scraper - A machine learning based product information extractor
"""
from .core import train, extract, extractAll
from .models import ProductData

__version__ = "0.1.0"
__all__ = ["train", "extract", "extractAll", "ProductData"]