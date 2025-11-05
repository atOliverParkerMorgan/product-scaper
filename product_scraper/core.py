"""
Core functionality for the product scraper
"""
from typing import Union, List
from pathlib import Path
from .models.product import ProductData
from .training.trainer import ModelTrainer
from .extractors.extractor import ProductExtractor

def train(training_data: Union[str, Path]) -> None:
    """
    Train the model using provided training data.
    
    Args:
        training_data: Path to JSON file containing training data
    """
    trainer = ModelTrainer()
    trainer.train(training_data)

def extract(url: str) -> ProductData:
    """
    Extract product information from a single URL.
    
    Args:
        url: URL of the product page
        
    Returns:
        ProductData object containing extracted information
    """
    extractor = ProductExtractor()
    return extractor.extract_single(url)

def extractAll(url_file: Union[str, Path]) -> List[ProductData]:
    """
    Extract product information from multiple URLs listed in a file.
    
    Args:
        url_file: Path to file containing URLs (one per line)
        
    Returns:
        List of ProductData objects
    """
    extractor = ProductExtractor()
    return extractor.extract_bulk(url_file)