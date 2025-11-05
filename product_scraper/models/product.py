"""
Product data model
"""
from dataclasses import dataclass
from typing import Optional, Dict, List

@dataclass
class ProductData:
    """
    Data class for storing product information
    """
    name: str
    price: float
    currency: str
    description: Optional[str] = None
    brand: Optional[str] = None
    category: Optional[str] = None
    images: Optional[List[str]] = None
    specifications: Optional[Dict[str, str]] = None
    url: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary representation"""
        return {k: v for k, v in self.__dict__.items() if v is not None}