"""Unified feature definitions and extraction for HTML element analysis."""

import regex as re
import logging
import lxml.html
import textstat
from typing import Dict, Any
from utils.utils import normalize_tag

# Configure logging
logger = logging.getLogger(__name__)

# Constants for feature extraction
UNWANTED_TAGS = {'script', 'style', 'noscript', 'form', 'iframe', 'header', 'footer', 'nav'}
OTHER_CATEGORY = 'other'
# Comprehensive currency list: symbols (\p{Sc}) + major currency codes and their variations
CURRENCY_HINTS = r'(?:\p{Sc}|(?:USD|EUR|GBP|JPY|CNY|CZK|CHF|Chf|AUD|CAD|NZD|SEK|NOK|DKK|PLN|HUF|RON|BGN|HRK|RSD|TRY|INR|BRL|MXN|ARS|ZAR|KRW|THB|MYR|SGD|IDR|PHP|VND|Kč|kr|zł|Rs|Ft|lei|kn|din|руб|₹|R\$|R)\b)'
NUMBER_PATTERN = r'(?:\d{1,3}(?:[., ]\d{3})+|\d+)(?:[.,]\d{1,2})?'
PRICE_REGEX = re.compile(
    fr'(?:{CURRENCY_HINTS}\s*{NUMBER_PATTERN}|{NUMBER_PATTERN}\s*{CURRENCY_HINTS})',
    re.UNICODE | re.IGNORECASE
)

# Feature column definitions - must match what train_model expects
NUMERIC_FEATURES = [
    'num_children',
    'num_siblings',
    'dom_depth',
    'text_len',
    'text_word_count',
    'text_digit_count',
    'text_density',
    'reading_ease',
    'has_currency_symbol',
    'is_price_format',
    'has_href',
    'is_image',
    'has_src',
    'has_alt',
    'alt_len',
    'has_dimensions',
    'parent_is_link',
    'sibling_image_count'
]

CATEGORICAL_FEATURES = [
    'tag',
    'parent_tag',
    'gparent_tag',
    'ggparent_tag'
]

TEXT_FEATURES = [
    'class_str',
    'id_str'
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + TEXT_FEATURES


def extract_element_features(
    element: lxml.html.HtmlElement, 
    category: str = OTHER_CATEGORY
) -> Dict[str, Any]:
    """
    Extract comprehensive features from a single HTML element.
    
    This function extracts all features used by the ML model:
    - Structural features (tag hierarchy, DOM position)
    - Attribute semantic features (class, id)
    - Text content features (length, word count, patterns)
    - Density and readability metrics
    - Image-specific features
    - Parent context features
    
    Args:
        element: lxml HTML element to extract features from
        category: Category label for this element (default: OTHER_CATEGORY)
        
    Returns:
        Dictionary containing all extracted features with consistent naming
    """
    try:
        text = element.text_content().strip()
        parent = element.getparent()
        # Get grandparent and great-grandparent
        gparent = parent.getparent() if parent is not None else None
        ggparent = gparent.getparent() if gparent is not None else None
        
        # Structural features
        features = {
            'Category': category,
            'tag': normalize_tag(element.tag),
            'parent_tag': normalize_tag(parent.tag) if parent is not None else 'root',
            'gparent_tag': normalize_tag(gparent.tag) if gparent is not None else 'root',
            'ggparent_tag': normalize_tag(ggparent.tag) if ggparent is not None else 'root',
            'num_children': len(element),
            'num_siblings': len(parent) - 1 if parent is not None else 0,
            'dom_depth': len(list(element.iterancestors())),
        }

        # Attribute semantic features
        features['class_str'] = " ".join(element.get('class', '').split()) if element.get('class') else ""
        features['id_str'] = element.get('id', '')
        
        # Text content features
        features['text_len'] = len(text)
        features['text_word_count'] = len(text.split())
        features['text_digit_count'] = sum(c.isdigit() for c in text)
        features['has_currency_symbol'] = 1 if re.search(CURRENCY_HINTS, text) else 0
        features['is_price_format'] = 1 if PRICE_REGEX.search(text) else 0
        
        # Density & readability
        num_descendants = len(list(element.iterdescendants())) + 1
        features['text_density'] = len(text) / num_descendants
        
        try:
            features['reading_ease'] = textstat.flesch_reading_ease(text) if text else 0
        except Exception:
            features['reading_ease'] = 0

        # Hyperlink context
        features['has_href'] = 1 if element.get('href') else 0
        features['is_image'] = 1 if element.tag == 'img' else 0
        
        # Image-specific features
        features['has_src'] = 1 if element.get('src') else 0
        features['has_alt'] = 1 if element.get('alt') else 0
        features['alt_len'] = len(element.get('alt', ''))
        
        # Image dimension hints (width/height attributes)
        try:
            width = element.get('width', '')
            height = element.get('height', '')
            features['has_dimensions'] = 1 if (width or height) else 0
        except Exception:
            features['has_dimensions'] = 0
        
        # Parent context for images (images in links are often product images)
        features['parent_is_link'] = 1 if (parent is not None and normalize_tag(parent.tag) == 'a') else 0
        
        # Count nearby images (siblings that are images)
        if parent is not None:
            sibling_images = sum(1 for sibling in parent if normalize_tag(getattr(sibling, 'tag', '')) == 'img')
            features['sibling_image_count'] = sibling_images
        else:
            features['sibling_image_count'] = 0

        return features

    except Exception as e:
        # Log unexpected errors but don't crash
        logger.debug(f"Error extracting features from element: {e}")
        return {}


def get_feature_columns():
    """
    Get the complete list of feature columns in the correct order.
    
    Returns:
        Dictionary with keys 'numeric', 'categorical', 'text', and 'all'
    """
    return {
        'numeric': NUMERIC_FEATURES,
        'categorical': CATEGORICAL_FEATURES,
        'text': TEXT_FEATURES,
        'all': ALL_FEATURES
    }


def validate_features(df) -> bool:
    """
    Validate that a DataFrame contains all required features.
    
    Args:
        df: pandas DataFrame to validate
        
    Returns:
        True if all required features are present, False otherwise
    """
    if df.empty:
        return False
        
    missing_features = []
    for feature in ALL_FEATURES:
        if feature not in df.columns:
            missing_features.append(feature)
    
    if missing_features:
        logger.warning(f"Missing features in DataFrame: {missing_features}")
        return False
    
    return True
