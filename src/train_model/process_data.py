"""Data processing utilities for HTML element feature extraction."""

import re
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

import lxml.html
import pandas as pd
import textstat
import yaml
from sklearn.preprocessing import LabelEncoder

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
PRICE_REGEX = re.compile(r'[$€£¥₹]|(\d+[.,]\d{2})')
CURRENCY_SYMBOLS = re.compile(r'[$€£¥₹]')

def normalize_tag(tag: Any) -> str:
    """Normalize HTML tag to string."""
    if not isinstance(tag, str):
        return 'unknown'
    return str(tag).lower()

def _extract_element_features(
    element: lxml.html.HtmlElement, 
    category: str = 'other'
) -> Dict[str, Any]:
    """
    Extract comprehensive features from a single HTML element.
    """
    try:
        text = element.text_content().strip()
        parent = element.getparent()
        
        # Structural / DOM Features
        features = {
            'Category': category,
            'tag': normalize_tag(element.tag),
            'parent_tag': normalize_tag(parent.tag) if parent is not None else 'root',
            'num_children': len(element),
            'num_siblings': len(parent) - 1 if parent is not None else 0,
            'dom_depth': len(list(element.iterancestors())),
        }

        # Attribute Semantic Features
        features['class_str'] = " ".join(element.get('class', '').split()) if element.get('class') else ""
        features['id_str'] = element.get('id', '')
        
        # Text Content Features
        features['text_len'] = len(text)
        features['text_word_count'] = len(text.split())
        features['text_digit_count'] = sum(c.isdigit() for c in text)
        features['has_currency_symbol'] = 1 if CURRENCY_SYMBOLS.search(text) else 0
        features['is_price_format'] = 1 if PRICE_REGEX.search(text) else 0
        
        # Density & Readability
        num_descendants = len(list(element.iterdescendants())) + 1
        features['text_density'] = len(text) / num_descendants
        
        try:
            features['reading_ease'] = textstat.flesch_reading_ease(text) if text else 0
        except Exception:
            features['reading_ease'] = 0

        # Hyperlink context
        features['has_href'] = 1 if element.get('href') else 0
        features['is_image'] = 1 if element.tag == 'img' else 0

        return features

    except Exception as e:
        # Suppress warnings for expected non-element issues if any slip through
        return {}


def html_to_dataframe(
    html_content: str, 
    selectors: Optional[Dict[str, List[str]]] = None
) -> pd.DataFrame:
    """
    Parse HTML and extract features into a DataFrame.
    """
    try:
        tree = lxml.html.fromstring(html_content)
    except Exception as e:
        logger.error(f"Failed to parse HTML: {e}")
        return pd.DataFrame()

    all_data = []
    labeled_elements = set()

    # 1. Extract Positive Labels (if selectors provided)
    if selectors:
        for category, css_selectors in selectors.items():
            for selector in css_selectors:
                try:
                    elements = tree.cssselect(selector)
                    for elem in elements:
                        if elem not in labeled_elements:
                            data = _extract_element_features(elem, category=category)
                            if data:
                                all_data.append(data)
                                labeled_elements.add(elem)
                except Exception as e:
                    logger.warning(f"Invalid selector {selector}: {e}")

    # 2. Extract Negative Samples (Everything else is 'other')
    for elem in tree.iter():
        # FIX: Skip comments (HtmlComment) and ProcessingInstructions
        if not isinstance(elem.tag, str):
            continue

        if elem in labeled_elements:
            continue
        
        # Optimization: Skip empty structural tags that have no attributes
        try:
            if not elem.text_content().strip() and not elem.attrib:
                continue
        except Exception:
            continue
            
        # Treat as 'other'
        data = _extract_element_features(elem, category='other')
        if data:
            all_data.append(data)

    df = pd.DataFrame(all_data)
    
    # Fill NAs for text columns with empty strings
    text_cols = ['class_str', 'id_str']
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna("")
            
    return df.fillna(0)

def selector_data_to_csv(data_domain_dir: Path) -> None:
    """Convert YAML/HTML pair to CSV."""
    yaml_path = data_domain_dir / 'selectors.yaml'
    html_path = data_domain_dir / 'page.html'
    csv_path = data_domain_dir / 'data.csv'

    if not yaml_path.exists() or not html_path.exists():
        return

    try:
        with open(yaml_path, 'r') as f:
            selectors = yaml.safe_load(f)
        with open(html_path, 'r') as f:
            html = f.read()

        df = html_to_dataframe(html, selectors)
        if not df.empty:
            df.to_csv(csv_path, index=False)
            logger.info(f"Saved {len(df)} rows to {csv_path}")
    except Exception as e:
        logger.error(f"Failed to process {data_domain_dir}: {e}")

def data_to_csv(project_root: Path = Path.cwd()) -> None:
    """Batch process all data."""
    data_dir = project_root / 'src' / 'data'
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return

    for site_dir in data_dir.iterdir():
        if site_dir.is_dir():
            selector_data_to_csv(site_dir)

if __name__ == "__main__":
    data_to_csv()