"""Data processing utilities for HTML element feature extraction."""

import regex as re
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional


import lxml.html
import pandas as pd
import textstat
import yaml

UNWANTED_TAGS = ['script', 'style', 'meta', 'link', 'noscript', 'iframe', 'head', 'input']

OTHER_CATEGORY = 'other'

CURRENCY_HINTS = r'(?:\p{Sc}|(?:USD|EUR|GBP|JPY|CNY|CZK|Kč|kr|zł|Rs)\b)'

NUMBER_PATTERN = r'(?:\d{1,3}(?:[., ]\d{3})+|\d+)(?:[.,]\d{1,2})?'

PRICE_REGEX = re.compile(
    fr'(?:{CURRENCY_HINTS}\s*{NUMBER_PATTERN}|{NUMBER_PATTERN}\s*{CURRENCY_HINTS})',
    re.UNICODE | re.IGNORECASE
)


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
def normalize_tag(tag: Any) -> str:
    """Normalize HTML tag to string."""
    if not isinstance(tag, str):
        return 'unknown'
    return str(tag).lower()

def _extract_element_features(
    element: lxml.html.HtmlElement, 
    category: str = OTHER_CATEGORY
) -> Dict[str, Any]:
    """
    Extract comprehensive features from a single HTML element.
    """
    try:
        text = element.text_content().strip()
        parent = element.getparent()
        # Get grandparent and great-grandparent
        gparent = parent.getparent() if parent is not None else None
        ggparent = gparent.getparent() if gparent is not None else None
        
        # Structural
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

        # Attribute Semantic Features
        features['class_str'] = " ".join(element.get('class', '').split()) if element.get('class') else ""
        features['id_str'] = element.get('id', '')
        
        # Text Content Features
        features['text_len'] = len(text)
        features['text_word_count'] = len(text.split())
        features['text_digit_count'] = sum(c.isdigit() for c in text)
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
    

def get_main_html_content_tag(html_content: str) -> Optional[lxml.html.HtmlElement]:
    """
    Identify the specific element that wraps the main content in the HTML document.
    Calculates score based on:
        (total_text_length + total_img_tags * IMG_IMPORTANCE) / total_tags
    """
    IMG_IMPORTANCE = 50
    
    if not html_content:
        return None

    try:
        tree = lxml.html.fromstring(html_content)
    except Exception as e:
        logger.error(f"Failed to parse HTML: {e}")
        return None

    best_elem = None
    best_score = -1.0

    # Iterate over every element in the tree
    for elem in tree.iter():
        # Skip comments and processing instructions
        if not isinstance(elem.tag, str):
            continue
        
        # Calculate Text Length
        text_content = elem.text_content()
        if not text_content:
            continue
        text_len = len(text_content.strip())

        # Efficiently count images
        img_count = sum(1 for _ in elem.iter('img'))

        # Count Total Tags (Descendants + Self)
        total_tags = sum(1 for _ in elem.iterdescendants()) + 1

        # Calculate Score
        score = (text_len + (img_count * IMG_IMPORTANCE)) / total_tags

        # Update Best Candidate
        if score > best_score:
            best_score = score
            best_elem = elem

    return best_elem


def html_to_dataframe(
    html_content: str, 
    selectors: Optional[Dict[str, List[str]]] = None
) -> pd.DataFrame:
    """
    Parse HTML and extract features into a DataFrame.
    Uses get_main_html_content_tag to narrow scope to relevant content.
    """
    main_content = get_main_html_content_tag(html_content)
    
    try:
        root = lxml.html.fromstring(html_content)
    except Exception:
        return pd.DataFrame()
    
    if main_content is None:
        main_content = root

    all_data = []
    labeled_elements = set()

    # Extract Positive Labels (scoped to root_node)
    if selectors:
        for category, css_selectors in selectors.items():
            for selector in css_selectors:
                try:
                    # cssselect will search within the full document root
                    elements = root.cssselect(selector)
                    for elem in elements:
                        if elem not in labeled_elements:
                            data = _extract_element_features(elem, category=category)
                            if data:
                                all_data.append(data)
                                labeled_elements.add(elem)
                except Exception as e:
                    logger.warning(f"Invalid selector {selector}: {e}")

    # Extract Negative Samples (Only from main_content to avoid noise)
    for elem in main_content.iter():
        # Skip comments
        if not isinstance(elem.tag, str) or normalize_tag(elem.tag) in UNWANTED_TAGS:
            continue

        if elem in labeled_elements:
            continue
        
        # Skip empty structural tags that have no attributes
        try:
            if not elem.text_content().strip() and not elem.attrib:
                continue
        except Exception:
            continue
            
        # Treat as 'other'
        data = _extract_element_features(elem, category=OTHER_CATEGORY)
        if data:
            all_data.append(data)

    df = pd.DataFrame(all_data)
    
    if df.empty:
        return df

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
    sys.path.insert(0, str(Path(__file__).parent.parent))

    data_to_csv()