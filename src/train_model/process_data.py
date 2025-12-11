"""Data processing utilities for HTML element feature extraction."""

import regex as re
import logging
import sys
from utils import normalize_tag
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import lxml.html
import pandas as pd
import textstat
import yaml

UNWANTED_TAGS = {'script', 'style', 'noscript', 'form', 'iframe', 'header', 'footer', 'nav'}

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
        features['has_currency_symbol'] = 1 if re.search(CURRENCY_HINTS, text) else 0
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

    except Exception:
        # Suppress warnings for expected non-element issues if any slip through
        return {}
    


def get_main_html_content_tag(
    html_content,
    IMG_IMPORTANCE=150,
    MIN_TAG_TEXT_LENGTH=0,
    MIN_IMAGE_COUNT=0,
    LINK_DENSITY_WEIGHT=0.4,
    DEPTH_SCORE_COEFFICIENT=30,
    PARENT_IMPROVEMENT_THRESHOLD=5
    ) -> Optional[lxml.html.HtmlElement]:

    if not html_content:
        return None

    try:
        tree = lxml.html.fromstring(html_content)
    except Exception as e:
        logger.error(f"Failed to parse HTML: {e}")
        return None

    # Track: [Best Element, Best Score]
    best_candidate = [None, -1.0]

    def process_node(elem: lxml.html.HtmlElement, current_depth: int, is_in_link: bool) -> Tuple[int, int, int]:
        """
        Returns: (total_text_len, total_img_count, total_link_text_len)
        """
        tag = normalize_tag(elem.tag)
        
        # 1. Invalid Tag Check
        if not isinstance(elem.tag, str) or tag in UNWANTED_TAGS:
            return (0, 0, 0)

        # 2. Link Status
        current_is_link = is_in_link or (tag == 'a')

        # 3. Local Stats
        own_text = (elem.text or "").strip() + (elem.tail or "").strip()
        local_text_len = len(own_text)
        local_img_count = 1 if tag == 'img' else 0
        local_link_text_len = local_text_len if current_is_link else 0

        # 4. Recursion
        child_text_len = 0
        child_img_count = 0
        child_link_text_len = 0

        for child in elem:
            c_text, c_img, c_link_len = process_node(child, current_depth + 1, current_is_link)
            child_text_len += c_text
            child_img_count += c_img
            child_link_text_len += c_link_len

        # 5. Aggregation
        total_text = local_text_len + child_text_len
        total_imgs = local_img_count + child_img_count
        total_link_text = local_link_text_len + child_link_text_len

        # 6. Scoring
        if total_text == 0:
            link_density = 1.0 
        else:
            link_density = total_link_text / total_text

        base_score = total_text + (total_imgs * IMG_IMPORTANCE)
        
        # --- Gallery Exception ---
        # If an element has many images (e.g., > 5), it is likely a product grid or gallery.
        # We should IGNORE or reduce link density penalty for these, because product grids are usually 100% links.
        if total_imgs > 5:
            effective_link_weight = 0.1  # Very low penalty for galleries
        else:
            effective_link_weight = LINK_DENSITY_WEIGHT

        penalty_factor = 1.0 - (link_density * effective_link_weight)
        
        # --- Depth Bonus (Additive) ---
        # We want to favor the specific container (depth 5) over the body (depth 1)
        depth_score = current_depth * DEPTH_SCORE_COEFFICIENT
        
        final_score = (base_score * max(0.01, penalty_factor)) + depth_score

        # --- Hard Body Penalty ---
        # The body tag accumulates everything. Unless the page is very flat, 
        # we almost never want to return 'body' as the specific main content.
        if tag == 'body' or tag == 'html':
            final_score *= 0.1 # Nuke the body score

        # 7. Update Candidate
        if total_text > MIN_TAG_TEXT_LENGTH or total_imgs >= MIN_IMAGE_COUNT:
            current_best_score = best_candidate[1]
            
            if current_best_score == -1.0:
                best_candidate[0] = elem
                best_candidate[1] = final_score
            
            # Parent vs Child Threshold Check
            elif final_score > (current_best_score * PARENT_IMPROVEMENT_THRESHOLD):
                best_candidate[0] = elem
                best_candidate[1] = final_score

        return (total_text, total_imgs, total_link_text)

    process_node(tree, 0, False)
    
    # Fallback: if nothing was selected or only head was selected, try to find body, then html, then tree
    if best_candidate[0] is None or normalize_tag(best_candidate[0].tag) == 'head':
        body = tree.find('.//body')
        if body is not None:
            return body
        html = tree.find('.//html')
        if html is not None:
            return html
        return tree
    
    return best_candidate[0]


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
    text_cols = ['class_str', 'id_str', 'tag', 'parent_tag', 'gparent_tag', 'ggparent_tag']
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