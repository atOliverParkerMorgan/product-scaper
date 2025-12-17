"""Data processing utilities for HTML element feature extraction."""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import lxml.html
import pandas as pd
import yaml
from utils.features import (
    extract_element_features,
    UNWANTED_TAGS,
    OTHER_CATEGORY
)
import random
from train_model.process_data import RANDOM_SEED
from utils.utils import normalize_tag

OTHER_TO_CATEGORY_RATIO = 10


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        
        # --- Gallery handling ---
        # Elements with many images are often product grids. Reduce the
        # link-density penalty for these so galleries aren't unfairly down-weighted.
        if total_imgs > 5:
            effective_link_weight = 0.1  # lower penalty for gallery-like containers
        else:
            effective_link_weight = LINK_DENSITY_WEIGHT

        penalty_factor = 1.0 - (link_density * effective_link_weight)
        
        # --- Depth bonus (additive) ---
        # Favor deeper, more specific containers over very shallow ones.
        depth_score = current_depth * DEPTH_SCORE_COEFFICIENT
        
        final_score = (base_score * max(0.01, penalty_factor)) + depth_score

        # --- Reduce score for document-level tags ---
        # Deprioritize 'body' and 'html' so more specific containers are preferred.
        if tag == 'body' or tag == 'html':
            final_score *= 0.1  # reduce body/html score

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

    # Process the tree to find the best content container
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
    selectors: Optional[Dict[str, List[str]]] = None,
    url: str = None
) -> pd.DataFrame:
    """
    Parse HTML and extract features into a DataFrame.
    Uses get_main_html_content_tag to narrow scope to relevant content.
    
    Args:
        html_content: HTML string to parse.
        selectors: Dictionary mapping categories to XPath selectors.
        url: Source URL to track origin of training data (for removal if needed).
    """
    main_content = get_main_html_content_tag(html_content)
    
    try:
        root = lxml.html.fromstring(html_content)
    except Exception:
        return pd.DataFrame()
    
    if main_content is None:
        main_content = root

    labeled_elements = set()
    positive_data = []

    # Extract Positive Labels using XPath
    if selectors:
        for category, xpath_selectors in selectors.items():
            for xpath in xpath_selectors:
                try:
                    # Use XPath to find elements
                    elements = root.xpath(xpath)
                    
                    for elem in elements:
                        # Make sure it's an Element, not text or comment
                        if not isinstance(elem, lxml.html.HtmlElement):
                            logger.debug(f"Skipping non-element: {type(elem)}")
                            continue
                            
                        if elem not in labeled_elements:
                            data = extract_element_features(elem, category=category)
                            if data:
                                logger.info(f"Added element with category='{category}', tag='{data.get('tag')}'")
                                positive_data.append(data)
                                labeled_elements.add(elem)
                except Exception as e:
                    logger.warning(f"Invalid XPath '{xpath}': {e}")

    # Extract Negative Samples (Only from main_content to avoid noise)
    negative_data = []
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
        data = extract_element_features(elem, category=OTHER_CATEGORY)
        if data:
            negative_data.append(data)

    if positive_data:
        max_negatives = len(positive_data) * OTHER_TO_CATEGORY_RATIO
        
        if len(negative_data) > max_negatives:
            # Randomly sample to reduce the count
            random.seed(RANDOM_SEED) # Ensure reproducibility
            negative_data = random.sample(negative_data, max_negatives)

    all_data = positive_data + negative_data

    df = pd.DataFrame(all_data)
    
    if df.empty:
        return df

    # Add source URL column for tracking and removal
    if url is not None:
        df['SourceURL'] = url

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