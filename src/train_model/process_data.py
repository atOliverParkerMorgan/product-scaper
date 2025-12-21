"""Data processing utilities for HTML element feature extraction."""

import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import lxml.html
import pandas as pd
import yaml

# Assuming these are available from your previous file
from utils.features import OTHER_CATEGORY, UNWANTED_TAGS, extract_element_features, process_page_features
from utils.utils import normalize_tag

RANDOM_SEED = 42
OTHER_TO_CATEGORY_RATIO = 10 # not too high to avoid the model being overwhelmed by negatives


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)




def get_main_html_content_tag(
    html_content: str,
    IMG_IMPORTANCE: int = 150,
    MIN_TAG_TEXT_LENGTH: int = 0,
    MIN_IMAGE_COUNT: int = 0,
    LINK_DENSITY_WEIGHT: float = 0.4,
    DEPTH_SCORE_COEFFICIENT: int = 30,
    PARENT_IMPROVEMENT_THRESHOLD: int = 5
) -> Optional[lxml.html.HtmlElement]:
    """
    Identify the main content tag in HTML by scoring elements based on text, images, and structure.
    """
    if not html_content:
        return None
    try:
        tree = lxml.html.fromstring(html_content)
    except Exception as e:
        logger.error(f"Failed to parse HTML: {e}")
        return None

    best_candidate = [None, -1.0]

    def process_node(elem: lxml.html.HtmlElement, current_depth: int, is_in_link: bool) -> Tuple[int, int, int]:
        tag = normalize_tag(elem.tag)
        if not isinstance(elem.tag, str) or tag in UNWANTED_TAGS:
            return (0, 0, 0)

        current_is_link = is_in_link or (tag == 'a')
        own_text = (elem.text or "").strip() + (elem.tail or "").strip()
        local_text_len = len(own_text)
        local_img_count = 1 if tag == 'img' else 0
        local_link_text_len = local_text_len if current_is_link else 0

        child_text_len = 0
        child_img_count = 0
        child_link_text_len = 0

        for child in elem:
            c_text, c_img, c_link_len = process_node(child, current_depth + 1, current_is_link)
            child_text_len += c_text
            child_img_count += c_img
            child_link_text_len += c_link_len

        total_text = local_text_len + child_text_len
        total_imgs = local_img_count + child_img_count
        total_link_text = local_link_text_len + child_link_text_len

        if total_text == 0:
            link_density = 1.0
        else:
            link_density = total_link_text / total_text

        base_score = total_text + (total_imgs * IMG_IMPORTANCE)

        if total_imgs > 5:
            effective_link_weight = 0.1
        else:
            effective_link_weight = LINK_DENSITY_WEIGHT

        penalty_factor = 1.0 - (link_density * effective_link_weight)
        depth_score = current_depth * DEPTH_SCORE_COEFFICIENT
        final_score = (base_score * max(0.01, penalty_factor)) + depth_score

        if tag == 'body' or tag == 'html':
            final_score *= 0.1

        if total_text > MIN_TAG_TEXT_LENGTH or total_imgs >= MIN_IMAGE_COUNT:
            current_best_score = best_candidate[1]
            if current_best_score == -1.0:
                best_candidate[0] = elem
                best_candidate[1] = final_score
            elif final_score > (current_best_score * PARENT_IMPROVEMENT_THRESHOLD):
                best_candidate[0] = elem
                best_candidate[1] = final_score

        return (total_text, total_imgs, total_link_text)

    process_node(tree, 0, False)

    if best_candidate[0] is None or normalize_tag(best_candidate[0].tag) == 'head':
        body = tree.find('.//body')
        if body is not None: return body
        html = tree.find('.//html')
        if html is not None: return html
        return tree

    return best_candidate[0]


def html_to_dataframe(
    html_content: str,
    selectors: Optional[Dict[str, List[str]]] = None,
    url: Optional[str] = None
) -> pd.DataFrame:
    """
    Parse HTML and extract features into a DataFrame.
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

    # --- 1. Extract Positive Labels ---
    if selectors:
        for category, xpath_selectors in selectors.items():
            for xpath in xpath_selectors:
                try:
                    elements = root.xpath(xpath)
                    for elem in elements:
                        if not isinstance(elem, lxml.html.HtmlElement):
                            continue

                        if elem not in labeled_elements:
                            # Pass selectors here so positive elements get distance context
                            data = extract_element_features(elem, category=category, selectors=selectors)
                            if data:
                                positive_data.append(data)
                                labeled_elements.add(elem)
                except Exception as e:
                    logger.warning(f"Invalid XPath '{xpath}': {e}")

    # --- 2. Extract Negative Samples ---
    negative_data = []
    for elem in main_content.iter():
        if not isinstance(elem.tag, str) or normalize_tag(elem.tag) in UNWANTED_TAGS:
            continue

        if elem in labeled_elements:
            continue

        # Skip empty structural tags
        try:
            if not elem.text_content().strip() and not elem.attrib:
                continue
        except Exception:
            continue

        # Pass selectors to negative samples too!
        data = extract_element_features(elem, category=OTHER_CATEGORY, selectors=selectors)
        if data:
            negative_data.append(data)

    # --- Balancing & Sampling ---
    if positive_data:
        max_negatives = len(positive_data) * OTHER_TO_CATEGORY_RATIO
        if len(negative_data) > max_negatives:
            random.seed(RANDOM_SEED)
            negative_data = random.sample(negative_data, max_negatives)

    all_data = positive_data + negative_data

    if not all_data:
        return pd.DataFrame()

    # This calculates ranks and scores based on the full page context
    processed_data = process_page_features(all_data)

    df = pd.DataFrame(processed_data)

    if url is not None:
        df['SourceURL'] = url

    # Clean up NAs
    text_cols = ['class_str', 'id_str', 'tag', 'parent_tag', 'gparent_tag']
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna("")

    return df.fillna(0)


def selector_data_to_csv(data_domain_dir: Path) -> None:
    """Convert YAML/HTML pair to CSV for a given domain directory."""
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
    """Batch process all data directories."""
    data_dir = project_root / 'src' / 'data'
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return

    for site_dir in data_dir.iterdir():
        if site_dir.is_dir():
            selector_data_to_csv(site_dir)
