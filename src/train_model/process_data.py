"""Data processing utilities for HTML element feature extraction."""

import random
from typing import Dict, List, Optional, Tuple

import lxml.html
import pandas as pd

from utils.console import log_error

# Assuming these are available from your previous file
from utils.features import OTHER_CATEGORY, UNWANTED_TAGS, extract_element_features, process_page_features
from utils.utils import get_unique_xpath, normalize_tag  # Added get_unique_xpath

RANDOM_SEED = 42
OTHER_TO_CATEGORY_RATIO = 10


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
        log_error(f"Failed to parse HTML: {e}")
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
    Includes explicit XPath extraction to ensure data alignment during prediction.
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
                            data = extract_element_features(elem, category=category, selectors=selectors)
                            if data:
                                # CRITICAL: Add XPath here to ensure it exists for prediction mapping
                                data['xpath'] = get_unique_xpath(elem)
                                positive_data.append(data)
                                labeled_elements.add(elem)
                except Exception as e:
                    log_error(f"Invalid XPath '{xpath}': {e}")

    # --- 2. Extract Negative Samples ---
    negative_data = []
    for elem in main_content.iter():
        if not isinstance(elem.tag, str) or normalize_tag(elem.tag) in UNWANTED_TAGS:
            continue

        if elem in labeled_elements:
            continue

        try:
            if not elem.text_content().strip() and not elem.attrib:
                continue
        except Exception:
            continue

        data = extract_element_features(elem, category=OTHER_CATEGORY, selectors=selectors)
        if data:
            # CRITICAL: Add XPath here to ensure it exists for prediction mapping
            data['xpath'] = get_unique_xpath(elem)
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

    # Process relative features
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
