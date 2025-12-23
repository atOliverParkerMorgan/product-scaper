"""Data processing utilities for HTML element feature extraction."""

# @generated "particially" Gemini 3: generated docs strings and refactored for linting.

import random
from typing import Any, Dict, List, Optional, Set, Tuple

import lxml.html
import numpy as np
import pandas as pd

from product_scraper.utils.console import log_error
from product_scraper.utils.features import (
    DEFAULT_DIST,
    OTHER_CATEGORY,
    UNWANTED_TAGS,
    extract_element_features,
    normalize_tag,
    process_page_features,
)

RANDOM_SEED = 42
OTHER_TO_CATEGORY_RATIO = 1

# Create a local random instance to avoid side effects on global state
rng = random.Random(RANDOM_SEED)


def get_main_html_content_tag(html_content: str) -> Optional[lxml.html.HtmlElement]:
    """
    Parses HTML content and returns the main body element or root tree.

    Args:
        html_content (str): The raw HTML string.

    Returns:
        Optional[lxml.html.HtmlElement]: The <body> element if found, otherwise the root element.
                                         Returns None if parsing fails or content is empty.
    """
    if not html_content:
        return None
    try:
        tree = lxml.html.fromstring(html_content)
        body = tree.find("body")
        return body if body is not None else tree
    except (TypeError, ValueError) as e:
        log_error(f"Failed to parse HTML: {e}")
        return None


def calculate_list_density(element: lxml.html.HtmlElement, max_depth: int = 5) -> float:
    """
    Calculates a density score based on the repetition of the element's tag among its siblings.

    This helps identify elements that are part of lists or grids (common in product listings).
    The score scales logarithmically with the count of siblings sharing the same tag.

    Args:
        element (lxml.html.HtmlElement): The target element.
        max_depth (int): How many levels up the DOM tree to check.

    Returns:
        float: The maximum calculated density score found within the depth range.
    """
    max_density = 0.0
    current = element
    for _ in range(max_depth):
        parent = current.getparent()
        if parent is None:
            break
        siblings = [child for child in parent if child.tag == current.tag]
        count = len(siblings)
        if count > 1:
            density_score = float(count) if count < 50 else 50.0 + np.log(count)
            max_density = max(max_density, density_score)
        current = parent
    return max_density


def _process_single_element(
    elem: Any,
    selectors: Dict[str, List[str]],
    category: str,
    augment_data: bool,
    dropout_threshold: float,
) -> Optional[Dict[str, Any]]:
    """Helper to extract features for a single element and apply augmentation."""
    if not isinstance(elem, lxml.html.HtmlElement):
        return None
    if normalize_tag(elem.tag) in UNWANTED_TAGS:
        return None

    base_data = extract_element_features(elem, selectors=selectors, category=category)
    if not base_data:
        return None

    should_dropout = augment_data and (rng.random() > dropout_threshold)
    if should_dropout:
        base_data["avg_distance_to_closest_categories"] = DEFAULT_DIST
    base_data["max_sibling_density"] = calculate_list_density(elem)
    return base_data


def _extract_positive_samples(
    main_content: lxml.html.HtmlElement,
    root_tree: lxml.html.HtmlElement,
    selectors: Dict[str, List[str]],
    augment_data: bool,
    dropout_threshold: float,
) -> Tuple[List[Dict[str, Any]], Set[Any]]:
    """Identifies and extracts features for labeled positive samples based on selectors."""
    labeled_elements = set()
    positive_data = []
    for category, xpaths in selectors.items():
        if category == OTHER_CATEGORY:
            continue
        for xpath in xpaths:
            try:
                found = main_content.xpath(xpath)
                if not found:
                    found = root_tree.xpath(xpath)
                for elem in found:
                    data = _process_single_element(
                        elem, selectors, category, augment_data, dropout_threshold
                    )
                    if data:
                        positive_data.append(data)
                        labeled_elements.add(elem)
            except (ValueError, KeyError, TypeError) as e:
                log_error(f"Error extracting positive sample: {e}")
                continue
    return positive_data, labeled_elements


# https://stackoverflow.com/questions/34655628/how-to-handle-class-imbalance-in-sklearn-random-forests-should-i-use-sample-wei
def _extract_negative_samples(
    main_content: lxml.html.HtmlElement,
    selectors: Dict[str, List[str]],
    labeled_elements: Set[Any],
    augment_data: bool,
    dropout_threshold: float,
) -> List[Dict[str, Any]]:
    """Iterates through unlabeled elements to extract negative samples."""
    negative_data = []
    for elem in main_content.iter():
        if not isinstance(elem.tag, str):
            continue
        if elem in labeled_elements:
            continue
        if normalize_tag(elem.tag) in UNWANTED_TAGS:
            continue

        try:
            text = elem.text_content()
            if not text.strip() and not elem.attrib:
                continue
        except (ValueError, KeyError, TypeError) as e:
            log_error(f"Error extracting negative sample: {e}")
            continue

        data = _process_single_element(
            elem, selectors, OTHER_CATEGORY, augment_data, dropout_threshold
        )
        if data:
            negative_data.append(data)
    return negative_data


def _finalize_dataframe(df: pd.DataFrame, url: Optional[str]) -> pd.DataFrame:
    """Cleans up the DataFrame, filling NaNs and adding metadata."""
    if url is not None:
        df["SourceURL"] = url
    text_cols = ["class_str", "id_str", "tag", "parent_tag", "gparent_tag"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna("")
    if "avg_distance_to_closest_categories" in df.columns:
        df["avg_distance_to_closest_categories"] = df[
            "avg_distance_to_closest_categories"
        ].fillna(50.0)
    if "max_sibling_density" in df.columns:
        df["max_sibling_density"] = df["max_sibling_density"].fillna(0.0)
    return df.fillna(0)


def html_to_dataframe(
    html_content: str,
    selectors: Dict[str, List[str]],
    url: Optional[str] = None,
    augment_data: bool = False,
) -> pd.DataFrame:
    """
    Extracts features from HTML, creates positive/negative samples, and returns a DataFrame.

    This function processes the HTML content to:
    1. Identify positive samples based on provided selectors (labeled data).
    2. Extract negative samples (unlabeled elements) from the rest of the page.
    3. Balance the dataset by subsampling negative examples.
    4. Compute page-level features and return the result as a pandas DataFrame.

    Args:
        html_content (str): The raw HTML string of the page.
        selectors (Dict[str, List[str]]): Dictionary mapping categories to lists of XPaths for positive labeling.
        url (Optional[str]): Source URL of the page (added as a metadata column).
        augment_data (bool): If True, applies random feature dropout to simulate noise and improve generalization.

    Returns:
        pd.DataFrame: A DataFrame containing extracted features for all processed elements.
    """
    main_content = get_main_html_content_tag(html_content)
    if main_content is None:
        return pd.DataFrame()

    root_tree = main_content.getroottree()
    num_selectors = len(selectors) if selectors else 1
    dropout_threshold = 1.0 / num_selectors

    positive_data, labeled_elements = _extract_positive_samples(
        main_content, root_tree, selectors, augment_data, dropout_threshold
    )

    negative_data = _extract_negative_samples(
        main_content, selectors, labeled_elements, augment_data, dropout_threshold
    )

    # Balance datasets
    if positive_data:
        max_negatives = int(len(positive_data) * OTHER_TO_CATEGORY_RATIO)
        max_negatives = max(max_negatives, 50)
        if len(negative_data) > max_negatives:
            negative_data = rng.sample(negative_data, max_negatives)

    all_data = positive_data + negative_data
    if not all_data:
        return pd.DataFrame()

    processed_data = process_page_features(all_data)
    df = pd.DataFrame(processed_data)

    return _finalize_dataframe(df, url)
