"""Data processing utilities for HTML element feature extraction."""

import random
from typing import Dict, List, Optional

import lxml.html
import numpy as np
import pandas as pd

from utils.console import log_error
from utils.features import (
    DEFAULT_DIST,
    OTHER_CATEGORY,
    UNWANTED_TAGS,
    extract_element_features,
    normalize_tag,
    process_page_features,
)

RANDOM_SEED = 42
OTHER_TO_CATEGORY_RATIO = 10.0

# Create a local random instance to avoid side effects on global state
rng = random.Random(RANDOM_SEED)


def get_main_html_content_tag(html_content: str) -> Optional[lxml.html.HtmlElement]:
    if not html_content:
        return None
    try:
        tree = lxml.html.fromstring(html_content)
        body = tree.find("body")
        return body if body is not None else tree
    except Exception as e:
        log_error(f"Failed to parse HTML: {e}")
        return None


def calculate_list_density(element: lxml.html.HtmlElement, max_depth: int = 5) -> float:
    """Checks if the element is part of a repeated list structure."""
    max_density = 0.0
    current = element

    for _ in range(max_depth):
        parent = current.getparent()
        if parent is None:
            break

        # Count siblings with the same tag
        siblings = [child for child in parent if child.tag == current.tag]
        count = len(siblings)

        if count > 1:
            # Logarithmic scaling for large lists
            density_score = float(count) if count < 50 else 50.0 + np.log(count)
            if density_score > max_density:
                max_density = density_score

        current = parent

    return max_density


def html_to_dataframe(
    html_content: str, selectors: Dict[str, List[str]], url: Optional[str] = None, augment_data: bool = False
) -> pd.DataFrame:
    """
    Extracts features from HTML, creates positive/negative samples, and returns a DataFrame.
    """
    main_content = get_main_html_content_tag(html_content)
    if main_content is None:
        return pd.DataFrame()

    root_tree = main_content.getroottree()
    labeled_elements = set()
    positive_data = []

    # Augmentation settings
    num_selectors = len(selectors) if selectors else 1
    dropout_threshold = 1.0 / num_selectors

    # --- 1. Extract Positive Samples ---
    for category, xpaths in selectors.items():
        if category == OTHER_CATEGORY:
            continue

        for xpath in xpaths:
            try:
                # Try finding in main_content first, then fallback to root
                found = main_content.xpath(xpath)
                if not found:
                    found = root_tree.xpath(xpath)

                for elem in found:
                    if isinstance(elem, lxml.html.HtmlElement):
                        if normalize_tag(elem.tag) in UNWANTED_TAGS:
                            continue

                        # Extract base features
                        base_data = extract_element_features(elem, selectors=selectors, category=category)

                        if base_data:
                            # Add specific context features
                            should_dropout = augment_data and (rng.random() > dropout_threshold)

                            # Note: avg_distance is already calculated in extract_element_features
                            # but we can modify it here for dropout simulation
                            if should_dropout:
                                base_data["avg_distance_to_closest_categories"] = DEFAULT_DIST

                            base_data["max_sibling_density"] = calculate_list_density(elem)

                            positive_data.append(base_data)
                            labeled_elements.add(elem)
            except Exception:
                continue

    # --- 2. Extract Negative Samples ---
    negative_data = []

    for elem in main_content.iter():
        if not isinstance(elem.tag, str):
            continue
        if normalize_tag(elem.tag) in UNWANTED_TAGS:
            continue
        if elem in labeled_elements:
            continue

        # Filter empty elements to reduce noise
        try:
            text = elem.text_content()
            if not text.strip() and not elem.attrib:
                continue
        except Exception:
            continue

        data = extract_element_features(elem, selectors=selectors, category=OTHER_CATEGORY)
        if data:
            should_dropout = augment_data and (rng.random() > dropout_threshold)
            if should_dropout:
                data["avg_distance_to_closest_categories"] = DEFAULT_DIST

            data["max_sibling_density"] = calculate_list_density(elem)
            negative_data.append(data)

    # --- 3. Balancing ---
    if positive_data:
        max_negatives = int(len(positive_data) * OTHER_TO_CATEGORY_RATIO)
        max_negatives = max(max_negatives, 50)  # Ensure at least some negatives

        if len(negative_data) > max_negatives:
            negative_data = rng.sample(negative_data, max_negatives)

    all_data = positive_data + negative_data
    if not all_data:
        return pd.DataFrame()

    # Process page-level relative features (ranking, global stats)
    processed_data = process_page_features(all_data)
    df = pd.DataFrame(processed_data)

    if url is not None:
        df["SourceURL"] = url

    # Fill Missing Values securely
    text_cols = ["class_str", "id_str", "tag", "parent_tag", "gparent_tag"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna("")

    if "avg_distance_to_closest_categories" in df.columns:
        df["avg_distance_to_closest_categories"] = df["avg_distance_to_closest_categories"].fillna(50.0)

    if "max_sibling_density" in df.columns:
        df["max_sibling_density"] = df["max_sibling_density"].fillna(0.0)

    return df.fillna(0)
