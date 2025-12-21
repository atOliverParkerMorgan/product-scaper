from typing import Any, Dict, List, Optional

import lxml.html
import numpy as np

from train_model.process_data import get_main_html_content_tag, html_to_dataframe
from utils.features import NON_TRAINING_FEATURES, TARGET_FEATURE, UNWANTED_TAGS, calculate_proximity_score
from utils.utils import get_unique_xpath, normalize_tag


def predict_category_selectors(model: Dict[str, Any], html_content: str, category: str, existing_selectors: Optional[Dict[str, List[str]]] = None,
) -> List[Dict[str, Any]]:
    """
    Predict selectors for a given category from HTML content using a trained model.

    Args:
        model (Dict[str, Any]): Trained model dictionary.
        html_content (str): HTML content to predict from.
        category (str): Category to predict selectors for.
    Returns:
        List[Dict[str, Any]]: List of predicted selector dictionaries.
    """

    pipeline = model['pipeline']
    label_encoder = model['label_encoder']

    try:
        target_class_idx = label_encoder.transform([category])[0]
    except ValueError:
        raise ValueError(f"Category '{category}' was not seen during training. Available: {label_encoder.classes_}")

    tree = lxml.html.fromstring(html_content)
    main_content = get_main_html_content_tag(html_content) or tree

    elements = []
    for elem in main_content.iter():
        if not isinstance(elem.tag, str) or normalize_tag(elem.tag) in UNWANTED_TAGS:
            continue
        try:
            if not elem.text_content().strip() and not elem.attrib:
                continue
        except Exception:
            continue
        elements.append(elem)

    X = html_to_dataframe(html_content, selectors=existing_selectors)

    if X.empty:
        return []

    # DataFrame should have same number of rows as elements
    if len(X) != len(elements):
        raise ValueError(f"Mismatch: DataFrame has {len(X)} rows but elements list has {len(elements)} items")

    if TARGET_FEATURE in X.columns:
        # Only drop columns that actually exist in the dataframe
        cols_to_drop = [col for col in NON_TRAINING_FEATURES if col in X.columns]
        X = X.drop(columns=cols_to_drop)

    predictions = pipeline.predict(X)
    match_indices = np.where(predictions == target_class_idx)[0]

    candidates = []
    for i in match_indices:
        element = elements[i]
        text_content = element.text_content().strip()
        preview = text_content[:50] + "..." if len(text_content) > 50 else text_content
        xpath = get_unique_xpath(element)

        candidates.append({
            'index': i,
            'xpath': xpath,
            'preview': preview,
            'tag': element.tag,
            'class': element.get('class', ''),
            'id': element.get('id', '')
        })

    return candidates


def calculate_distance(item1, item2):
    """Wrapper to safely get distance between two selector dictionaries."""
    return calculate_proximity_score(item1['xpath'], item2['xpath'])

def group_prediction_to_products(
    html_content: str,
    selectors: Dict[str, List[Dict[str, Any]]],
    categories: List[str],
    max_distance_threshold: int = 50  # Max 'hops' allowed between items
) -> List[Dict[str, Any]]:
    """
    Group predicted selectors into product dictionaries using mutually exclusive pairing.
    """
    if not categories or not selectors:
        return []

    # 1. Determine Anchor Category (The one with the most items, usually Price or Image)
    # Filter out empty categories first
    valid_categories = [c for c in categories if c in selectors and selectors[c]]
    if not valid_categories:
        return []

    anchor_category = max(valid_categories, key=lambda c: len(selectors[c]))
    anchor_items = selectors[anchor_category]

    # Initialize products with the anchor items
    # We use a list of dicts, where each dict represents a product being built
    products = []
    for item in anchor_items:
        products.append({anchor_category: item})

    # 2. Match other categories to the Anchors
    for cat in valid_categories:
        if cat == anchor_category:
            continue

        candidates = selectors[cat]
        if not candidates:
            continue

        # --- A. Calculate ALL possible distances between current products and candidates ---
        # We build a list of edges: (distance, product_index, candidate_index)
        edges = []
        for p_idx, product in enumerate(products):
            # The 'location' of the product is defined by its anchor item
            anchor_item = product[anchor_category]

            for c_idx, candidate in enumerate(candidates):
                # Calculate distance (tree distance + index delta)
                dist_score_tuple = calculate_proximity_score(anchor_item['xpath'], candidate['xpath'])
                # Sum the tuple components for a single scalar score
                score = dist_score_tuple[0] + dist_score_tuple[1]

                if score <= max_distance_threshold:
                    edges.append((score, p_idx, c_idx))

        # --- B. Sort edges by distance (closest pairs first) ---
        edges.sort(key=lambda x: x[0])

        # --- C. Assign Greedily (Mutual Exclusion) ---
        assigned_products = set()
        assigned_candidates = set()

        for score, p_idx, c_idx in edges:
            if p_idx in assigned_products or c_idx in assigned_candidates:
                continue

            # Lock this assignment
            products[p_idx][cat] = candidates[c_idx]
            assigned_products.add(p_idx)
            assigned_candidates.add(c_idx)

    return products
