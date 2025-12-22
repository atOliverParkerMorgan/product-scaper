from typing import Any, Dict, List, Optional

import lxml.html
import numpy as np

from train_model.process_data import get_main_html_content_tag, html_to_dataframe
from utils.features import NON_TRAINING_FEATURES, TARGET_FEATURE, UNWANTED_TAGS, calculate_proximity_score
from utils.utils import get_unique_xpath, normalize_tag


def predict_category_selectors(
    model: Dict[str, Any],
    html_content: str,
    category: str,
    existing_selectors: Optional[Dict[str, List[str]]] = None,
) -> List[Dict[str, Any]]:

    pipeline = model['pipeline']
    label_encoder = model['label_encoder']
    training_features = model.get('features', {})

    if category not in label_encoder.classes_:
        return []

    try:
        target_class_idx = label_encoder.transform([category])[0]
    except ValueError:
        return []

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

    # NO augmentation during prediction
    X = html_to_dataframe(html_content, selectors=existing_selectors, augment_data=False)

    if X.empty:
        return []

    cols_to_drop = [col for col in NON_TRAINING_FEATURES if col in X.columns]
    if TARGET_FEATURE in X.columns:
        cols_to_drop.append(TARGET_FEATURE)

    X_pred = X.drop(columns=cols_to_drop, errors='ignore')

    # Force align columns with training data
    if training_features:
        # Add missing numeric columns with defaults
        for col in training_features.get('numeric', []):
            if col not in X_pred.columns:
                if 'density' in col:
                    X_pred[col] = 0.0
                else:
                    X_pred[col] = 100.0

        # Add missing text columns
        for col in training_features.get('text', []):
            if col not in X_pred.columns:
                X_pred[col] = 'empty'

    try:
        # Get probabilities to filter low confidence if needed
        # predictions = pipeline.predict(X_pred)
        # Using predict_proba allows for threshold tuning, but strictly we use predict here
        predictions = pipeline.predict(X_pred)
    except Exception:
        return []

    match_indices = np.where(predictions == target_class_idx)[0]
    candidates = []

    xpaths_in_df = X['xpath'].values if 'xpath' in X.columns else []

    for idx in match_indices:
        if idx < len(elements):
            element = elements[idx]

            # Sanity check alignment
            if len(xpaths_in_df) > idx:
                if xpaths_in_df[idx] != get_unique_xpath(element):
                    continue

            text_content = element.text_content().strip()
            preview = text_content[:50] + "..." if len(text_content) > 50 else text_content

            candidates.append({
                'index': int(idx),
                'xpath': get_unique_xpath(element),
                'preview': preview,
                'tag': element.tag,
                'class': element.get('class', ''),
                'id': element.get('id', '')
            })

    return candidates

# ... (rest of predict_data.py remains the same)
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
