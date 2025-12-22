from typing import Any, Dict, List, Optional, Tuple

import lxml.html
import numpy as np
from train_model.process_data import html_to_dataframe

from product_scraper.utils.features import (
    NON_TRAINING_FEATURES,
    TARGET_FEATURE,
    calculate_proximity_score,
)


def predict_category_selectors(
    model: Dict[str, Any],
    html_content: str,
    category: str,
    existing_selectors: Optional[Dict[str, List[str]]] = None,
) -> List[Dict[str, Any]]:
    """
    Predicts elements in the HTML that match the given category.
    """
    pipeline = model["pipeline"]
    label_encoder = model["label_encoder"]
    training_features = model.get("features", {})

    if category not in label_encoder.classes_:
        return []

    try:
        target_class_idx = label_encoder.transform([category])[0]
    except ValueError:
        return []

    # Parse HTML once to look up elements later
    tree = lxml.html.fromstring(html_content)

    # 1. Generate Features (No Augmentation for prediction)
    X = html_to_dataframe(
        html_content, selectors=existing_selectors or {}, augment_data=False
    )

    if X.empty:
        return []

    # 2. Align Columns with Training Data
    # Drop non-training columns present in extraction
    cols_to_drop = [col for col in NON_TRAINING_FEATURES if col in X.columns]
    if TARGET_FEATURE in X.columns:
        cols_to_drop.append(TARGET_FEATURE)

    X_pred = X.drop(columns=cols_to_drop, errors="ignore")

    # Fill missing columns expected by the model
    if training_features:
        for col in training_features.get("numeric", []):
            if col not in X_pred.columns:
                X_pred[col] = 0.0 if "density" in col else 100.0

        for col in training_features.get("text", []):
            if col not in X_pred.columns:
                X_pred[col] = "empty"

    # 3. Predict
    try:
        predictions = pipeline.predict(X_pred)
    except Exception:
        return []

    # 4. Extract Matches securely using XPath
    # We iterate over the DataFrame indices where prediction matches target
    match_indices = np.where(predictions == target_class_idx)[0]
    candidates = []

    # Ensure 'xpath' column exists to map back to elements
    if "xpath" not in X.columns:
        return []

    for idx in match_indices:
        # Get the xpath from the dataframe row
        xpath = X.iloc[idx]["xpath"]

        # Find the element in the tree
        found_elements = tree.xpath(xpath)
        if not found_elements:
            continue

        element = found_elements[0]
        if not isinstance(element, lxml.html.HtmlElement):
            continue

        text_content = element.text_content().strip()
        preview = text_content[:50] + "..." if len(text_content) > 50 else text_content

        candidates.append(
            {
                "index": int(idx),
                "xpath": xpath,
                "preview": preview,
                "tag": str(element.tag),
                "class": element.get("class", ""),
                "id": element.get("id", ""),
            }
        )

    return candidates


def group_prediction_to_products(
    html_content: str,
    selectors: Dict[str, List[Dict[str, Any]]],
    categories: List[str],
    max_distance_threshold: int = 50,
) -> List[Dict[str, Any]]:
    """
    Group predicted selectors into product dictionaries using greedy nearest-neighbor clustering.
    """
    if not categories or not selectors:
        return []

    # 1. Determine Anchor Category (The one with the most items, usually Price or Image)
    valid_categories = [c for c in categories if c in selectors and selectors[c]]
    if not valid_categories:
        return []

    anchor_category = max(valid_categories, key=lambda c: len(selectors[c]))
    anchor_items = selectors[anchor_category]

    # Initialize products with the anchor items
    products = [{anchor_category: item} for item in anchor_items]

    # 2. Match other categories to the Anchors
    for cat in valid_categories:
        if cat == anchor_category:
            continue

        candidates = selectors[cat]
        if not candidates:
            continue

        # --- Calculate Distances ---
        # Edges: (score, product_index, candidate_index)
        edges: List[Tuple[int, int, int]] = []

        for p_idx, product in enumerate(products):
            anchor_item = product[anchor_category]

            for c_idx, candidate in enumerate(candidates):
                # Calculate distance (tree distance + index delta)
                dist_tree, dist_index = calculate_proximity_score(
                    anchor_item["xpath"], candidate["xpath"]
                )

                # Weighted score: Tree distance is usually more significant than index delta
                score = dist_tree + dist_index

                if score <= max_distance_threshold:
                    edges.append((score, p_idx, c_idx))

        # --- Sort & Assign Greedily ---
        edges.sort(key=lambda x: x[0])

        assigned_products = set()
        assigned_candidates = set()

        for _, p_idx, c_idx in edges:
            if p_idx in assigned_products or c_idx in assigned_candidates:
                continue

            products[p_idx][cat] = candidates[c_idx]
            assigned_products.add(p_idx)
            assigned_candidates.add(c_idx)

    return products
