"""
Prediction utilities for extracting and grouping HTML elements by category using a trained model.
"""

# @generated "partially" Gemini 3: refactored for linting errors and add docstrings.

from typing import Any, Dict, List, Optional, Set, Tuple

import lxml.html
import numpy as np

from product_scraper.train_model.process_data import html_to_dataframe
from product_scraper.utils.console import log_error
from product_scraper.utils.features import (
    NON_TRAINING_FEATURES,
    TARGET_FEATURE,
    calculate_proximity_score,
)


def _get_target_class_idx(label_encoder: Any, category: str) -> Optional[int]:
    """Helper to get the index of the target category from the label encoder."""
    if category not in label_encoder.classes_:
        return None
    try:
        return label_encoder.transform([category])[0]
    except ValueError:
        return None


def _prepare_X_pred(X: Any, training_features: Dict[str, List[str]]) -> Any:
    """Helper to prepare the features DataFrame for prediction."""
    cols_to_drop = [col for col in NON_TRAINING_FEATURES if col in X.columns]
    if TARGET_FEATURE in X.columns:
        cols_to_drop.append(TARGET_FEATURE)

    X_pred = X.drop(columns=cols_to_drop, errors="ignore")

    if training_features:
        for col in training_features.get("numeric", []):
            if col not in X_pred.columns:
                X_pred[col] = 0.0 if "density" in col else 100.0
        for col in training_features.get("text", []):
            if col not in X_pred.columns:
                X_pred[col] = "empty"
    return X_pred


def _extract_candidates(
    tree: lxml.html.HtmlElement, X: Any, match_indices: np.ndarray
) -> List[Dict[str, Any]]:
    """Helper to extract candidate elements from the HTML tree based on prediction indices."""
    candidates = []
    if "xpath" not in X.columns:
        log_error("Missing 'xpath' column in feature DataFrame.")
        return []

    for idx in match_indices:
        xpath = X.iloc[idx]["xpath"]
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


def predict_category_selectors(
    model: Dict[str, Any],
    html_content: str,
    category: str,
    existing_selectors: Optional[Dict[str, List[str]]] = None,
) -> List[Dict[str, Any]]:
    """
    Predicts elements in the HTML that match the given category using the trained model.

    This function extracts features from the HTML content, preprocesses them to match
    the training data format, and uses the model pipeline to predict the category of each element.
    It returns a list of candidate elements that match the target category.

    Args:
        model (Dict[str, Any]): The trained model dictionary containing 'pipeline', 'label_encoder', etc.
        html_content (str): The raw HTML string of the page to predict on.
        category (str): The target category label to predict (e.g., "price", "title").
        existing_selectors (Optional[Dict[str, List[str]]]): Dictionary of known selectors to aid feature extraction (context).

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, where each dictionary represents a predicted element
                              and contains keys like 'index', 'xpath', 'preview', 'tag', 'class', and 'id'.
    """
    pipeline = model["pipeline"]
    label_encoder = model["label_encoder"]
    training_features = model.get("features", {})

    target_class_idx = _get_target_class_idx(label_encoder, category)
    if target_class_idx is None:
        return []

    tree = lxml.html.fromstring(html_content)
    X = html_to_dataframe(
        html_content, selectors=existing_selectors or {}, augment_data=False
    )
    if X.empty:
        return []

    X_pred = _prepare_X_pred(X, training_features)

    try:
        predictions = pipeline.predict(X_pred)
    except (ValueError, KeyError, TypeError) as e:
        log_error(f"Prediction error: {e}")
        return []
    except Exception as e:  # pylint: disable=broad-exception-caught
        log_error(f"Unexpected prediction error: {e}")
        return []

    match_indices = np.where(predictions == target_class_idx)[0]
    return _extract_candidates(tree, X, match_indices)


def _compute_proximity_edges(
    products: List[Dict[str, Any]],
    candidates: List[Dict[str, Any]],
    anchor_category: str,
    max_distance_threshold: int,
) -> List[Tuple[int, int, int]]:
    """Helper to compute edges between anchor products and candidates based on distance."""
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
    return edges


# https://stackoverflow.com/questions/47974874/algorithm-for-grouping-points-in-given-distance
def group_prediction_to_products(
    selectors: Dict[str, List[Any]],
    categories: List[str],
    max_distance_threshold: int = 50,
) -> List[Dict[str, Any]]:
    """
    Group predicted selectors into product dictionaries using greedy nearest-neighbor clustering.

    Robust to input being simple strings (XPaths) or dictionary objects.
    """
    if not categories or not selectors:
        return []

    # Ensure all items are dictionaries with an 'xpath' key.
    normalized_selectors = {}
    for cat, items in selectors.items():
        cleaned_items = []
        for item in items:
            if isinstance(item, str):
                cleaned_items.append({"xpath": item})
            else:
                cleaned_items.append(item)
        normalized_selectors[cat] = cleaned_items

    selectors = normalized_selectors

    valid_categories = [c for c in categories if c in selectors and selectors[c]]
    if not valid_categories:
        return []

    # Anchor is the category that is first in the list of valid categories
    anchor_category = valid_categories[0]
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
        edges = _compute_proximity_edges(
            products, candidates, anchor_category, max_distance_threshold
        )

        # --- Sort & Assign Greedily ---
        edges.sort(key=lambda x: x[0])

        assigned_candidates: Set[int] = set()

        for _, p_idx, c_idx in edges:
            if c_idx in assigned_candidates:
                continue

            # Allow multiple items per product by default
            if cat not in products[p_idx]:
                products[p_idx][cat] = []

            # Append item to the list
            products[p_idx][cat].append(candidates[c_idx])

            assigned_candidates.add(c_idx)

    return products
