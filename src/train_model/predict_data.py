from typing import Any, Dict, List

import lxml.html
import numpy as np

from train_model.process_data import get_main_html_content_tag, html_to_dataframe
from utils.features import NON_TRAINING_FEATURES, TARGET_FEATURE, UNWANTED_TAGS, calculate_proximity_score
from utils.utils import get_unique_xpath, normalize_tag


def predict_selectors(model: Dict[str, Any], html_content: str, category: str) -> List[Dict[str, Any]]:
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

    X = html_to_dataframe(html_content, selectors=None)

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


def group_prediction_to_products(
    html_content: str,
    selectors: Dict[str, List[Dict[str, Any]]],
    categories: List[str]
) -> List[Dict[str, Any]]:
    """
    Group predicted selectors into product dictionaries by proximity.

    Args:
        html_content (str): HTML content.
        selectors (Dict[str, List[Dict[str, Any]]]): Predicted selectors by category.
        categories (List[str]): List of categories.
    Returns:
        List[Dict[str, Any]]: List of product dictionaries.
    """
    if not categories or not selectors:
        return []
    anchor_category = max(categories, key=lambda c: len(selectors.get(c, [])))
    anchor_items = selectors.get(anchor_category, [])
    products = []
    for anchor_item in anchor_items:
        product = {}
        product[anchor_category] = anchor_item
        anchor_xpath = anchor_item['xpath']
        for cat in categories:
            if cat == anchor_category:
                continue
            candidates = selectors.get(cat, [])
            if not candidates:
                continue
            best_candidate = min(
                candidates,
                key=lambda x: calculate_proximity_score(anchor_xpath, x['xpath'])
            )
            product[cat] = best_candidate
        products.append(product)
    return products
