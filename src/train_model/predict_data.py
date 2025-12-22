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
    
    # Check if category exists in model
    if category not in label_encoder.classes_:
        # If the category wasn't in training, we can't predict it.
        # However, for 'Other', we might handle it differently, but usually we predict target classes.
        return []

    try:
        target_class_idx = label_encoder.transform([category])[0]
    except ValueError:
        return []

    # Prepare elements list to map back predictions to HTML elements
    tree = lxml.html.fromstring(html_content)
    main_content = get_main_html_content_tag(html_content) or tree

    elements = []
    for elem in main_content.iter():
        if not isinstance(elem.tag, str) or normalize_tag(elem.tag) in UNWANTED_TAGS:
            continue
        # Optional: Skipping purely empty elements to match training logic
        try:
            if not elem.text_content().strip() and not elem.attrib:
                continue
        except Exception:
            continue
        elements.append(elem)

    # Pass existing selectors to calculate distance features relative to KNOWN items
    X = html_to_dataframe(html_content, selectors=existing_selectors)

    if X.empty:
        return []

    # Ensure alignment
    if len(X) != len(elements):
        # Fallback: if html_to_dataframe filters differently, we can't map reliably by index.
        # Strict alignment logic required or rely on 'xpath' column if preserved.
        # For now, assuming html_to_dataframe logic matches the loop above.
        # If mismatched, we trust X's xpaths.
        pass

    # Drop non-training columns
    cols_to_drop = [col for col in NON_TRAINING_FEATURES if col in X.columns]
    if TARGET_FEATURE in X.columns:
        cols_to_drop.append(TARGET_FEATURE)
        
    X_pred = X.drop(columns=cols_to_drop, errors='ignore')
    
    # Handle missing columns that might have been in training but not here
    # (e.g., dist_to_closest_Price if Price wasn't found yet)
    # The pipeline handles standard scaling, but if column is missing, we must add it.
    
    # Get feature names expected by the model (if available via column transformer)
    # This is complex with Pipelines. Simplest way is to ensure html_to_dataframe produces consistent columns
    # or add missing ones with 0/default.
    
    # PREDICT
    try:
        predictions = pipeline.predict(X_pred)
    except ValueError as e:
        # Often happens if columns mismatch
        # Force align columns
        # This part requires access to the training columns list stored in model['features']
        if 'features' in model:
            train_numeric = model['features']['numeric']
            for col in train_numeric:
                if col not in X_pred.columns:
                    X_pred[col] = 9999.0 # Missing distance
            
            # Reorder
            all_train_cols = train_numeric + model['features']['categorical'] + model['features']['text']
            # Filter to only those that exist or let ColumnTransformer handle by name
            # Generally ColumnTransformer is robust if columns are missing provided they are not required
            # but usually it's better to provide them.
            pass
        predictions = pipeline.predict(X_pred)

    match_indices = np.where(predictions == target_class_idx)[0]

    candidates = []
    
    # Map back using DataFrame index or logic
    # Since X is derived from elements list, indices *should* align if logic is identical.
    # To be safe, rely on X['xpath'] if available
    
    xpaths_in_df = X['xpath'].values if 'xpath' in X.columns else []

    for idx in match_indices:
        # Retrieve element
        if idx < len(elements):
            element = elements[idx]
            
            # double check xpath alignment if possible
            if len(xpaths_in_df) > idx:
                df_xpath = xpaths_in_df[idx]
                el_xpath = get_unique_xpath(element)
                if df_xpath != el_xpath:
                    # Misalignment detected
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
