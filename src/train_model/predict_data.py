from train_model.process_data import html_to_dataframe, get_main_html_content_tag
from utils.features import UNWANTED_TAGS, NON_TRAINING_FEATURES, TARGET_FEATURE
from utils.utils import get_unique_xpath, normalize_tag
import lxml.html
import numpy as np
from typing import Any, Dict, List
import re

def predict_selectors(model: Dict[str, Any], html_content: str, category: str) -> List[Dict[str, Any]]:
    """
    Predicts selectors of the `category` from the html_content 
    
    Args:
        model:
        html_content: str
        category
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

def get_xpath_segments(xpath: str) -> List[str]:
    """Helper to split xpath into clean segments."""
    return [s for s in xpath.split('/') if s]

def extract_index(segment: str) -> int:
    """Extracts the integer index from a segment like 'div[3]', default to 1."""
    match = re.search(r'\[(\d+)\]', segment)
    return int(match.group(1)) if match else 1

def calculate_proximity_score(xpath1: str, xpath2: str) -> tuple:
    """
    Calculates a proximity score tuple: (Tree Distance, Index Delta).
    Lower values for both mean 'closer'.
    """
    path1 = get_xpath_segments(xpath1)
    path2 = get_xpath_segments(xpath2)
    
    min_len = min(len(path1), len(path2))
    divergence_index = 0
    
    # Find the Lowest Common Ancestor
    for i in range(min_len):
        if path1[i] == path2[i]:
            divergence_index += 1
        else:
            break
            
    # Calculate Tree Distance
    # Steps up from xpath1 to LCA + Steps down from LCA to xpath2
    dist_up = len(path1) - divergence_index
    dist_down = len(path2) - divergence_index
    tree_distance = dist_up + dist_down
    
    # Tie-Breaker: Calculate Index Delta (Tie-Breaker)
    index_delta = 0
    if divergence_index < len(path1) and divergence_index < len(path2):
        # Extract indices from the segments where they diverge
        idx1 = extract_index(path1[divergence_index])
        idx2 = extract_index(path2[divergence_index])
        index_delta = abs(idx1 - idx2)
        
    return (tree_distance, index_delta)

def group_prediction_to_products(html_content: str, selectors: Dict[str, List[Dict[str, Any]]], categories: List[str]) -> List[Dict[str, Any]]:
    
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
            
            # Find Best Match using the Proximity Tuple
            best_candidate = min(
                candidates, 
                key=lambda x: calculate_proximity_score(anchor_xpath, x['xpath'])
            )
            
            product[cat] = best_candidate
            
        products.append(product)

    return products