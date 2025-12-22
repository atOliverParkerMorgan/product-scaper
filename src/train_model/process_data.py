"""Data processing utilities for HTML element feature extraction."""

import random
from typing import Dict, List, Optional, Any
import numpy as np

import lxml.html
import pandas as pd

from utils.console import log_error
from utils.features import OTHER_CATEGORY, UNWANTED_TAGS, extract_element_features, process_page_features, calculate_proximity_score
from utils.utils import get_unique_xpath, normalize_tag

RANDOM_SEED = 42
# Increased ratio to provide more negative samples for better "Other" classification
OTHER_TO_CATEGORY_RATIO = 3.0 

def get_main_html_content_tag(html_content: str) -> Optional[lxml.html.HtmlElement]:
    """
    Returns the full HTML tree to ensure no Titles/Prices are cropped out.
    """
    if not html_content:
        return None
    try:
        tree = lxml.html.fromstring(html_content)
        body = tree.find('body')
        return body if body is not None else tree
    except Exception as e:
        log_error(f"Failed to parse HTML: {e}")
        return None

def calculate_list_density(element: lxml.html.HtmlElement, max_depth: int = 4) -> int:
    """
    Checks ancestors up to `max_depth` to see if the element is part of a list structure.
    
    Logic:
    1. Check element's siblings (same tag).
    2. Check parent's siblings (same tag as parent).
    3. Return the MAXIMUM count found.
    
    This detects both direct lists (<li>...</li>) and wrapped lists (<div><a>Item</a></div>).
    """
    max_density = 0
    current = element
    
    for _ in range(max_depth):
        parent = current.getparent()
        if parent is None:
            break
            
        # Count children of parent that share the 'current' tag
        count = 0
        target_tag = current.tag
        
        # Iterating only direct children is fast in lxml
        for child in parent:
            if child.tag == target_tag:
                count += 1
        
        if count > max_density:
            max_density = count
            
        # Move up tree
        current = parent
        
    return max_density

def calculate_context_features(element: lxml.html.HtmlElement, selectors: Dict[str, List[str]], root_tree) -> Dict[str, float]:
    """
    Calculates spatial features (distances) relative to KNOWN anchors only.
    """
    features = {}
    
    # 1. List Density Feature
    features['max_sibling_density'] = float(calculate_list_density(element))

    # 2. Distance Features
    if not selectors:
        return features

    current_xpath = get_unique_xpath(element)

    for category, xpaths in selectors.items():
        if category == OTHER_CATEGORY or not xpaths:
            continue
            
        min_dist = 1000.0 
        
        for ref_xpath in xpaths:
            if ref_xpath == current_xpath:
                continue

            try:
                d_score = calculate_proximity_score(current_xpath, ref_xpath)
                total_dist = d_score[0] + d_score[1]
                if total_dist < min_dist:
                    min_dist = total_dist
            except Exception:
                continue
        
        features[f'dist_to_closest_{category}'] = min_dist

    return features

def html_to_dataframe(
    html_content: str,
    selectors: Dict[str, List[str]],
    url: Optional[str] = None
) -> pd.DataFrame:
    """
    Extracts features from ALL valid elements in the HTML body.
    """
    main_content = get_main_html_content_tag(html_content)
    if main_content is None:
        return pd.DataFrame()
    
    root_tree = main_content.getroottree()
    labeled_elements = set()
    positive_data = []

    # --- 1. Extract Positive Samples ---
    for category, xpaths in selectors.items():
        if category == OTHER_CATEGORY:
            continue
            
        for xpath in xpaths:
            try:
                found = main_content.xpath(xpath)
                if not found:
                    found = root_tree.xpath(xpath)
                
                for elem in found:
                    if isinstance(elem, lxml.html.HtmlElement):
                        if normalize_tag(elem.tag) in UNWANTED_TAGS: continue
                        
                        data = extract_element_features(elem, category=category, selectors=selectors)
                        if data:
                            data['xpath'] = get_unique_xpath(elem)
                            # Add Context & List Features
                            data.update(calculate_context_features(elem, selectors, root_tree))
                            
                            positive_data.append(data)
                            labeled_elements.add(elem)
            except Exception:
                continue

    # --- 2. Extract Negative Samples ---
    negative_data = []
    
    for elem in main_content.iter():
        if not isinstance(elem.tag, str): continue
        if normalize_tag(elem.tag) in UNWANTED_TAGS: continue
        if elem in labeled_elements: continue

        text = elem.text_content()
        if not text.strip() and not elem.attrib: continue

        data = extract_element_features(elem, category=OTHER_CATEGORY, selectors=selectors)
        if data:
            data['xpath'] = get_unique_xpath(elem)
            # Add Context & List Features
            data.update(calculate_context_features(elem, selectors, root_tree))
            
            negative_data.append(data)

    # --- 3. Balancing ---
    if positive_data:
        max_negatives = int(len(positive_data) * OTHER_TO_CATEGORY_RATIO)
        max_negatives = max(max_negatives, 50) 
        
        if len(negative_data) > max_negatives:
            random.seed(RANDOM_SEED)
            negative_data = random.sample(negative_data, max_negatives)
    
    all_data = positive_data + negative_data
    if not all_data: return pd.DataFrame()

    processed_data = process_page_features(all_data)
    df = pd.DataFrame(processed_data)

    if url is not None: df['SourceURL'] = url
    
    for col in ['class_str', 'id_str', 'tag', 'parent_tag', 'gparent_tag']:
        if col in df.columns: df[col] = df[col].fillna('')
    
    # Fill numeric NaNs
    dist_cols = [c for c in df.columns if 'dist_to_' in c or 'density' in c]
    for col in dist_cols:
        df[col] = df[col].fillna(9999.0)

    return df.fillna(0)