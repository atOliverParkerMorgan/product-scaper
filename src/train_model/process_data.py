"""Data processing utilities for HTML element feature extraction."""

import random
from typing import Dict, List, Optional

import lxml.html
import numpy as np
import pandas as pd

from utils.console import log_error
from utils.features import OTHER_CATEGORY, UNWANTED_TAGS, extract_element_features, process_page_features
from utils.utils import get_unique_xpath, normalize_tag

RANDOM_SEED = 42
OTHER_TO_CATEGORY_RATIO = 3.0

def get_main_html_content_tag(html_content: str) -> Optional[lxml.html.HtmlElement]:
    if not html_content:
        return None
    try:
        tree = lxml.html.fromstring(html_content)
        body = tree.find('body')
        return body if body is not None else tree
    except Exception as e:
        log_error(f"Failed to parse HTML: {e}")
        return None

def get_dom_distance(elem1: lxml.html.HtmlElement, elem2: lxml.html.HtmlElement) -> float:
    """Calculates structural distance (Tree Hops) between two elements."""
    if elem1 == elem2:
        return 0.0

    path1 = list(elem1.iterancestors()) + [elem1]
    path2 = list(elem2.iterancestors()) + [elem2]
    path1.reverse()
    path2.reverse()

    i = 0
    while i < len(path1) and i < len(path2) and path1[i] == path2[i]:
        i += 1

    # Distance = steps up to common ancestor + steps down to target
    hops = (len(path1) - i) + (len(path2) - i)

    # Small penalty for visual distance within the same parent
    sibling_penalty = 0.0
    if i > 0 and (len(path1) == i + 1) and (len(path2) == i + 1):
        try:
            parent = path1[i-1]
            idx1 = parent.index(path1[i])
            idx2 = parent.index(path2[i])
            sibling_penalty = abs(idx1 - idx2) * 0.1
        except ValueError:
            pass

    return float(hops) + min(sibling_penalty, 2.0)

def calculate_list_density(element: lxml.html.HtmlElement, max_depth: int = 3) -> float:
    """Checks if the element is part of a repeated list structure."""
    max_density = 0.0
    current = element

    for _ in range(max_depth):
        parent = current.getparent()
        if parent is None:
            break

        # Count direct children of parent that have the same tag as 'current'
        siblings = [child for child in parent if child.tag == current.tag]
        count = len(siblings)

        if count > 1:
            density_score = float(count) if count < 10 else 10.0 + np.log(count)
            if density_score > max_density:
                max_density = density_score

        current = parent

    return max_density

def calculate_context_features(
    element: lxml.html.HtmlElement,
    selectors: Dict[str, List[str]],
    root_tree,
    dropout: bool = False
) -> Dict[str, float]:
    """
    Calculates spatial features.
    Args:
        dropout (bool): If True, ignores selectors (simulates "not found yet" state).
    """
    features = {}

    # 1. Structural Density (Intrinsic feature, always calculated)
    features['max_sibling_density'] = calculate_list_density(element)

    # 2. Distance to known categories (Extrinsic context)
    # If dropout is True, we return default MAX distance to simulate a cold start.
    if not selectors or dropout:
        return features

    current_xpath = get_unique_xpath(element)

    for category, xpaths in selectors.items():
        if category == OTHER_CATEGORY or not xpaths:
            continue

        min_dist = 100.0 # Default "Far Away"

        for ref_xpath in xpaths:
            if ref_xpath == current_xpath: continue

            try:
                found = root_tree.xpath(ref_xpath)
                if found:
                    dist = get_dom_distance(element, found[0])
                    if dist < min_dist:
                        min_dist = dist
            except Exception:
                continue

        features[f'dist_to_closest_{category}'] = min_dist

    return features

def html_to_dataframe(
    html_content: str,
    selectors: Dict[str, List[str]],
    url: Optional[str] = None,
    augment_data: bool = False
) -> pd.DataFrame:
    """
    Extracts features from HTML.
    
    Args:
        augment_data (bool): If True, applies random dropout to context features 
                             to make the model robust to missing neighbors.
    """
    main_content = get_main_html_content_tag(html_content)
    if main_content is None:
        return pd.DataFrame()

    root_tree = main_content.getroottree()
    labeled_elements = set()
    positive_data = []

    # --- 1. Extract Positive Samples ---
    for category, xpaths in selectors.items():
        if category == OTHER_CATEGORY: continue

        for xpath in xpaths:
            try:
                found = main_content.xpath(xpath)
                if not found: found = root_tree.xpath(xpath)

                for elem in found:
                    if isinstance(elem, lxml.html.HtmlElement):
                        if normalize_tag(elem.tag) in UNWANTED_TAGS: continue

                        base_data = extract_element_features(elem, category=category, selectors=selectors)
                        if base_data:
                            base_data['xpath'] = get_unique_xpath(elem)

                            # --- DATA AUGMENTATION ---
                            # Logic: 50% of the time, provide full context.
                            # 50% of the time, pretend we have NO context (dropout=True).
                            # This teaches the model to work in both scenarios.
                            if augment_data and random.random() > 0.5:
                                context = calculate_context_features(elem, selectors, root_tree, dropout=True)
                            else:
                                context = calculate_context_features(elem, selectors, root_tree, dropout=False)

                            base_data.update(context)
                            positive_data.append(base_data)
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
            # For negatives, we also apply augmentation so the distribution matches positives
            if augment_data and random.random() > 0.5:
                context = calculate_context_features(elem, selectors, root_tree, dropout=True)
            else:
                context = calculate_context_features(elem, selectors, root_tree, dropout=False)

            data.update(context)
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

    # Fill defaults
    for col in ['class_str', 'id_str', 'tag', 'parent_tag', 'gparent_tag']:
        if col in df.columns: df[col] = df[col].fillna('')

    # Critical: Fill missing distances with LARGE number, not 0
    dist_cols = [c for c in df.columns if 'dist_to_' in c]
    for col in dist_cols:
        df[col] = df[col].fillna(100.0)

    if 'max_sibling_density' in df.columns:
        df['max_sibling_density'] = df['max_sibling_density'].fillna(0.0)

    return df.fillna(0)
