from train_model.process_data import html_to_dataframe, get_main_html_content_tag
from utils.features import UNWANTED_TAGS, NON_TRAINIG_FEATURES, TARGET_FEATURE
from utils.utils import get_unique_xpath, normalize_tag
import lxml.html
import numpy as np
from typing import Any, Dict, List


def predict_selectors(model: Dict[str, Any], html_content: str, category: str) -> List[Dict[str, Any]]:
        
    pipeline = model['pipeline']
    label_encoder = model['label_encoder']
    
    try:
        target_class_idx = label_encoder.transform([category])[0]
    except ValueError:
        raise ValueError(f"Category '{category}' was not seen during training. Available: {label_encoder.classes_}")
    
    X = html_to_dataframe(html_content, selectors=None)
    
    if X.empty:
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
    
    if TARGET_FEATURE in X.columns:
        X = X.drop(columns=NON_TRAINIG_FEATURES)
    
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