import pickle
import lxml.html
import numpy as np
from pathlib import Path
from typing import List, Optional
from train_model.process_data import html_to_dataframe
from utils.utils import get_unique_xpath


def predict_selectors(
    html_content: str, 
    category: str, 
    model_path: Optional[Path] = None,
    only_main_content: bool = True,
    threshold: float = 0.0
) -> List[dict]:
    """
    Predict selectors for a given category using the trained Pipeline.
    """
    if model_path is None:
        model_path = Path.cwd() / 'models' / 'category_classifier.pkl'
    
    if not model_path.exists():
        print(f"❌ Model not found at {model_path}")
        return []

    with open(model_path, 'rb') as f:
        model_artifact = pickle.load(f)
    
    # Extract components from the artifact dictionary
    pipeline = model_artifact['pipeline']
    label_encoder = model_artifact['label_encoder']
    
    try:
        # Transform category string to integer (e.g., 'price' -> 2)
        target_class_idx = label_encoder.transform([category])[0]
    except ValueError:
        print(f"⚠️ Category '{category}' was not seen during training. Available: {label_encoder.classes_}")
        return []

 
    X = html_to_dataframe(html_content, selectors=None)
    
    if X.empty:
        print("⚠️ No valid elements found in HTML.")
        return []
    
    # Get the parsed tree to retrieve element references
    try:
        tree = lxml.html.fromstring(html_content)
    except Exception as e:
        print(f"Error parsing HTML: {e}")
        return []
    
    # Rebuild element list by iterating the same way html_to_dataframe does
    from train_model.process_data import get_main_html_content_tag, UNWANTED_TAGS, normalize_tag
    
    main_content = get_main_html_content_tag(html_content) if only_main_content else tree
    if main_content is None:
        main_content = tree
    
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
    
    if len(elements) != len(X):
        print(f"⚠️ Mismatch between elements ({len(elements)}) and features ({len(X)})")
        return []

    # Remove the Category column as it's the target
    if 'Category' in X.columns:
        X = X.drop(columns=['Category'])

    try:
        predictions = pipeline.predict(X)
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return []

    # 4. Filter Candidates where prediction matches target category
    candidates = []
    
    # Get indices where prediction equals our target_class_idx
    match_indices = np.where(predictions == target_class_idx)[0]
    
    for i in match_indices:
        element = elements[i]
        
        # Generate preview
        text_content = element.text_content().strip()
        preview = text_content[:50] + "..." if len(text_content) > 50 else text_content
        
        # Generate XPath for reliable highlighting
        xpath = get_unique_xpath(element)
        
        candidates.append({
            'index': i,  # Keep index as fallback
            'xpath': xpath,
            'preview': preview,
            'tag': element.tag,
            'class': element.get('class', '')
        })

    if candidates:
        print(f"🔮 Found {len(candidates)} candidates for '{category}'")
    else:
        print(f"🔮 No candidates found for '{category}'")

    return candidates
