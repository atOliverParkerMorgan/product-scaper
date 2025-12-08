import pickle
import lxml.html
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict

# Import the feature extractor from your process_data script
# Ensure this import works based on your folder structure
from train_model.process_data import _extract_element_features

def get_unique_xpath(element) -> str:
    """
    Generate a robust XPath for an lxml element.
    """
    try:
        tree = element.getroottree()
        return tree.getpath(element)
    except Exception:
        return ""

def predict_selectors(
    html_content: str, 
    category: str, 
    model_path: Optional[Path] = None
) -> List[dict]:
    """
    Predict selectors for a given category using the trained Pipeline.
    """
    if model_path is None:
        model_path = Path.cwd() / 'src' / 'models' / 'category_classifier.pkl'
    
    if not model_path.exists():
        print(f"❌ Model not found at {model_path}")
        return []

    # 1. Load the Model Artifact
    with open(model_path, 'rb') as f:
        model_artifact = pickle.load(f)
    
    # Extract components from the artifact dictionary
    pipeline = model_artifact['pipeline']
    label_encoder = model_artifact['label_encoder']
    
    # Check if the requested category exists in the model
    try:
        # Transform category string to integer (e.g., 'price' -> 2)
        target_class_idx = label_encoder.transform([category])[0]
    except ValueError:
        print(f"⚠️ Category '{category}' was not seen during training. Available: {label_encoder.classes_}")
        return []

    # 2. Parse HTML & Extract Features
    # We must replicate the iteration logic from process_data EXACTLY 
    # to ensure 'elements' list aligns with 'features' list.
    try:
        tree = lxml.html.fromstring(html_content)
    except Exception as e:
        print(f"Error parsing HTML: {e}")
        return []

    elements = []
    feature_rows = []

    for elem in tree.iter():
        # CRITICAL: Must skip non-element tags (Comments, etc.) just like process_data.py
        if not isinstance(elem.tag, str):
            continue
            
        # Optimization: Skip empty structural tags (must match process_data logic)
        try:
            if not elem.text_content().strip() and not elem.attrib:
                continue
        except Exception:
            continue

        # Extract features using the function from process_data.py
        # We pass 'other' as category since we are predicting
        features = _extract_element_features(elem, category='other')
        
        if features:
            feature_rows.append(features)
            elements.append(elem)

    if not feature_rows:
        print("⚠️ No valid elements found in HTML.")
        return []

    # 3. Create DataFrame
    # The pipeline expects a DataFrame with columns like 'class_str', 'tag', etc.
    X = pd.DataFrame(feature_rows)
    
    # Ensure text columns are strings (fixes NaN issues)
    text_cols = ['class_str', 'id_str', 'tag', 'parent_tag']
    for col in text_cols:
        if col in X.columns:
            X[col] = X[col].astype(str).replace('nan', '')

    # 4. Predict using the Pipeline
    try:
        # Get probabilities for all classes
        # Pipeline handles all scaling and encoding automatically!
        probabilities = pipeline.predict_proba(X)
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return []

    # 5. Filter Candidates
    candidates = []
    
    # probabilities is an array of shape (n_samples, n_classes)
    # We want the column corresponding to our target_class_idx
    target_probs = probabilities[:, target_class_idx]
    
    # Filter based on threshold
    THRESHOLD = 0.4  # Adjust sensitivity here
    
    # Get indices where probability > threshold
    match_indices = np.where(target_probs > THRESHOLD)[0]
    
    for i in match_indices:
        prob = target_probs[i]
        element = elements[i]
        
        # Generate preview
        text_content = element.text_content().strip()
        preview = text_content[:50] + "..." if len(text_content) > 50 else text_content
        
        # Generate XPath for reliable highlighting
        xpath = get_unique_xpath(element)
        
        candidates.append({
            'index': i,  # Keep index as fallback
            'xpath': xpath,
            'confidence': float(prob),
            'preview': preview,
            'tag': element.tag,
            'class': element.get('class', '')
        })

    # Sort by confidence descending
    candidates.sort(key=lambda x: x['confidence'], reverse=True)
    
    if candidates:
        print(f"🔮 Found {len(candidates)} candidates for '{category}' (Top confidence: {candidates[0]['confidence']:.2f})")
    else:
        print(f"🔮 No candidates found for '{category}' above threshold {THRESHOLD}")

    return candidates[:15] # Return top 15