import lxml.html
import pandas as pd
from pathlib import Path
from typing import List, Optional

from train_model.train_model import load_trained_model
from train_model.process_data import _extract_element_features
from utils.constants import ALL_FEATURES, UNWANTED_TAGS
from utils.utils import normalize_tag, generate_selector_for_element



def predict_selectors(html_content: str, category: str, model_path: Optional[Path] = None) -> List[str]:
    """Predict selectors for a given category using the trained model."""
    
    if model_path is None:
        model_path = Path.cwd() / 'src' / 'models' / 'category_classifier.pkl'
    
    if not model_path.exists():
        print(f"⚠️ Model not found at {model_path}")
        return []
    
    try:
        # Load model
        model_data = load_trained_model(model_path)
        model = model_data['model']
        label_encoder = model_data['label_encoder']
        feature_names = model_data['feature_names']
        tag_encoder = model_data.get('tag_encoder')  # Get tag encoder if available
        
        # Extract features from all elements
        document = lxml.html.fromstring(html_content)
        all_features = []
        element_map = []
        
        for element in document.iter():
            tag_name = normalize_tag(element.tag)
            if tag_name in UNWANTED_TAGS:
                continue
            
            features = _extract_element_features(element, ALL_FEATURES)
            all_features.append(features)
            element_map.append(element)
        
        if not all_features:
            return []
        
        # Create DataFrame
        df = pd.DataFrame(all_features)
        
        # Encode tag and parent_tag features if tag_encoder is available
        if tag_encoder is not None:
            if 'tag' in df.columns:
                # Handle unknown tags by mapping to 'unknown'
                df['tag'] = df['tag'].apply(lambda x: x if x in tag_encoder.classes_ else 'unknown')
                df['tag'] = tag_encoder.transform(df['tag'])
            if 'parent_tag' in df.columns:
                df['parent_tag'] = df['parent_tag'].apply(lambda x: x if x in tag_encoder.classes_ else 'unknown')
                df['parent_tag'] = tag_encoder.transform(df['parent_tag'])
        
        # Filter to only include features that the model was trained on
        X = df[feature_names].fillna(0)
        predictions = model.predict(X)
        predicted_labels = label_encoder.inverse_transform(predictions)
        
        # Get probability scores for confidence
        probabilities = model.predict_proba(X)
        
        # Find elements predicted as the target category
        predicted_selectors = []
        for i, (pred_label, element, prob) in enumerate(zip(predicted_labels, element_map, probabilities)):
            if pred_label == category:
                # Get confidence for this prediction
                pred_idx = list(label_encoder.classes_).index(pred_label)
                confidence = prob[pred_idx]
                
                # Only include predictions with reasonable confidence (>30%)
                if confidence > 0.3:
                    try:
                        # Generate selector for this element
                        selector = generate_selector_for_element(element)
                        if selector:
                            predicted_selectors.append(selector)
                    except Exception:
                        continue
        print(predicted_selectors)
        print(f"🔮 Predicted {len(predicted_selectors)} potential matches for '{category}'")
        return predicted_selectors[:20]  # Limit to top 20 predictions
        
    except Exception as e:
        print(f"⚠️ Prediction error: {e}")
        return []
