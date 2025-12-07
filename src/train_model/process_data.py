from pathlib import Path
import csv
import yaml
import lxml.html
from lxml.etree import XPathEvalError
from enum import Enum
import regex as re
from typing import List, Dict, Any, Optional
import textstat
from urllib.parse import urlparse
from sklearn.preprocessing import LabelEncoder
import pickle
import os
# --- Feature Definitions ---

class Features(Enum):
    """Enumeration of all possible features we can extract for an element."""
    
    TAG = "tag"
    ID_NAME = 'id_name'
    CLASS_NAME = 'class_name'
    ALL_ATTRIBUTES = 'all_attributes'
    
    # Text Content Features
    TEXT_CONTENT = 'text_content' # TODO: convert to transformed features only?

    TEXT_CONTENT_LENGTH = 'text_content_length'
    TEXT_CONTENT_NUM_WORDS = 'text_content_num_words'
    TEXT_CONTENT_NUM_DIGITS = 'text_content_num_digits'
    TEXT_CONTENT_PRICE_FORMAT_PROBABILITY = 'text_content_price_format'
    TEXT_CONTENT_DIFFICULTY = 'text_content_difficulty' 

    PARENT_TAG = 'parent_tag'
    NUM_CHILDREN = 'num_children'
    NUM_SIBLINGS = 'num_siblings' 
    POSITION_IN_SIBLINGS = 'position_in_siblings' 
    CHILD_TAGS = 'child_tags'
    SIBLING_TAGS = 'sibling_tags' 
    DOM_DISTANCE_FROM_ROOT = 'dom_distance_from_root'
    
    # --- Other Obvious Feature Ideas ---
    HAS_HREF = 'has_href' # Boolean: is it a link?
    HREF_DOMAIN = 'href_domain' # The netloc of the link
    IS_IMAGE = 'is_image' # Boolean: is it an img tag?
    IMAGE_SRC = 'image_src' # The 'src' attribute
    
    #  TODO: Future features
    IS_TAG_A_PART_OF_A_LINK = 'is_tag_a_part_of_a_link'
    DOM_DISTANCE_FROM_NEAREST_IMAGE = 'dom_distance_from_nearest_image'

ALL_FEATURES = [feat for feat in Features]
UNWANTED_TAGS = ['script', 'style', 'meta', 'link', 'noscript', 'iframe', 'head', 'input',]

OTHER_CATEGORY = 'other'


PRICE_REGEX = re.compile(r'(\p{Sc}\s*\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?|\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?\s*\p{Sc})', re.UNICODE)

class FeatureEncoders:
    """Container for all feature encoders used in the pipeline."""
    def __init__(self):
        self.tag_encoder = LabelEncoder()
        self.category_encoder = LabelEncoder()
        self.selector_encoder = LabelEncoder()
        
    def save(self, filepath: Path):
        """Save encoders to disk for reuse during inference."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(filepath: Path):
        """Load encoders from disk."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)

COMMON_TAGS = ['div', 'span', 'p', 'a', 'img', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'ul', 'ol', 'li', 
               'table', 'tr', 'td', 'th', 'form', 'input', 'button', 'select', 'option', 'textarea', 
               'label', 'nav', 'header', 'footer', 'section', 'article', 'aside', 'main', 'body', 'html', 'unknown']

def _normalize_tag(tag_name) -> str:
    """Normalize tag name to string."""
    if not tag_name or not hasattr(tag_name, 'lower'):
        return 'unknown'
    return str(tag_name).lower()

def _count_unique_tags(tag_list: List[str]) -> int:
    """Count unique tags."""
    return len(set(tag_list)) if tag_list else 0


def _extract_element_features(element: lxml.html.HtmlElement, requested_features: List[Features] = ALL_FEATURES, features: Dict[str, Any] = None) -> Dict[str, Any]:
    """Extract features for a single lxml element."""
    if features is None:
        features = {}
    
    try:
        text = element.text_content().strip()
        parent = element.getparent()
    except ValueError:
        text = ''
        parent = None
        
    for feat in requested_features:
        try:
            if feat == Features.TAG:
                # Store normalized tag name - will be encoded by LabelEncoder later
                features[feat.value] = _normalize_tag(element.tag)
            
            elif feat == Features.ID_NAME:
                # Binary: has ID or not (IDs are usually unique, not good for encoding)
                features[feat.value] = 1 if element.get('id') else 0
            
            elif feat == Features.CLASS_NAME:
                # Count number of classes (better than encoding class names)
                class_attr = element.get('class')
                features[feat.value] = len(class_attr.split()) if class_attr else 0
            
            elif feat == Features.ALL_ATTRIBUTES:
                # Count number of attributes
                features[feat.value] = len(element.attrib)
            
            elif feat == Features.TEXT_CONTENT:
                features[feat.value] = len(text) / max(len(text.split()), 1) if text else 0
            
            elif feat == Features.TEXT_CONTENT_LENGTH:
                features[feat.value] = len(text)
            
            elif feat == Features.TEXT_CONTENT_NUM_WORDS:
                features[feat.value] = len(text.split())
            
            elif feat == Features.TEXT_CONTENT_NUM_DIGITS:
                features[feat.value] = sum(c.isdigit() for c in text)
            
            elif feat == Features.TEXT_CONTENT_PRICE_FORMAT_PROBABILITY:
                features[feat.value] = 1 if PRICE_REGEX.search(text) else 0
            
            elif feat == Features.TEXT_CONTENT_DIFFICULTY:
                try:
                    features[feat.value] = textstat.flesch_reading_ease(text) if text else 0
                except Exception:
                    features[feat.value] = 0

            elif feat == Features.PARENT_TAG:
                features[feat.value] = _normalize_tag(parent.tag) if parent is not None else 'unknown'

            elif feat == Features.NUM_CHILDREN:
                features[feat.value] = len(element)
            
            elif feat == Features.NUM_SIBLINGS:
                features[feat.value] = len(parent) - 1 if parent is not None else 0
            
            elif feat == Features.POSITION_IN_SIBLINGS:
                features[feat.value] = element.getparent().index(element) if parent is not None else 0

            elif feat == Features.CHILD_TAGS:
                # Count unique child tags
                child_tags = [child.tag for child in element.iterchildren(tag='*')]
                features[feat.value] = _count_unique_tags(child_tags)
            
            elif feat == Features.SIBLING_TAGS:
                # Count unique sibling tags
                if parent is not None:
                    sibling_tags = [child.tag for child in parent.iterchildren(tag='*') if child != element]
                    features[feat.value] = _count_unique_tags(sibling_tags)
                else:
                    features[feat.value] = 0

            elif feat == Features.DOM_DISTANCE_FROM_ROOT:
                features[feat.value] = len(element.xpath('ancestor::*'))
            
            elif feat == Features.HAS_HREF:
                features[feat.value] = 1 if element.get('href') is not None else 0
            
            elif feat == Features.HREF_DOMAIN:
                href = element.get('href')
                try:
                    features[feat.value] = 1 if href and urlparse(href).netloc else 0
                except Exception:
                    features[feat.value] = 0
            
            elif feat == Features.IS_IMAGE:
                features[feat.value] = 1 if element.tag == 'img' else 0
            
            elif feat == Features.IMAGE_SRC:
                features[feat.value] = 1 if element.tag == 'img' and element.get('src') else 0

        except Exception as e:
            print(f"Warning: Could not extract feature '{feat.value}': {e}")
            features[feat.value] = 0
            
    return features


def extract_features_from_html(html_content: str, selectors: dict, features_to_extract: List[Features] = ALL_FEATURES, 
                               unwanted_tags: List[str] = None, encoders: Optional[FeatureEncoders] = None) -> tuple[List[Dict[str, Any]], FeatureEncoders]:
    """Extract features from HTML content for each category using provided CSS selectors."""
    if unwanted_tags is None:
        unwanted_tags = UNWANTED_TAGS
    try:
        document = lxml.html.fromstring(html_content)
    except Exception as e:
        print(f"Error parsing HTML: {e}")
        return [], FeatureEncoders()

    all_features_data = []
    encoders = encoders or FeatureEncoders()
    all_tags, all_parent_tags, all_categories, all_selectors = [], [], [], []
    selected_elements = set()

    for category, selector_list in selectors.items():
        for selector in selector_list:
            try:
                elements = document.cssselect(selector)
                if not elements:
                    print(f"Warning: Selector '{selector}' matched 0 elements.")
                    continue
                    
                for element in elements:
                    selected_elements.add(element)
                    feature_row = {'Category': category}
                    feature_row = _extract_element_features(element, features_to_extract, feature_row)
                    feature_row['Selector'] = selector
                    all_features_data.append(feature_row)
                    
                    all_categories.append(category)
                    all_selectors.append(selector)
                    if 'tag' in feature_row:
                        all_tags.append(feature_row['tag'])
                    if 'parent_tag' in feature_row:
                        all_parent_tags.append(feature_row['parent_tag'])
                    
            except XPathEvalError:
                print(f"Error: Invalid selector '{selector}'.")
            except Exception as e:
                print(f"Error: {e}")

    for element in document.iter():
        if element in selected_elements or element.tag in unwanted_tags or not isinstance(element.tag, str):
            continue

        feature_row = {'Category': OTHER_CATEGORY}
        feature_row = _extract_element_features(element, features_to_extract, feature_row)
        feature_row['Selector'] = 'none'
        all_features_data.append(feature_row)
        
        all_categories.append(OTHER_CATEGORY)
        all_selectors.append('none')
        if 'tag' in feature_row:
            all_tags.append(feature_row['tag'])
        if 'parent_tag' in feature_row:
            all_parent_tags.append(feature_row['parent_tag'])

    all_unique_tags = list(set(all_tags + all_parent_tags + COMMON_TAGS))
    if all_unique_tags:
        encoders.tag_encoder.fit(all_unique_tags)
    if all_categories:
        encoders.category_encoder.fit(list(set(all_categories)))
    if all_selectors:
        encoders.selector_encoder.fit(list(set(all_selectors)))
    
    for row in all_features_data:
        if 'tag' in row:
            row['tag'] = encoders.tag_encoder.transform([row['tag']])[0]
        if 'parent_tag' in row:
            row['parent_tag'] = encoders.tag_encoder.transform([row['parent_tag']])[0]
        if 'Selector' in row:
            row['Selector'] = encoders.selector_encoder.transform([row['Selector']])[0]

    return all_features_data, encoders

def selector_data_to_csv(data_domain_dir: Path) -> None:
    """Convert stored selector data (YAML + HTML) into 'data.csv' with extracted features."""
    yaml_path = data_domain_dir / 'selectors.yaml'
    html_path = data_domain_dir / 'page.html'
    data_csv_path = data_domain_dir / 'data.csv'
    encoders_path = data_domain_dir / 'encoders.pkl'

    if not yaml_path.exists() or not html_path.exists():
        raise FileNotFoundError(f"Missing files in {data_domain_dir}")

    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            selectors = yaml.safe_load(f)
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    print(f"Extracting features from {html_path}...")
    extracted_data, encoders = extract_features_from_html(html_content, selectors)

    if not extracted_data:
        print("No data extracted.")
        return
    
    encoders.save(encoders_path)
    print(f"Saved encoders to {encoders_path}")

    try:
        with open(data_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(extracted_data[0].keys()))
            writer.writeheader()
            writer.writerows(extracted_data)
        print(f"Saved {len(extracted_data)} rows to {data_csv_path}")
    except Exception as e:
        print(f"Error writing CSV: {e}")


def data_to_csv(project_root: Path = Path.cwd()) -> None:
    """Process all domain directories and generate 'data.csv' files."""
    if not (project_root / 'src' / 'data').exists():
        os.makedirs(project_root / 'src' / 'data')

    for domain_dir in (project_root / 'src' / 'data').iterdir():
        if domain_dir.is_dir():
            selector_data_to_csv(domain_dir)
