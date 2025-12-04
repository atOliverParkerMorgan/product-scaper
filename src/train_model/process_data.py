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
    TEXT_CONTENT_DIFFICULTY = 'text_content_difficulty' # (Requires a library like textstat)

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


SYMBOL = r'\p{Sc}' # https://stackoverflow.com/questions/14169820/regular-expression-to-match-all-currency-symbols
SPACE = r'\s*'
NUMBER = r'\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?' # https://stackoverflow.com/questions/1547574/regex-for-prices

PRICE_REGEX = re.compile(rf"""
    (
        (?:{SYMBOL}{SPACE}{NUMBER})|
        (?:{NUMBER}{SPACE}{SYMBOL})
    )
""", re.VERBOSE | re.UNICODE)

class FeatureEncoders:
    """Container for all feature encoders used in the pipeline."""
    def __init__(self):
        self.tag_encoder = LabelEncoder()
        self.category_encoder = LabelEncoder()
        self.selector_encoder = LabelEncoder()
        # For text-based features, we'll use frequency-based encoding or keep as hash
        # since text content varies too much for label encoding
        
    def save(self, filepath: Path):
        """Save encoders to disk for reuse during inference."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(filepath: Path):
        """Load encoders from disk."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)

# Common HTML tags (for initialization and unknown tag handling)
COMMON_TAGS = [
    'div', 'span', 'p', 'a', 'img', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
    'ul', 'ol', 'li', 'table', 'tr', 'td', 'th', 'form', 'input',
    'button', 'select', 'option', 'textarea', 'label', 'nav', 'header',
    'footer', 'section', 'article', 'aside', 'main', 'body', 'html', 'unknown'
]

def _normalize_tag(tag_name) -> str:
    """Normalize tag name to string, handling special cases."""
    if not tag_name or not hasattr(tag_name, 'lower'):
        return 'unknown'
    try:
        return str(tag_name).lower()
    except:
        return 'unknown'

def _count_unique_tags(tag_list: List[str]) -> int:
    """Count unique tags in a list."""
    return len(set(tag_list)) if tag_list else 0


def _extract_element_features(element: lxml.html.HtmlElement, requested_features: List[Features] = ALL_FEATURES, features: Dict[str, Any]= {}, raw_values: bool = True) -> Dict[str, Any]:
    """
    Extracts a set of features for a *single* lxml element.
    """
    
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
                # Use text length as proxy (text content itself not useful for ML)
                # Already have TEXT_CONTENT_LENGTH, so use character density
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
                # Use textstat to calculate reading difficulty (numerical)
                try:
                    features[feat.value] = textstat.flesch_reading_ease(text) if text else 0
                except:
                    features[feat.value] = 0

            elif feat == Features.PARENT_TAG:
                # Store normalized parent tag - will be encoded later
                features[feat.value] = _normalize_tag(parent.tag) if parent is not None else 'unknown'

            elif feat == Features.NUM_CHILDREN:
                features[feat.value] = len(element)
            
            elif feat == Features.NUM_SIBLINGS:
                # Count elements at the same level (excluding self)
                if parent is not None:
                    features[feat.value] = len(parent) - 1 
                else:
                    features[feat.value] = 0
            
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
                # Binary: is external link (has domain) or internal/no link
                href = element.get('href')
                if href:
                    try:
                        parsed = urlparse(href)
                        features[feat.value] = 1 if parsed.netloc else 0
                    except:
                        features[feat.value] = 0
                else:
                    features[feat.value] = 0
            
            elif feat == Features.IS_IMAGE:
                features[feat.value] = 1 if element.tag == 'img' else 0
            
            elif feat == Features.IMAGE_SRC:
                # Binary: has image source or not
                if element.tag == 'img':
                    src = element.get('src')
                    features[feat.value] = 1 if src else 0
                else:
                    features[feat.value] = 0

        except Exception as e:
            # Safely handle errors (e.g., no parent)
            print(f"Warning: Could not extract feature '{feat.value}': {e}")
            features[feat.value] = 0  # Default to 0 for numerical consistency
            
    return features


def extract_features_from_html(html_content: str, selectors: dict, features_to_extract: List[Features] = ALL_FEATURES, unwanted_tags:List[str]=UNWANTED_TAGS, encoders: Optional[FeatureEncoders] = None) -> tuple[List[Dict[str, Any]], FeatureEncoders]:
    """Extract features from HTML content for each category using provided CSS selectors.

    Args:
        html_content (str): The HTML content of the page.
        selectors (dict): Mapping from category to list of CSS selectors.
        features_to_extract (list): List of Features enum members to extract.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, where each dict is a "row"
                                containing all extracted features for one element.
    """
    try:
        document = lxml.html.fromstring(html_content)
    except Exception as e:
        print(f"Error: lxml failed to parse HTML. {e}")
        return [], FeatureEncoders()

    all_features_data = []
    
    # Initialize encoders if not provided
    if encoders is None:
        encoders = FeatureEncoders()
    
    # Collect all categorical values for fitting encoders
    all_tags = []
    all_parent_tags = []
    all_categories = []
    all_selectors = []
    
    # Track all selected elements to exclude them from 'other' category
    selected_elements = set()

    for category, selector_list in selectors.items():
        for selector in selector_list:
            try:
                elements = document.cssselect(selector)
                
                if not elements:
                    print(f"Warning: Selector '{selector}' for category '{category}' matched 0 elements.")
                    continue
                    
                for element in elements:
                    # Track this element as selected
                    selected_elements.add(element)
                    
                    # Get all features for this one element
                    feature_row = {}
                    feature_row['Category'] = category  # Keep raw category name
                    feature_row = _extract_element_features(element, features_to_extract, feature_row)
                    feature_row['Selector'] = selector  # Keep raw selector
                    
                    all_features_data.append(feature_row)
                    
                    # Collect categorical values
                    all_categories.append(category)
                    all_selectors.append(selector)
                    if 'tag' in feature_row:
                        all_tags.append(feature_row['tag'])
                    if 'parent_tag' in feature_row:
                        all_parent_tags.append(feature_row['parent_tag'])
                    
            except XPathEvalError:
                print(f"Error: Invalid CSS selector '{selector}'. Skipping.")
            except Exception as e:
                print(f"Error processing selector '{selector}': {e}")

    # Collect all other elements that weren't selected
    for element in document.iter():
        # Skip if this element was already selected
        if element in selected_elements:
            continue
            
        # Skip unwanted tags (script, style, etc.)
        if element.tag in unwanted_tags:
            continue
        
        # Skip if tag is not a string (comments, etc.)
        if not isinstance(element.tag, str):
            continue

        feature_row = {}
        feature_row['Category'] = OTHER_CATEGORY  # Keep raw category
        feature_row = _extract_element_features(element, features_to_extract, feature_row)
        feature_row['Selector'] = 'none'  # Use string for consistency
        
        all_features_data.append(feature_row)
        
        # Collect categorical values
        all_categories.append(OTHER_CATEGORY)
        all_selectors.append('none')
        if 'tag' in feature_row:
            all_tags.append(feature_row['tag'])
        if 'parent_tag' in feature_row:
            all_parent_tags.append(feature_row['parent_tag'])

    # Fit encoders on collected categorical data
    # Combine all tags (from tag and parent_tag fields) with common tags
    all_unique_tags = list(set(all_tags + all_parent_tags + COMMON_TAGS))
    if all_unique_tags:
        encoders.tag_encoder.fit(all_unique_tags)
    if all_categories:
        encoders.category_encoder.fit(list(set(all_categories)))
    if all_selectors:
        encoders.selector_encoder.fit(list(set(all_selectors)))
    
    # Encode tag and selector features (keep Category as readable string)
    for row in all_features_data:
        if 'tag' in row:
            row['tag'] = encoders.tag_encoder.transform([row['tag']])[0]
        if 'parent_tag' in row:
            row['parent_tag'] = encoders.tag_encoder.transform([row['parent_tag']])[0]
        if 'Selector' in row:
            row['Selector'] = encoders.selector_encoder.transform([row['Selector']])[0]
        # Note: Category is kept as string for readability in CSV

    return all_features_data, encoders

def selector_data_to_csv(data_domain_dir: Path) -> None:
    """
    Converts stored selector data (YAML + HTML) into a rich 'data.csv' file
    containing all extracted features for each selected element.
    
    Args:
        data_domain_dir (Path): Path to the domain-specific data directory
                                (e.g., 'data/my_site').
    """

    yaml_path = data_domain_dir / 'selectors.yaml'
    html_path = data_domain_dir / 'page.html'
    data_csv_path = data_domain_dir / 'data.csv'
    encoders_path = data_domain_dir / 'encoders.pkl'

    if not yaml_path.exists() or not html_path.exists():
        raise FileNotFoundError(f"Error: Missing 'selectors.yaml' or 'page.html' in {data_domain_dir}")
        

    try:
        with open(yaml_path, 'r', encoding='utf-8') as yaml_file:
            selectors = yaml.safe_load(yaml_file)
    except Exception as e:
        print(f"Error loading YAML: {e}")
        return

    try:
        with open(html_path, 'r', encoding='utf-8') as html_file:
            html_content = html_file.read()
    except Exception as e:
        print(f"Error loading HTML: {e}")
        return


    print(f"Extracting features from {html_path}...")
    extracted_data, encoders = extract_features_from_html(html_content, selectors)

    if not extracted_data:
        print("No data was extracted. 'data.csv' will not be created.")
        return
    
    # Save encoders for later use during inference
    encoders.save(encoders_path)
    print(f"Saved feature encoders to {encoders_path}")

    try:
        headers = list(extracted_data[0].keys())

        with open(data_csv_path, 'w', newline='', encoding='utf-8') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=headers)
            
            # Write the header row (e.g., 'Category', 'Selector', 'tag', 'text_content'...)
            writer.writeheader()
            
            # Write all the feature rows
            writer.writerows(extracted_data)
        
        print(f"Successfully saved {len(extracted_data)} feature rows to {data_csv_path}")

    except Exception as e:
        print(f"Error writing CSV file: {e}")


def data_to_csv(project_root: Path = Path.cwd()) -> None:
    """
    Process all domain-specific data directories and generate 'data.csv' files
    with extracted features for each.
    Args:
        project_root (Path): Root directory containing 'data' folder with domain subfolders.
    """

    for domain_dir in (project_root / 'src' / 'data').iterdir():
        if domain_dir.is_dir():
            selector_data_to_csv(domain_dir)
