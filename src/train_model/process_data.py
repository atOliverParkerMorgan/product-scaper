from pathlib import Path
import csv
import yaml
import lxml.html
from lxml.etree import XPathEvalError
from enum import Enum
import regex as re
from typing import List, Dict, Any
import textstat

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
        (?:{SYMBOL}{SPACE}{NUMBER})
                
        (?:{NUMBER}{SPACE}{SYMBOL})
    )
""", re.VERBOSE | re.UNICODE)


def _extract_element_features(element: lxml.html.HtmlElement, requested_features: List[Features] = ALL_FEATURES, features: Dict[str, Any]= {}) -> Dict[str, Any]:
    """
    Extracts a set of features for a *single* lxml element.
    """
    
    try:
        text = element.text_content().strip()
        parent = element.getparent()
    
    except ValueError:
        # for some elements (<input/>...), text_content() may fail
        text = ''
        parent = None
        
    for feat in requested_features:
        try:
            if feat == Features.TAG:
                features[feat.value] = element.tag
            
            elif feat == Features.ID_NAME:
                features[feat.value] = element.get('id')
            
            elif feat == Features.CLASS_NAME:
                features[feat.value] = element.get('class')
            
            elif feat == Features.ALL_ATTRIBUTES:
                features[feat.value] = dict(element.attrib)
            
            elif feat == Features.TEXT_CONTENT:
                features[feat.value] = text
            
            elif feat == Features.TEXT_CONTENT_LENGTH:
                features[feat.value] = len(text)
            
            elif feat == Features.TEXT_CONTENT_NUM_WORDS:
                features[feat.value] = len(text.split())
            
            elif feat == Features.TEXT_CONTENT_NUM_DIGITS:
                features[feat.value] = sum(c.isdigit() for c in text)
            
            elif feat == Features.TEXT_CONTENT_PRICE_FORMAT_PROBABILITY:
                features[feat.value] = 1 if PRICE_REGEX.search(text) else 0

            elif feat == Features.PARENT_TAG:
                features[feat.value] = parent.tag if parent is not None else None

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
                features[feat.value] = [child.tag for child in element.iterchildren(tag='*')]

            elif feat == Features.DOM_DISTANCE_FROM_ROOT:
                features[feat.value] = len(element.xpath('ancestor::*'))
            
            elif feat == Features.HAS_HREF:
                features[feat.value] = element.get('href') is not None
            
            elif feat == Features.IS_IMAGE:
                features[feat.value] = element.tag == 'img'
            
            elif feat == Features.IMAGE_SRC:
                if element.tag == 'img':
                    features[feat.value] = element.get('src')
                else:
                    features[feat.value] = None # Ensure key exists if requested

        except Exception as e:
            # Safely handle errors (e.g., no parent)
            print(f"Warning: Could not extract feature '{feat.value}': {e}")
            features[feat.value] = None # Add key as None so CSV columns match
            
    return features


def extract_features_from_html(html_content: str, selectors: dict, features_to_extract: List[Features] = ALL_FEATURES, unwanted_tags:List[str]=UNWANTED_TAGS) -> List[Dict[str, Any]]:
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
        return []

    all_features_data = []

    all_selectors = [sel for sel_list in selectors.values() for sel in sel_list]
    for category, selector_list in selectors.items():
        for selector in selector_list:
            try:
                elements = document.cssselect(selector)
                
                if not elements:
                    print(f"Warning: Selector '{selector}' for category '{category}' matched 0 elements.")
                    continue
                    
                for element in elements:
                    # Get all features for this one element
                    feature_row = {}
                    feature_row['Category'] = category
                    feature_row = _extract_element_features(element, features_to_extract, feature_row)
                    feature_row['Selector'] = selector
                    
                    all_features_data.append(feature_row)
                    
            except XPathEvalError:
                print(f"Error: Invalid CSS selector '{selector}'. Skipping.")
            except Exception as e:
                print(f"Error processing selector '{selector}': {e}")

    # all other element have the category 'other'
    other_elements = set(document.iter()) - {el for sel_list in selectors.values() for sel in sel_list for el in document.cssselect(sel)}
    for element in other_elements:
        if element.tag in unwanted_tags:
            continue

        feature_row = {}
        feature_row['Category'] = OTHER_CATEGORY
        feature_row = _extract_element_features(element, features_to_extract, feature_row)
        feature_row['Selector'] = None
        
        all_features_data.append(feature_row)

    return all_features_data

def selector_data_to_csv(data_domain_dir: Path) -> None:
    """
    Converts stored selector data (YAML + HTML) into a rich 'train.csv' file
    containing all extracted features for each selected element.
    
    Args:
        data_domain_dir (Path): Path to the domain-specific data directory
                                (e.g., 'data/my_site').
    """

    yaml_path = data_domain_dir / 'selectors.yaml'
    html_path = data_domain_dir / 'page.html'
    train_csv_path = data_domain_dir / 'train.csv'

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
    extracted_data = extract_features_from_html(html_content, selectors)

    if not extracted_data:
        print("No data was extracted. 'train.csv' will not be created.")
        return

    try:
        headers = list(extracted_data[0].keys())

        with open(train_csv_path, 'w', newline='', encoding='utf-8') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=headers)
            
            # Write the header row (e.g., 'Category', 'Selector', 'tag', 'text_content'...)
            writer.writeheader()
            
            # Write all the feature rows
            writer.writerows(extracted_data)
        
        print(f"Successfully saved {len(extracted_data)} feature rows to {train_csv_path}")

    except Exception as e:
        print(f"Error writing CSV file: {e}")


def data_to_csv(project_root: Path = Path.cwd()) -> None:
    """
    Process all domain-specific data directories and generate 'train.csv' files
    with extracted features for each.
    Args:
        project_root (Path): Root directory containing 'data' folder with domain subfolders.
    """

    for domain_dir in (project_root / 'data').iterdir():
        if domain_dir.is_dir():
            selector_data_to_csv(domain_dir)
