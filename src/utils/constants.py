"""Constants used across the product scraper project."""

try:
    import regex as re
except ImportError:
    import re
from enum import Enum

# --- Feature Definitions ---

class Features(Enum):
    """Enumeration of all possible features we can extract for an element."""
    
    TAG = "tag"
    ID_NAME = 'id_name'
    CLASS_NAME = 'class_name'
    ALL_ATTRIBUTES = 'all_attributes'
    
    # Text Content Features
    TEXT_CONTENT = 'text_content'

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
    HAS_HREF = 'has_href'
    HREF_DOMAIN = 'href_domain'
    IS_IMAGE = 'is_image'
    IMAGE_SRC = 'image_src'
    
    #  TODO: Future features
    IS_TAG_A_PART_OF_A_LINK = 'is_tag_a_part_of_a_link'
    DOM_DISTANCE_FROM_NEAREST_IMAGE = 'dom_distance_from_nearest_image'


ALL_FEATURES = [feat for feat in Features]

# Tags to exclude from feature extraction
UNWANTED_TAGS = ['script', 'style', 'meta', 'link', 'noscript', 'iframe', 'head', 'input']

# Common HTML tags for encoder initialization
COMMON_TAGS = [
    'div', 'span', 'p', 'a', 'img', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 
    'ul', 'ol', 'li', 'table', 'tr', 'td', 'th', 'form', 'input', 'button', 
    'select', 'option', 'textarea', 'label', 'nav', 'header', 'footer', 
    'section', 'article', 'aside', 'main', 'body', 'html', 'unknown'
]

# Category for unselected elements
OTHER_CATEGORY = 'other'

# Regex for detecting price patterns
PRICE_REGEX = re.compile(
    r'(\p{Sc}\s*\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?|\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?\s*\p{Sc})', 
    re.UNICODE
)

# Model training constants
RANDOM_STATE = 42
EXCLUDED_FEATURES = ['Category']
DATA_NAME = 'data.csv'
