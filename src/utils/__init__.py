"""Utilities module for product scraper."""

from .constants import (
    ALL_FEATURES,
    COMMON_TAGS,
    DATA_NAME,
    EXCLUDED_FEATURES,
    Features,
    OTHER_CATEGORY,
    PRICE_REGEX,
    RANDOM_STATE,
    UNWANTED_TAGS,
)
from .utils import (
    count_unique_tags,
    generate_selector_for_element,
    normalize_tag,
)

__all__ = [
    # Constants
    'ALL_FEATURES',
    'COMMON_TAGS',
    'DATA_NAME',
    'EXCLUDED_FEATURES',
    'Features',
    'OTHER_CATEGORY',
    'PRICE_REGEX',
    'RANDOM_STATE',
    'UNWANTED_TAGS',
    # Utils
    'count_unique_tags',
    'generate_selector_for_element',
    'normalize_tag',
]
