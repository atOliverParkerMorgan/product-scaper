"""Utilities module for product scraper."""

from .utils import (
    count_unique_tags,
    generate_selector_for_element,
    normalize_tag,
)

__all__ = [
    # Constants
    'DATA_NAME',
    'OTHER_CATEGORY',
    'PRICE_REGEX',
    'RANDOM_STATE',
    'UNWANTED_TAGS',
    # Utils
    'count_unique_tags',
    'generate_selector_for_element',
    'normalize_tag',
]
