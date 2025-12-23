"""
Utility functions used across the product scraper project.

This module provides helpers for tag normalization, xpath generation, unique tag counting,
and CSS selector generation for lxml elements.
"""

# @generated "partially" Gemini 3: docstrings.

from typing import Any, List


def normalize_tag(tag_name: Any) -> str:
    """
    Normalize a tag name to a lowercase string.

    Args:
        tag_name: The tag name to normalize.
    Returns:
        str: Normalized tag name, or 'unknown' if not valid.
    """
    if not tag_name or not hasattr(tag_name, "lower"):
        return "unknown"
    return str(tag_name).lower()


def get_unique_xpath(element: Any) -> str:
    """
    Get the unique XPath for an lxml element.

    Args:
        element: lxml element.
    Returns:
        str: XPath string.
    """
    tree = element.getroottree()
    return tree.getpath(element)


def count_unique_tags(tag_list: List[str]) -> int:
    """
    Count the number of unique tags in a list.

    Args:
        tag_list: List of tag names.
    Returns:
        int: Number of unique tags.
    """
    return len(set(tag_list)) if tag_list else 0


def generate_selector_for_element(element: Any) -> str:
    """
    Generate a CSS selector for an lxml element.

    Args:
        element: lxml element.
    Returns:
        str: CSS selector string.
    """
    parts = []
    current = element
    while current is not None and hasattr(current, "tag"):
        tag = current.tag.lower() if hasattr(current.tag, "lower") else str(current.tag)
        if tag == "html":
            break
        # Try to use ID for uniqueness
        elem_id = current.get("id")
        if elem_id:
            parts.insert(0, f"{tag}#{elem_id}")
            break
        # Use class if available
        classes = current.get("class", "")
        if classes:
            class_list = classes.split()
            if class_list:
                parts.insert(0, f"{tag}.{class_list[0]}")
        else:
            parts.insert(0, tag)
        current = current.getparent()
        # Limit depth to avoid overly long selectors
        if len(parts) > 5:
            break
    return " > ".join(parts) if parts else ""
