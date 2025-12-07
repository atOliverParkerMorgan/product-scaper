"""Utility functions used across the product scraper project."""

from typing import List


def normalize_tag(tag_name) -> str:
    """Normalize tag name to string."""
    if not tag_name or not hasattr(tag_name, 'lower'):
        return 'unknown'
    return str(tag_name).lower()


def count_unique_tags(tag_list: List[str]) -> int:
    """Count unique tags."""
    return len(set(tag_list)) if tag_list else 0


def generate_selector_for_element(element) -> str:
    """Generate a CSS selector for an lxml element."""
    parts = []
    current = element
    
    while current is not None and hasattr(current, 'tag'):
        tag = current.tag.lower() if hasattr(current.tag, 'lower') else str(current.tag)
        if tag == 'html':
            break
        
        # Try to use ID for uniqueness
        elem_id = current.get('id')
        if elem_id:
            parts.insert(0, f"{tag}#{elem_id}")
            break
        
        # Use class if available
        classes = current.get('class', '')
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
    
    return ' > '.join(parts) if parts else ''
