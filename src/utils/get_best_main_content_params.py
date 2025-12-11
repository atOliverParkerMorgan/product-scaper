"""
Parameter optimization script for get_main_html_content_tag function.

This script:
1. Extracts ground truth main content identifiers from test data
2. Performs grid search over parameter combinations
3. Finds the best parameter set that maximizes matching accuracy
"""

import sys
from pathlib import Path
from typing import Dict, Tuple, Optional, Any
from itertools import product
import lxml.html

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.train_model.process_data import get_main_html_content_tag
from src.utils.console import CONSOLE, log_info


# Ground truth: Expected main content tag identifiers for each test page
# Format: {page_name: (expected_tag, expected_class, expected_id)}
GROUND_TRUTH = {
    'page_1': ('div', 'w3-row-padding', None),
    'page_2': ('div', None, 'content'),
    'page_3': ('div', None, 'content'),
    'page_4': ('div', 'items-list', None),
    'page_5': ('div', None, 'content'),
    'page_6': ('ul', 'products', None),
    'page_7': ('div', None, 'content'),
    'page_8': ('div', None, 'content'),
    'page_9': ('div', None, 'content'),
    'page_10': ('div', 'product-list__item', None),
    'page_11': ('div', None, 'content'),
    'page_12': ('div', None, 'items-list'),
    'page_13': ('div', None, None),
    'page_14': ('div', None, 'incenterpage'),
    'page_15': ('div', None, 'center_column'),

}


def element_matches_expectation(
    element: Optional[lxml.html.HtmlElement],
    expected_tag: str,
    expected_class: Optional[str],
    expected_id: Optional[str],
) -> bool:
    """
    Check if element matches the expected criteria.
    
    Args:
        element: The HTML element to check
        expected_tag: Expected tag name (must match exactly)
        expected_class: Expected class (must be in element.classes if provided)
        expected_id: Expected id attribute (must match exactly if provided)
    
    Returns:
        True if element matches all provided criteria
    """
    if element is None:
        return False
    
    # Check tag
    if element.tag != expected_tag:
        return False
    
    # Check class
    if expected_class is not None:
        if expected_class not in element.classes:
            return False
    
    # Check id
    if expected_id is not None:
        if element.get('id') != expected_id:
            return False
    
    return True


def load_test_pages() -> Dict[str, str]:
    """Load all test HTML pages from test_data directory."""
    test_data_dir = Path(__file__).parent.parent.parent / 'tests' / 'test_data'
    pages = {}
    
    for page_file in sorted(test_data_dir.glob('page_*.html')):
        page_name = page_file.stem  # e.g., 'page_1'
        try:
            with open(page_file, 'r', encoding='utf-8') as f:
                pages[page_name] = f.read()
        except Exception as e:
            logger.error(f"Failed to load {page_file}: {e}")
    
    return pages


def evaluate_params(
    pages: Dict[str, str],
    img_importance: int,
    min_tag_text_length: int,
    min_image_count: int,
    link_density_weight: float,
    depth_score_coefficient: float,
    parent_improvement_threshold: float,
) -> Tuple[float, Dict[str, bool]]:
    """
    Evaluate a parameter set against all test pages.
    
    Returns:
        (accuracy_score, {page_name: matched})
    """
    matches = {}
    
    for page_name, html_content in pages.items():
        if page_name not in GROUND_TRUTH:
            continue
        
        expected_tag, expected_class, expected_id = GROUND_TRUTH[page_name]
        
        # Get the main content element with current params
        main_content = get_main_html_content_tag(
            html_content,
            IMG_IMPORTANCE=img_importance,
            MIN_TAG_TEXT_LENGTH=min_tag_text_length,
            MIN_IMAGE_COUNT=min_image_count,
            LINK_DENSITY_WEIGHT=link_density_weight,
            DEPTH_SCORE_COEFFICIENT=depth_score_coefficient,
            PARENT_IMPROVEMENT_THRESHOLD=parent_improvement_threshold
        )
        
        # Check if it matches expectations
        matched = element_matches_expectation(
            main_content, expected_tag, expected_class, expected_id
        )
        matches[page_name] = matched
    
    # Calculate accuracy
    if matches:
        accuracy = sum(matches.values()) / len(matches)
    else:
        accuracy = 0.0
    
    return accuracy, matches


def grid_search(pages: Dict[str, str]) -> Tuple[Dict[str, Any], float, Dict[str, bool]]:
    """
    Perform grid search over parameter combinations.
    
    Returns:
        (best_params, best_accuracy, best_matches)
    """
    # Define parameter grids
    # Adjust ranges based on your expected parameter space
    img_importance_values = [10,20,40, 50, 100, 150, 200,  250, 300, 350]
    min_tag_text_length_values = [0, 10, 12, 13, 15, 16, 18, 25, 80, 150]
    min_image_count_values = [0, 1, 2, 3, 4, 5, 6, 8, 9]
    link_density_weight_values = [ 0.3, 0.4,0.5, 0.65, 0.75, ]
    depth_score_coefficient_values = [5.0, 7.5, 8 ,9, 10.0, 12, 15.0 ,30,32]
    parent_improvement_threshold_values = [1.9, 2.0, 2.1,2.2, 2.5, 3, 5]
    
    total_combinations = (
        len(img_importance_values) *
        len(min_tag_text_length_values) *
        len(min_image_count_values) *
        len(link_density_weight_values) *
        len(depth_score_coefficient_values) *
        len(parent_improvement_threshold_values)
    )
    
    logger.info(f"Starting grid search with {total_combinations} combinations...")
    
    best_params = None
    best_accuracy = -1.0
    best_matches = {}
    
    combination_count = 0
    
    # Generate all combinations
    param_combinations = product(
        img_importance_values,
        min_tag_text_length_values,
        min_image_count_values,
        link_density_weight_values,
        depth_score_coefficient_values,
        parent_improvement_threshold_values
    )
    
    for (img_imp, min_text, min_img, link_dens, depth_coef, parent_thresh) in param_combinations:
        combination_count += 1
        
        if combination_count % 100 == 0:
            logger.info(f"Progress: {combination_count}/{total_combinations}")
        
        accuracy, matches = evaluate_params(
            pages,
            img_importance=img_imp,
            min_tag_text_length=min_text,
            min_image_count=min_img,
            link_density_weight=link_dens,
            depth_score_coefficient=depth_coef,
            parent_improvement_threshold=parent_thresh
        )
        
        # Track best parameters
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = {
                'IMG_IMPORTANCE': img_imp,
                'MIN_TAG_TEXT_LENGTH': min_text,
                'MIN_IMAGE_COUNT': min_img,
                'LINK_DENSITY_WEIGHT': link_dens,
                'DEPTH_SCORE_COEFFICIENT': depth_coef,
                'PARENT_IMPROVEMENT_THRESHOLD': parent_thresh,
            }
            best_matches = matches
            
            logger.info(
                f"New best accuracy: {best_accuracy:.2%} "
                f"| IMG_IMPORTANCE={img_imp}, MIN_TEXT={min_text}, MIN_IMG={min_img}, "
                f"LINK_DENS={link_dens}, DEPTH_COEF={depth_coef}, PARENT_THRESH={parent_thresh}"
            )
    
    logger.info(f"Grid search completed. Evaluated {combination_count} combinations.")
    
    return best_params, best_accuracy, best_matches


def print_results(best_params: Dict[str, Any], best_accuracy: float, best_matches: Dict[str, bool]):
    """Print optimization results in a formatted way."""
    CONSOLE.print("\n" + "="*80)
    CONSOLE.print("PARAMETER OPTIMIZATION RESULTS")
    CONSOLE.print("="*80)
    
    log_info(f"Best Accuracy: {best_accuracy:.2%}")
    
    CONSOLE.print("\nBest Parameters:")
    CONSOLE.print("-" * 80)
    for param_name, param_value in best_params.items():
        CONSOLE.print(f"  {param_name:<35} = {param_value}")
    
    CONSOLE.print("\nMatches per page:")
    CONSOLE.print("-" * 80)
    for page_name, matched in best_matches.items():
        status = "PASS" if matched else "FAIL"
        CONSOLE.print(f"  {page_name:<35} {status}")
    
    CONSOLE.print("\nFunction call template:")
    CONSOLE.print("-" * 80)
    CONSOLE.print("get_main_html_content_tag(")
    CONSOLE.print("    html_content,")
    for param_name, param_value in best_params.items():
        if isinstance(param_value, float):
            CONSOLE.print(f"    {param_name}={param_value},")
        else:
            CONSOLE.print(f"    {param_name}={param_value},")
    CONSOLE.print(")")
    
    CONSOLE.print("="*80 + "\n")


def main():
    """Main entry point."""
    
    pages = load_test_pages()
    
    if not pages:
        return
    
    CONSOLE.print(f"Loaded {len(pages)} test pages")
    
    # Run grid search
    best_params, best_accuracy, best_matches = grid_search(pages)
    
    # Print results
    print_results(best_params, best_accuracy, best_matches)
    
    return best_params, best_accuracy


if __name__ == "__main__":
    main()
    