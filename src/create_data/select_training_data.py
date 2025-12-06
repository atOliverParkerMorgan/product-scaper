"""
Simple Playwright-based element selector using native hover and click capabilities.
No complex custom JavaScript - just clean Playwright interactions.
"""
import yaml
from pathlib import Path
from typing import Dict, List
from urllib.parse import urlparse
import logging
from playwright.sync_api import sync_playwright, Page


logging.basicConfig(level=logging.INFO)

CATEGORIES = ['product_name', 'price', 'image_url']
UI_PATH = Path(__file__).parent / 'injections'


def inject_styles(page: Page):
    """Inject minimal CSS for highlighting and UI."""
    styles_path = UI_PATH / 'styles.css'
    css_content = styles_path.read_text(encoding='utf-8')
    page.add_style_tag(content=css_content)


def generate_selector(page: Page, element_js: str) -> str:
    """Generate CSS selector for an element using JavaScript."""
    selector_script = UI_PATH / 'selector_generator.js'
    js_content = selector_script.read_text(encoding='utf-8')
    # Wrap the function and call it with the element
    selector = page.evaluate(f"(element) => {{ return ({js_content})(element); }}", element_js)
    return selector


def create_ui(page: Page, category: str, selected_count: int, category_idx: int, total: int):
    """Create or update the UI panel."""
    create_ui_script = UI_PATH / 'create_ui.js'
    js_content = create_ui_script.read_text(encoding='utf-8')
    page.evaluate(js_content, {
        'category': category,
        'selectedCount': selected_count,
        'categoryIdx': category_idx,
        'total': total
    })


def select_data(url: str, categories: List[str]) -> None:
    """
    Interactive element selector using Playwright's native capabilities.
    
    Args:
        url: Target webpage URL to scrape
        categories: Data categories to extract (e.g., ["price", "title"])
    """
    with sync_playwright() as p:
        # Launch browser
        browser = p.chromium.launch(
            headless=False,
            args=['--start-maximized']
        )
        context = browser.new_context(
            viewport={'width': 1920, 'height': 1080}
        )
        page = context.new_page()
        
        # Navigate to URL
        logging.info(f"Navigating to {url}...")
        page.goto(url, wait_until='domcontentloaded', timeout=60000)
        page.wait_for_load_state('networkidle', timeout=10000)
        
        # Inject styles
        inject_styles(page)
        
        # Inject optimal-select library for better CSS selector generation
        page.add_script_tag(url='https://unpkg.com/optimal-select@latest/dist/optimal-select.js')
        
        # Storage for selections
        selections: Dict[str, List[str]] = {}
        current_category_idx = 0
        
        # Add hover highlighting
        hover_script = UI_PATH / 'mouse_handler.js'
        hover_js = hover_script.read_text(encoding='utf-8')
        page.evaluate(hover_js)
        
        def update_ui():
            """Update UI with current category info."""
            category = categories[current_category_idx]
            count = len(selections.get(category, []))
            create_ui(page, category, count, current_category_idx, len(categories))
            
            # Highlight selected elements
            clear_script = UI_PATH / 'clear_highlights.js'
            clear_js = clear_script.read_text(encoding='utf-8')
            page.evaluate(clear_js)
            
            highlight_script = UI_PATH / 'highlight_selector.js'
            highlight_js = highlight_script.read_text(encoding='utf-8')
            
            for selector in selections.get(category, []):
                try:
                    page.evaluate(f"(sel) => {{ return ({highlight_js})(sel); }}", selector)
                except Exception as e:
                    logging.warning(f"Could not highlight selector {selector}: {e}")
        
        def save_and_close():
            """Save selections and close browser."""
            if not selections:
                logging.warning("No selections made")
                return
            
            try:
                project_root = Path.cwd()
                parsed_url = urlparse(url)
                domain = parsed_url.netloc.replace('www.', '')
                base_name = domain.split('.')[0] or 'default_site'
                
                save_dir = project_root / 'src' / 'data' / base_name
                save_dir.mkdir(parents=True, exist_ok=True)
                
                # Save selectors
                yaml_path = save_dir / 'selectors.yaml'
                with open(yaml_path, 'w', encoding='utf-8') as f:
                    yaml.dump(selections, f, default_flow_style=False, sort_keys=False)
                
                # Save HTML
                html_content = page.content()
                html_path = save_dir / 'page.html'
                with open(html_path, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                
                logging.info(f"✓ Saved to {save_dir}")
            except Exception as e:
                logging.error(f"Error saving: {e}")
        
        # Initial UI
        update_ui()
        
        logging.info("✓ Ready! Hover and click elements to select them.")
        
        # Main interaction loop
        try:
            while current_category_idx < len(categories):
                category = categories[current_category_idx]
                
                # Wait for user interaction 
                interaction_script = UI_PATH / 'interaction_handler.js'
                interaction_js = interaction_script.read_text(encoding='utf-8')
                action = page.evaluate(interaction_js)
                
                if action['type'] == 'element':
                    # Get selector from the action (already generated in JavaScript)
                    selector = action.get('selector')
                    
                    if selector:
                        # Toggle selection
                        if category not in selections:
                            selections[category] = []
                        
                        if selector in selections[category]:
                            selections[category].remove(selector)
                            logging.info(f"❌ Removed: {selector}")
                        else:
                            selections[category].append(selector)
                            logging.info(f"✓ Selected: {selector}")
                        
                        update_ui()
                
                elif action['type'] == 'prev':
                    current_category_idx = max(0, current_category_idx - 1)
                    update_ui()
                
                elif action['type'] == 'next':
                    current_category_idx += 1
                    if current_category_idx >= len(categories):
                        break
                    update_ui()
                
                elif action['type'] == 'done':
                    break
        
        except KeyboardInterrupt:
            logging.info("\nInterrupted by user")
        except Exception as e:
            logging.error(f"Error: {e}")
        finally:
            save_and_close()
            browser.close()
            logging.info("✓ Done!")
