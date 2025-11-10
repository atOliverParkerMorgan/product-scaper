import webview
import yaml
from pathlib import Path
import json
from typing import Dict, Optional, List
from webview import Window
from urllib.parse import urlparse

CATEGORIES = ['product_name', 'price', 'image_url']

class Api:
    def __init__(self, url: str, categories: List[str]):
        """Initialize API with target URL and data categories to extract."""
        self.window: Optional[Window] = None
        self.selections: Dict[str, List[str]] = {}
        self.current_category_index: int = 0
        self.categories: List[str] = categories
        self.url = url
        
    def _ensure_window(self) -> Window:
        """Return initialized window or raise RuntimeError."""
        if self.window is None:
            raise RuntimeError("Window is not initialized")
        return self.window

    def start_workflow(self) -> None:
        """Begin the element selection workflow."""
        self.current_category_index = 0
        self.prompt_next_category()

    def prompt_next_category(self) -> None:
        """Display prompt for current category or save if complete."""
        if self.current_category_index >= len(self.categories):
            self.save_file()
            return

        category = self.categories[self.current_category_index]
        existing_selectors = self.selections.get(category, [])
        existing_selectors_json = json.dumps(existing_selectors)
        
        window = self._ensure_window()
        window.evaluate_js(f'promptForSelection("{category}", {existing_selectors_json})')

    def accept_selection(self, category: str, selector_to_save: str, run_prediction: bool = True) -> None:
        """Add new CSS selector for the given category."""
        if category not in self.selections:
            self.selections[category] = []
        
        if selector_to_save not in self.selections[category]:
            self.selections[category].append(selector_to_save)

        self.prompt_next_category()

    def unselect_selector(self, category: str, selector_to_remove: str) -> None:
        """Remove a previously selected CSS selector."""
        if category in self.selections and selector_to_remove in self.selections[category]:
            self.selections[category].remove(selector_to_remove)
            if not self.selections[category]:
                del self.selections[category]

        self.prompt_next_category()

    def user_clicked_previous_category(self) -> None:
        """Navigate to previous category in the workflow."""
        self.current_category_index = max(0, self.current_category_index - 1)
        self.prompt_next_category()

    def user_clicked_next_category(self) -> None:
        """Navigate to next category in the workflow."""
        self.current_category_index += 1
        self.prompt_next_category()

    def refresh_prompt(self) -> None:
        """Redisplay the current category prompt."""
        self.prompt_next_category()

    def save_file(self) -> None:
        """Save selectors and page source to domain-specific directory."""
        window = self._ensure_window()

        if not self.selections:
            window.evaluate_js('alert("No selectors chosen. Add some, then save.")')
            self.current_category_index = 0
            self.prompt_next_category()
            return

        try:
            project_root = Path.cwd()
            parsed_url = urlparse(self.url)
            domain = parsed_url.netloc.replace('www.', '')
            base_name = domain.split('.')[0] or 'default_site'

            save_dir = project_root / 'data' / base_name
            save_dir.mkdir(parents=True, exist_ok=True)

            yaml_path = save_dir / 'selectors.yaml'
            html_path = save_dir / 'page.html'

            with open(yaml_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.selections, f, default_flow_style=False, sort_keys=False)

            html_content = window.evaluate_js('document.documentElement.outerHTML')
            if html_content:
                with open(html_path, 'w', encoding='utf-8') as f:
                    f.write(html_content)

            window.evaluate_js(f'alert("Files saved successfully to:\\n{save_dir.as_posix()}")')

        except Exception as e:
            error_msg = str(e).replace('"', "'").replace('\n', ' ')
            window.evaluate_js(f'alert("Error saving file: {error_msg}")')
        
        finally:
            if self.window:
                self.window.destroy()



def load_custom_js(window: Window) -> None:
    """Load and inject custom JavaScript for element selection."""
    selector_path = Path(__file__).resolve().parent / 'selector.js'
    try:
        with selector_path.open('r', encoding='utf-8') as f:
            selector_logic = f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Required selector.js not found at {selector_path}")
    
    window.evaluate_js(selector_logic)
    window.evaluate_js('initApp()')


def select_data(url: str, categories: List[str]) -> None:
    """
    Open an interactive window for selecting HTML elements by category.
    
    Args:
        url: Target webpage URL to scrape
        categories: Data categories to extract (e.g., ["price", "title"])
    """
    api = Api(url, categories)
    
    window = webview.create_window(
        'HTML Element Selector',
        url,
        js_api=api,
        resizable=True,
        width=1200,
        height=800,
        maximized=True
    )
    
    if window is None:
        raise RuntimeError("Failed to create webview window")

    api.window = window
    window.events.loaded += lambda: load_custom_js(window)
    webview.start()