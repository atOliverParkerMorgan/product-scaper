import webview
import yaml
from pathlib import Path
import json
from typing import Dict, Optional, List
from webview import Window
from urllib.parse import urlparse
import logging
import pickle
try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

# Suppress pywebview debug messages
logging.getLogger('webview').setLevel(logging.WARNING)

CATEGORIES = ['product_name', 'price', 'image_url']

class Api:
    def __init__(self, url: str, categories: List[str]):
        """Initialize API with target URL and data categories to extract."""
        self.window: Optional[Window] = None
        self.selections: Dict[str, List[str]] = {}
        self.current_category_index: int = 0
        self.categories: List[str] = categories
        self.url = url
        self.model = None
        self._load_model()
        
    def _ensure_window(self) -> Window:
        """Return initialized window or raise RuntimeError."""
        if self.window is None:
            raise RuntimeError("Window is not initialized")
        return self.window
    
    def _load_model(self) -> None:
        """Load the trained model if available."""
        try:
            model_path = Path.cwd() / 'src' / 'models' / 'category_classifier.pkl'
            if model_path.exists():
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                logging.info("Model loaded successfully")
            else:
                logging.warning(f"Model not found at {model_path}")
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            self.model = None

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
    
    def get_model_predictions(self, category: str) -> List[str]:
        """Get model predictions for elements matching the given category."""
        if self.model is None or BeautifulSoup is None:
            return []
        
        try:
            window = self._ensure_window()
            html_content = window.evaluate_js('document.documentElement.outerHTML')
            
            if not html_content:
                return []
            
            soup = BeautifulSoup(html_content, 'html.parser')
            predictions = []
            
            # Find all elements and get their features
            for element in soup.find_all(True):
                try:
                    # Extract features similar to training data
                    features = self._extract_element_features(element)
                    
                    # Predict category
                    predicted_category = self.model.predict([features])[0]
                    
                    # If prediction matches current category, generate selector
                    if predicted_category == category:
                        selector = self._generate_css_selector(element, soup)
                        if selector:
                            predictions.append(selector)
                except Exception:
                    continue
            
            # Limit predictions to top candidates
            return predictions[:10]
            
        except Exception as e:
            logging.error(f"Error getting model predictions: {e}")
            return []
    
    def _extract_element_features(self, element) -> List[float]:
        """Extract features from an element for model prediction."""
        features = []
        
        # Tag name features
        common_tags = ['div', 'span', 'p', 'a', 'img', 'h1', 'h2', 'h3', 'ul', 'li']
        tag_features = [1 if element.name == tag else 0 for tag in common_tags]
        features.extend(tag_features)
        
        # Class features
        classes = element.get('class', [])
        class_text = ' '.join(classes).lower()
        price_keywords = ['price', 'cost', 'amount', 'value']
        name_keywords = ['name', 'title', 'product', 'item']
        image_keywords = ['image', 'img', 'photo', 'picture']
        
        features.append(1 if any(kw in class_text for kw in price_keywords) else 0)
        features.append(1 if any(kw in class_text for kw in name_keywords) else 0)
        features.append(1 if any(kw in class_text for kw in image_keywords) else 0)
        
        # ID features
        element_id = element.get('id', '').lower()
        features.append(1 if any(kw in element_id for kw in price_keywords) else 0)
        features.append(1 if any(kw in element_id for kw in name_keywords) else 0)
        features.append(1 if any(kw in element_id for kw in image_keywords) else 0)
        
        # Text content features
        text = element.get_text(strip=True)
        features.append(1 if text and any(c.isdigit() for c in text) else 0)
        features.append(len(text) if text else 0)
        
        # Attribute features
        features.append(1 if element.get('href') else 0)
        features.append(1 if element.get('src') else 0)
        features.append(1 if element.name == 'img' else 0)
        
        return features
    
    def _generate_css_selector(self, element, soup) -> Optional[str]:
        """Generate a CSS selector for an element."""
        try:
            # Try to use ID first
            if element.get('id'):
                return f"[id='{element.get('id')}']"
            
            # Build selector with tag and classes
            selector = element.name
            classes = element.get('class', [])
            if classes:
                selector += '.' + '.'.join(classes)
            
            # Check if selector is unique enough
            matches = soup.select(selector)
            if len(matches) <= 5:  # Reasonable number of matches
                return selector
            
            # Add nth-of-type if needed
            parent = element.parent
            if parent:
                siblings = [sib for sib in parent.find_all(element.name, recursive=False)]
                if len(siblings) > 1:
                    index = siblings.index(element) + 1
                    selector += f':nth-of-type({index})'
            
            return selector
        except Exception as e:
            logging.debug(f"Error generating selector: {e}")
            return None

    def save_file(self) -> None:
        """Save selectors and page source to domain-specific directory."""
        try:
            window = self._ensure_window()
        except RuntimeError:
            return

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

            save_dir = project_root / 'src' / 'data' / base_name
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
    
    # Temporarily suppress stderr to hide pywebview debug messages
    import sys
    import os
    old_stderr = sys.stderr
    sys.stderr = open(os.devnull, 'w')
    
    try:
        webview.start()
    finally:
        sys.stderr.close()
        sys.stderr = old_stderr