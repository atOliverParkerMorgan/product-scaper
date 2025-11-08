import webview
from pathlib import Path
import re
from typing import Dict, Optional, List
from webview import Window

# --- Define your data categories here ---
CATEGORIES: List[str] = [
    "product_name",
    "price",
    "description",
    "image_url"
]

class Api:
    def __init__(self):
        self.window: Optional[Window] = None
        self.selections: Dict[str, str] = {}
        self.current_category_index: int = 0
        
    def _ensure_window(self) -> Window:
        """Ensure window is available and return it"""
        if self.window is None:
            raise RuntimeError("Window is not initialized")
        return self.window

    def start_workflow(self):
        """
        Called by JavaScript once the page and all scripts are loaded.
        Starts the selection process.
        """
        self.prompt_next_category()

    def prompt_next_category(self) -> None:
        """
        Prompts the user to select the next category in the list.
        If done, it triggers the save process.
        """
        if self.current_category_index >= len(CATEGORIES):
            self.save_file()
            return

        category = CATEGORIES[self.current_category_index]
        print(f"--> Prompting for category: {category}")
        window = self._ensure_window()
        # This JS function is defined in selector.js
        window.evaluate_js(f'promptForSelection("{category}")') 


    def save_selector(self, category: str, selector: str) -> None:
        """
        Called by JavaScript when an element is clicked.
        Saves the selector and triggers the "predict similar" step.
        """
        self.selections[category] = selector
        print(f"✅ Saved '{category}': {selector}")
        
        # Now, predict similar elements
        simple_selector = self.generate_simple_selector(selector)
        print(f"    -> Predicting similar with: {simple_selector}")
        
        # This JS function will highlight all matches and ask for confirmation
        # Escape the selector properly for JavaScript
        escaped_selector = simple_selector.replace('"', '\\"')
        window = self._ensure_window()
        window.evaluate_js(f'highlightAndConfirm("{escaped_selector}", "{category}")')

    def generate_simple_selector(self, selector: str) -> str:
        """
        Generates a simpler, more general selector for "predicting"
        similar elements. It prioritizes IDs and classes.
        
        Example: 'div > div.content > h1:nth-of-type(1)'
        Becomes: 'div.content > h1'
        
        Args:
            selector: The original CSS selector string
        
        Returns:
            A simplified CSS selector string
        """
        parts = selector.split(' > ')
        
        # Try to find the most specific part with an ID or class
        for i in range(len(parts) - 1, -1, -1):
            if '#' in parts[i] or '.' in parts[i]:
                # Found a good anchor. Use it and all parts after it.
                simple_selector = ' > '.join(parts[i:])
                # Remove :nth-of-type, etc.
                simple_selector = re.sub(r':nth-of-type\(\d+\)', '', simple_selector)
                simple_selector = re.sub(r':nth-child\(\d+\)', '', simple_selector)
                return simple_selector
        
        # If no class/ID, just use the last two tags
        return ' > '.join(parts[-2:])


    def prediction_confirmed(self, category: str, was_good: bool) -> None:
        """
        Called by JavaScript with the result of the confirmation modal
        (True = OK, False = Redo).
        """
        if was_good:
            print(f"    -> Prediction approved for {category}.")
            self.current_category_index += 1
            self.prompt_next_category() # Move to the next category
        else:
            print(f"    -> Prediction rejected. Redoing {category}.")
            # Ask to select the same category again
            window = self._ensure_window()
            window.evaluate_js(f'promptForSelection("{category}")')

    def save_file(self) -> None:
        """
        Triggers a "Save File" dialog in the pywebview window.
        """
        print("\nAll categories selected. Triggering save dialog...")
        window = self._ensure_window()
        window.create_file_dialog(
            dialog_type=1,  # SAVE_DIALOG = 1
            directory=str(Path.cwd()),
            save_filename='selectors.yaml',
            file_types=('YAML Files (*.yaml;*.yml)',)
        )

    def _on_save_dialog_result(self, file_path: Optional[str]) -> None:
        """
        Callback for when the save dialog is closed.
        If a path is chosen, it saves the YAML file.
        """
        window = self._ensure_window()
        if not file_path:
            print("Save cancelled.")
            # Ask to save again
            window.evaluate_js('showModal("Save cancelled. <button onclick=\'window.pywebview.api.save_file()\'>Save Again</button>", false)')
            return

        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.selections, f, default_flow_style=False, sort_keys=False)
            print(f"\n🎉 Successfully saved selectors to: {file_path}")
            window.evaluate_js('showModal("Selectors saved successfully!", false)')
        except Exception as e:
            print(f"Error saving file: {e}")
            window.evaluate_js(f'showModal("Error saving file: {e}. <button onclick=\'window.pywebview.api.save_file()\'>Try Again</button>", false)')


def load_custom_js(window: Window) -> None:
    """
    Injects our custom JavaScript logic into the loaded webpage.
    """
    selector_path = Path(__file__).resolve().parent / 'selector.js'
    with selector_path.open('r', encoding='utf-8') as f:
        selector_logic = f.read()
    
    window.evaluate_js(selector_logic)
    # After JS is loaded, tell it to initialize and contact Python
    window.evaluate_js('initApp()')


# --- Main Application ---
if __name__ == '__main__':
    # We need pyyaml to run
    try:
        import yaml
    except ImportError:
        print("Error: PyYAML is not installed. Please run 'pip install pyyaml'")
        exit(1)

    api = Api()
    
    window = webview.create_window(
        'HTML Element Selector Workflow',
        'https://www.artonpaper.ch/new',  # Start with a default URL
        js_api=api
    )
    
    if window is not None:  # Type guard for window
        api.window = window  # Give the API a reference to the window
        # Capture window in a local variable for type safety
        win = window  # This variable is now known to be non-None
        window.events.loaded += lambda: load_custom_js(win)
    
    webview.start(debug=True)  # debug=True is helpful for development