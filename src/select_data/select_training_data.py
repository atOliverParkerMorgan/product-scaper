import webview
import yaml  # Make sure pyyaml is installed: pip install pyyaml
from pathlib import Path
import re
import json
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
        self.selections: Dict[str, List[str]] = {}
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
        print("Starting workflow...")
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
        existing_selectors = self.selections.get(category, [])
        existing_selectors_json = json.dumps(existing_selectors)
        
        print(f"--> Prompting for category: {category} (Has {len(existing_selectors)} existing)")
        window = self._ensure_window()
        
        # JS function handles updating the UI
        window.evaluate_js(f'promptForSelection("{category}", {existing_selectors_json})') 

    def accept_selection(self, category: str, selector_to_save: str) -> None:
        """
        Called by JS to add a new selector.
        """
        if category not in self.selections:
            self.selections[category] = []
        
        if selector_to_save not in self.selections[category]:
            self.selections[category].append(selector_to_save)
            print(f"    -> Added to '{category}': {selector_to_save}")
        else:
            print(f"    -> Selector already exists for '{category}'.")

        # ALWAYS re-prompt to refresh UI highlights
        self.prompt_next_category()

    def unselect_selector(self, category: str, selector_to_remove: str) -> None:
        """
        Called by JS when a user clicks an already-selected element.
        """
        print(f"❌ User unselected item from '{category}': {selector_to_remove}")
        if category in self.selections and selector_to_remove in self.selections[category]:
            self.selections[category].remove(selector_to_remove)
            if not self.selections[category]:
                del self.selections[category]
        else:
            print(f"    -> Warning: Tried to unselect, but selector not found.")

        # ALWAYS re-prompt to refresh UI highlights
        self.prompt_next_category()

    def user_clicked_previous_category(self) -> None:
        """
        Called by JavaScript when user clicks 'Previous'.
        """
        print("User moving to PREVIOUS category.")
        self.current_category_index = max(0, self.current_category_index - 1)
        self.prompt_next_category() # Re-prompt with the new (or same) index

    def user_clicked_next_category(self) -> None:
        """
        Called by JavaScript when the user clicks the "Next Category" button.
        """
        print("User moving to NEXT category.")
        self.current_category_index += 1
        self.prompt_next_category() # Move to the next category (or save)

    def save_file(self) -> None:
        """
        Triggers a "Save File" dialog.
        """
        if not self.selections:
            print("No selectors chosen. Aborting save.")
            window = self._ensure_window()
            # Show a simple alert
            window.evaluate_js('alert("No selectors chosen. Add some, then save.")')
            self.current_category_index = 0 # Restart from first category
            self.prompt_next_category()
            return

        print("\nAll categories selected. Triggering save dialog...")
        window = self._ensure_window()
        
        file_path = window.create_file_dialog(
            dialog_type=webview.FileDialog.OPEN.SAVE,
            directory=str(Path.cwd()),
            allow_multiple=False,
            save_filename='selectors.yaml',
            file_types=('YAML Files (*.yaml;*.yml)',),
        )

        self._on_save_dialog_result(file_path[0])


    def _on_save_dialog_result(self, file_path: str) -> None:
        """
        Callback for when the save dialog is closed.
        """
        window = self._ensure_window()

        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.selections, f, default_flow_style=False, sort_keys=False)
            window.evaluate_js(f'alert("Selectors saved successfully to {file_path}!")')
        except Exception as e:
            print(f"Error saving file: {e}")
            window.evaluate_js(f'alert("Error saving file: {e}")')
        

        # exiting after save to avoid confusion
        if self.window:
            self.window.destroy()            

def load_custom_js(window: Window) -> None:
    """
    Injects our custom JavaScript logic.
    """
    # Assuming 'selector.js' is in the same directory as this Python script
    selector_path = Path(__file__).resolve().parent / 'selector.js'
    try:
        with selector_path.open('r', encoding='utf-8') as f:
            selector_logic = f.read()

    except FileNotFoundError:
        raise FileNotFoundError("Please create a 'selector.js' file in the same directory.")
    
    window.evaluate_js(selector_logic)
    window.evaluate_js('initApp()')


# --- Main Application ---
if __name__ == '__main__':
    try:
        import yaml
    except ImportError:
        print("Error: PyYAML is not installed. Please run 'pip install pyyaml'")
        exit(1)

    api = Api()
    
    window = webview.create_window(
        'HTML Element Selector',
        'https://www.artonpaper.ch/new', # Target URL
        js_api=api,
        width=1200,
        height=800
    )
    
    if window is None:
        print("Error: Could not create webview window.")
        exit(1)
        
    api.window = window
    window.toggle_fullscreen()

    
    # We inject JS after the page is loaded
    window.events.loaded += lambda: load_custom_js(window)
    
    