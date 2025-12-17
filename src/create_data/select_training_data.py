import time
import copy
from typing import Dict, List, TYPE_CHECKING
from pathlib import Path
from playwright.sync_api import sync_playwright, Error as PlaywrightError
from train_model.predict_data import predict_selectors
from utils.console import log_info, log_warning, log_error, log_success

if TYPE_CHECKING:
    from ProductScaper import ProductScraper
# --- CONFIGURATION ---

UI_PATH = Path(__file__).parent / 'ui'

# Create UI directory/files if they don't exist to prevent immediate crash
if not UI_PATH.exists():
    UI_PATH.mkdir(parents=True, exist_ok=True)
    (UI_PATH / 'styles.css').touch()
    (UI_PATH / 'core.js').touch()
    (UI_PATH / 'update.js').touch()

CSS_CONTENT = (UI_PATH / 'styles.css').read_text(encoding='utf-8')
JS_CORE_LOGIC = (UI_PATH / 'core.js').read_text(encoding='utf-8')
JS_UPDATE_UI = (UI_PATH / 'update.js').read_text(encoding='utf-8')


def highlight_selectors(page, selectors: List[str], force_update=False):
    """Helper to re-apply green outlines securely and remove predicted highlights."""
    try:
        # 1. Clean up old 'selected' classes if forcing update
        if force_update:
            page.evaluate("document.querySelectorAll('.pw-selected').forEach(el => el.classList.remove('pw-selected'))")
        
        # 2. Apply new selections using XPath
        # Process selectors in small batches so one invalid selector doesn't abort the whole operation
        for xpath in selectors:
            safe_xpath = xpath.replace('"', '\\"').replace('\n', ' ')
            page.evaluate(f"""
                (() => {{
                    try {{
                        const result = document.evaluate("{safe_xpath}", document, null, XPathResult.ORDERED_NODE_SNAPSHOT_TYPE, null);
                        for (let i = 0; i < result.snapshotLength; i++) {{
                            const el = result.snapshotItem(i);
                            el.classList.add('pw-selected');
                            // If it was predicted, remove the purple prediction style
                            el.classList.remove('pw-predicted'); 
                        }}
                    }} catch(e) {{
                        console.log("Invalid XPath skipped:", "{safe_xpath}");
                    }}
                }})()
            """)
    except PlaywrightError:
        pass


def inject_ui_scripts(page):
    """Inject CSS and JavaScript into the page."""
    try:
        # Add styles
        if CSS_CONTENT.strip():
            page.add_style_tag(content=CSS_CONTENT)
        
        # Add Core Logic (listener, selector generation)
        if JS_CORE_LOGIC.strip():
            page.evaluate(JS_CORE_LOGIC)
        
        # Inject a fallback selector generator if core.js didn't provide it
        # This ensures 'Select Predicted' works even if core.js is missing logic
        page.evaluate("""
            if (typeof window._generateSelector === 'undefined') {
                window._generateSelector = function(el) {
                    if (el.id) return '#' + el.id;
                    if (el.className) {
                        const classes = Array.from(el.classList).join('.');
                        if (classes) return '.' + classes;
                    }
                    return el.tagName.toLowerCase();
                }
            }
        """)
        return True
    except PlaywrightError:
        log_warning("Could not inject scripts (page might be loading)")
        return False


def update_ui_state(page, category, selection_count, current_idx, total_categories):
    """Update the UI with current category and progress."""
    try:
        page.evaluate(JS_UPDATE_UI, {
            'category': category,
            'count': selection_count,
            'idx': current_idx,
            'total': total_categories
        })
        return True
    except PlaywrightError:
        return False


def poll_for_action(page, timeout=0.2):
    """Poll for user actions (click, button, keyboard) within timeout."""
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            # Check for element clicks
            clicked = page.evaluate("window._clickedSelector")
            if clicked:
                page.evaluate("window._clickedSelector = null")
                return 'toggle', clicked
            
            # Check for UI buttons (Next, Prev, Done, Predict, Select All)
            ui_btn = page.evaluate("window._action")
            if ui_btn:
                page.evaluate("window._action = null")
                return 'navigate', ui_btn
            
            # Check for keyboard shortcuts (Undo/Redo)
            key_act = page.evaluate("window._keyAction")
            if key_act:
                page.evaluate("window._keyAction = null")
                return 'history', key_act
        except PlaywrightError:
            break
        
        time.sleep(0.05)
    
    return None, None


def handle_toggle_action(selector, category, selections, undo_stack, redo_stack, current_idx):
    """Handle selector toggle action."""
    # Push state to undo stack
    undo_stack.append((current_idx, copy.deepcopy(selections)))
    redo_stack.clear()
    
    if category not in selections:
        selections[category] = []
    
    if selector in selections[category]:
        selections[category].remove(selector)
        log_info(f"Removed: {selector}")
    else:
        selections[category].append(selector)
        log_success(f"Added: {selector}")


def handle_navigate_action(direction, current_idx, undo_stack, redo_stack, page):
    """Handle navigation action (next, prev, done)."""
    if direction == 'next':
        undo_stack.clear()
        redo_stack.clear()
        # Clear predictions when navigating
        try:
            page.evaluate("document.querySelectorAll('.pw-predicted').forEach(el => el.classList.remove('pw-predicted'))")
        except Exception:
            pass
        return current_idx + 1, False
    elif direction == 'prev' and current_idx > 0:
        undo_stack.clear()
        redo_stack.clear()
        # Clear predictions when navigating
        try:
            page.evaluate("document.querySelectorAll('.pw-predicted').forEach(el => el.classList.remove('pw-predicted'))")
        except Exception:
            pass
        return current_idx - 1, False
    elif direction == 'done':
        return current_idx, True
    # 'select_predicted' is handled in the main loop
    return current_idx, False


def handle_history_action(cmd, current_idx, selections, undo_stack, redo_stack):
    """Handle undo/redo action."""
    if cmd == 'undo' and undo_stack:
        redo_stack.append((current_idx, copy.deepcopy(selections)))
        prev_idx, prev_selections = undo_stack.pop()
        # Allow undo only if we're still on the same category step (UX choice)
        if prev_idx == current_idx:
            log_info("Undo")
            return prev_selections
        # If undo would change the category step, restore state and warn (simple history behavior)
        undo_stack.append((prev_idx, prev_selections))
        log_warning("Cannot undo across category changes (navigation clears history)")
        
    elif cmd == 'redo' and redo_stack:
        undo_stack.append((current_idx, copy.deepcopy(selections)))
        next_idx, next_selections = redo_stack.pop()
        if next_idx == current_idx:
            log_info("Redo")
            return next_selections
        redo_stack.append((next_idx, next_selections))
    
    return selections


def navigate_to_url(page, url):
    """Navigate to the target URL."""
    log_info(f"Navigating to {url}")
    try:
        # 'domcontentloaded' is faster than 'networkidle'
        page.goto(url, wait_until='domcontentloaded', timeout=60000)
    except Exception as e:
        log_warning(f"Navigation warning: {e}")


def select_data(product_scraper: 'ProductScraper', url: str)-> Dict[str, List[str]]:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False, args=['--start-maximized'])
        context = browser.new_context(no_viewport=True)
        page = context.new_page()

        navigate_to_url(page, url)
        inject_ui_scripts(page)

        selections: Dict[str, List[str]] = {}
        undo_stack = [] 
        redo_stack = []
        current_idx = 0
        last_selections_hash = None
        last_category = None  # Track category changes to re-run predictions
        
        try:
            should_exit = False
            while current_idx < len(product_scraper.categories) and not should_exit:
                category = product_scraper.categories[current_idx]
                current_selection_list = selections.get(category, [])

                # 1. Auto-run predictions for this category (when category changes)
                if category != last_category:
                    log_info(f"Auto-predicting for '{category}'")
                    html_content = page.content()
                    if product_scraper.model is None:
                        log_warning("No model provided for predictions")
                    else:
                        try:
                            predicted = predict_selectors(product_scraper.model, html_content, category)
                        except ValueError as ve:
                            log_warning(str(ve))
                            predicted = []
                        
                        if predicted:
                            page.evaluate("document.querySelectorAll('.pw-predicted').forEach(el => el.classList.remove('pw-predicted'))")
                            highlighted_count = 0
                            
                            for candidate in predicted:
                                try:
                                    xpath = candidate.get('xpath')
                                    idx = candidate.get('index')
                                    
                                    js_highlight_script = ""
                                    
                                    if xpath:
                                        escaped_xpath = xpath.replace("'", "\\'")
                                        js_highlight_script = f"""
                                            const iterator = document.evaluate('{escaped_xpath}', document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null);
                                            const el = iterator.singleNodeValue;
                                            if (el && !el.classList.contains('pw-selected')) {{
                                                el.classList.add('pw-predicted');
                                                return 1;
                                            }}
                                            return 0;
                                        """
                                    elif idx is not None:
                                        js_highlight_script = f"""
                                            const allElements = document.querySelectorAll('*');
                                            const el = allElements[{idx}];
                                            if (el && !el.classList.contains('pw-selected')) {{
                                                el.classList.add('pw-predicted');
                                                return 1;
                                            }}
                                            return 0;
                                        """

                                    if js_highlight_script:
                                        result = page.evaluate(f"(() => {{ try {{ {js_highlight_script} }} catch(e) {{ return 0; }} }})()")
                                        highlighted_count += result

                                except Exception as e:
                                    log_warning(f"Error highlighting candidate: {e}")

                            log_success(f"Highlighted {highlighted_count} predicted elements")
                        else:
                            log_warning("No predictions found for this category")
                    
                    last_category = category

                # 2. Update UI Overlay
                ui_updated = update_ui_state(page, category, len(current_selection_list), 
                                            current_idx, len(product_scraper.categories))
                
                # If UI failed to update (page refresh/navigation), reinject
                if not ui_updated:
                    time.sleep(1)
                    if inject_ui_scripts(page):
                        continue
                    # If injection fails repeatedly, break loop
                    log_error("Lost connection to page UI")
                    break
                
                # 3. Highlight selections (only if changed)
                current_hash = hash(tuple(current_selection_list))
                if current_hash != last_selections_hash:
                    highlight_selectors(page, current_selection_list, force_update=True)
                    last_selections_hash = current_hash

                # 4. Poll for interactions
                action_type, action_payload = poll_for_action(page)

                if action_type == 'toggle':
                    handle_toggle_action(action_payload, category, selections, 
                                       undo_stack, redo_stack, current_idx)
                    last_selections_hash = None

                elif action_type == 'navigate':
                    if action_payload == 'select_predicted':
                        # --- SELECT ALL PREDICTED ---
                        log_info("Selecting all predicted elements")
                        try:
                            selectors_added = page.evaluate("""
                                (() => {
                                    const predicted = document.querySelectorAll('.pw-predicted');
                                    const selectors = [];
                                    predicted.forEach(el => {
                                        const selector = window._generateSelector(el);
                                        if (selector && !el.classList.contains('pw-selected')) {
                                            el.classList.remove('pw-predicted');
                                            el.classList.add('pw-selected');
                                            selectors.push(selector);
                                        }
                                    });
                                    return selectors;
                                })()
                            """)
                            
                            if selectors_added:
                                undo_stack.append((current_idx, copy.deepcopy(selections)))
                                redo_stack.clear()
                                
                                if category not in selections:
                                    selections[category] = []
                                
                                for selector in selectors_added:
                                    if selector not in selections[category]:
                                        selections[category].append(selector)
                                
                                log_success(f"Added {len(selectors_added)} elements")
                                last_selections_hash = None
                            else:
                                log_warning("No highlighted predictions to select")
                                
                        except Exception as e:
                            log_error(f"Error selecting predicted: {e}")
                    
                    else:
                        # Normal Navigation (Next/Prev/Done)
                        current_idx, should_exit = handle_navigate_action(action_payload, current_idx, undo_stack, redo_stack, page)
                        last_selections_hash = None

                elif action_type == 'history':
                    selections = handle_history_action(
                        action_payload, current_idx, selections, undo_stack, redo_stack)
                    last_selections_hash = None

        except PlaywrightError as e:
            if "Target closed" in str(e):
                log_warning("Window closed")
            else:
                log_error(f"Playwright Error: {e}")
        except KeyboardInterrupt:
            log_warning("Interrupted")
        except Exception as e:
            log_error(f"Error: {e}")
        finally:
            try:
                browser.close()
            except Exception:
                pass
            finally:
                return selections
