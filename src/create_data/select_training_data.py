import logging
import time
import yaml
import copy
from typing import Dict, List
from urllib.parse import urlparse
from pathlib import Path
from playwright.sync_api import sync_playwright, Error as PlaywrightError

# --- CONFIGURATION ---
logging.basicConfig(level=logging.INFO, format='%(message)s')

UI_PATH = Path(__file__).parent / 'ui'

if not UI_PATH.exists():
    raise FileNotFoundError(f"UI directory not found: {UI_PATH}")

CSS_CONTENT = (UI_PATH / 'styles.css').read_text(encoding='utf-8')
JS_CORE_LOGIC = (UI_PATH / 'core.js').read_text(encoding='utf-8')
JS_UPDATE_UI = (UI_PATH / 'update.js').read_text(encoding='utf-8')

def highlight_selectors(page, selectors):
    """Helper to re-apply green outlines securely."""
    try:
        page.evaluate("document.querySelectorAll('.pw-selected').forEach(el => el.classList.remove('pw-selected'))")
        for sel in selectors:
            safe_sel = sel.replace('"', '\\"')
            page.evaluate(f"""
                try {{
                    const els = document.querySelectorAll("{safe_sel}");
                    els.forEach(el => el.classList.add('pw-selected'));
                }} catch(e) {{}}
            """)
    except PlaywrightError:
        pass


def inject_ui_scripts(page):
    """Inject CSS and JavaScript into the page."""
    try:
        page.add_style_tag(content=CSS_CONTENT)
        page.evaluate(JS_CORE_LOGIC)
        return True
    except PlaywrightError:
        print("⚠️ Could not inject scripts (page might be loading).")
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
            clicked = page.evaluate("window._clickedSelector")
            if clicked:
                page.evaluate("window._clickedSelector = null")
                return 'toggle', clicked
            
            ui_btn = page.evaluate("window._action")
            if ui_btn:
                page.evaluate("window._action = null")
                return 'navigate', ui_btn
                
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
    undo_stack.append((current_idx, copy.deepcopy(selections)))
    redo_stack.clear()
    
    if category not in selections:
        selections[category] = []
    
    if selector in selections[category]:
        selections[category].remove(selector)
        print(f"[-] Removed: {selector}")
    else:
        selections[category].append(selector)
        print(f"[+] Added: {selector}")


def handle_navigate_action(direction, current_idx, categories, undo_stack, redo_stack):
    """Handle navigation action (next, prev, done)."""
    if direction == 'next':
        undo_stack.clear()
        redo_stack.clear()
        return current_idx + 1, False
    elif direction == 'prev' and current_idx > 0:
        undo_stack.clear()
        redo_stack.clear()
        return current_idx - 1, False
    elif direction == 'done':
        return current_idx, True
    return current_idx, False


def handle_history_action(cmd, current_idx, selections, undo_stack, redo_stack):
    """Handle undo/redo action."""
    if cmd == 'undo' and undo_stack:
        redo_stack.append((current_idx, copy.deepcopy(selections)))
        prev_idx, prev_selections = undo_stack.pop()
        if prev_idx == current_idx:
            print("↺ Undo")
            return prev_selections
        undo_stack.append((prev_idx, prev_selections))
    elif cmd == 'redo' and redo_stack:
        undo_stack.append((current_idx, copy.deepcopy(selections)))
        next_idx, next_selections = redo_stack.pop()
        if next_idx == current_idx:
            print("↻ Redo")
            return next_selections
        redo_stack.append((next_idx, next_selections))
    
    return selections


def navigate_to_url(page, url):
    """Navigate to the target URL."""
    print(f"\n🌐 Navigating to {url}...")
    try:
        page.goto(url, wait_until='domcontentloaded', timeout=60000)
    except Exception as e:
        print(f"⚠️ Navigation warning: {e}")


def select_data(url: str, categories: List[str]):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False, args=['--start-maximized'])
        context = browser.new_context(no_viewport=True)
        page = context.new_page()

        print(f"\n🌐 Navigating to {url}...")
        try:
            page.goto(url, wait_until='domcontentloaded', timeout=60000)
        except Exception as e:
            print(f"⚠️ Navigation warning: {e}")

        # Inject CSS/JS
        try:
            page.add_style_tag(content=CSS_CONTENT)
            page.evaluate(JS_CORE_LOGIC)
        except PlaywrightError:
            print("⚠️ Could not inject scripts (page might be loading).")

        selections: Dict[str, List[str]] = {}
        undo_stack = [] 
        redo_stack = []
        current_idx = 0
        
        print("✓ Ready! Interaction loop started.")

        try:
            while current_idx < len(categories):
                category = categories[current_idx]
                current_selection_list = selections.get(category, [])

                # 1. Update UI
                try:
                    page.evaluate(JS_UPDATE_UI, {
                        'category': category,
                        'count': len(current_selection_list),
                        'idx': current_idx,
                        'total': len(categories)
                    })
                except PlaywrightError:
                    # Retry injection if nav happened
                    try:
                        time.sleep(1)
                        page.add_style_tag(content=CSS_CONTENT)
                        page.evaluate(JS_CORE_LOGIC)
                        continue
                    except Exception:
                        break

                # 2. Highlight
                highlight_selectors(page, current_selection_list)

                # 3. Poll Loop
                action_type = None
                action_payload = None
                start_loop = time.time()
                
                while time.time() - start_loop < 0.2:
                    try:
                        # Check Click (Select)
                        clicked = page.evaluate("window._clickedSelector")
                        if clicked:
                            page.evaluate("window._clickedSelector = null")
                            action_type = 'toggle'
                            action_payload = clicked
                            break
                        
                        # Check Buttons
                        ui_btn = page.evaluate("window._action")
                        if ui_btn:
                            page.evaluate("window._action = null")
                            action_type = 'navigate'
                            action_payload = ui_btn
                            break
                            
                        # Check Keyboard
                        key_act = page.evaluate("window._keyAction")
                        if key_act:
                            page.evaluate("window._keyAction = null")
                            action_type = 'history'
                            action_payload = key_act
                            break
                    except PlaywrightError:
                        break
                    
                    time.sleep(0.05)

                # 4. Handle Logic
                if action_type == 'toggle':
                    undo_stack.append((current_idx, copy.deepcopy(selections)))
                    redo_stack.clear()
                    
                    selector = action_payload
                    if category not in selections:
                        selections[category] = []
                    
                    if selector in selections[category]:
                        selections[category].remove(selector)
                        print(f"[-] Removed: {selector}")
                    else:
                        selections[category].append(selector)
                        print(f"[+] Added: {selector}")

                elif action_type == 'navigate':
                    direction = action_payload
                    if direction == 'next':
                        current_idx += 1
                        undo_stack.clear(); redo_stack.clear()
                    elif direction == 'prev' and current_idx > 0:
                        current_idx -= 1
                        undo_stack.clear(); redo_stack.clear()
                    elif direction == 'done':
                        break
                
                elif action_type == 'history':
                    cmd = action_payload
                    if cmd == 'undo' and undo_stack:
                        redo_stack.append((current_idx, copy.deepcopy(selections)))
                        prev_idx, prev_selections = undo_stack.pop()
                        if prev_idx == current_idx:
                            selections = prev_selections
                            print("↺ Undo")
                        else:
                            undo_stack.append((prev_idx, prev_selections))
                    elif cmd == 'redo' and redo_stack:
                        undo_stack.append((current_idx, copy.deepcopy(selections)))
                        next_idx, next_selections = redo_stack.pop()
                        if next_idx == current_idx:
                            selections = next_selections
                            print("↻ Redo")
                        else:
                            redo_stack.append((next_idx, next_selections))

        except PlaywrightError as e:
            if "Target closed" in str(e):
                print("\n⚠️ Window closed.")
            else:
                print(f"\n❌ Playwright Error: {e}")
        except KeyboardInterrupt:
            print("\n🛑 Interrupted.")
        except Exception as e:
            print(f"\n❌ Error: {e}")
        finally:
            save_results(selections, url, page)
            try:
                browser.close()
            except Exception:
                pass

def save_results(selections, url, page):
    if not selections:
        return
    try:
        domain = urlparse(url).netloc.replace('www.', '')
        base_name = domain.split('.')[0] or 'site'
        
        # Use absolute path relative to this script
        save_dir = Path(__file__).parent.parent / 'data' / base_name
        
        # Ensure dir exists
        if not save_dir.exists():
            save_dir.mkdir(parents=True, exist_ok=True)
            print(f"📁 Created directory: {save_dir}")

        with open(save_dir / 'selectors.yaml', 'w', encoding='utf-8') as f:
            yaml.dump(selections, f, sort_keys=False)
        
        try:
            with open(save_dir / 'page.html', 'w', encoding='utf-8') as f:
                f.write(page.content())
        except Exception: 
            print("⚠️ Could not save HTML (browser context closed)")

        print(f"\n💾 Saved to: {save_dir}")
        print(yaml.dump(selections, sort_keys=False))
    except Exception as e:
        print(f"Save error: {e}")

if __name__ == "__main__":
    target_url = "https://books.toscrape.com/" 
    cats = ['product_name', 'price', 'image']
    select_data(target_url, cats)