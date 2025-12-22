import copy
import time
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import sync_playwright

# Assuming these imports exist in your project structure
from train_model.predict_data import predict_category_selectors
from utils.console import log_error, log_info, log_warning

if TYPE_CHECKING:
    from ..core import ProductScraper
UI_PATH = Path(__file__).parent / "ui"

# Load files (Ensure these files exist)
try:
    CSS_CONTENT = (UI_PATH / "styles.css").read_text(encoding="utf-8")
    # JS_CORE_LOGIC should now contain the JavaScript provided above
    JS_CORE_LOGIC = (UI_PATH / "core.js").read_text(encoding="utf-8")
    JS_UPDATE_UI = (UI_PATH / "update.js").read_text(encoding="utf-8")
except FileNotFoundError as e:
    log_error(f"UI Resource missing: {e}")
    CSS_CONTENT, JS_CORE_LOGIC, JS_UPDATE_UI = "", "", ""


def highlight_selectors(page, selectors: List[str], force_update: bool = False) -> None:
    try:
        if force_update:
            page.evaluate(
                "document.querySelectorAll('.pw-selected').forEach(el => el.classList.remove('pw-selected'))"
            )

        for xpath in selectors:
            safe_xpath = xpath.replace('"', '\\"').replace("\n", " ")
            page.evaluate(f"""
                (() => {{
                    try {{
                        const result = document.evaluate("{safe_xpath}", document, null, XPathResult.ORDERED_NODE_SNAPSHOT_TYPE, null);
                        for (let i = 0; i < result.snapshotLength; i++) {{
                            const el = result.snapshotItem(i);
                            el.classList.add('pw-selected');
                            el.classList.remove('pw-predicted'); 
                        }}
                    }} catch(e) {{}}
                }})()
            """)
    except Exception:
        pass


def inject_ui_scripts(page) -> bool:
    """Injects scripts and returns True if successful, False if page is dead/navigating."""
    try:
        if CSS_CONTENT.strip():
            page.add_style_tag(content=CSS_CONTENT)

        if JS_CORE_LOGIC.strip():
            page.evaluate(JS_CORE_LOGIC)

        # Fallback generator check
        page.evaluate("""
            if (typeof window._generateSelector === 'undefined') {
                window._generateSelector = function(el) { return el.tagName.toLowerCase(); }
            }
        """)
        return True
    except PlaywrightError:
        return False


def update_ui_state(
    page, category: str, selection_count: int, current_idx: int, total_categories: int
) -> bool:
    try:
        page.evaluate(
            JS_UPDATE_UI,
            {
                "category": category,
                "count": selection_count,
                "idx": current_idx,
                "total": total_categories,
            },
        )
        return True
    except PlaywrightError:
        return False


def poll_for_action(page, timeout: float = 0.1) -> tuple:
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            # 1. Clicked Element
            clicked = page.evaluate("window._clickedSelector")
            if clicked:
                page.evaluate("window._clickedSelector = null")
                return "toggle", clicked

            # 2. UI Buttons
            ui_btn = page.evaluate("window._action")
            if ui_btn:
                page.evaluate("window._action = null")
                return "navigate", ui_btn

            # 3. Keyboard
            key_act = page.evaluate("window._keyAction")
            if key_act:
                page.evaluate("window._keyAction = null")
                return "history", key_act
        except PlaywrightError:
            return "error", None

        time.sleep(0.02)
    return None, None


def select_data(product_scraper: "ProductScraper", url: str) -> Dict[str, List[str]]:
    selections: Dict[str, List[str]] = {}

    with sync_playwright() as p:
        # Launch browser with specific args to help prevent crashes/hanging
        browser = p.chromium.launch(
            headless=False,
            args=["--start-maximized", "--disable-blink-features=AutomationControlled"],
        )
        context = browser.new_context(no_viewport=True)

        # Enable bypassing CSP if scripts are blocked
        context.set_extra_http_headers({"Accept-Language": "en-US,en;q=0.9"})

        page = context.new_page()

        log_info(f"Navigating to {url}")
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=60000)
        except Exception as e:
            log_warning(f"Initial navigation issue: {e}")

        if not inject_ui_scripts(page):
            log_error("Failed to inject UI scripts on load.")
            return {}

        undo_stack = []
        redo_stack = []
        current_idx = 0
        last_selections_hash = None
        last_category = None

        try:
            should_exit = False
            while current_idx < len(product_scraper.categories) and not should_exit:
                category = product_scraper.categories[current_idx]
                current_selection_list = selections.get(category, [])

                # --- 1. PREDICTIONS ---
                if category != last_category:
                    log_info(f"Category: {category}")
                    # Clear previous predictions first
                    try:
                        page.evaluate(
                            "document.querySelectorAll('.pw-predicted').forEach(el => el.classList.remove('pw-predicted'))"
                        )
                    except:
                        pass

                    # Run predictions logic...
                    if product_scraper.model:
                        try:
                            # Assuming predict_category_selectors is defined elsewhere
                            html = page.content()
                            predicted = predict_category_selectors(
                                product_scraper.model,
                                html,
                                category,
                                existing_selectors=selections,
                            )

                            # Simple Highlighting Logic for predictions
                            for cand in predicted:
                                xpath = cand.get("xpath", "").replace("'", "\\'")
                                if xpath:
                                    page.evaluate(f"""
                                        const r = document.evaluate('{xpath}', document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null);
                                        if (r.singleNodeValue && !r.singleNodeValue.classList.contains('pw-selected')) {{
                                            r.singleNodeValue.classList.add('pw-predicted');
                                        }}
                                    """)
                        except Exception as e:
                            log_warning(f"Prediction error: {e}")

                    last_category = category

                # --- 2. UPDATE UI ---
                if not update_ui_state(
                    page,
                    category,
                    len(current_selection_list),
                    current_idx,
                    len(product_scraper.categories),
                ):
                    # Connection lost? Try re-injecting
                    log_warning("UI lost. Re-injecting...")
                    time.sleep(0.5)
                    if not inject_ui_scripts(page):
                        log_error("Cannot recover connection to page.")
                        break

                # --- 3. REFRESH HIGHLIGHTS ---
                # Calculate hash to avoid expensive DOM calls every loop
                current_hash = hash(tuple(sorted(current_selection_list)))
                if current_hash != last_selections_hash:
                    highlight_selectors(page, current_selection_list, force_update=True)
                    last_selections_hash = current_hash

                # --- 4. POLL USER ACTIONS ---
                action_type, action_payload = poll_for_action(page)

                if action_type == "error":
                    # Page probably refreshed or closed
                    log_warning("Page context invalid, attempting to reconnect...")
                    time.sleep(1)
                    inject_ui_scripts(page)
                    last_selections_hash = None  # Force re-draw
                    continue

                if action_type == "toggle":
                    # Add/Remove logic
                    undo_stack.append((current_idx, copy.deepcopy(selections)))
                    redo_stack.clear()

                    if category not in selections:
                        selections[category] = []

                    if action_payload in selections[category]:
                        selections[category].remove(action_payload)
                    else:
                        selections[category].append(action_payload)

                    last_selections_hash = None

                elif action_type == "navigate":
                    if action_payload == "next":
                        current_idx += 1
                    elif action_payload == "prev":
                        if current_idx > 0:
                            current_idx -= 1
                    elif action_payload == "done":
                        should_exit = True
                    elif action_payload == "select_predicted":
                        # Grab all .pw-predicted items
                        new_sels = page.evaluate("""
                            (() => {
                                const res = [];
                                document.querySelectorAll('.pw-predicted').forEach(el => {
                                    const sel = window._generateSelector(el);
                                    if(sel) res.push(sel);
                                    el.classList.remove('pw-predicted');
                                    el.classList.add('pw-selected');
                                });
                                return res;
                            })()
                        """)
                        if new_sels:
                            undo_stack.append((current_idx, copy.deepcopy(selections)))
                            if category not in selections:
                                selections[category] = []
                            selections[category].extend(
                                [s for s in new_sels if s not in selections[category]]
                            )
                            last_selections_hash = None

                elif action_type == "history":
                    # Basic Undo/Redo logic
                    if action_payload == "undo" and undo_stack:
                        redo_stack.append((current_idx, copy.deepcopy(selections)))
                        idx, s = undo_stack.pop()
                        if idx == current_idx:
                            selections = s
                        else:
                            undo_stack.append((idx, s))  # Don't undo category change

                    elif action_payload == "redo" and redo_stack:
                        undo_stack.append((current_idx, copy.deepcopy(selections)))
                        idx, s = redo_stack.pop()
                        if idx == current_idx:
                            selections = s
                        else:
                            redo_stack.append((idx, s))

                    last_selections_hash = None

        except Exception as e:
            log_error(f"Selection loop error: {e}")
        finally:
            browser.close()

    return selections
