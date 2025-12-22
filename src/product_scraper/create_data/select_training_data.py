"""
Interactive UI and browser automation for selecting training data for ProductScraper.
"""

import copy
import time
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import sync_playwright
from train_model.predict_data import predict_category_selectors
from utils.console import log_error, log_info, log_success, log_warning

if TYPE_CHECKING:
    from ProductScraper import ProductScraper

UI_PATH = Path(__file__).parent / "ui"

# Load UI resources safely
try:
    CSS_CONTENT = (UI_PATH / "styles.css").read_text(encoding="utf-8")
    JS_CORE_LOGIC = (UI_PATH / "core.js").read_text(encoding="utf-8")
    JS_UPDATE_UI = (UI_PATH / "update.js").read_text(encoding="utf-8")
except FileNotFoundError:
    CSS_CONTENT, JS_CORE_LOGIC, JS_UPDATE_UI = "", "", ""
    log_warning(
        "UI resource files not found. Ensure 'ui/core.js' and 'ui/styles.css' exist."
    )


def highlight_selectors(page, selectors: List[str], force_update: bool = False) -> None:
    """
    Helper to re-apply green outlines securely and remove predicted highlights.
    """
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
                    }} catch(e) {{
                        console.log("Invalid XPath skipped");
                    }}
                }})()
            """)
    except Exception:
        pass


def inject_ui_scripts(page) -> bool:
    """
    Inject CSS and JavaScript into the page. Returns False if page is dead.
    """
    try:
        if CSS_CONTENT.strip():
            page.add_style_tag(content=CSS_CONTENT)

        if JS_CORE_LOGIC.strip():
            page.evaluate(JS_CORE_LOGIC)

        # Fallback generator if JS file failed
        page.evaluate("""
            if (typeof window._generateSelector === 'undefined') {
                window._generateSelector = function(el) { return el.tagName.toLowerCase(); }
            }
        """)
        return True
    except PlaywrightError:
        log_warning("Could not inject scripts (page might be loading)")
        return False


def update_ui_state(
    page, category: str, selection_count: int, current_idx: int, total_categories: int
) -> bool:
    """
    Update the UI with current category and progress.
    """
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
    """
    Poll for user actions (click, button, keyboard).
    Returns ('type', payload) or (None, None).
    """
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


# --- Helper Functions (Extracted for Testing) ---


def handle_toggle_action(
    selector: str,
    category: str,
    selections: Dict[str, List[str]],
    undo_stack: list,
    redo_stack: list,
    current_idx: int,
) -> None:
    """Adds or removes a selector from the current category."""
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


def handle_history_action(
    action: str,
    current_idx: int,
    selections: Dict[str, List[str]],
    undo_stack: list,
    redo_stack: list,
) -> Dict[str, List[str]]:
    """Handles Undo and Redo operations."""
    if action == "undo" and undo_stack:
        redo_stack.append((current_idx, copy.deepcopy(selections)))
        idx, prev_selections = undo_stack.pop()

        # Only allow undo if we are in the same category step
        if idx == current_idx:
            log_info("Undo")
            return prev_selections
        else:
            undo_stack.append((idx, prev_selections))
            log_warning("Cannot undo across category changes")

    elif action == "redo" and redo_stack:
        undo_stack.append((current_idx, copy.deepcopy(selections)))
        idx, next_selections = redo_stack.pop()
        if idx == current_idx:
            log_info("Redo")
            return next_selections
        else:
            redo_stack.append((idx, next_selections))

    return selections


# --- Main Loop ---


def select_data(product_scraper: "ProductScraper", url: str) -> Dict[str, List[str]]:
    selections: Dict[str, List[str]] = {}

    with sync_playwright() as p:
        # Launch options to prevent hanging on complex sites
        browser = p.chromium.launch(
            headless=False,
            args=["--start-maximized", "--disable-blink-features=AutomationControlled"],
        )
        context = browser.new_context(no_viewport=True)

        # Bypass some bot protections
        context.set_extra_http_headers({"Accept-Language": "en-US,en;q=0.9"})

        page = context.new_page()

        log_info(f"Navigating to {url}")
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=60000)
        except Exception as e:
            log_warning(f"Navigation warning: {e}")

        if not inject_ui_scripts(page):
            log_error("Failed to inject UI scripts. Exiting.")
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

                # --- 1. Auto-Predictions on Category Change ---
                if category != last_category:
                    log_info(f"Category: {category}")
                    try:
                        # Clear old predictions
                        page.evaluate(
                            "document.querySelectorAll('.pw-predicted').forEach(el => el.classList.remove('pw-predicted'))"
                        )

                        if product_scraper.model:
                            predicted = predict_category_selectors(
                                product_scraper.model,
                                page.content(),
                                category,
                                existing_selectors=selections,
                            )
                            if predicted:
                                count = 0
                                for cand in predicted:
                                    xpath = cand.get("xpath", "").replace("'", "\\'")
                                    if xpath:
                                        res = page.evaluate(f"""
                                            (() => {{
                                                const r = document.evaluate('{xpath}', document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null);
                                                if (r.singleNodeValue && !r.singleNodeValue.classList.contains('pw-selected')) {{
                                                    r.singleNodeValue.classList.add('pw-predicted');
                                                    return 1;
                                                }}
                                                return 0;
                                            }})()
                                        """)
                                        count += res
                                log_success(f"Found {count} predicted elements")
                    except Exception as e:
                        log_warning(f"Prediction error: {e}")

                    last_category = category

                # --- 2. Update UI & Connection Check ---
                if not update_ui_state(
                    page,
                    category,
                    len(current_selection_list),
                    current_idx,
                    len(product_scraper.categories),
                ):
                    log_warning("UI connection lost. Re-injecting scripts...")
                    time.sleep(1)
                    if not inject_ui_scripts(page):
                        log_error("Lost connection to page. Please restart.")
                        break

                # --- 3. Refresh Highlights ---
                # Only re-evaluate XPath if the list actually changed
                current_hash = hash(tuple(sorted(current_selection_list)))
                if current_hash != last_selections_hash:
                    highlight_selectors(page, current_selection_list, force_update=True)
                    last_selections_hash = current_hash

                # --- 4. Poll User Actions ---
                action_type, action_payload = poll_for_action(page)

                if action_type == "error":
                    time.sleep(0.5)
                    continue

                if action_type == "toggle":
                    handle_toggle_action(
                        action_payload,
                        category,
                        selections,
                        undo_stack,
                        redo_stack,
                        current_idx,
                    )
                    last_selections_hash = None

                elif action_type == "navigate":
                    if action_payload == "next":
                        undo_stack.clear()
                        redo_stack.clear()
                        current_idx += 1
                    elif action_payload == "prev":
                        undo_stack.clear()
                        redo_stack.clear()
                        if current_idx > 0:
                            current_idx -= 1
                    elif action_payload == "done":
                        should_exit = True
                    elif action_payload == "select_predicted":
                        # Javascript grabs all highlighted predictions and returns their selectors
                        new_selectors = page.evaluate("""
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
                        if new_selectors:
                            undo_stack.append((current_idx, copy.deepcopy(selections)))
                            if category not in selections:
                                selections[category] = []
                            # Merge unique selectors
                            for s in new_selectors:
                                if s not in selections[category]:
                                    selections[category].append(s)
                            log_success(f"Added {len(new_selectors)} predicted items")
                            last_selections_hash = None

                elif action_type == "history":
                    selections = handle_history_action(
                        action_payload, current_idx, selections, undo_stack, redo_stack
                    )
                    last_selections_hash = None

        except Exception as e:
            log_error(f"Error in selection loop: {e}")
        finally:
            try:
                browser.close()
            except:
                pass

    return selections
