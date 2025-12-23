import copy
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import Page, Playwright, Route, sync_playwright

from product_scraper.train_model.predict_data import predict_category_selectors
from product_scraper.utils.console import log_error, log_info, log_warning

if TYPE_CHECKING:
    from ..core import ProductScraper

UI_PATH = Path(__file__).parent / "ui"

# Load UI resources
try:
    CSS_CONTENT = (UI_PATH / "styles.css").read_text(encoding="utf-8")
    JS_CORE_LOGIC = (UI_PATH / "core.js").read_text(encoding="utf-8")
    JS_UPDATE_UI = (UI_PATH / "update.js").read_text(encoding="utf-8")
except FileNotFoundError as e:
    log_error(f"UI Resource missing: {e}")
    CSS_CONTENT, JS_CORE_LOGIC, JS_UPDATE_UI = "", "", ""


class SelectionState:
    """
    Manages the state of the interactive selection session.
    """

    def __init__(self, categories: List[str]):
        self.categories = categories
        self.selections: Dict[str, List[str]] = {}
        self.undo_stack: List = []
        self.redo_stack: List = []
        self.current_idx: int = 0
        self.last_category: Optional[str] = None
        self.current_predictions: List[str] = []
        self.should_exit: bool = False

        # Optimization / Caching state
        self.last_sent_selections: Optional[Tuple] = None
        self.last_sent_predictions: Optional[Tuple] = None
        self.last_calc_class_candidates: List[str] = []
        self.last_calc_class_xpath: str = ""

    @property
    def current_category(self) -> str:
        """Returns the currently active category name."""
        if 0 <= self.current_idx < len(self.categories):
            return self.categories[self.current_idx]
        return ""

    @property
    def current_selection_list(self) -> List[str]:
        """Returns the list of selected XPaths for the current category."""
        return self.selections.get(self.current_category, [])

    @property
    def is_finished(self) -> bool:
        """Checks if the session should end."""
        return self.current_idx >= len(self.categories) or self.should_exit


def update_highlights(
    page: Page, selectors: List[str], predicted: Optional[List[str]] = None
) -> None:
    """
    Calls the optimized JS function to update highlights without flickering.

    Args:
        page (Page): The Playwright page object.
        selectors (List[str]): List of currently selected XPaths.
        predicted (Optional[List[str]]): List of predicted XPaths.

    Returns:
        None
    """
    try:
        page.evaluate(
            "([sel, pred]) => window._updateHighlights(sel, pred)",
            [selectors, predicted if predicted else []],
        )
    except Exception:
        pass


def inject_ui_scripts(page: Page) -> bool:
    """
    Injects the custom CSS and JS into the webpage.

    Args:
        page (Page): The Playwright page object.

    Returns:
        bool: True if injection was successful, False otherwise.
    """
    try:
        if CSS_CONTENT.strip():
            page.add_style_tag(content=CSS_CONTENT)
        if JS_CORE_LOGIC.strip():
            page.evaluate(JS_CORE_LOGIC)
        return page.evaluate("typeof window._generateSelector !== 'undefined'")
    except PlaywrightError:
        return False


def _setup_page(p: Playwright, url: str) -> Tuple[Optional[Page], Optional[Any]]:  # noqa: F821
    """
    Configures the browser, context, and page with navigation locks.

    Args:
        p (Playwright): The Playwright context.
        url (str): The target URL to scrape.

    Returns:
        Tuple[Optional[Page], Optional[BrowserContext]]: The configured page and context, or (None, None) on failure.
    """
    browser = p.chromium.launch(
        headless=False,
        args=["--start-maximized", "--disable-blink-features=AutomationControlled"],
    )
    context = browser.new_context(no_viewport=True)
    context.set_extra_http_headers({"Accept-Language": "en-US,en;q=0.9"})

    page = context.new_page()

    # Block navigation away from the target URL
    def route_handler(route: Route):
        request = route.request
        if (
            request.is_navigation_request()
            and request.frame == page.main_frame
            and request.url != url
            and request.url != "about:blank"
        ):
            log_warning(f"Blocked navigation to: {request.url}")
            route.abort()
        else:
            route.continue_()

    context.route("**/*", route_handler)

    log_info(f"Navigating to {url}")
    try:
        page.goto(url, wait_until="domcontentloaded", timeout=60000)
    except Exception as e:
        log_warning(f"Initial navigation warning: {e}")

    if not inject_ui_scripts(page):
        log_error("Failed to initialize UI.")
        browser.close()
        return None, None

    return page, browser


def _update_predictions_if_needed(
    state: SelectionState, scraper: "ProductScraper", page: Page
) -> None:
    """
    Runs the ML model to predict selectors if the category has changed.

    Args:
        state (SelectionState): The current session state.
        scraper (ProductScraper): The scraper instance with the model.
        page (Page): The Playwright page object.

    Returns:
        None
    """
    if state.current_category == state.last_category:
        return

    state.current_predictions = []
    if scraper.model:
        try:
            preds = predict_category_selectors(
                scraper.model,  # pyright: ignore[reportArgumentType]
                page.content(),
                state.current_category,
                existing_selectors=state.selections,
            )
            state.current_predictions = [  # pyright: ignore[reportAttributeAccessIssue]
                p.get("xpath") for p in preds if p.get("xpath")
            ]  # pyright: ignore[reportAssignmentType]
        except Exception:
            pass

    state.last_category = state.current_category
    # Reset class cache on category change
    state.last_calc_class_candidates = []
    state.last_calc_class_xpath = ""


def _calculate_class_candidates(state: SelectionState, page: Page) -> Set[str]:
    """
    Determines which elements match the CSS class of the last selected item.

    Args:
        state (SelectionState): The current session state.
        page (Page): The Playwright page object.

    Returns:
        Set[str]: A set of XPaths that match the class of the last selection.
    """
    current_list = state.current_selection_list

    if not current_list:
        state.last_calc_class_candidates = []
        return set()

    last_sel = current_list[-1]

    # Return cached result if the last selected item hasn't changed
    if last_sel == state.last_calc_class_xpath and state.last_calc_class_candidates:
        return set(state.last_calc_class_candidates)

    # Calculate new candidates
    try:
        safe_xpath = last_sel.replace('"', '\\"')
        class_selector = page.evaluate(f'window._getSameClassSelector("{safe_xpath}")')

        if class_selector:
            new_cands = page.evaluate(f"""
                (() => {{
                    const res = [];
                    document.querySelectorAll('.{class_selector}').forEach(el => {{
                        const xp = window._generateSelector(el);
                        if(xp) res.push(xp);
                    }});
                    return res;
                }})()
            """)
            state.last_calc_class_candidates = new_cands
        else:
            state.last_calc_class_candidates = []

        state.last_calc_class_xpath = last_sel
        return set(state.last_calc_class_candidates)

    except Exception:
        state.last_calc_class_candidates = []
        return set()


def _sync_ui_visuals(state: SelectionState, page: Page) -> None:
    """
    Updates the UI overlay text and element highlights on the page.

    Args:
        state (SelectionState): The current session state.
        page (Page): The Playwright page object.

    Returns:
        None
    """
    pred_set = set(state.current_predictions)
    sel_set = set(state.current_selection_list)

    # Check button states
    all_ai_selected = len(pred_set) > 0 and pred_set.issubset(sel_set)

    class_candidates_set = _calculate_class_candidates(state, page)
    all_class_selected = len(
        class_candidates_set
    ) > 0 and class_candidates_set.issubset(sel_set)

    # Update Text UI
    try:
        page.evaluate(
            JS_UPDATE_UI,
            {
                "category": state.current_category,
                "count": len(state.current_selection_list),
                "idx": state.current_idx,
                "total": len(state.categories),
                "totalPredictions": len(state.current_predictions),
                "allAiSelected": all_ai_selected,
                "allClassSelected": all_class_selected,
            },
        )
    except PlaywrightError:
        return

    # Efficient Highlight Update (Flicker-Free)
    state_sig = (tuple(state.current_selection_list), tuple(state.current_predictions))
    last_sig = (state.last_sent_selections, state.last_sent_predictions)

    if state_sig != last_sig:
        update_highlights(page, state.current_selection_list, state.current_predictions)
        state.last_sent_selections = tuple(state.current_selection_list)
        state.last_sent_predictions = tuple(state.current_predictions)


def _handle_toggle_action(state: SelectionState, xpath: str) -> None:
    """
    Handles a manual click on an element to toggle its selection status.

    Args:
        state (SelectionState): The current session state.
        xpath (str): The XPath of the clicked element.

    Returns:
        None
    """
    state.undo_stack.append((state.current_idx, copy.deepcopy(state.selections)))
    state.redo_stack.clear()

    if state.current_category not in state.selections:
        state.selections[state.current_category] = []

    current_list = state.selections[state.current_category]
    if xpath in current_list:
        current_list.remove(xpath)
    else:
        current_list.append(xpath)


def _handle_ui_buttons(state: SelectionState, payload: str, page: Page) -> None:
    """
    Handles clicks on the UI overlay buttons (Next, Prev, AI Toggle, Class Toggle).

    Args:
        state (SelectionState): The current session state.
        payload (str): The action payload identifier.
        page (Page): The Playwright page object.

    Returns:
        None
    """
    if payload == "next":
        state.current_idx += 1
    elif payload == "prev":
        if state.current_idx > 0:
            state.current_idx -= 1
    elif payload == "done":
        state.should_exit = True

    elif payload == "toggle_ai":
        _handle_toggle_ai(state)

    elif payload == "toggle_class":
        _handle_toggle_class(state)


def _handle_toggle_ai(state: SelectionState) -> None:
    """
    Toggles the selection of all AI-predicted elements.

    Args:
        state (SelectionState): The current session state.

    Returns:
        None
    """
    if state.current_category not in state.selections:
        state.selections[state.current_category] = []

    current_list = state.selections[state.current_category]
    pred_set = set(state.current_predictions)
    sel_set = set(current_list)

    all_ai_selected = len(pred_set) > 0 and pred_set.issubset(sel_set)

    if all_ai_selected:
        # Unselect AI
        state.selections[state.current_category] = [
            s for s in current_list if s not in pred_set
        ]
        log_info("Unselected AI predictions.")
    else:
        # Select AI
        for p in state.current_predictions:
            if p not in current_list:
                current_list.append(p)
        log_info("Selected AI predictions.")


def _handle_toggle_class(state: SelectionState) -> None:
    """
    Toggles the selection of all elements matching the class of the last selected item.

    Args:
        state (SelectionState): The current session state.

    Returns:
        None
    """
    if state.current_category not in state.selections:
        state.selections[state.current_category] = []

    candidates_set = set(state.last_calc_class_candidates)
    if not candidates_set:
        return

    current_list = state.selections[state.current_category]
    sel_set = set(current_list)

    all_class_selected = candidates_set.issubset(sel_set)

    if all_class_selected:
        # Unselect Class
        state.selections[state.current_category] = [
            s for s in current_list if s not in candidates_set
        ]
        log_info("Unselected class matches.")
    else:
        # Select Class
        added_count = 0
        for c in state.last_calc_class_candidates:
            if c not in current_list:
                current_list.append(c)
                added_count += 1
        log_info(f"Selected {added_count} class matches.")


def _handle_history_action(state: SelectionState, action: str) -> None:
    """
    Handles Undo/Redo keyboard shortcuts.

    Args:
        state (SelectionState): The current session state.
        action (str): Either "undo" or "redo".

    Returns:
        None
    """
    if action == "undo" and state.undo_stack:
        state.redo_stack.append((state.current_idx, copy.deepcopy(state.selections)))
        idx, prev_selections = state.undo_stack.pop()

        if idx == state.current_idx:
            state.selections = prev_selections
        else:
            # Prevent undoing across category navigation to avoid confusion
            state.undo_stack.append((idx, prev_selections))

    elif action == "redo" and state.redo_stack:
        state.undo_stack.append((state.current_idx, copy.deepcopy(state.selections)))
        idx, next_selections = state.redo_stack.pop()

        if idx == state.current_idx:
            state.selections = next_selections
        else:
            state.redo_stack.append((idx, next_selections))


def select_data(product_scraper: "ProductScraper", url: str) -> Dict[str, List[str]]:
    """
    Main entry point for interactive data selection.

    Initializes the browser, manages the event loop, and handles user interactions
    for selecting scraping targets on a webpage.

    Args:
        product_scraper (ProductScraper): The main scraper instance containing configuration and models.
        url (str): The URL of the page to label.

    Returns:
        Dict[str, List[str]]: A dictionary mapping categories to lists of selected XPaths.
    """
    state = SelectionState(product_scraper.categories)

    with sync_playwright() as p:
        page, browser = _setup_page(p, url)
        if not page:
            return {}

        while not state.is_finished:
            # 1. Update Context
            _update_predictions_if_needed(state, product_scraper, page)

            # 2. Update UI
            try:
                _sync_ui_visuals(state, page)
            except PlaywrightError:
                # If page is closed/refreshed, try to recover
                if not inject_ui_scripts(page):
                    break

            # 3. Poll for Actions
            action_type, action_payload = None, None
            for _ in range(10):  # Poll window ~200ms
                try:
                    if page.evaluate("window._clickedSelector"):
                        action_type = "toggle"
                        action_payload = page.evaluate("window._clickedSelector")
                        page.evaluate("window._clickedSelector = null")
                        break
                    if page.evaluate("window._action"):
                        action_type = "ui"
                        action_payload = page.evaluate("window._action")
                        page.evaluate("window._action = null")
                        break
                    if page.evaluate("window._keyAction"):
                        action_type = "history"
                        action_payload = page.evaluate("window._keyAction")
                        page.evaluate("window._keyAction = null")
                        break
                except PlaywrightError:
                    break
                time.sleep(0.02)

            if not action_type:
                continue

            # 4. Handle Actions
            if action_type == "toggle" and action_payload:
                _handle_toggle_action(state, action_payload)
            elif action_type == "ui" and action_payload:
                _handle_ui_buttons(state, action_payload, page)
            elif action_type == "history" and action_payload:
                _handle_history_action(state, action_payload)

        if browser:
            browser.close()

    return state.selections
