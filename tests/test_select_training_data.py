from pathlib import Path

import pytest
from playwright.sync_api import sync_playwright

# Import the module so we can locate it on the filesystem
from product_scraper.create_data import select_training_data

# --- UNIT TESTS ---


def test_inject_ui_scripts_runs():
    """Test that scripts are injected without error."""

    class DummyPage:
        def add_style_tag(self, content):
            pass

        def evaluate(self, js, *args, **kwargs):
            return True

    page = DummyPage()
    assert select_training_data.inject_ui_scripts(page) is True


def test_highlight_selectors_handles_invalid():
    """Test that invalid selectors do not crash the script."""

    class DummyPage:
        def evaluate(self, js):
            if "INVALID" in js:
                raise Exception("Playwright Error")
            return None

    page = DummyPage()
    try:
        select_training_data.highlight_selectors(page, ["//div", "INVALID"])
    except Exception as e:
        pytest.fail(f"highlight_selectors should not propagate exceptions: {e}")


def test_handle_toggle_action_add_and_remove():
    """Test adding and removing selections via the helper."""
    selections = {}
    undo_stack = []
    redo_stack = []

    # 1. Add
    select_training_data.handle_toggle_action(
        "sel", "cat", selections, undo_stack, redo_stack, 0
    )
    assert "sel" in selections["cat"]
    assert len(undo_stack) == 1

    # 2. Remove
    select_training_data.handle_toggle_action(
        "sel", "cat", selections, undo_stack, redo_stack, 0
    )
    assert "sel" not in selections["cat"]
    assert len(undo_stack) == 2


def test_handle_history_action_undo_redo():
    """Test undo/redo logic via the helper."""
    selections = {"cat": ["a"]}
    undo_stack = [(0, {"cat": []})]
    redo_stack = []

    # 1. Undo
    out = select_training_data.handle_history_action(
        "undo", 0, selections, undo_stack, redo_stack
    )
    assert out == {"cat": []}
    assert len(redo_stack) == 1

    # 2. Redo
    out2 = select_training_data.handle_history_action(
        "redo", 0, out, undo_stack, redo_stack
    )
    assert out2 == {"cat": ["a"]}
    assert len(redo_stack) == 0


# --- INTEGRATION TESTS (PLAYWRIGHT) ---


@pytest.fixture(scope="session")
def corejs_code():
    """
    Locates core.js by looking at the actual file location of
    the imported select_training_data module.
    """
    # 1. Get the directory containing select_training_data.py
    #    (e.g., .../src/product_scraper/create_data)
    module_dir = Path(select_training_data.__file__).parent

    # 2. Resolve ui/core.js relative to that directory
    core_js_path = module_dir / "ui" / "core.js"

    if not core_js_path.exists():
        pytest.fail(f"File not found: {core_js_path}")

    return core_js_path.read_text(encoding="utf-8")


@pytest.fixture(scope="function")
def browser_page(corejs_code):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        # Minimal HTML structure
        page.set_content("""
            <html>
                <body>
                    <div id="target" class="item">Click Me</div>
                    <div class="pw-predicted" id="p1">Predicted</div>
                </body>
            </html>
        """)
        # Inject the core logic manually for testing
        page.evaluate(corejs_code)
        yield page
        browser.close()


def test_select_predicted_action(browser_page):
    """Test the 'Select All Predicted' logic."""
    # Ensure button exists before clicking
    btn = browser_page.locator('[data-testid="pw-btn-select-predicted"]')
    if not btn.is_visible():
        pytest.fail("Predicted button not visible - check core.js injection")

    btn.click()

    # Verify window._action was set
    action = browser_page.evaluate("window._action")
    assert action == "select_predicted"

    # Simulate the Python-side response (swapping classes)
    browser_page.evaluate("""
        document.querySelectorAll('.pw-predicted').forEach(el => {
            el.classList.remove('pw-predicted');
            el.classList.add('pw-selected');
        });
    """)

    classes = browser_page.locator("#p1").get_attribute("class")
    assert "pw-selected" in classes
    assert "pw-predicted" not in classes


def test_ui_renders(browser_page):
    """Test that the UI renders and testids are present."""
    assert browser_page.locator('[data-testid="pw-ui-header"]').is_visible()
    assert browser_page.locator('[data-testid="pw-ui-body"]').is_visible()
    assert browser_page.locator('[data-testid="pw-btn-next"]').is_visible()


def test_selector_box_updates_on_hover(browser_page):
    """Test hover updates the box with the new FULL XPath format."""
    browser_page.hover("#target")
    selector_text = browser_page.locator('[data-testid="pw-selector-box"]').inner_text()

    # Assert it uses the new detailed XPath format
    assert selector_text.startswith("/")
    assert "div" in selector_text
    assert "[" in selector_text
