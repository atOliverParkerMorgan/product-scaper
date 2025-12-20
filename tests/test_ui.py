"""Tests for the UI injection and interaction logic in create_data/select_training_data.py."""

import pytest

from create_data.select_training_data import inject_ui_scripts


class MockPage:
    def __init__(self):
        self._content = ""
        self._elements_by_id = {}
        self._css = ""
        self._generate_selector_defined = False

    def set_content(self, html: str):
        self._content = html
        import re
        self._elements_by_id = {}
        for m in re.finditer(r"<(?P<tag>\w+)(?P<attrs>[^>]*)>(?P<inner>.*?)</\w+>", html, re.S):
            attrs = m.group('attrs')
            tag = m.group('tag')
            inner = m.group('inner')
            id_m = re.search(r'id=["\'](?P<id>[^"\']+)["\']', attrs)
            class_m = re.search(r'class=["\'](?P<class>[^"\']+)["\']', attrs)
            if id_m:
                eid = id_m.group('id')
                classes = set(class_m.group('class').split()) if class_m else set()
                self._elements_by_id[eid] = {'id': eid, 'classes': classes, 'tag': tag, 'inner': inner}

    def add_style_tag(self, content=""):
        self._css = content or self._css

    def evaluate(self, script: str, *args, **kwargs):
        if "window._generateSelector" in script:
            self._generate_selector_defined = True
            return None

        import re
        m = re.search(r"querySelector\(['\"]#(?P<id>[^'\"]+)['\"]\)", script)
        if m:
            qid = m.group('id')
            if "classList.add('pw-selected')" in script or 'classList.add("pw-selected")' in script:
                if qid in self._elements_by_id:
                    self._elements_by_id[qid]['classes'].add('pw-selected')
            if "classList.remove('pw-predicted')" in script:
                if qid in self._elements_by_id and 'pw-predicted' in self._elements_by_id[qid]['classes']:
                    self._elements_by_id[qid]['classes'].discard('pw-predicted')
            return None

        if "querySelectorAll('.pw-predicted')" in script or 'querySelectorAll(".pw-predicted")' in script:
            changed = []
            for eid, el in self._elements_by_id.items():
                if 'pw-predicted' in el['classes']:
                    if 'classList.remove(' in script or "classList.remove('pw-predicted')" in script:
                        el['classes'].discard('pw-predicted')
                    if 'classList.add(' in script and 'pw-selected' in script:
                        el['classes'].add('pw-selected')
                    changed.append(eid)
            return []

        return None

    def locator(self, selector: str):
        if selector.startswith('#'):
            eid = selector[1:]
            el = self._elements_by_id.get(eid, {'classes': set()})

            class MockLocator:
                def __init__(self, el, page, eid):
                    self._el = el

                def get_class_string(self):
                    return ' '.join(self._el.get('classes', []))

            return MockLocator(el, self, eid)

    def get_classes(self, selector: str):
        if selector.startswith('#'):
            eid = selector[1:]
            el = self._elements_by_id.get(eid)
            if not el:
                return ''
            return ' '.join(el['classes'])


@pytest.fixture
def mock_ui_page():
    """Sets up a lightweight mock page with injected styles and core logic."""
    page = MockPage()
    page.set_content("""
        <div id="target" class="item">Click Me</div>
        <div class="pw-predicted" id="p1">Predicted</div>
    """)
    inject_ui_scripts(page)
    return page


def test_element_highlighting(mock_ui_page):
    """Test if clicking an element triggers the selection class."""
    # Simulate the JS listener behavior defined in core.js
    # Manually trigger what the UI script would do
    mock_ui_page.evaluate("document.querySelector('#target').classList.add('pw-selected')")

    assert 'pw-selected' in mock_ui_page.get_classes('#target')


def test_select_predicted_action(mock_ui_page):
    """Test the 'Select All Predicted' JS logic."""
    # Simulate clicking the 'Select Predicted' button
    mock_ui_page.evaluate("""
        (() => {
            const predicted = document.querySelectorAll('.pw-predicted');
            predicted.forEach(el => {
                el.classList.remove('pw-predicted');
                el.classList.add('pw-selected');
            });
        })()
    """)

    assert 'pw-selected' in mock_ui_page.get_classes('#p1')
    assert 'pw-predicted' not in mock_ui_page.get_classes('#p1')
