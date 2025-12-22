import pytest

from product_scraper.create_data import select_training_data


def test_inject_ui_scripts_runs(monkeypatch):
    class DummyPage:
        def add_style_tag(self, content):
            self.style = content

        def evaluate(self, js, *args, **kwargs):
            return True

    page = DummyPage()
    assert select_training_data.inject_ui_scripts(page) is True


def test_highlight_selectors_handles_invalid(monkeypatch):
    class DummyPage:
        def evaluate(self, js):
            if "INVALID" in js:
                raise Exception("bad xpath")
            return None

    page = DummyPage()
    try:
        select_training_data.highlight_selectors(page, ["//div", "INVALID"])
    except Exception:
        pytest.fail("highlight_selectors should not propagate exceptions")
