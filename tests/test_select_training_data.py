import pytest

from create_data import select_training_data


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
            if 'INVALID' in js:
                raise Exception('bad xpath')
            return None
    page = DummyPage()
    try:
        select_training_data.highlight_selectors(page, ["//div", "INVALID"])
    except Exception:
        pytest.fail("highlight_selectors should not propagate exceptions")

def test_handle_toggle_action_add_and_remove():
    selections = {}
    undo_stack = []
    redo_stack = []
    select_training_data.handle_toggle_action('sel', 'cat', selections, undo_stack, redo_stack, 0)
    assert 'sel' in selections['cat']
    select_training_data.handle_toggle_action('sel', 'cat', selections, undo_stack, redo_stack, 0)
    assert 'sel' not in selections['cat']

def test_handle_history_action_undo_redo():
    selections = {'cat': ['a']}
    undo_stack = [(0, {'cat': []})]
    redo_stack = []
    out = select_training_data.handle_history_action('undo', 0, selections, undo_stack, redo_stack)
    assert out == {'cat': []}
    out2 = select_training_data.handle_history_action('redo', 0, out, undo_stack, redo_stack)
    assert out2 == {'cat': ['a']}
