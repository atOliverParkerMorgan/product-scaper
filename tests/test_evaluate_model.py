import numpy as np

from train_model.evaluate_model import calculate_meaningful_metrics


def test_calculate_meaningful_metrics_basic():
    y_true = np.array(['a', 'b', 'other', 'a'])
    y_pred = np.array(['a', 'b', 'other', 'b'])
    acc, f1, per_class = calculate_meaningful_metrics(y_true, y_pred)
    assert acc is not None and f1 is not None
    assert isinstance(per_class, dict)
    assert 'a' in per_class['f1'] and 'b' in per_class['f1']

def test_calculate_meaningful_metrics_all_excluded():
    y_true = np.array(['other', 'other'])
    y_pred = np.array(['other', 'other'])
    acc, f1, per_class = calculate_meaningful_metrics(y_true, y_pred)
    assert acc is None and f1 is None and per_class is None
