import lxml.html
import pandas as pd
from utils.features import extract_element_features, get_feature_columns, validate_features

def test_extract_element_features_basic():
    html = '<div style="font-size:18px;font-weight:bold">Test <span>content</span></div>'
    elem = lxml.html.fromstring(html)
    features = extract_element_features(elem)
    assert isinstance(features, dict)
    assert features['tag'] == 'div'
    assert features['font_size'] == 18.0
    assert features['font_weight'] == 700
    assert features['num_children'] == 1

def test_get_feature_columns():
    cols = get_feature_columns()
    assert 'numeric' in cols and 'categorical' in cols and 'text' in cols and 'all' in cols
    assert isinstance(cols['all'], list)

def test_validate_features():
    cols = get_feature_columns()['all']
    df = pd.DataFrame([{col: 1 for col in cols}])
    assert validate_features(df)
    df_missing = pd.DataFrame([{'foo': 1}])
    assert not validate_features(df_missing)
    df_empty = pd.DataFrame([])
    assert not validate_features(df_empty)
