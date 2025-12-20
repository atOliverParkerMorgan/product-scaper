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

def test_regex_features_extraction():
    html = '''<div>Price: $19.99 <span>Sold Out</span> <span>Review</span> <span>Buy now</span></div>'''
    elem = lxml.html.fromstring(html)
    features = extract_element_features(elem)
    assert features['has_currency_symbol'] == 1
    assert features['is_price_format'] == 1
    assert features['has_sold_keyword'] == 1
    assert features['has_review_keyword'] == 1
    assert features['has_cta_keyword'] == 1

def test_image_feature_extraction():
    html = '<img src="foo.jpg" alt="product" width="100" height="50" style="width:100px;height:50px">'
    elem = lxml.html.fromstring(html)
    features = extract_element_features(elem)
    assert features['is_image'] == 1
    assert features['has_src'] == 1
    assert features['has_alt'] == 1
    assert features['alt_len'] == len('product')
    assert features['image_area'] == 5000
    # Test extraction from style only
    html2 = '<img src="foo.jpg" style="width:60px;height:40px">'
    elem2 = lxml.html.fromstring(html2)
    features2 = extract_element_features(elem2)
    assert features2['image_area'] == 2400
