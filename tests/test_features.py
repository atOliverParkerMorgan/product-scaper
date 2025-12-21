import pytest

try:
    from pytest_lazyfixture import lazy_fixture
except ImportError:
    lazy_fixture = None
import lxml
import pandas as pd

from utils.features import extract_element_features, get_feature_columns, validate_features

# Direct parameterization for feature extraction cases (no fixture needed)
feature_cases = [
    # Basic div
    ('<div>Simple text</div>', {'tag': 'div', 'text_len': 11, 'is_block_element': 1}),
    # Header
    ('<h1>Header</h1>', {'tag': 'h1', 'is_header': 1}),
    # Bold
    ('<b style="font-weight:bold">Bold</b>', {'is_bold': 1, 'font_weight': 700}),
    # Italic
    ('<i style="font-style:italic">Italic</i>', {'is_italic': 1}),
    # Hidden
    ('<span style="display:none">Hidden</span>', {'is_hidden': 1}),
    # Price
    ('<span>$99.99</span>', {'has_currency_symbol': 1, 'is_price_format': 1}),
    # CTA
    ('<button>Buy now</button>', {'has_cta_keyword': 1}),
    # Review
    ('<div>5 stars review</div>', {'has_review_keyword': 1}),
    # Image
    ('<img src="foo.jpg" alt="bar" width="10" height="20">', {'is_image': 1, 'image_area': 200}),
    # List item
    ('<li>Item</li>', {'is_list_item': 1}),
]

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
    html = '''<div>Price: 19,99 ,- <span>Sold Out</span> <span>Review</span> <span>Buy now</span></div>'''
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
    assert features['img_area_raw'] == 5000
    # Test extraction from style only

    html2 = '<img src="foo.jpg" style="width:60px;height:40px">'
    elem2 = lxml.html.fromstring(html2)
    features2 = extract_element_features(elem2)
    assert features2['img_area_raw'] == 2400



# Parametrized test for various feature extraction cases using direct parameterization
@pytest.mark.parametrize("html,expected", feature_cases)
def test_feature_extraction_various(html, expected):
    elem = lxml.html.fromstring(html)
    features = extract_element_features(elem)
    for key, val in expected.items():
        assert features[key] == val, f"Feature {key} expected {val}, got {features[key]}"

def test_feature_extraction_csv_columns():
    # Simulate a row from the CSV and check feature mapping
    html = '<h2 class="product-title" id="7424" style="font-size:16px;font-weight:400">Art on Paper</h2>'
    elem = lxml.html.fromstring(html)
    features = extract_element_features(elem, category="title")
    # Check key features
    assert features['tag'] == 'h2'
    assert features['font_size'] == 16.0
    assert features['font_weight'] == 400
    assert features['is_header'] == 1
    assert features['class_str'] == 'product-title'
    assert features['id_str'] == '7424'
    assert features['Category'] == 'title'

def test_feature_extraction_edge_cases():
    # Empty element
    html = '<div></div>'
    elem = lxml.html.fromstring(html)
    features = extract_element_features(elem)
    assert features['text_len'] == 0
    assert features['text_word_count'] == 0
    assert features['avg_word_length'] == 0.0
    # Hidden element
    html = '<span style="display:none">Hidden</span>'
    elem = lxml.html.fromstring(html)
    features = extract_element_features(elem)
    assert features['is_hidden'] == 1
    # Block element
    html = '<section>Block</section>'
    elem = lxml.html.fromstring(html)
    features = extract_element_features(elem)
    assert features['is_block_element'] == 1
