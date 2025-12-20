"""Tests for feature extraction and consistency in utils/features.py."""

import lxml.html
import pytest

from train_model.process_data import html_to_dataframe
from utils.features import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET_FEATURE,
    TEXT_FEATURES,
    extract_element_features,
    validate_features,
)


@pytest.fixture
def sample_html():
    return """
    <html>
        <body>
            <div class="product-card" id="p123">
                <h1 class="title">Test Product</h1>
                <span class="price">$10.00</span>
                <img src="test.jpg" alt="test">
            </div>
        </body>
    </html>
    """

def test_feature_extraction_completeness():
    """Verify that extract_element_features returns every expected key."""
    html = '<div class="test-class" id="test-id">Content</div>'
    elem = lxml.html.fromstring(html)
    features = extract_element_features(elem)

    extracted_keys = {k for k in features.keys() if k != TARGET_FEATURE}
    expected_keys = set(ALL_FEATURES)

    assert extracted_keys == expected_keys, f"Missing: {expected_keys - extracted_keys}"

def test_dataframe_structure(sample_html):
    """Ensure html_to_dataframe generates a valid schema."""
    df = html_to_dataframe(sample_html)

    assert not df.empty
    assert TARGET_FEATURE in df.columns
    for feature in ALL_FEATURES:
        assert feature in df.columns

def test_feature_category_uniqueness():
    """Ensure no overlap between feature types."""
    combined = NUMERIC_FEATURES + CATEGORICAL_FEATURES + TEXT_FEATURES
    assert len(combined) == len(set(combined)), "Duplicate features across categories"
    assert set(combined) == set(ALL_FEATURES)

def test_validation_logic(sample_html):
    """Test the validate_features utility."""
    df = html_to_dataframe(sample_html)
    assert validate_features(df) is True

    invalid_df = df.drop(columns=[NUMERIC_FEATURES[0]])
    assert validate_features(invalid_df) is False
