"""Test script to verify feature consistency between extraction and training."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from utils.features import (
    extract_element_features, 
    NUMERIC_FEATURES, 
    CATEGORICAL_FEATURES, 
    TEXT_FEATURES, 
    ALL_FEATURES,
    validate_features
)
from train_model.process_data import html_to_dataframe
import lxml.html


def test_feature_extraction():
    """Test that extract_element_features returns all expected features."""
    html = '<div class="test-class" id="test-id">Test <span>content</span></div>'
    elem = lxml.html.fromstring(html)
    features = extract_element_features(elem)
    
    print('='*60)
    print('TEST 1: Feature Extraction Completeness')
    print('='*60)
    
    extracted_keys = [k for k in features.keys() if k != 'Category']
    
    print(f'\nExtracted features count: {len(extracted_keys)}')
    print(f'Expected features count: {len(ALL_FEATURES)}')
    
    missing = set(ALL_FEATURES) - set(extracted_keys)
    extra = set(extracted_keys) - set(ALL_FEATURES)
    
    if missing:
        print(f'\n❌ MISSING in extraction: {missing}')
        return False
    if extra:
        print(f'\n❌ EXTRA in extraction: {extra}')
        return False
    
    print('\n✓ All features match perfectly!')
    return True


def test_dataframe_features():
    """Test that html_to_dataframe creates DataFrames with correct features."""
    html = '''
    <html>
        <body>
            <div class="price">$99.99</div>
            <h1 class="title">Product Name</h1>
            <img src="product.jpg" alt="Product" />
        </body>
    </html>
    '''
    
    print('\n' + '='*60)
    print('TEST 2: DataFrame Feature Completeness')
    print('='*60)
    
    df = html_to_dataframe(html)
    
    if df.empty:
        print('\n❌ DataFrame is empty')
        return False
    
    print(f'\nDataFrame shape: {df.shape}')
    print(f'Expected features: {len(ALL_FEATURES) + 1}')  # +1 for Category
    print(f'Actual columns: {len(df.columns)}')
    
    # Check all expected features are present
    missing_cols = []
    for feature in ALL_FEATURES:
        if feature not in df.columns:
            missing_cols.append(feature)
    
    if missing_cols:
        print(f'\n❌ Missing columns in DataFrame: {missing_cols}')
        return False
    
    if 'Category' not in df.columns:
        print('\n❌ Missing Category column in DataFrame')
        return False
    
    print('\n✓ DataFrame has all required features!')
    return True


def test_feature_categories():
    """Test that feature categories are correctly defined."""
    print('\n' + '='*60)
    print('TEST 3: Feature Category Definitions')
    print('='*60)
    
    print(f'\nNumeric features ({len(NUMERIC_FEATURES)}):')
    for f in NUMERIC_FEATURES:
        print(f'  - {f}')
    
    print(f'\nCategorical features ({len(CATEGORICAL_FEATURES)}):')
    for f in CATEGORICAL_FEATURES:
        print(f'  - {f}')
    
    print(f'\nText features ({len(TEXT_FEATURES)}):')
    for f in TEXT_FEATURES:
        print(f'  - {f}')
    
    # Check for duplicates
    all_listed = NUMERIC_FEATURES + CATEGORICAL_FEATURES + TEXT_FEATURES
    if len(all_listed) != len(set(all_listed)):
        print('\n❌ Duplicate features found across categories!')
        return False
    
    if len(all_listed) != len(ALL_FEATURES):
        print(f'\n❌ Feature count mismatch: {len(all_listed)} vs {len(ALL_FEATURES)}')
        return False
    
    print('\n✓ Feature categories are correctly defined!')
    return True


def test_validation_function():
    """Test the validate_features function."""
    print('\n' + '='*60)
    print('TEST 4: Feature Validation Function')
    print('='*60)
    
    # Create a valid DataFrame
    html = '<div class="test">Test</div>'
    df = html_to_dataframe(html)
    
    if df.empty:
        print('\n⚠️  Cannot test with empty DataFrame')
        return True
    
    # Test with valid DataFrame
    if validate_features(df):
        print('\n✓ Validation correctly passes valid DataFrame')
    else:
        print('\n❌ Validation incorrectly fails valid DataFrame')
        return False
    
    # Test with missing column
    df_invalid = df.drop(columns=['text_len'])
    if not validate_features(df_invalid):
        print('✓ Validation correctly rejects DataFrame with missing features')
    else:
        print('❌ Validation incorrectly passes invalid DataFrame')
        return False
    
    return True


if __name__ == '__main__':
    print('\n' + '='*60)
    print('FEATURE CONSISTENCY VALIDATION TEST SUITE')
    print('='*60)
    
    results = []
    results.append(('Feature Extraction', test_feature_extraction()))
    results.append(('DataFrame Features', test_dataframe_features()))
    results.append(('Feature Categories', test_feature_categories()))
    results.append(('Validation Function', test_validation_function()))
    
    print('\n' + '='*60)
    print('TEST RESULTS SUMMARY')
    print('='*60)
    
    for test_name, passed in results:
        status = '✓ PASS' if passed else '❌ FAIL'
        print(f'{status}: {test_name}')
    
    all_passed = all(result[1] for result in results)
    
    print('\n' + '='*60)
    if all_passed:
        print('🎉 ALL TESTS PASSED!')
    else:
        print('❌ SOME TESTS FAILED')
    print('='*60)
    
    sys.exit(0 if all_passed else 1)
