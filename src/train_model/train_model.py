"""Advanced Model Training Pipeline."""

import logging
import warnings
from typing import List

from rich.console import Console
from rich.panel import Panel

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Constants
RANDOM_STATE = 42
CONSOLE = Console()

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


def build_pipeline(num_cols: List[str], cat_cols: List[str], text_cols: List[str]) -> Pipeline:
    """
    Constructs a robust preprocessing and training pipeline.
    """
    
    # 1. Preprocessing Steps
    preprocessor = ColumnTransformer(
        transformers=[
            # Numeric: Standard Scaling
            ('num', StandardScaler(), num_cols),
            
            # Categorical: One Hot Encoding
            ('cat', OneHotEncoder(handle_unknown='ignore', min_frequency=5), cat_cols),
            
            # Text: Class names often contain 'title', 'price', etc. 
            ('txt_class', TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), max_features=1000), 'class_str'),
            ('txt_id', TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), max_features=1000), 'id_str'),
        ],
        remainder='drop'
    )
    
    # class_weight='balanced' automatically handles the "Other" category dominance
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,     # Let trees grow deep to catch specific HTML patterns
        min_samples_leaf=2, # Prevent overfitting
        n_jobs=-1,          # Use all CPU cores
        random_state=RANDOM_STATE,
        class_weight='balanced' # Crucial for handling class imbalance
    )
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', clf)
    ])
    
    return pipeline

def train_model(df, pipeline: Pipeline = None):
    
    CONSOLE.print(Panel("[bold cyan]Starting Training Pipeline (Random Forest)[/bold cyan]"))
    
    target_col = 'Category'
    
    # Basic numeric features
    numeric_features = [
        'num_children', 'num_siblings', 'dom_depth', 'text_len', 
        'text_word_count', 'text_digit_count', 'text_density', 'reading_ease',
        'has_currency_symbol', 'is_price_format', 'has_href', 'is_image',
        'has_src', 'has_alt', 'alt_len', 'has_dimensions', 'parent_is_link', 'sibling_image_count'
    ]
    
    categorical_features = ['tag', 'parent_tag']
    text_features = ['class_str', 'id_str'] 
    
    # Filter only existing columns
    numeric_features = [c for c in numeric_features if c in df.columns]
    
    if target_col not in df.columns:
        logger.error(f"Target column '{target_col}' not found in data.")
        return

    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Label Encoding Target
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Split
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=RANDOM_STATE, stratify=y_encoded
        )
    except ValueError as e:
        logger.error(f"Error during train-test split: {e}")
        return
    
    if pipeline is None:
        pipeline = build_pipeline(numeric_features, categorical_features, text_features)
    
    CONSOLE.print(f"Training on {len(X_train)} samples using Random Forest...")
    
    # Random Forest uses class_weight='balanced' in init, so we don't need to pass weights here
    pipeline.fit(X_train, y_train)
    
    CONSOLE.print("[bold]Evaluating...[/bold]")
    y_pred = pipeline.predict(X_test)
    y_test_decoded = label_encoder.inverse_transform(y_test)
    y_pred_decoded = label_encoder.inverse_transform(y_pred)
    

    report = classification_report(y_test_decoded, y_pred_decoded)
    CONSOLE.print(Panel(report, title="Classification Report"))
    

    model_artifact = {
        'pipeline': pipeline,
        'label_encoder': label_encoder,
        'features': {
            'numeric': numeric_features,
            'categorical': categorical_features,
            'text': text_features
        }
    }

    return model_artifact
        