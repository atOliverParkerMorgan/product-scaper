"""Model Training Pipeline."""

import warnings
from typing import List

from rich.panel import Panel

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from utils.console import CONSOLE, log_info, log_error
from utils.features import NUMERIC_FEATURES, CATEGORICAL_FEATURES, TEXT_FEATURES, NON_TRAINING_FEATURES, TARGET_FEATURE
# Constants
from train_model.process_data import RANDOM_SEED
warnings.filterwarnings('ignore')


def build_pipeline(num_cols: List[str], cat_cols: List[str], text_cols: List[str]) -> Pipeline:
    """
    Constructs a robust preprocessing and training pipeline.
    
    Args:
        num_cols: List of numeric feature column names
        cat_cols: List of categorical feature column names
        text_cols: List of text feature column names (for TF-IDF)
    
    Returns:
        Sklearn Pipeline with preprocessing and Random Forest classifier
    """
    
    # 1. Preprocessing Steps
    transformers = [
        # Numeric: Standard Scaling
        ('num', StandardScaler(), num_cols),
        
        # Categorical: One Hot Encoding
        ('cat', OneHotEncoder(handle_unknown='ignore', min_frequency=5), cat_cols),
    ]
    
    # 2. Add TF-IDF transformers for text features
    # Text features like class_str and id_str often contain 'title', 'price', etc.
    # min_df=1 allows single document vocabularies (for small datasets)
    if 'class_str' in text_cols:
        transformers.append(
            ('txt_class', TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), max_features=1000, min_df=1), 'class_str')
        )
    if 'id_str' in text_cols:
        transformers.append(
            ('txt_id', TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), max_features=1000, min_df=1), 'id_str')
        )
    
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='drop'
    )
    
    # class_weight='balanced' automatically handles the "Other" category dominance
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,     # Let trees grow deep to catch specific HTML patterns
        min_samples_leaf=2, # Prevent overfitting
        n_jobs=-1,          # Use all CPU cores
        random_state=RANDOM_SEED,
        class_weight='balanced' # Crucial for handling class imbalance
    )
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', clf)
    ])
    
    return pipeline

def train_model(df, pipeline: Pipeline = None, test: bool = False):
    
    log_info("Starting Training Pipeline (Random Forest)")
        
    # Use feature definitions from unified features module
    numeric_features = NUMERIC_FEATURES.copy()
    categorical_features = CATEGORICAL_FEATURES.copy()
    text_features = TEXT_FEATURES.copy()
    
    # Filter only existing columns
    numeric_features = [c for c in numeric_features if c in df.columns]
    categorical_features = [c for c in categorical_features if c in df.columns]
    text_features = [c for c in text_features if c in df.columns]
    
    if TARGET_FEATURE not in df.columns:
        log_error(f"Target column '{TARGET_FEATURE}' not found in data")
        return

    # Only drop columns that actually exist in the dataframe
    cols_to_drop = [col for col in NON_TRAINING_FEATURES if col in df.columns]
    X = df.drop(columns=cols_to_drop)
    y = df[TARGET_FEATURE]
    
    # Label Encoding Target
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    if pipeline is None:
        pipeline = build_pipeline(numeric_features, categorical_features, text_features)
    
    # Only split and test if test=True
    if test:
        # Split
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=0.2, random_state=RANDOM_SEED, stratify=y_encoded
            )
        except ValueError as e:
            log_error(f"Error during train-test split: {e}")
            return
        
        log_info(f"Training on {len(X_train)} samples using Random Forest")
        
        # Random Forest uses class_weight='balanced' in init, so we don't need to pass weights here
        pipeline.fit(X_train, y_train)
        
        log_info("Evaluating")
        y_pred = pipeline.predict(X_test)
        y_test_decoded = label_encoder.inverse_transform(y_test)
        y_pred_decoded = label_encoder.inverse_transform(y_pred)
        
        report = classification_report(y_test_decoded, y_pred_decoded)
        CONSOLE.print(Panel(report, title="Classification Report"))
    else:
        # Train on full dataset without split
        log_info(f"Training on {len(X)} samples using Random Forest")
        pipeline.fit(X, y_encoded)
    

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
        