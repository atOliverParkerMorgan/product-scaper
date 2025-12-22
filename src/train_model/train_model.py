"""Model Training Pipeline."""

import warnings
from typing import Any, Dict, List, Optional

import pandas as pd
from rich.panel import Panel
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, ParameterGrid, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

# Constants
try:
    from train_model.process_data import RANDOM_SEED
    from utils.console import CONSOLE, log_error, log_info
    from utils.features import (
        CATEGORICAL_FEATURES,
        NON_TRAINING_FEATURES,
        NUMERIC_FEATURES,
        TARGET_FEATURE,
        TEXT_FEATURES,
    )
except ImportError:
    RANDOM_SEED = 42
    from rich.console import Console
    CONSOLE = Console()
    def log_info(msg): CONSOLE.log(f"[green]{msg}[/]")
    def log_error(msg): CONSOLE.log(f"[red]{msg}[/]")
    NUMERIC_FEATURES = []
    CATEGORICAL_FEATURES = []
    TEXT_FEATURES = ['class_str', 'id_str']
    NON_TRAINING_FEATURES = []
    TARGET_FEATURE = 'target'

warnings.filterwarnings('ignore')


def build_pipeline(num_cols: List[str], cat_cols: List[str], text_cols: List[str]) -> Pipeline:
    """
    Constructs a robust preprocessing and training pipeline.
    """
    transformers = []

    if num_cols:
        transformers.append(('num', StandardScaler(), num_cols))

    if cat_cols:
        transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', min_frequency=5), cat_cols))

    if 'class_str' in text_cols:
        transformers.append(
            ('txt_class', TfidfVectorizer(
                analyzer='char_wb',
                ngram_range=(2, 4),
                max_features=1000, 
                min_df=1 
            ), 'class_str')
        )
    if 'id_str' in text_cols:
        transformers.append(
            ('txt_id', TfidfVectorizer(
                analyzer='char_wb',
                ngram_range=(2, 4),
                max_features=1000,
                min_df=1
            ), 'id_str')
        )

    preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')

    clf = RandomForestClassifier(
        n_estimators=500,
        max_depth=20,
        min_samples_leaf=2,
        min_samples_split=5,
        max_features='sqrt',
        random_state=RANDOM_SEED,
        class_weight='balanced_subsample',
        n_jobs=-1
    )

    return Pipeline([('preprocessor', preprocessor), ('classifier', clf)])


def train_model(
    df: pd.DataFrame,
    pipeline: Optional[Pipeline] = None,
    test_size: float = 0.2,
    param_grid: Optional[Dict[str, Any]] = None,
    grid_search_cv: int = 3
) -> Optional[Dict[str, Any]]:
    
    log_info("Starting Training Pipeline (Random Forest)")

    if df is None or df.empty:
        log_error("Input DataFrame is empty.")
        return None

    if TARGET_FEATURE not in df.columns:
        log_error(f"Target column '{TARGET_FEATURE}' not found in data")
        return None

    data = df.copy()

    # --- Dynamic Feature Detection ---
    # Automatically include known numeric features, distance features, and the new density feature
    numeric_features = [c for c in data.columns if c in NUMERIC_FEATURES or 'dist_to_' in c or 'density' in c]
    # Ensure standard numeric features from constants are included if they exist
    for c in NUMERIC_FEATURES:
        if c in data.columns and c not in numeric_features:
            numeric_features.append(c)
            
    categorical_features = [c for c in CATEGORICAL_FEATURES if c in data.columns]
    text_features = [c for c in TEXT_FEATURES if c in data.columns]

    cols_to_drop = [col for col in NON_TRAINING_FEATURES if col in data.columns]
    if TARGET_FEATURE not in cols_to_drop:
        cols_to_drop.append(TARGET_FEATURE)

    X = data.drop(columns=cols_to_drop, errors='ignore')
    y = data[TARGET_FEATURE]

    # Handle text NaNs
    for col in text_features:
        if col in X.columns:
            X[col] = X[col].fillna('empty')
            X[col] = X[col].astype(str).replace(r'^\s*$', 'empty', regex=True)

    # Handle numeric NaNs (e.g. missing distance = far away)
    for col in numeric_features:
        if col in X.columns:
            # For density, 0 is a better default than 9999 (implies no siblings)
            if 'density' in col:
                X[col] = X[col].fillna(0.0)
            else:
                X[col] = X[col].fillna(9999.0)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    if pipeline is None:
        pipeline = build_pipeline(numeric_features, categorical_features, text_features)

    # Splitting
    if test_size > 0.0:
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=test_size, random_state=RANDOM_SEED, stratify=y_encoded
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=test_size, random_state=RANDOM_SEED
            )
    else:
        X_train, y_train = X, y_encoded
        X_test, y_test = None, None

    log_info(f"Training on {len(X_train)} samples | Features: {X_train.shape[1]}")

    if param_grid is not None:
        combinations = list(ParameterGrid(param_grid))
        CONSOLE.print(Panel(f"Grid Search: {len(combinations)} candidates, {grid_search_cv} folds.", title="Workload"))
        search = GridSearchCV(
            pipeline, param_grid, cv=grid_search_cv, verbose=1, scoring='f1_weighted', n_jobs=-1
        )
        search.fit(X_train, y_train)
        pipeline = search.best_estimator_
        CONSOLE.print(Panel(str(search.best_params_), title="Best Hyperparameters"))
    else:
        pipeline.fit(X_train, y_train)

    if X_test is not None:
        log_info("Evaluating on Test Set")
        y_pred = pipeline.predict(X_test)
        
        test_report = classification_report(
            label_encoder.inverse_transform(y_test),
            label_encoder.inverse_transform(y_pred),
            zero_division=0
        )
        CONSOLE.print(Panel(test_report, title="Test Set Report"))

    return {
        'pipeline': pipeline,
        'label_encoder': label_encoder,
        'features': {'numeric': numeric_features, 'categorical': categorical_features, 'text': text_features}
    }