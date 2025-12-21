"""Model Training Pipeline."""

import warnings
from typing import Any, Dict, List

from rich.panel import Panel
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

# Constants
from train_model.process_data import RANDOM_SEED
from utils.console import CONSOLE, log_error, log_info
from utils.features import (
    CATEGORICAL_FEATURES,
    NON_TRAINING_FEATURES,
    NUMERIC_FEATURES,
    TARGET_FEATURE,
    TEXT_FEATURES,
)

warnings.filterwarnings('ignore')


def build_pipeline(num_cols: List[str], cat_cols: List[str], text_cols: List[str]) -> Pipeline:
    """
    Constructs a robust preprocessing and training pipeline.
    """

    # 1. Preprocessing Steps
    transformers = [
        # Numeric: Standard Scaling is important for features like 'text_len' vs 'is_bold'
        ('num', StandardScaler(), num_cols),

        # Categorical: One Hot Encoding
        # handle_unknown='ignore' is crucial for web data where new tags might appear
        ('cat', OneHotEncoder(handle_unknown='ignore', min_frequency=5), cat_cols),
    ]

    # 2. Add TF-IDF transformers for text features
    # max_features=500 is enough for class names; 1000 adds too much noise.
    if 'class_str' in text_cols:
        transformers.append(
            ('txt_class', TfidfVectorizer(
                analyzer='char_wb',
                ngram_range=(3, 5),
                max_features=500,
                min_df=2
            ), 'class_str')
        )
    if 'id_str' in text_cols:
        transformers.append(
            ('txt_id', TfidfVectorizer(
                analyzer='char_wb',
                ngram_range=(3, 5),
                max_features=500,
                min_df=2
            ), 'id_str')
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='drop'
    )

    # 3. Classifier
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=25,       # Deep enough for DOM, but prevents memorization
        min_samples_leaf=2, # Requires at least 2 samples to make a decision rule
        n_jobs=-1,
        random_state=RANDOM_SEED,
        class_weight='balanced'
    )

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', clf)
    ])

    return pipeline


def train_model(
    df,
    pipeline: Pipeline = None,
    test: bool = True,
    validation: bool = True,
    param_search: bool = True,
    param_grid: Dict[str, Any] = None,
    min_samples_for_validation: int = 200,
    validation_size: float = 0.1,
    grid_search_cv: int = 3
):
    """
    Main training execution function with expanded Hyperparameter tuning.
    """
    log_info("Starting Training Pipeline (Random Forest)")

    # --- 1. Feature Selection & Safety Checks ---
    numeric_features = [c for c in NUMERIC_FEATURES if c in df.columns]
    categorical_features = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    text_features = [c for c in TEXT_FEATURES if c in df.columns]

    if TARGET_FEATURE not in df.columns:
        log_error(f"Target column '{TARGET_FEATURE}' not found in data")
        return None

    # Drop non-training columns
    cols_to_drop = [col for col in NON_TRAINING_FEATURES if col in df.columns]
    X = df.drop(columns=cols_to_drop)
    y = df[TARGET_FEATURE]

    # Clean text columns (NaN -> 'empty')
    for col in text_features:
        if col in X.columns:
            X[col] = X[col].fillna('empty')
            X[col] = X[col].replace(r'^\s*$', 'empty', regex=True)

    # Encode Labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Build Pipeline if not provided
    if pipeline is None:
        pipeline = build_pipeline(numeric_features, categorical_features, text_features)

    # --- 2. Splitting Logic (Robust) ---
    X_train, y_train = X, y_encoded
    X_val, y_val, X_test, y_test = None, None, None, None

    # (Split logic remains the same as your original code...)
    if test or validation:
        test_size = 0.2 if test else 0.0
        val_size = validation_size if validation and len(X) >= min_samples_for_validation else 0.0
        total_test_size = test_size + val_size

        if total_test_size > 0:
            try:
                X_train, X_temp, y_train, y_temp = train_test_split(
                    X, y_encoded, test_size=total_test_size, random_state=RANDOM_SEED, stratify=y_encoded
                )
            except ValueError:
                X_train, X_temp, y_train, y_temp = train_test_split(
                    X, y_encoded, test_size=total_test_size, random_state=RANDOM_SEED, stratify=None
                )

            if val_size > 0:
                rel_val_size = val_size / total_test_size
                X_val, X_test, y_val, y_test = train_test_split(
                    X_temp, y_temp, test_size=(1 - rel_val_size), random_state=RANDOM_SEED
                )
            else:
                X_test, y_test = X_temp, y_temp

    # --- 3. Training & Grid Search ---
    log_info(f"Training on {len(X_train)} samples")

    if param_search:
        if param_grid is None:
            # --- IMPROVED PARAMETER GRID ---
            # We tune the Model AND the Text Extraction simultaneously
            param_grid = {
                # Random Forest Regularization
                'classifier__n_estimators': [100, 200, 400, 600, 800, 1000, 1800],
                'classifier__max_depth': [8, 15, 25, 30,],
                'classifier__min_samples_split': [2, 5],
                'classifier__min_samples_leaf': [1, 2],
                'classifier__max_features': ['sqrt', 'log2'], # Critical for high-dim data

                # TF-IDF Tuning (Accessing via the step name 'preprocessor' -> transformer name 'txt_class')
                # NOTE: Only include these if 'class_str' is actually in text_features
            }

            # Dynamically add text tuning if features exist
            if 'class_str' in text_features:
                param_grid.update({
                    'preprocessor__txt_class__ngram_range': [(2, 4), (3, 5)],
                    'preprocessor__txt_class__max_features': [500, 1000],
                })

            # Optional: Tune numeric scaling (rarely changes much for RF, but good for completeness)
            # param_grid['preprocessor__num__with_mean'] = [True, False]

        log_info("Starting GridSearchCV...")
        search = GridSearchCV(
            pipeline,
            param_grid,
            cv=grid_search_cv,
            n_jobs=-1,
            verbose=1,
            scoring='f1_weighted' # optimizing for F1 is usually better than accuracy for scraping
        )
        search.fit(X_train, y_train)
        pipeline = search.best_estimator_

        # Pretty print best params
        CONSOLE.print(Panel(str(search.best_params_), title="Best Hyperparameters"))
    else:
        pipeline.fit(X_train, y_train)

    # --- 4. Evaluation ---
    if X_val is not None:
        log_info("Evaluating on Validation Set")
        y_val_pred = pipeline.predict(X_val)
        val_report = classification_report(
            label_encoder.inverse_transform(y_val),
            label_encoder.inverse_transform(y_val_pred),
            zero_division=0
        )
        CONSOLE.print(Panel(val_report, title="Validation Report"))

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
