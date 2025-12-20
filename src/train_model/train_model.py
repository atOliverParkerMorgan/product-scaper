"""Model Training Pipeline."""

import warnings
from typing import List

from rich.panel import Panel
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
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


def train_model(
    df,
    pipeline: Pipeline = None,
    test: bool = False,
    validation: bool = False,
    param_search: bool = False,
    param_grid: dict = None,
    min_samples_for_validation: int = 200,
    validation_size: float = 0.1,
    grid_search_cv: int = 3
):

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

    # Preprocess text columns to avoid empty vocabulary errors
    for col in text_features:
        if col in X.columns:
            X[col] = X[col].fillna('empty')
            X[col] = X[col].replace(r'^\s*$', 'empty', regex=True)

    # Label Encoding Target
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    if pipeline is None:
        pipeline = build_pipeline(numeric_features, categorical_features, text_features)


    # Split into train/validation/test if requested and enough samples
    if test or validation:
        test_size = 0.2 if test else 0.0
        val_size = validation_size if validation and len(X) >= min_samples_for_validation else 0.0
        total_size = test_size + val_size
        if total_size > 0:
            try:
                X_train, X_temp, y_train, y_temp = train_test_split(
                    X, y_encoded, test_size=total_size, random_state=RANDOM_SEED, stratify=y_encoded
                )
                if val_size > 0:
                    rel_val_size = val_size / total_size
                    X_val, X_test, y_val, y_test = train_test_split(
                        X_temp, y_temp, test_size=(1 - rel_val_size), random_state=RANDOM_SEED, stratify=y_temp
                    )
                else:
                    X_val, y_val = None, None
                    X_test, y_test = X_temp, y_temp
            except ValueError as e:
                log_error(f"Error during train/val/test split: {e}")
                return
        else:
            X_train, y_train = X, y_encoded
            X_val, y_val, X_test, y_test = None, None, None, None

        log_info(f"Training on {len(X_train)} samples using Random Forest")

        if X_train is None or len(X_train) == 0 or y_train is None or len(y_train) == 0:
            log_error("Training set is empty after split.")
            return

        # Hyperparameter search if requested
        if param_search:
            if param_grid is None:
                param_grid = {
                    'classifier__n_estimators': [100, 200],
                    'classifier__max_depth': [None, 10, 20],
                    'classifier__min_samples_leaf': [1, 2, 4],
                }
            log_info("Starting GridSearchCV for RandomForestClassifier parameters...")
            search = GridSearchCV(pipeline, param_grid, cv=grid_search_cv, n_jobs=-1, verbose=1)
            search.fit(X_train, y_train)
            pipeline = search.best_estimator_
            log_info(f"Best params: {search.best_params_}")

        else:
            pipeline.fit(X_train, y_train)

        # Validation set evaluation
        if X_val is not None and y_val is not None:
            log_info("Evaluating on validation set")
            y_val_pred = pipeline.predict(X_val)
            y_val_decoded = label_encoder.inverse_transform(y_val)
            y_val_pred_decoded = label_encoder.inverse_transform(y_val_pred)
            val_report = classification_report(y_val_decoded, y_val_pred_decoded)
            CONSOLE.print(Panel(val_report, title="Validation Set Report"))

        # Test set evaluation
        if test and X_test is not None and y_test is not None:
            log_info("Evaluating on test set")
            y_pred = pipeline.predict(X_test)
            y_test_decoded = label_encoder.inverse_transform(y_test)
            y_pred_decoded = label_encoder.inverse_transform(y_pred)
            report = classification_report(y_test_decoded, y_pred_decoded)
            CONSOLE.print(Panel(report, title="Test Set Classification Report"))
    else:
        # Train on full dataset without split
        log_info(f"Training on {len(X)} samples using Random Forest")
        if param_search:
            if param_grid is None:
                param_grid = {
                    'classifier__n_estimators': [100, 200, 300, 400],
                    'classifier__max_depth': [None, 5, 10, 20],
                    'classifier__min_samples_leaf': [1, 2, 4],
                }
            log_info("Starting GridSearchCV for RandomForestClassifier parameters...")
            search = GridSearchCV(pipeline, param_grid, cv=grid_search_cv, n_jobs=-1, verbose=1)
            search.fit(X, y_encoded)
            pipeline = search.best_estimator_
            log_info(f"Best params: {search.best_params_}")
        else:
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
