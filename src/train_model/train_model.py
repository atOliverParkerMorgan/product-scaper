"""Model Training Pipeline."""

import warnings

# ADDED 'cast' to imports
from typing import Any, Dict, List, Optional, cast

import pandas as pd
from rich.panel import Panel
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

# Correct Imports
from utils.console import CONSOLE, log_error, log_info
from utils.features import (
    CATEGORICAL_FEATURES,
    NON_TRAINING_FEATURES,
    NUMERIC_FEATURES,
    TARGET_FEATURE,
    TEXT_FEATURES,
)

# Use local random state for reproducibility
RANDOM_STATE = 42

warnings.filterwarnings("ignore")


def build_pipeline(num_cols: List[str], cat_cols: List[str], text_cols: List[str]) -> Pipeline:
    transformers = []

    if num_cols:
        transformers.append(("num", StandardScaler(), num_cols))

    if cat_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=5), cat_cols))

    # TF-IDF for class names and IDs
    for col in ["class_str", "id_str"]:
        if col in text_cols:
            transformers.append(
                (f"txt_{col}", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), max_features=1000, min_df=1), col)
            )

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    clf = RandomForestClassifier(
        n_estimators=400,
        max_depth=15,
        min_samples_leaf=2,
        min_samples_split=5,
        max_features="sqrt",
        random_state=RANDOM_STATE,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )

    return Pipeline([("preprocessor", preprocessor), ("classifier", clf)])


def train_model(
    df: pd.DataFrame,
    pipeline: Optional[Pipeline] = None,
    test_size: float = 0.2,
    param_grid: Optional[Dict[str, Any]] = None,
    grid_search_cv: int = 3,
) -> Optional[Dict[str, Any]]:
    log_info("Starting Training Pipeline (Random Forest)")

    if df is None or df.empty:
        log_error("Input DataFrame is empty.")
        return None

    data = df.copy()

    # Dynamic feature selection based on what's available in the dataframe
    numeric_features = [c for c in data.columns if c in NUMERIC_FEATURES or "dist_to_" in c or "density" in c]
    # Ensure defined numeric features are included if they exist
    for c in NUMERIC_FEATURES:
        if c in data.columns and c not in numeric_features:
            numeric_features.append(c)

    categorical_features = [c for c in CATEGORICAL_FEATURES if c in data.columns]
    text_features = [c for c in TEXT_FEATURES if c in data.columns]

    cols_to_drop = [col for col in NON_TRAINING_FEATURES if col in data.columns]
    if TARGET_FEATURE not in cols_to_drop:
        cols_to_drop.append(TARGET_FEATURE)

    X = data.drop(columns=cols_to_drop, errors="ignore")
    y = data[TARGET_FEATURE]

    # Pre-filling NaN
    for col in text_features:
        if col in X.columns:
            X[col] = X[col].fillna("empty")
            X[col] = X[col].astype(str).replace(r"^\s*$", "empty", regex=True)

    for col in numeric_features:
        if col in X.columns:
            if "density" in col:
                X[col] = X[col].fillna(0.0)
            else:
                X[col] = X[col].fillna(100.0)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    if pipeline is None:
        pipeline = build_pipeline(numeric_features, categorical_features, text_features)

    # Split Data
    if test_size > 0.0:
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=test_size, random_state=RANDOM_STATE, stratify=y_encoded
            )
        except ValueError:
            # Fallback if stratify fails (e.g., too few samples for a class)
            X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=test_size, random_state=RANDOM_STATE)
    else:
        X_train, y_train = X, y_encoded
        X_test, y_test = None, None

    log_info(f"Training on {len(X_train)} samples | Features: {X_train.shape[1]}")

    if param_grid is not None:
        search = GridSearchCV(pipeline, param_grid, cv=grid_search_cv, verbose=1, scoring="f1_weighted", n_jobs=-1)
        # Fix: Cast y_train to Any to satisfy the type checker
        search.fit(X_train, cast(Any, y_train))
        pipeline = search.best_estimator_
    else:
        # Fix: Cast y_train to Any to satisfy the type checker
        pipeline.fit(X_train, cast(Any, y_train))

    if X_test is not None:
        log_info("Evaluating on Test Set")
        y_pred = pipeline.predict(X_test)
        test_report = classification_report(
            label_encoder.inverse_transform(cast(Any, y_test)), label_encoder.inverse_transform(y_pred), zero_division=0
        )
        CONSOLE.print(Panel(cast(Any, test_report), title="Test Set Report"))

    return {
        "pipeline": pipeline,
        "label_encoder": label_encoder,
        "features": {"numeric": numeric_features, "categorical": categorical_features, "text": text_features},
    }
