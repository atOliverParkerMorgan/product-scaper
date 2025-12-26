"""
Model Training Pipeline.

Handles the creation and training of the Random Forest Classifier used
to identify HTML elements based on their features.
"""

from typing import Any, Dict, List, Optional, Tuple, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich.panel import Panel
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

# Relative imports
from product_scraper.utils.console import CONSOLE, log_error, log_info
from product_scraper.utils.features import (
    CATEGORICAL_FEATURES,
    NON_TRAINING_FEATURES,
    NUMERIC_FEATURES,
    TARGET_FEATURE,
    TEXT_FEATURES,
)

RANDOM_STATE = 42


def build_pipeline(
    num_cols: List[str], cat_cols: List[str], text_cols: List[str]
) -> Pipeline:
    """
    Constructs the Scikit-Learn processing and classification pipeline.

    Args:
        num_cols: List of numeric feature names (scaled via StandardScaler).
        cat_cols: List of categorical feature names (encoded via OneHotEncoder).
        text_cols: List of text feature names (vectorized via TF-IDF).

    Returns:
        Pipeline: A configured sklearn Pipeline object.
    """
    transformers = []

    if num_cols:
        transformers.append(("num", StandardScaler(), num_cols))

    if cat_cols:
        transformers.append(
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=5), cat_cols)
        )

    # TF-IDF for specific class/id strings
    for col in ["class_str", "id_str"]:
        if col in text_cols:
            transformers.append(
                (
                    f"txt_{col}",
                    TfidfVectorizer(
                        analyzer="char_wb",
                        ngram_range=(2, 4),
                        max_features=1000,
                        min_df=1,
                    ),
                    col,
                )
            )

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    # Balanced Subsample is crucial for imbalanced web data (lots of 'other' tags)
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


def select_and_prepare_features(df):
    """Select and prepare feature columns from the DataFrame."""
    numeric = [
        c
        for c in df.columns
        if c in NUMERIC_FEATURES or "dist_to_" in c or "density" in c
    ]
    for c in NUMERIC_FEATURES:
        if c in df.columns and c not in numeric:
            numeric.append(c)
    categorical = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    text = [c for c in TEXT_FEATURES if c in df.columns]
    return numeric, categorical, text


def fill_missing(X, text_features, numeric_features):
    """Fill missing values in the feature DataFrame."""
    for col in text_features:
        if col in X.columns:
            X[col] = (
                X[col]
                .fillna("empty")
                .astype(str)
                .replace(r"^\s*$", "empty", regex=True)
            )
    for col in numeric_features:
        if col in X.columns:
            fill_val = 0.0 if "density" in col else 100.0
            X[col] = X[col].fillna(fill_val)
    return X


def _prepare_data(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Any, LabelEncoder, Dict[str, List[str]], pd.Series]:
    """Internal helper to select features, fill missing data, and encode target."""
    data = df.copy()

    # Capture the SourceURL for group-based splitting (Site-level split)
    if "SourceURL" in data.columns:
        groups = data["SourceURL"]
    else:
        # Fallback if SourceURL is missing (unlikely in this pipeline)
        groups = pd.Series(range(len(data)))

    numeric, categorical, text = select_and_prepare_features(data)

    cols_to_drop = [col for col in NON_TRAINING_FEATURES if col in data.columns]
    if TARGET_FEATURE not in cols_to_drop:
        cols_to_drop.append(TARGET_FEATURE)

    X = data.drop(columns=cols_to_drop, errors="ignore")
    y = data[TARGET_FEATURE]

    X = fill_missing(X, text, numeric)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    features = {
        "numeric": numeric,
        "categorical": categorical,
        "text": text,
    }

    return X, y_encoded, label_encoder, features, groups


def _split_data(
    X: pd.DataFrame, y_encoded: Any, groups: pd.Series, test_size: float
) -> Tuple[Any, Any, Any, Any]:
    """
    Internal helper to split data by Website URL (Groups).
    This ensures the test set contains completely unseen websites.
    """
    if test_size <= 0.0:
        return X, None, y_encoded, None

    unique_urls = groups.unique()

    # If we don't have enough unique sites, fall back to random row splitting
    if len(unique_urls) < 2:
        log_info(
            "Insufficient unique URLs for site-split. Falling back to random row split."
        )
        try:
            return tuple(
                train_test_split(
                    X,
                    y_encoded,
                    test_size=test_size,
                    random_state=RANDOM_STATE,
                    stratify=y_encoded,
                )
            )
        except ValueError:
            return tuple(
                train_test_split(
                    X, y_encoded, test_size=test_size, random_state=RANDOM_STATE
                )
            )

    # 1. Split the Sites (Groups)
    train_urls, test_urls = train_test_split(
        unique_urls, test_size=test_size, random_state=RANDOM_STATE
    )

    log_info(
        f"Splitting by URL: {len(train_urls)} Train Sites / {len(test_urls)} Test Sites"
    )

    # 2. Create Masks for rows belonging to those sites
    train_mask = groups.isin(train_urls)
    test_mask = groups.isin(test_urls)

    # 3. Apply masks
    X_train = X[train_mask]
    X_test = X[test_mask]
    y_train = y_encoded[train_mask]
    y_test = y_encoded[test_mask]

    return X_train, X_test, y_train, y_test


def _ensure_pipeline(
    pipeline: Optional[Pipeline], features: Dict[str, List[str]]
) -> Pipeline:
    """Internal helper to return the existing pipeline or build a new one."""
    if pipeline is None:
        return build_pipeline(
            features["numeric"], features["categorical"], features["text"]
        )
    return pipeline


def _fit_model(
    pipeline: Pipeline,
    split_data: Tuple[Any, Any, Any, Any],
    param_grid: Optional[Dict[str, Any]],
    cv: int,
) -> Pipeline:
    """Internal helper to handle GridSearchCV or standard fit."""
    X_train, _, y_train, _ = split_data

    if param_grid is not None:
        search = GridSearchCV(
            pipeline,
            param_grid,
            cv=cv,
            verbose=1,
            scoring="f1_weighted",
            n_jobs=-1,
        )
        search.fit(X_train, cast(Any, y_train))
        return search.best_estimator_

    pipeline.fit(X_train, cast(Any, y_train))
    return pipeline


def _plot_model_diagnostics(
    pipeline: Pipeline,
    X_test: Any,
    y_test: Any,
    label_encoder: LabelEncoder,
    features_dict: Dict[str, List[str]],
) -> None:
    """
    Internal helper to generate and display Feature Importance and Confusion Matrix.
    """
    log_info("Generating Model Diagnostics Figures...")

    # 1. Extract Feature Names from Pipeline
    # This is complex because the pipeline transforms features (OneHot, TF-IDF)
    preprocessor = pipeline.named_steps["preprocessor"]
    clf = pipeline.named_steps["classifier"]

    feature_names = []

    # Iterate through transformers in the ColumnTransformer
    for name, trans, cols in preprocessor.transformers_:
        if name == "drop" or trans == "drop":
            continue
        if name == "num":
            # Numeric columns are kept as is (just scaled)
            feature_names.extend(cols)
        elif hasattr(trans, "get_feature_names_out"):
            # Categorical and Text transformers provide new names
            try:
                names = trans.get_feature_names_out(cols)
                feature_names.extend(names)
            except (AttributeError, ValueError):
                # Fallback if specific transformer doesn't support it
                feature_names.extend(cols)

    # 2. Plot Feature Importance
    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
        # Ensure lengths match before plotting
        if len(importances) == len(feature_names):
            indices = np.argsort(importances)[::-1]
            top_n = 5

            plt.figure(figsize=(12, 6))
            plt.title(f"Top {top_n} Feature Importances")
            plt.bar(range(top_n), importances[indices[:top_n]], align="center")
            plt.xticks(
                range(top_n),
                [feature_names[i] for i in indices[:top_n]],
                rotation=45,
                ha="right",
            )
            plt.tight_layout()
            plt.show()


def _evaluate_performance(
    pipeline: Pipeline,
    split_data: Tuple[Any, Any, Any, Any],
    label_encoder: LabelEncoder,
) -> None:
    """Internal helper to evaluate the model on the test set."""
    _, X_test, _, y_test = split_data

    if X_test is not None:
        log_info("Evaluating on Test Set (Unseen Websites)")
        y_pred = pipeline.predict(X_test)
        report = classification_report(
            label_encoder.inverse_transform(cast(Any, y_test)),
            label_encoder.inverse_transform(y_pred),
            zero_division=0,
        )
        CONSOLE.print(Panel(cast(str, report), title="Test Set Report"))


def train_model(
    df: pd.DataFrame,
    pipeline: Optional[Pipeline] = None,
    test_size: float = 0.2,
    param_grid: Optional[Dict[str, Any]] = None,
    grid_search_cv: int = 3,
    show_model_figure: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Main training function.

    Prepares data, splits it by Website URL, trains the model, and evaluates performance.

    Args:
        show_model_figure (bool): If True, displays matplotlib figures for feature importance
                                  and confusion matrix after training.
    """
    log_info("Starting Training Pipeline (Random Forest)")

    if df is None or df.empty:
        log_error("Input DataFrame is empty.")
        return None

    # 1. Prepare Data
    X, y_encoded, label_encoder, features, groups = _prepare_data(df)

    # 2. Pipeline setup
    pipeline = _ensure_pipeline(pipeline, features)

    # 3. Split Data (kept as tuple to save local variables)
    split_data = _split_data(X, y_encoded, groups, test_size)

    # Log training size
    log_info(
        f"Training on {len(split_data[0])} samples | Features: {split_data[0].shape[1]}"
    )

    # 4. Train
    pipeline = _fit_model(pipeline, split_data, param_grid, grid_search_cv)

    # 5. Evaluate
    _evaluate_performance(pipeline, split_data, label_encoder)

    # 6. Visualize
    if show_model_figure and split_data[1] is not None:
        # split_data[1] is X_test, split_data[3] is y_test
        _plot_model_diagnostics(
            pipeline, split_data[1], split_data[3], label_encoder, features
        )

    return {
        "pipeline": pipeline,
        "label_encoder": label_encoder,
        "features": features,
    }
