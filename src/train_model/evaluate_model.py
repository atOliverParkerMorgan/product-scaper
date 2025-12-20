"""Model evaluation utilities for classification models."""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich import box
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)
console = Console()

EXCLUDED_FROM_EVAL = ['other']


def calculate_meaningful_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    excluded_categories: Optional[List[str]] = None
) -> Tuple[Optional[float], Optional[float], Optional[Dict[str, Dict[str, float]]]]:
    """
    Calculate metrics excluding specified categories (e.g., 'other').
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        excluded_categories: List of category names to exclude from evaluation
        
    Returns:
        Tuple of (accuracy, f1_score, per_class_metrics)
        Returns (None, None, None) if no meaningful categories found
    """
    if excluded_categories is None:
        excluded_categories = EXCLUDED_FROM_EVAL
    
    # Create mask to filter out excluded categories
    mask = ~np.isin(np.array(y_true), excluded_categories)
    
    if mask.sum() == 0:
        logger.warning("No meaningful categories found for evaluation!")
        return None, None, None
    
    y_true_filtered = np.array(y_true)[mask]
    y_pred_filtered = np.array(y_pred)[mask]
    
    # Calculate overall metrics
    acc = accuracy_score(y_true_filtered, y_pred_filtered)
    f1 = f1_score(y_true_filtered, y_pred_filtered, average='weighted', zero_division=0)
    
    # Calculate per-class metrics
    labels = sorted(set(y_true_filtered))
    precision, recall, f1_per_class, support = precision_recall_fscore_support(
        y_true_filtered, y_pred_filtered, labels=labels, average=None, zero_division=0
    )
    
    per_class_metrics = {
        'precision': dict(zip(labels, precision)),
        'recall': dict(zip(labels, recall)),
        'f1': dict(zip(labels, f1_per_class)),
        'support': dict(zip(labels, support))
    }
    
    return acc, f1, per_class_metrics


def display_performance_table(
    split_name: str,
    acc_all: float,
    f1_all: float,
    acc_meaningful: Optional[float] = None,
    f1_meaningful: Optional[float] = None
) -> None:
    """
    Display a performance summary table.
    
    Args:
        split_name: Name of the data split (e.g., "Training", "Validation", "Test")
        acc_all: Accuracy on all categories
        f1_all: F1 score on all categories
        acc_meaningful: Accuracy on meaningful categories only
        f1_meaningful: F1 score on meaningful categories only
    """
    table = Table(title=f"{split_name} Set Performance", box=box.DOUBLE)
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("All Categories", justify="right", style="yellow")
    table.add_column("Meaningful Only", justify="right", style="green")
    
    acc_str = f"{acc_meaningful:.4f}" if acc_meaningful is not None else "N/A"
    f1_str = f"{f1_meaningful:.4f}" if f1_meaningful is not None else "N/A"
    
    table.add_row("Accuracy", f"{acc_all:.4f}", acc_str)
    table.add_row("F1 Score (weighted)", f"{f1_all:.4f}", f1_str)
    
    console.print(table)


def display_per_class_metrics(
    split_name: str,
    per_class_metrics: Dict[str, Dict[str, float]]
) -> None:
    """
    Display per-class metrics table.
    
    Args:
        split_name: Name of the data split
        per_class_metrics: Dictionary containing precision, recall, f1, and support per class
    """
    table = Table(
        title=f"{split_name} - Per-Class Metrics (Meaningful Categories)",
        box=box.ROUNDED
    )
    table.add_column("Category", style="cyan")
    table.add_column("Precision", justify="right", style="blue")
    table.add_column("Recall", justify="right", style="magenta")
    table.add_column("F1-Score", justify="right", style="green")
    table.add_column("Support", justify="right", style="yellow")
    
    for category in sorted(per_class_metrics['f1'].keys()):
        table.add_row(
            category,
            f"{per_class_metrics['precision'][category]:.3f}",
            f"{per_class_metrics['recall'][category]:.3f}",
            f"{per_class_metrics['f1'][category]:.3f}",
            f"{per_class_metrics['support'][category]:.0f}"
        )
    
    console.print(table)


def evaluate_model(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    label_encoder: LabelEncoder,
    split_name: str = "Validation",
    display_results: bool = True
) -> Dict[str, Any]:
    """
    Evaluate a model on a given dataset.
    
    Args:
        model: Trained model with predict method
        X: Feature matrix
        y: True labels (original, not encoded)
        label_encoder: Label encoder to transform predictions back to original labels
        split_name: Name of the split for display purposes
        display_results: Whether to print results to console
        
    Returns:
        Dictionary containing evaluation metrics and results
    """
    # Make predictions
    pred_encoded = model.predict(X)
    pred = label_encoder.inverse_transform(pred_encoded)
    
    # Calculate metrics on all categories
    acc_all = accuracy_score(y, pred)
    f1_all = f1_score(y, pred, average='weighted', zero_division=0)
    
    # Calculate metrics on meaningful categories only
    acc_meaningful, f1_meaningful, per_class_metrics = calculate_meaningful_metrics(y, pred)
    
    if display_results:
        display_performance_table(split_name, acc_all, f1_all, acc_meaningful, f1_meaningful)
        
        if per_class_metrics:
            display_per_class_metrics(split_name, per_class_metrics)
        
        console.print()
    
    return {
        'accuracy_all': acc_all,
        'f1_all': f1_all,
        'accuracy': acc_meaningful,
        'f1': f1_meaningful,
        'per_class': per_class_metrics,
        'predictions': pred,
        'confusion_matrix': confusion_matrix(y, pred),
        'classification_report': classification_report(y, pred)
    }




