"""Training and evaluation module for product category classification."""

from .evaluate_model import (
    calculate_meaningful_metrics,
    display_per_class_metrics,
    display_performance_table,
    evaluate_model,
)
from .process_data import data_to_csv
from .train_model import train_model

__all__ = [
    # Data processing
    'data_to_csv',
    # Model training
    'train_model',
    # Model evaluation
    'evaluate_model',
    'calculate_meaningful_metrics',
    'display_performance_table',
    'display_per_class_metrics',
]