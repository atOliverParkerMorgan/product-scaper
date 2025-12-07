"""Training and evaluation module for product category classification."""

from .evaluate_model import (
    calculate_meaningful_metrics,
    compare_models,
    display_per_class_metrics,
    display_performance_table,
    evaluate_model,
)
from .process_data import data_to_csv
from .train_model import (
    load_data,
    load_trained_model,
    run,
    preprocess_data,
    save_model,
    train_model,
)

__all__ = [
    # Data processing
    'data_to_csv',
    'load_data',
    'preprocess_data',
    # Model training
    'train_model',
    'save_model',
    'load_trained_model',
    'run',
    # Model evaluation
    'evaluate_model',
    'calculate_meaningful_metrics',
    'compare_models',
    'display_performance_table',
    'display_per_class_metrics',
]