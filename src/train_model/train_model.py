"""Model training utilities for product category classification."""

import logging
import pickle
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from rich import box
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from utils.constants import RANDOM_STATE, EXCLUDED_FEATURES, DATA_NAME

from .evaluate_model import compare_models, evaluate_model

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)]
)
logger = logging.getLogger(__name__)
console = Console()


# Configuration constants are now imported from utils.constants


def load_data(
    project_root: Optional[Path] = None,
    test_domains: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load training and test data from CSV files.
    
    Args:
        project_root: Root directory of the project
        test_domains: List of domain names to use as test set
        
    Returns:
        Tuple of (train_df, test_df)
        
    Raises:
        FileNotFoundError: If data directory doesn't exist
        ValueError: If no training data found
    """
    if project_root is None:
        project_root = Path.cwd()
    
    data_dir = project_root / 'src' / 'data'
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found at {data_dir}")
    
    train_data = []
    test_data = []

    for site_dir in data_dir.iterdir():
        if not site_dir.is_dir():
            continue
            
        data_file = site_dir / DATA_NAME
        if not data_file.exists():
            logger.warning(f"No {DATA_NAME} found in {site_dir}, skipping.")
            continue
        
        df = pd.read_csv(data_file)
        if test_domains and site_dir.name in test_domains:
            test_data.append(df)
        else:
            train_data.append(df)

    if not train_data:
        raise ValueError("No training data found.")

    train_df = pd.concat(train_data, ignore_index=True)
    test_df = pd.concat(test_data, ignore_index=True) if test_data else pd.DataFrame()
    
    return train_df, test_df





def preprocess_data(
    df: pd.DataFrame,
    label_encoder: Optional[LabelEncoder] = None
) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str], LabelEncoder]:
    """
    Preprocess data with label encoding for target variable.
    
    Args:
        df: Input DataFrame with features and 'Category' column
        label_encoder: Optional pre-fitted LabelEncoder for transform-only mode
        
    Returns:
        Tuple of (X, y_encoded, feature_names, class_names, label_encoder)
    """
    X = df.drop(columns=EXCLUDED_FEATURES, errors='ignore').fillna(0)
    y = df['Category']
    
    if label_encoder is None:
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
    else:
        y_encoded = label_encoder.transform(y)
    
    logger.info(
        f"[bold]Features:[/bold] {len(X.columns)} | "
        f"[bold]Classes:[/bold] {len(label_encoder.classes_)}\n"
    )
    
    return (
        X,
        pd.Series(y_encoded),
        X.columns.tolist(),
        label_encoder.classes_.tolist(),
        label_encoder
    )





def _create_base_model(model_type: str) -> Any:
    """
    Create a base classification model.
    
    Args:
        model_type: Type of model ('xgboost' or 'random_forest')
        
    Returns:
        Base model instance
    """
    if model_type == 'xgboost':
        return XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=RANDOM_STATE,
            eval_metric='mlogloss',
            n_jobs=-1,
            enable_categorical=False
        )
    else:
        return RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight='balanced'
        )


def _get_param_grid(model_type: str) -> Dict[str, List]:
    """
    Get hyperparameter grid for model tuning.
    
    Args:
        model_type: Type of model ('xgboost' or 'random_forest')
        
    Returns:
        Dictionary of hyperparameter options
    """
    if model_type == 'xgboost':
        return {
            'n_estimators': [100, 200],
            'max_depth': [4, 6, 8],
            'learning_rate': [0.05, 0.1]
        }
    else:
        return {
            'n_estimators': [100, 200],
            'max_depth': [None, 20],
            'min_samples_split': [2, 5]
        }


def train_model(
    X_train: pd.DataFrame,
    y_train_encoded: pd.Series,
    y_train_original: pd.Series,
    X_val: pd.DataFrame,
    y_val_encoded: pd.Series,
    y_val_original: pd.Series,
    label_encoder: LabelEncoder,
    use_smote: bool = True,
    model_type: str = 'xgboost',
    tune_hyperparams: bool = False
) -> Tuple[Any, Dict[str, Any]]:
    """
    Train a classification model with optional SMOTE and hyperparameter tuning.
    
    Args:
        X_train: Training features
        y_train_encoded: Training labels (encoded as numbers)
        y_train_original: Training labels (original strings)
        X_val: Validation features
        y_val_encoded: Validation labels (encoded as numbers)
        y_val_original: Validation labels (original strings)
        label_encoder: Label encoder for inverse transform
        use_smote: Whether to use SMOTE for handling class imbalance
        model_type: Type of model ('xgboost' or 'random_forest')
        tune_hyperparams: Whether to perform hyperparameter tuning
        
    Returns:
        Tuple of (trained_model, metrics_dict)
    """
    console.print(Panel(
        f"[bold cyan]Training {model_type.upper()}[/bold cyan]\n"
        f"SMOTE: {use_smote} | Tuning: {tune_hyperparams}",
        box=box.DOUBLE
    ))
    
    # Create base model
    base_model = _create_base_model(model_type)
    
    # Create pipeline with optional SMOTE
    if use_smote:
        pipeline = ImbPipeline([
            ('smote', SMOTE(random_state=RANDOM_STATE, k_neighbors=3)),
            ('classifier', base_model)
        ])
    else:
        pipeline = base_model
    
    # Hyperparameter tuning
    if tune_hyperparams:
        param_grid = _get_param_grid(model_type)
        
        # Adjust param names for pipeline
        if use_smote:
            param_grid = {f'classifier__{k}': v for k, v in param_grid.items()}
        
        logger.info("Tuning hyperparameters...")
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=3,
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=0
        )
        grid_search.fit(X_train, y_train_encoded)
        model = grid_search.best_estimator_
        
        logger.info(
            f"[bold green]Best params:[/bold green] {grid_search.best_params_} | "
            f"CV score: {grid_search.best_score_:.4f}\n"
        )
    else:
        model = pipeline
        model.fit(X_train, y_train_encoded)
    
    # Evaluate on training set (use original labels for evaluation)
    logger.info("[bold]Training Set:[/bold]")
    train_metrics = evaluate_model(model, X_train, y_train_original, label_encoder, "Training")
    
    # Evaluate on validation set (use original labels for evaluation)
    logger.info("[bold]Validation Set:[/bold]")
    val_metrics = evaluate_model(model, X_val, y_val_original, label_encoder, "Validation")
    
    console.print(Panel(
        val_metrics['classification_report'],
        title="[bold]Validation Report[/bold]",
        box=box.ROUNDED
    ))
    
    # Return model and consolidated metrics
    return model, {
        'train_acc': train_metrics['accuracy'],
        'train_f1': train_metrics['f1'],
        'val_acc': val_metrics['accuracy'],
        'val_f1': val_metrics['f1'],
        'train_acc_all': train_metrics['accuracy_all'],
        'train_f1_all': train_metrics['f1_all'],
        'val_acc_all': val_metrics['accuracy_all'],
        'val_f1_all': val_metrics['f1_all'],
        'per_class_metrics': val_metrics['per_class']
    }


def save_model(
    model: Any,
    feature_names: List[str],
    class_names: List[str],
    label_encoder: LabelEncoder,
    save_path: Path
) -> None:
    """
    Save the trained model and associated metadata.
    
    Args:
        model: Trained model object
        feature_names: List of feature column names
        class_names: List of class labels
        label_encoder: Fitted LabelEncoder
        save_path: Path where to save the model
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    model_data = {
        'model': model,
        'feature_names': feature_names,
        'class_names': class_names,
        'label_encoder': label_encoder,
        'random_state': RANDOM_STATE
    }
    
    with open(save_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    logger.info(f"[bold green]✓[/bold green] Model saved to {save_path}")


def load_trained_model(model_path: Path) -> Dict[str, Any]:
    """
    Load a trained model from disk.
    
    Args:
        model_path: Path to the saved model file
        
    Returns:
        Dictionary containing model and metadata
    """
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    logger.info(f"[bold green]✓[/bold green] Model loaded from {model_path}")
    return model_data


def _display_data_split(X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: Optional[pd.DataFrame]) -> None:
    """Display data split information."""
    split_table = Table(title="Data Split", box=box.ROUNDED)
    split_table.add_column("Split", style="cyan")
    split_table.add_column("Samples", justify="right", style="green")
    split_table.add_row("Train", str(len(X_train)))
    split_table.add_row("Val", str(len(X_val)))
    if X_test is not None:
        split_table.add_row("Test", str(len(X_test)))
    console.print(split_table)
    console.print()


def _train_multiple_models(
    X_train: pd.DataFrame,
    y_train_encoded: pd.Series,
    y_train_original: pd.Series,
    X_val: pd.DataFrame,
    y_val_encoded: pd.Series,
    y_val_original: pd.Series,
    label_encoder: LabelEncoder
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, float]]]:
    """Train multiple models and return them with their metrics."""
    models = {}
    metrics = {}
    
    # Train XGBoost with SMOTE
    try:
        model1, metrics1 = train_model(
            X_train, y_train_encoded, y_train_original,
            X_val, y_val_encoded, y_val_original,
            label_encoder, use_smote=True, model_type='xgboost'
        )
        models['xgboost_smote'] = model1
        metrics['xgboost_smote'] = metrics1
    except Exception as e:
        logger.error(f"XGBoost failed: {e}")

    # Train Random Forest
    try:
        model2, metrics2 = train_model(
            X_train, y_train_encoded, y_train_original,
            X_val, y_val_encoded, y_val_original,
            label_encoder, use_smote=False, model_type='random_forest'
        )
        models['random_forest'] = model2
        metrics['random_forest'] = metrics2
    except Exception as e:
        logger.error(f"RandomForest failed: {e}")
    
    return models, metrics


class ClassificationModel:
    def __init__(self):
        self.saved_at_path = None
        self.trained_on_website = []
        



def run(
    project_root: Optional[Path] = None,
    save_dir: Optional[Path] = None,
    test_domains: Optional[List[str]] = None
) -> Optional[Tuple[Any, List[str], List[str], Dict]]:
    """
    Main training pipeline with domain-specific test set.
    
    Args:
        project_root: Root directory of the project
        save_dir: Directory to save trained models
        test_domains: List of domain names to use as test set
        
    Returns:
        Tuple of (best_model, feature_names, class_names, metrics) or None if failed
    """
    project_root = project_root or Path.cwd()
    save_dir = save_dir or project_root / 'src' / 'models'
    
    console.print(Panel.fit(
        "[bold cyan]Product Scraper Category Classification[/bold cyan]\n"
        "[yellow]Model Training Pipeline[/yellow]",
        box=box.DOUBLE_EDGE
    ))
    
    # Load data
    try:
        train_df, test_df = load_data(project_root, test_domains)
    except Exception as e:
        logger.error(f"[bold red]ERROR:[/bold red] {e}")
        return None

    # Preprocess training/validation data
    X_train_val, y_train_val_encoded, feature_names, class_names, label_encoder = preprocess_data(train_df)
    y_train_val_original = train_df['Category']  # Keep original string labels
    
    if len(X_train_val) < 10:
        logger.error("Not enough training data.")
        return None

    # Split into train and validation sets
    try:
        X_train, X_val, y_train_encoded, y_val_encoded, y_train_original, y_val_original = train_test_split(
            X_train_val, y_train_val_encoded, y_train_val_original,
            test_size=0.2,
            random_state=RANDOM_STATE,
            stratify=y_train_val_encoded
        )
    except ValueError:
        # Fallback if stratification fails
        X_train, X_val, y_train_encoded, y_val_encoded, y_train_original, y_val_original = train_test_split(
            X_train_val, y_train_val_encoded, y_train_val_original,
            test_size=0.2,
            random_state=RANDOM_STATE
        )
    
    # Preprocess test data if available
    if len(test_df) > 0:
        X_test, y_test_encoded, _, _, _ = preprocess_data(test_df, label_encoder)
        y_test_original = test_df['Category']
    else:
        X_test, y_test_encoded, y_test_original = None, None, None
    
    # Display data split
    _display_data_split(X_train, X_val, X_test)
    
    # Train multiple models
    models, metrics = _train_multiple_models(
        X_train, y_train_encoded, y_train_original,
        X_val, y_val_encoded, y_val_original,
        label_encoder
    )
    
    if not models:
        logger.error("No models trained successfully.")
        return None

    # Select best model
    best_model_name, best_model = compare_models(models, metrics, 'val_f1')
    
    # Evaluate on test set if available
    if X_test is not None and len(X_test) > 0:
        console.print(Panel(
            "[bold yellow]Test Set Evaluation[/bold yellow]",
            box=box.DOUBLE_EDGE
        ))
        test_metrics = evaluate_model(best_model, X_test, y_test_original, label_encoder, "Test")
        console.print(Panel(
            test_metrics['classification_report'],
            title="[bold]Test Report[/bold]",
            box=box.ROUNDED
        ))
    
    # Save best model
    model_path = save_dir / 'category_classifier.pkl'
    save_model(best_model, feature_names, class_names, label_encoder, model_path)
    
    console.print(Panel.fit(
        f"[bold green]✓ Training Complete![/bold green]\n"
        f"Best: [cyan]{best_model_name}[/cyan]\n"
        f"Saved: [yellow]{model_path}[/yellow]",
        box=box.DOUBLE_EDGE
    ))
    
    return best_model, feature_names, class_names, metrics


if __name__ == '__main__':
    run()