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

# Configuration constants
RANDOM_STATE = 42
EXCLUDED_FEATURES = ['Category']
DATA_NAME = 'data.csv'


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


def _calculate_meaningful_metrics(y_true: np.ndarray, y_pred: np.ndarray, excluded_categories: List[str] = None) -> Tuple[float, float, Optional[Dict]]:
    """Calculate metrics excluding specified categories (like 'other')."""
    if excluded_categories is None:
        excluded_categories = EXCLUDED_FROM_EVAL
    
    mask = ~np.isin(np.array(y_true), excluded_categories)
    if mask.sum() == 0:
        logger.warning("No meaningful categories found!")
        return None, None, None
    
    y_true_filtered = np.array(y_true)[mask]
    y_pred_filtered = np.array(y_pred)[mask]
    
    acc = accuracy_score(y_true_filtered, y_pred_filtered)
    f1 = f1_score(y_true_filtered, y_pred_filtered, average='weighted', zero_division=0)
    
    labels = sorted(set(y_true_filtered))
    precision, recall, f1_per_class, support = precision_recall_fscore_support(
        y_true_filtered, y_pred_filtered, labels=labels, average=None, zero_division=0
    )
    
    return acc, f1, {'precision': dict(zip(labels, precision)), 'recall': dict(zip(labels, recall)), 
                     'f1': dict(zip(labels, f1_per_class)), 'support': dict(zip(labels, support))}


def _evaluate_model(model: Any, X: pd.DataFrame, y: pd.Series, label_encoder: LabelEncoder, split_name: str = "Validation") -> Dict[str, Any]:
    """Evaluate a model on a given dataset."""
    pred_encoded = model.predict(X)
    pred = label_encoder.inverse_transform(pred_encoded)
    
    acc_all = accuracy_score(y, pred)
    f1_all = f1_score(y, pred, average='weighted', zero_division=0)
    acc, f1, per_class = _calculate_meaningful_metrics(y, pred)
    
    table = Table(title=f"{split_name} Set Performance", box=box.DOUBLE)
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("All Categories", justify="right", style="yellow")
    table.add_column("Meaningful Only", justify="right", style="green")
    
    table.add_row("Accuracy", f"{acc_all:.4f}", f"{acc:.4f}" if acc else "N/A")
    table.add_row("F1 Score (weighted)", f"{f1_all:.4f}", f"{f1:.4f}" if f1 else "N/A")
    
    console.print(table)
    
    # Display per-class metrics if available
    if per_class:
        per_class_table = Table(title=f"{split_name} - Per-Class Metrics (Meaningful Categories)", box=box.ROUNDED)
        per_class_table.add_column("Category", style="cyan")
        per_class_table.add_column("Precision", justify="right", style="blue")
        per_class_table.add_column("Recall", justify="right", style="magenta")
        per_class_table.add_column("F1-Score", justify="right", style="green")
        per_class_table.add_column("Support", justify="right", style="yellow")
        
        for category in sorted(per_class['f1'].keys()):
            per_class_table.add_row(
                category,
                f"{per_class['precision'][category]:.3f}",
                f"{per_class['recall'][category]:.3f}",
                f"{per_class['f1'][category]:.3f}",
                f"{per_class['support'][category]:.0f}"
            )
        
        console.print(per_class_table)
    
    console.print()
    
    return {'accuracy_all': acc_all, 'f1_all': f1_all, 'accuracy': acc, 'f1': f1, 'per_class': per_class, 
            'predictions': pred, 'confusion_matrix': confusion_matrix(y, pred), 'classification_report': classification_report(y, pred)}


def train_model(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, 
                label_encoder: LabelEncoder, use_smote: bool = True, model_type: str = 'xgboost', 
                tune_hyperparams: bool = False) -> Tuple[Any, Dict[str, Any]]:
    """Train a classification model with optional SMOTE and hyperparameter tuning."""
    console.print(Panel(f"[bold cyan]Training {model_type.upper()}[/bold cyan]\nSMOTE: {use_smote} | Tuning: {tune_hyperparams}", box=box.DOUBLE))
    
    if model_type == 'xgboost':
        base_model = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=random_state, 
                                    eval_metric='mlogloss', n_jobs=-1, enable_categorical=False)
        param_grid = {'n_estimators': [100, 200], 'max_depth': [4, 6, 8], 'learning_rate': [0.05, 0.1]} if tune_hyperparams else None
    else:
        base_model = RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1, class_weight='balanced')
        param_grid = {'n_estimators': [100, 200], 'max_depth': [None, 20], 'min_samples_split': [2, 5]} if tune_hyperparams else None
    
    if use_smote:
        pipeline = ImbPipeline([('smote', SMOTE(random_state=random_state, k_neighbors=3)), ('classifier', base_model)])
        if param_grid:
            param_grid = {f'classifier__{k}': v for k, v in param_grid.items()}
    else:
        pipeline = base_model
    
    if tune_hyperparams and param_grid:
        logger.info("Tuning hyperparameters...")
        grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='f1_weighted', n_jobs=-1, verbose=0)
        grid_search.fit(X_train, y_train)
        model = grid_search.best_estimator_
        logger.info(f"[bold green]Best params:[/bold green] {grid_search.best_params_} | CV score: {grid_search.best_score_:.4f}\n")
    else:
        model = pipeline
        model.fit(X_train, y_train)
    
    logger.info("[bold]Training Set:[/bold]")
    train_metrics = _evaluate_model(model, X_train, y_train, label_encoder, "Training")
    
    logger.info("[bold]Validation Set:[/bold]")
    val_metrics = _evaluate_model(model, X_val, y_val, label_encoder, "Validation")
    
    console.print(Panel(val_metrics['classification_report'], title="[bold]Validation Report[/bold]", box=box.ROUNDED))
    
    return model, {'train_acc': train_metrics['accuracy'], 'train_f1': train_metrics['f1'], 
                   'val_acc': val_metrics['accuracy'], 'val_f1': val_metrics['f1'],
                   'train_acc_all': train_metrics['accuracy_all'], 'train_f1_all': train_metrics['f1_all'],
                   'val_acc_all': val_metrics['accuracy_all'], 'val_f1_all': val_metrics['f1_all'],
                   'per_class_metrics': val_metrics['per_class']}


def save_model(model: Any, feature_names: List[str], class_names: List[str], label_encoder: LabelEncoder, save_path: Path) -> None:
    """Save the trained model and associated metadata."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model_data = {'model': model, 'feature_names': feature_names, 'class_names': class_names, 
                  'label_encoder': label_encoder, 'random_state': random_state}
    with open(save_path, 'wb') as f:
        pickle.dump(model_data, f)
    logger.info(f"[bold green]✓[/bold green] Model saved to {save_path}")


def load_trained_model(model_path: Path) -> Dict[str, Any]:
    """Load a trained model from disk."""
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    logger.info(f"[bold green]✓[/bold green] Model loaded from {model_path}")
    return model_data


def main(project_root: Optional[Path] = None, save_dir: Optional[Path] = None, 
         test_domains: Optional[List[str]] = None) -> Optional[Tuple[Any, List[str], List[str], Dict]]:
    """Main training pipeline with domain-specific test set."""
    project_root = project_root or Path.cwd()
    save_dir = save_dir or project_root / 'src' / 'models'
    
    console.print(Panel.fit("[bold cyan]Product Scraper Category Classification[/bold cyan]\n[yellow]Model Training Pipeline[/yellow]", box=box.DOUBLE_EDGE))
    
    try:
        train_df, test_df = load_data(project_root, test_domains)
    except Exception as e:
        logger.error(f"[bold red]ERROR:[/bold red] {e}")
        return None

    X_train_val, y_train_val, feature_names, class_names, label_encoder = preprocess_data(train_df)
    
    if len(X_train_val) < 10:
        logger.error("Not enough training data.")
        return None

    try:
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=random_state, stratify=y_train_val)
    except ValueError:
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=random_state)
    
    if len(test_df) > 0:
        X_test, y_test, _, _, _ = preprocess_data(test_df, label_encoder)
    else:
        X_test, y_test = None, None
    
    split_table = Table(title="Data Split", box=box.ROUNDED)
    split_table.add_column("Split", style="cyan")
    split_table.add_column("Samples", justify="right", style="green")
    split_table.add_row("Train", str(len(X_train)))
    split_table.add_row("Val", str(len(X_val)))
    if X_test is not None:
        split_table.add_row("Test", str(len(X_test)))
    console.print(split_table)
    console.print()
    
    models, metrics = {}, {}
    
    try:
        model1, metrics1 = train_model(X_train, y_train, X_val, y_val, label_encoder, use_smote=True, model_type='xgboost')
        models['xgboost_smote'] = model1
        metrics['xgboost_smote'] = metrics1
    except Exception as e:
        logger.error(f"XGBoost failed: {e}")

    try:
        model2, metrics2 = train_model(X_train, y_train, X_val, y_val, label_encoder, use_smote=False, model_type='random_forest')
        models['random_forest'] = model2
        metrics['random_forest'] = metrics2
    except Exception as e:
        logger.error(f"RandomForest failed: {e}")

    if not models:
        logger.error("No models trained.")
        return None

    best_model_name = max(metrics, key=lambda k: metrics[k]['val_f1'] or 0)
    best_model = models[best_model_name]
    
    summary_table = Table(title="Model Comparison", box=box.DOUBLE)
    summary_table.add_column("Model", style="cyan")
    summary_table.add_column("Val F1", justify="right", style="green")
    
    for model_name, model_metrics in metrics.items():
        style = "bold green" if model_name == best_model_name else ""
        summary_table.add_row(f"{'★ ' if model_name == best_model_name else ''}{model_name}", 
                              f"{model_metrics['val_f1'] or 0:.4f}", style=style)
    console.print(summary_table)
    console.print()
    
    if X_test is not None and len(X_test) > 0:
        console.print(Panel("[bold yellow]Test Set Evaluation[/bold yellow]", box=box.DOUBLE_EDGE))
        test_metrics = _evaluate_model(best_model, X_test, y_test, label_encoder, "Test")
        console.print(Panel(test_metrics['classification_report'], title="[bold]Test Report[/bold]", box=box.ROUNDED))
    
    model_path = save_dir / 'category_classifier.pkl'
    save_model(best_model, feature_names, class_names, label_encoder, model_path)
    
    console.print(Panel.fit(f"[bold green]✓ Training Complete![/bold green]\nBest: [cyan]{best_model_name}[/cyan]\nSaved: [yellow]{model_path}[/yellow]", box=box.DOUBLE_EDGE))
    
    return best_model, feature_names, class_names, metrics


if __name__ == '__main__':
    main()