import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any
import pickle
import logging
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

logger = logging.getLogger(__name__)
console = Console()


def load_trained_model(model_path: Path) -> Dict[str, Any]:
    """Load a trained model from disk."""
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    logger.info(f"Model loaded from {model_path}")
    return model_data


def load_data(data_path: Path) -> pd.DataFrame:
    """Load data from CSV file."""
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} samples from {data_path}")
    return df


def preprocess_data(df: pd.DataFrame, feature_names: List[str]) -> pd.DataFrame:
    """Preprocess data to match training feature set."""
    # Keep only the features used during training
    X = df[feature_names].fillna(0)
    return X


def predict(model: Any, X: pd.DataFrame, label_encoder: Any) -> np.ndarray:
    """Make predictions and decode labels."""
    pred_encoded = model.predict(X)
    pred = label_encoder.inverse_transform(pred_encoded)
    return pred


def predict_data(model_path: Path, data_path: Path, output_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Main prediction pipeline.
    
    Args:
        model_path: Path to the trained model file
        data_path: Path to the CSV file with data to predict
        output_path: Optional path to save predictions
    
    Returns:
        DataFrame with predictions added
    """
    console.print(Panel.fit("[bold cyan]Product Category Prediction[/bold cyan]", box=box.DOUBLE_EDGE))
    
    # Load model and data
    model_data = load_trained_model(model_path)
    model = model_data['model']
    label_encoder = model_data['label_encoder']
    feature_names = model_data['feature_names']
    
    df = load_data(data_path)
    
    # Preprocess
    X = preprocess_data(df, feature_names)
    
    # Predict
    predictions = predict(model, X, label_encoder)
    
    # Add predictions to dataframe
    df['Predicted_Category'] = predictions
    
    # Display prediction summary
    table = Table(title="Prediction Summary", box=box.ROUNDED)
    table.add_column("Category", style="cyan")
    table.add_column("Count", justify="right", style="green")
    
    pred_counts = pd.Series(predictions).value_counts()
    for category, count in pred_counts.items():
        table.add_row(category, str(count))
    
    console.print(table)
    
    # Save if output path provided
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Predictions saved to {output_path}")
    
    console.print(Panel.fit(f"[bold green]✓ Prediction Complete![/bold green]\nProcessed {len(df)} samples", box=box.DOUBLE_EDGE))
    
    return df


def main(model_path: Optional[Path] = None, data_path: Optional[Path] = None, output_path: Optional[Path] = None):
    """Main entry point for prediction."""
    project_root = Path.cwd()
    
    if model_path is None:
        model_path = project_root / 'src' / 'models' / 'category_classifier.pkl'
    
    if data_path is None:
        # Default to first domain data for demo
        data_path = project_root / 'src' / 'data' / 'antik-bilevrany' / 'data.csv'
    
    if output_path is None:
        output_path = data_path.parent / 'predictions.csv'
    
    try:
        predict_data(model_path, data_path, output_path)
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise


if __name__ == '__main__':
    main()
