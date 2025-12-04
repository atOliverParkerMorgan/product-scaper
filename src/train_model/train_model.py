import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import pickle
import warnings

warnings.filterwarnings('ignore')

random_state = 42

# Features to exclude from training (non-predictive or identifiers)
EXCLUDED_FEATURES = ['Category']


def load_data(project_root: Path = None) -> pd.DataFrame:
    """
    Load all training data from CSV files in the data directory.
    """
    if project_root is None:
        project_root = Path.cwd()
    
    all_data = []
    data_dir = project_root / 'src' / 'data'
    
    if not data_dir.exists():
        # Fallback for simple directory structures
        data_dir = project_root / 'data'
        if not data_dir.exists():
             raise FileNotFoundError(f"Data directory not found at {data_dir}")
    
    # Check for direct CSVs or subdirectories
    files = list(data_dir.rglob('*.csv'))
    
    if not files:
        raise ValueError("No CSV files found in data directory!")

    for csv_path in files:
        # Skip hidden files or temporary files
        if csv_path.name.startswith('.'):
            continue
            
        print(f"Loading data from {csv_path}")
        try:
            domain_data = pd.read_csv(csv_path)
            all_data.append(domain_data)
        except Exception as e:
            print(f"Error loading {csv_path}: {e}")
    
    if not all_data:
        raise ValueError("No valid training data could be loaded!")
    
    combined_data = pd.concat(all_data, ignore_index=True)
    print(f"\nTotal samples loaded: {len(combined_data)}")
    
    if 'Category' in combined_data.columns:
        print(f"Class distribution:\n{combined_data['Category'].value_counts()}\n")
    else:
        raise ValueError("Target column 'Category' missing from loaded data.")
    
    return combined_data


def preprocess_data(df: pd.DataFrame):
    """
    Preprocess the data for training.
    """
    # Separate features and target
    # Ensure we don't try to drop columns that don't exist
    cols_to_drop = [c for c in EXCLUDED_FEATURES if c in df.columns]
    X = df.drop(columns=cols_to_drop)
    y = df['Category']
    
    # Handle any missing values
    X = X.fillna(0)
    
    # Ensure X is numeric (drop any remaining object columns that weren't excluded)
    X = X.select_dtypes(include=[np.number])
    
    feature_names = X.columns.tolist()
    class_names = sorted(y.unique())
    
    print(f"Features used for training: {len(feature_names)}")
    print(f"Feature names: {feature_names}")
    print(f"Classes: {class_names}\n")
    
    return X, y, feature_names, class_names


def train_model(X_train, y_train, X_val, y_val, use_smote=True, model_type='xgboost'):
    """
    Train a classification model with optional SMOTE for handling class imbalance.
    """
    print(f"Training {model_type.upper()} model (SMOTE: {use_smote})...")
    
    # Define model based on type
    if model_type == 'xgboost':
        base_model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=random_state,
            eval_metric='mlogloss',
            n_jobs=-1,
            enable_categorical=False
        )
    else:  # random_forest
        base_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=random_state,
            n_jobs=-1,
            class_weight='balanced'
        )
    
    # Create pipeline with or without SMOTE
    if use_smote:
        # Use SMOTE to oversample minority classes
        smote = SMOTE(random_state=random_state, k_neighbors=3)
        model = ImbPipeline([
            ('smote', smote),
            ('classifier', base_model)
        ])
    else:
        model = base_model
    
    # Train the model
    # Note: sklearn models usually handle string labels, but XGBoost might warn
    # If using XGBoost < 1.0 with string labels, LabelEncoder is needed separately
    model.fit(X_train, y_train)
    
    # Evaluate on training set
    train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, train_pred)
    train_f1 = f1_score(y_train, train_pred, average='weighted')
    
    # Evaluate on validation set
    val_pred = model.predict(X_val)
    val_acc = accuracy_score(y_val, val_pred)
    val_f1 = f1_score(y_val, val_pred, average='weighted')
    
    print(f"\nTraining Accuracy: {train_acc:.4f}")
    print(f"Training F1 Score: {train_f1:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}")
    print(f"Validation F1 Score: {val_f1:.4f}\n")
    
    print("Validation Classification Report:")
    print(classification_report(y_val, val_pred))
    
    print("Validation Confusion Matrix:")
    print(confusion_matrix(y_val, val_pred))
    print()
    
    return model, {
        'train_acc': train_acc,
        'train_f1': train_f1,
        'val_acc': val_acc,
        'val_f1': val_f1
    }


def save_model(model, feature_names, class_names, save_path: Path):
    """
    Save the trained model and associated metadata.
    """
    model_data = {
        'model': model,
        'feature_names': feature_names,
        'class_names': class_names,
        'random_state': random_state
    }
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"Model saved to {save_path}")


def load_trained_model(model_path: Path):
    """
    Load a trained model from disk.
    """
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    return model_data


def main(project_root: Path = None, save_dir: Path = None):
    """
    Main training pipeline.
    """
    if project_root is None:
        project_root = Path.cwd()
    
    if save_dir is None:
        save_dir = project_root / 'src' / 'models'
    
    print("="*70)
    print("Product Scraper Category Classification Model Training")
    
    # 1. Load data
    try:
        df = load_data(project_root)
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        return

    # 2. Preprocess
    # FIX: Removed duplicate faulty call
    X, y, feature_names, class_names = preprocess_data(df)
    
    if len(X) < 10:
        print("Not enough data to split. Aborting.")
        return

    # 3. Split data: 60% train, 20% validation, 20% test
    try:
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, random_state=random_state, stratify=y_temp
        )
    except ValueError as e:
        print(f"Error during splitting (likely class imbalance too high): {e}")
        # Fallback without stratify if classes are too small
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, random_state=random_state
        )
    
    print(f"Train set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")
    print(f"Test set size: {len(X_test)}\n")
    
    # 4. Train multiple models and compare
    models = {}
    metrics = {}
    
    # Model 1: XGBoost with SMOTE
    try:
        model1, metrics1 = train_model(X_train, y_train, X_val, y_val, 
                                       use_smote=True, model_type='xgboost')
        models['xgboost_smote'] = model1
        metrics['xgboost_smote'] = metrics1
    except Exception as e:
        print(f"Failed to train XGBoost: {e}")

    # Model 2: Random Forest with class weights
    try:
        model2, metrics2 = train_model(X_train, y_train, X_val, y_val, 
                                       use_smote=False, model_type='random_forest')
        models['random_forest'] = model2
        metrics['random_forest'] = metrics2
    except Exception as e:
        print(f"Failed to train Random Forest: {e}")

    if not models:
        print("No models were successfully trained.")
        return

    # Select best model based on validation F1 score
    best_model_name = max(metrics, key=lambda k: metrics[k]['val_f1'])
    best_model = models[best_model_name]
    
    print("="*70)
    print(f"Best Model: {best_model_name}")
    print(f"Validation F1 Score: {metrics[best_model_name]['val_f1']:.4f}")
    print("="*70 + "\n")
    
    # 5. Final evaluation on test set
    test_pred = best_model.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)
    test_f1 = f1_score(y_test, test_pred, average='weighted')
    
    print("Final Test Set Performance:")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}\n")
    print("Test Classification Report:")
    print(classification_report(y_test, test_pred))
    
    print("Test Confusion Matrix:")
    # FIX: Added missing print statement
    print(confusion_matrix(y_test, test_pred))
    
    # 6. Save the best model
    model_path = save_dir / 'category_classifier.pkl'
    save_model(best_model, feature_names, class_names, model_path)
    
    print(f"\n{'='*70}")
    print("Training Complete!")
    print(f"{'='*70}")
    
    # FIX: Removed the duplicate unreachable return statement
    return best_model, feature_names, class_names, metrics


if __name__ == '__main__':
    main()