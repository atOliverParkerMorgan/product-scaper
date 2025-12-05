import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import pickle
import warnings

warnings.filterwarnings('ignore')

random_state = 42

# Features to exclude from training (non-predictive or identifiers)
EXCLUDED_FEATURES = ['Category']

# Categories to exclude from evaluation metrics (too generic/noisy)
EXCLUDED_FROM_EVAL = ['other']


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
        print(f"Class distribution (all categories):\n{combined_data['Category'].value_counts()}\n")
        
        # Show distribution excluding 'other'
        meaningful_data = combined_data[~combined_data['Category'].isin(EXCLUDED_FROM_EVAL)]
        if len(meaningful_data) > 0:
            print(f"Class distribution (meaningful categories only, excluding 'other'):\n{meaningful_data['Category'].value_counts()}\n")
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


def calculate_meaningful_metrics(y_true, y_pred, excluded_categories=EXCLUDED_FROM_EVAL):
    """
    Calculate metrics excluding specified categories (like 'other').
    Returns metrics only for meaningful categories.
    """
    # Convert to numpy arrays for consistent handling
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    
    # Create mask for meaningful categories
    mask = ~np.isin(y_true_arr, excluded_categories)
    
    if mask.sum() == 0:
        print("Warning: No meaningful categories found in the data!")
        return None, None, None
    
    y_true_filtered = y_true_arr[mask]
    y_pred_filtered = y_pred_arr[mask]
    
    acc = accuracy_score(y_true_filtered, y_pred_filtered)
    f1 = f1_score(y_true_filtered, y_pred_filtered, average='weighted', zero_division=0)
    
    # Get per-class metrics
    labels = sorted(set(y_true_filtered))
    precision, recall, f1_per_class, support = precision_recall_fscore_support(
        y_true_filtered, y_pred_filtered, labels=labels, average=None, zero_division=0
    )
    
    return acc, f1, {
        'precision': dict(zip(labels, precision)),
        'recall': dict(zip(labels, recall)),
        'f1': dict(zip(labels, f1_per_class)),
        'support': dict(zip(labels, support))
    }


def train_model(X_train, y_train, X_val, y_val, use_smote=True, model_type='xgboost', tune_hyperparams=True):
    """
    Train a classification model with optional SMOTE and hyperparameter tuning.
    """
    print(f"Training {model_type.upper()} model (SMOTE: {use_smote}, Tuning: {tune_hyperparams})...")
    
    # XGBoost requires numeric labels, so encode them
    label_encoder = None
    if model_type == 'xgboost':
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)
        y_val_encoded = label_encoder.transform(y_val)
    else:
        y_train_encoded = y_train
        y_val_encoded = y_val
    
    # Define model based on type with hyperparameter grid
    if model_type == 'xgboost':
        if tune_hyperparams:
            base_model = XGBClassifier(
                random_state=random_state,
                eval_metric='mlogloss',
                n_jobs=-1,
                enable_categorical=False
            )
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [4, 6, 8],
                'learning_rate': [0.05, 0.1],
                'min_child_weight': [1, 3],
                'subsample': [0.8, 1.0]
            }
        else:
            base_model = XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=random_state,
                eval_metric='mlogloss',
                n_jobs=-1,
                enable_categorical=False
            )
            param_grid = None
    else:  # random_forest
        if tune_hyperparams:
            base_model = RandomForestClassifier(
                random_state=random_state,
                n_jobs=-1,
                class_weight='balanced'
            )
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [None, 20],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
                'max_features': ['sqrt']
            }
        else:
            base_model = RandomForestClassifier(
                n_estimators=200,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=random_state,
                n_jobs=-1,
                class_weight='balanced'
            )
            param_grid = None
    
    # Create pipeline with or without SMOTE
    if use_smote:
        smote = SMOTE(random_state=random_state, k_neighbors=3)
        pipeline = ImbPipeline([
            ('smote', smote),
            ('classifier', base_model)
        ])
        # Adjust param_grid for pipeline
        if param_grid:
            param_grid = {f'classifier__{k}': v for k, v in param_grid.items()}
    else:
        pipeline = base_model
    
    # Hyperparameter tuning with GridSearchCV
    if tune_hyperparams and param_grid:
        print("Performing hyperparameter tuning with GridSearchCV...")
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=3,
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X_train, y_train_encoded)
        model = grid_search.best_estimator_
        print(f"\nBest hyperparameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}\n")
    else:
        model = pipeline
        model.fit(X_train, y_train_encoded)
    
    # Evaluate on training set (all categories)
    train_pred_encoded = model.predict(X_train)
    # Decode predictions for XGBoost
    if label_encoder:
        train_pred = label_encoder.inverse_transform(train_pred_encoded)
    else:
        train_pred = train_pred_encoded
    
    train_acc_all = accuracy_score(y_train, train_pred)
    train_f1_all = f1_score(y_train, train_pred, average='weighted')
    
    # Evaluate on training set (meaningful categories only)
    train_acc, train_f1, train_per_class = calculate_meaningful_metrics(
        y_train, train_pred
    )
    
    # Evaluate on validation set (all categories)
    val_pred_encoded = model.predict(X_val)
    # Decode predictions for XGBoost
    if label_encoder:
        val_pred = label_encoder.inverse_transform(val_pred_encoded)
    else:
        val_pred = val_pred_encoded
    
    val_acc_all = accuracy_score(y_val, val_pred)
    val_f1_all = f1_score(y_val, val_pred, average='weighted')
    
    # Evaluate on validation set (meaningful categories only)
    val_acc, val_f1, val_per_class = calculate_meaningful_metrics(
        y_val, val_pred
    )
    
    print(f"\n{'='*70}")
    print("METRICS INCLUDING 'OTHER' CATEGORY:")
    print(f"Training Accuracy: {train_acc_all:.4f} | Training F1: {train_f1_all:.4f}")
    print(f"Validation Accuracy: {val_acc_all:.4f} | Validation F1: {val_f1_all:.4f}")
    
    print(f"\n{'='*70}")
    print("METRICS FOR MEANINGFUL CATEGORIES ONLY (excluding 'other'):")
    print(f"Training Accuracy: {train_acc:.4f} | Training F1: {train_f1:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f} | Validation F1: {val_f1:.4f}")
    
    if val_per_class:
        print("\nPer-Class Metrics (Validation Set, Meaningful Categories):")
        for category in sorted(val_per_class['f1'].keys()):
            print(f"  {category:15s} - P: {val_per_class['precision'][category]:.3f}, "
                  f"R: {val_per_class['recall'][category]:.3f}, "
                  f"F1: {val_per_class['f1'][category]:.3f}, "
                  f"Support: {val_per_class['support'][category]:.0f}")
    print(f"{'='*70}\n")
    
    # Show full classification report and confusion matrix
    print("Full Validation Classification Report (all categories):")
    print(classification_report(y_val, val_pred))
    
    print("\nValidation Confusion Matrix (all categories):")
    print(confusion_matrix(y_val, val_pred))
    print()
    
    return model, label_encoder, {
        'train_acc': train_acc,
        'train_f1': train_f1,
        'val_acc': val_acc,
        'val_f1': val_f1,
        'train_acc_all': train_acc_all,
        'train_f1_all': train_f1_all,
        'val_acc_all': val_acc_all,
        'val_f1_all': val_f1_all,
        'per_class_metrics': val_per_class
    }


def save_model(model, feature_names, class_names, save_path: Path, label_encoder=None):
    """
    Save the trained model and associated metadata.
    """
    model_data = {
        'model': model,
        'feature_names': feature_names,
        'class_names': class_names,
        'label_encoder': label_encoder,
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
    label_encoders = {}
    metrics = {}
    
    # Model 1: XGBoost with SMOTE and hyperparameter tuning
    try:
        model1, encoder1, metrics1 = train_model(X_train, y_train, X_val, y_val, 
                                       use_smote=True, model_type='xgboost', tune_hyperparams=False)
        models['xgboost_smote_tuned'] = model1
        label_encoders['xgboost_smote_tuned'] = encoder1
        metrics['xgboost_smote_tuned'] = metrics1
    except Exception as e:
        print(f"Failed to train XGBoost with tuning: {e}")
        # Fallback without tuning
        try:
            model1, encoder1, metrics1 = train_model(X_train, y_train, X_val, y_val, 
                                           use_smote=True, model_type='xgboost', tune_hyperparams=False)
            models['xgboost_smote'] = model1
            label_encoders['xgboost_smote'] = encoder1
            metrics['xgboost_smote'] = metrics1
        except Exception as e2:
            print(f"Failed to train XGBoost without tuning: {e2}")

    # Model 2: Random Forest with class weights and hyperparameter tuning
    try:
        model2, encoder2, metrics2 = train_model(X_train, y_train, X_val, y_val, 
                                       use_smote=False, model_type='random_forest', tune_hyperparams=False)
        models['random_forest_tuned'] = model2
        label_encoders['random_forest_tuned'] = encoder2
        metrics['random_forest_tuned'] = metrics2
    except Exception as e:
        print(f"Failed to train Random Forest with tuning: {e}")
        # Fallback without tuning
        try:
            model2, encoder2, metrics2 = train_model(X_train, y_train, X_val, y_val, 
                                           use_smote=False, model_type='random_forest', tune_hyperparams=False)
            models['random_forest'] = model2
            label_encoders['random_forest'] = encoder2
            metrics['random_forest'] = metrics2
        except Exception as e2:
            print(f"Failed to train Random Forest without tuning: {e2}")

    if not models:
        print("No models were successfully trained.")
        return

    # Select best model based on meaningful validation F1 score (excluding 'other')
    best_model_name = max(metrics, key=lambda k: metrics[k]['val_f1'])
    best_model = models[best_model_name]
    best_label_encoder = label_encoders[best_model_name]
    
    print("="*70)
    print(f"Best Model: {best_model_name}")
    print(f"Validation F1 Score (meaningful categories): {metrics[best_model_name]['val_f1']:.4f}")
    print(f"Validation F1 Score (all categories): {metrics[best_model_name]['val_f1_all']:.4f}")
    print("="*70 + "\n")
    
    # 5. Final evaluation on test set
    test_pred_encoded = best_model.predict(X_test)
    # Decode predictions if XGBoost
    if best_label_encoder:
        test_pred = best_label_encoder.inverse_transform(test_pred_encoded)
    else:
        test_pred = test_pred_encoded
    
    test_acc_all = accuracy_score(y_test, test_pred)
    test_f1_all = f1_score(y_test, test_pred, average='weighted')
    
    # Calculate meaningful metrics on test set
    test_acc, test_f1, test_per_class = calculate_meaningful_metrics(
        y_test, test_pred
    )
    
    print("="*70)
    print("FINAL TEST SET PERFORMANCE")
    print("="*70)
    print("\nMetrics including 'other' category:")
    print(f"Test Accuracy: {test_acc_all:.4f}")
    print(f"Test F1 Score: {test_f1_all:.4f}")
    
    print("\nMetrics for meaningful categories only (excluding 'other'):")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")
    
    if test_per_class:
        print("\nPer-Class Metrics (Test Set, Meaningful Categories):")
        for category in sorted(test_per_class['f1'].keys()):
            print(f"  {category:15s} - P: {test_per_class['precision'][category]:.3f}, "
                  f"R: {test_per_class['recall'][category]:.3f}, "
                  f"F1: {test_per_class['f1'][category]:.3f}, "
                  f"Support: {test_per_class['support'][category]:.0f}")
    
    print("\n" + "="*70)
    print("Test Classification Report (all categories):")
    print(classification_report(y_test, test_pred))
    
    print("\nTest Confusion Matrix (all categories):")
    print(confusion_matrix(y_test, test_pred))
    
    # 6. Save the best model
    model_path = save_dir / 'category_classifier.pkl'
    save_model(best_model, feature_names, class_names, model_path, best_label_encoder)
    
    print(f"\n{'='*70}")
    print("Training Complete!")
    print(f"{'='*70}")
    
    # FIX: Removed the duplicate unreachable return statement
    return best_model, feature_names, class_names, metrics


if __name__ == '__main__':
    main()