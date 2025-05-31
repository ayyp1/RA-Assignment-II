import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, confusion_matrix, f1_score, precision_score, balanced_accuracy_score, roc_curve
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import os
import warnings
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin  # ADDED MISSING IMPORT


warnings.filterwarnings('ignore')

np.random.seed(42)

class FeatureChange(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.training_features_ = None
        self.positive_cols_ = None
        self.poly_cols_ = None
        
    def fit(self, X, y=None):
      
        self.training_features_ = X.columns.tolist()
        
        
        self.positive_cols_ = [col for col in self.training_features_ 
                               if X[col].min() > 0]
        
       
        self.poly_cols_ = self.training_features_[:min(5, len(self.training_features_))]
        
        return self
        
    def transform(self, X):
        # Align features with training set
        aligned_X = pd.DataFrame(columns=self.training_features_)
        
        # Add existing features
        for col in self.training_features_:
            if col in X.columns:
                aligned_X[col] = X[col]
            else:
                aligned_X[col] = 0  # Fill missing features with zeros
                
        # Create interaction feature
        if len(self.training_features_) >= 2:
            col1, col2 = self.training_features_[:2]
            aligned_X['FEATURE_INTERACTION'] = aligned_X[col1] * aligned_X[col2]
        
        
        for col in self.positive_cols_:
            aligned_X[f'LOG_{col}'] = np.log1p(np.clip(aligned_X[col], 1e-10, None))
        
       
       
        for col in self.poly_cols_:
            aligned_X[f'POLY_{col}'] = aligned_X[col] ** 2
        
        return aligned_X.values

def load_and_preprocess_data(file_path, is_train=True, feature_aligner=None, 
                            selector=None, scaler=None):
    df = pd.read_csv(file_path)
    
    # Handle different file structures
    if 'CLASS' in df.columns:
        X = df.drop(columns=['ID', 'CLASS'])
        y = df['CLASS'] if is_train else None
    else:
        X = df.drop(columns=['ID'])
        y = None
        
    ids = df['ID']
    
    # Handle missing values and infinity
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))
    X = X.clip(lower=-1e10, upper=1e10)
    
    # Apply Winsorization
    def winsorize(series):
        q1 = series.quantile(0.05)
        q3 = series.quantile(0.95)
        return np.clip(series, q1, q3)
    
    X = X.apply(winsorize, axis=0)
    
    # Apply feature alignment and engineering
    if is_train:
        feature_aligner = FeatureChange()
        feature_aligner.fit(X)
        X_engineered = feature_aligner.transform(X)
        print(f"Training features after engineering: {X_engineered.shape[1]}")
    else:
        X_engineered = feature_aligner.transform(X)
        print(f"Test features after engineering: {X_engineered.shape[1]}")
    
    # Apply scaling
    if is_train:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_engineered)
    else:
        X_scaled = scaler.transform(X_engineered)
    
    # Feature selection
    if is_train:
        base_estimator = LogisticRegression(max_iter=1000)
        selector = RFE(base_estimator, n_features_to_select=50)
        X_selected = selector.fit_transform(X_scaled, y)
        selected_features = selector.get_support(indices=True)
    else:
        X_selected = selector.transform(X_scaled)
        selected_features = None
    
    return X_selected, y, ids, selector, scaler, feature_aligner

def calculate_metrics(y_true, y_pred, y_pred_proba):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Balanced Accuracy': balanced_accuracy_score(y_true, y_pred),
        'AUROC': roc_auc_score(y_true, y_pred_proba[:, 1]),
        'Sensitivity': recall_score(y_true, y_pred),
        'Specificity': specificity,
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'F1-score': f1_score(y_true, y_pred)
    }

def train_and_evaluate_models(X_train, y_train, X_val, y_val):
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000, class_weight='balanced'),
        'RandomForest': RandomForestClassifier(class_weight='balanced_subsample', random_state=42),
        'XGBoost': XGBClassifier(eval_metric='logloss', random_state=42),
        'LightGBM': LGBMClassifier(random_state=42, verbose=-1)
    }
    
    param_grids = {
    'LogisticRegression': {
        'classifier__C': np.logspace(-3, 3, 20),
        'classifier__solver': ['lbfgs', 'saga']
    },
    'RandomForest': {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [5, 10, None]
    },
    'XGBoost': {
        'classifier__n_estimators': [100, 200],
        'classifier__learning_rate': [0.01, 0.1],
        'classifier__max_depth': [3, 6]
    },
    'LightGBM': {
        'classifier__n_estimators': [100, 200],
        'classifier__learning_rate': [0.01, 0.1],
        'classifier__max_depth': [3, 6]
    }
    }
    
    results = {}
    best_models = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Create pipeline with SMOTE
        pipeline = Pipeline([
            ('smote', SMOTE(random_state=42)),
            ('classifier', model)
        ])
        
        # Hyperparameter tuning
        grid_search = RandomizedSearchCV(
            pipeline,
            param_grids[name],
            n_iter=10,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring='roc_auc',
            n_jobs=-1,
            random_state=42
        )
        
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        print(f"Best parameters: {best_params}")
        
        # Calibrate classifier for better probability estimates
        calibrated_model = CalibratedClassifierCV(best_model, method='isotonic', cv=5)
        calibrated_model.fit(X_train, y_train)
        
        # Validation predictions
        y_pred = calibrated_model.predict(X_val)
        y_pred_proba = calibrated_model.predict_proba(X_val)
        
        # Calculate metrics
        metrics = calculate_metrics(y_val, y_pred, y_pred_proba)
        
        # Store results
        results[name] = metrics
        best_models[name] = calibrated_model
        
        # Print metrics
        print(f"{name} Validation Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
    
    return results, best_models

def create_stacking_ensemble(base_models, X_train, y_train, X_val, y_val):
    estimators = [(name, model) for name, model in base_models.items()]
    
    stacking_model = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000, class_weight='balanced'),
        stack_method='predict_proba',
        n_jobs=-1
    )
    
    # Train stacking model
    stacking_model.fit(X_train, y_train)
    
    # Validation predictions
    y_pred = stacking_model.predict(X_val)
    y_pred_proba = stacking_model.predict_proba(X_val)
    
    # Calculate metrics
    metrics = calculate_metrics(y_val, y_pred, y_pred_proba)
    
    print("\nStacking Ensemble Validation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    return stacking_model, metrics

def plot_roc_curves(y_val, predictions, model_names):
    plt.figure(figsize=(10, 8))
    
    for name in model_names:
        fpr, tpr, _ = roc_curve(y_val, predictions[name]['proba'][:, 1])
        roc_auc = roc_auc_score(y_val, predictions[name]['proba'][:, 1])
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.4f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend()
    plt.savefig('roc_curves_comparison.png')
    plt.show()

def save_predictions(ids, probas, filename):
    output = pd.DataFrame({
        'ID': ids,
        'CLASS_0': probas[:, 0],
        'CLASS_1': probas[:, 1]
    })
    output.to_csv(filename, index=False)

def main():
    train_file = 'train_set.csv'
    test_file = 'test_set.csv'
    blinded_file = 'blinded_test_set.csv'
    
    # Load training data
    print("Loading training data...")
    (X_train, y_train, ids_train, selector, 
     scaler, feature_aligner) = load_and_preprocess_data(train_file, is_train=True)
    
    # Split into train and validation sets
    print("Splitting data...")
    # Create indices for splitting
    train_idx, val_idx = train_test_split(
        np.arange(len(X_train)), 
        test_size=0.2, 
        random_state=42, 
        stratify=y_train
    )
    
    # Split features, target, and IDs
    X_train_split, X_val = X_train[train_idx], X_train[val_idx]
    y_train_split, y_val = y_train[train_idx], y_train[val_idx]
    ids_train_split, ids_val = ids_train[train_idx], ids_train[val_idx]
    
    # Load test data
    print("Loading test data...")
    X_test, y_test, ids_test, _, _, _ = load_and_preprocess_data(
        test_file, is_train=False, 
        feature_aligner=feature_aligner,
        selector=selector, 
        scaler=scaler
    )
    
    
    # Load blinded data
    print("Loading blinded data...")
    X_blinded, _, ids_blinded, _, _, _ = load_and_preprocess_data(
        blinded_file, is_train=False, 
        feature_aligner=feature_aligner,
        selector=selector, 
        scaler=scaler
    )
    
    print("\nData loaded successfully. Feature counts:")
    print(f"- Training: {X_train.shape[1]}")
    print(f"- Validation: {X_val.shape[1]}")
    print(f"- Test: {X_test.shape[1]}")
    print(f"- Blinded: {X_blinded.shape[1]}")
    
    # Train and evaluate models
    print("\nTraining models on training set...")
    val_results, best_models = train_and_evaluate_models(X_train_split, y_train_split, X_val, y_val)
    
    # Create and evaluate stacking ensemble
    stacking_model, stacking_metrics = create_stacking_ensemble(
        best_models, X_train_split, y_train_split, X_val, y_val
    )
    val_results['Stacking'] = stacking_metrics
    
    # Print validation results
    print("\nValidation Set Performance:")
    results_df = pd.DataFrame(val_results).T
    print(results_df)
    
    # Collect predictions for ROC curve
    predictions = {}
    for name, model in best_models.items():
        y_pred_proba = model.predict_proba(X_val)
        predictions[name] = {
            'pred': model.predict(X_val),
            'proba': y_pred_proba
        }
    y_pred_proba_stack = stacking_model.predict_proba(X_val)
    predictions['Stacking'] = {
        'pred': stacking_model.predict(X_val),
        'proba': y_pred_proba_stack
    }
    
    # Plot ROC curves
    plot_roc_curves(y_val, predictions, list(best_models.keys()) + ['Stacking'])
    
    # Generate and save predictions
    os.makedirs('predictions', exist_ok=True)
    print("\nSaving predictions...")
    
    # Train final models on full training data
    for name, model in best_models.items():
        model.fit(X_train, y_train)
        
        # Save predictions
        save_predictions(ids_train, model.predict_proba(X_train), 
                        f'predictions/train_predictions_{name}.csv')
        save_predictions(ids_test, model.predict_proba(X_test), 
                        f'predictions/test_predictions_{name}.csv')
        save_predictions(ids_blinded, model.predict_proba(X_blinded), 
                        f'predictions/blinded_predictions_{name}.csv')
    
    # Train and save stacking predictions
    stacking_model.fit(X_train, y_train)
    save_predictions(ids_train, stacking_model.predict_proba(X_train), 
                    'predictions/train_predictions_stacking.csv')
    save_predictions(ids_test, stacking_model.predict_proba(X_test), 
                    'predictions/test_predictions_stacking.csv')
    save_predictions(ids_blinded, stacking_model.predict_proba(X_blinded), 
                    'predictions/blinded_predictions_stacking.csv')
    
    # Select best model based on validation AUROC
    best_model_name = max(val_results, key=lambda k: val_results[k]['AUROC'])
    best_model = best_models[best_model_name] if best_model_name != 'Stacking' else stacking_model
    print(f"\nBest model: {best_model_name} with Validation AUROC: {val_results[best_model_name]['AUROC']:.4f}")
    
    # Save best model predictions
    save_predictions(ids_blinded, best_model.predict_proba(X_blinded), 
                    'predictions/blinded_predictions_best.csv')
    
    # Print final summary
    print("\nTraining completed successfully!")
    print(f"Best model: {best_model_name}")
    print("Validation Metrics:")
    print(pd.DataFrame(val_results[best_model_name], index=[best_model_name]))
    
    # Feature importance for tree-based models
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
        sorted_idx = importances.argsort()[::-1][:10]
        
        print("\nTop 10 Features:")
        for i in sorted_idx:
            print(f"Feature {i}: {importances[i]:.4f}")
        
        # Plot feature importance
        plt.figure(figsize=(10, 6))
        plt.title("Feature Importances")
        plt.bar(range(10), importances[sorted_idx[:10]])
        plt.xticks(range(10), sorted_idx[:10])
        plt.tight_layout()
        plt.savefig('feature_importances.png')
        plt.show()
    else:
        print("\nFeature importances not available for this model type")

if __name__ == "__main__":
    main()