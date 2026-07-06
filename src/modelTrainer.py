"""
Model Trainer Module
Handles model training, evaluation, and selection
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
    precision_recall_curve
)
import joblib
import warnings
warnings.filterwarnings('ignore')


class ModelTrainer:
    """Class for training and evaluating fraud detection models"""
    
    def __init__(self, random_state: int = 42):
        """
        Initialize ModelTrainer
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.models = self._initialize_models()
        self.trained_models = {}
        self.results = {}
        
    def _initialize_models(self) -> Dict[str, Any]:
        """
        Initialize all models with default parameters
        
        Returns:
            Dictionary of model instances
        """
        return {
            'Logistic Regression': LogisticRegression(
                max_iter=1000,
                random_state=self.random_state,
                class_weight='balanced'
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=-1
            ),
            'XGBoost': XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                scale_pos_weight=10,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                eval_metric='logloss',
                use_label_encoder=False
            )
        }
    
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray, 
                   model_name: str) -> Any:
        """
        Train a specific model
        
        Args:
            X_train: Training features
            y_train: Training targets
            model_name: Name of model to train
            
        Returns:
            Trained model
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        model.fit(X_train, y_train)
        self.trained_models[model_name] = model
        return model
    
    def train_all_models(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """
        Train all models
        
        Args:
            X_train: Training features
            y_train: Training targets
            
        Returns:
            Dictionary of trained models
        """
        for name in self.models.keys():
            print(f"Training {name}...")
            self.train_model(X_train, y_train, name)
        return self.trained_models
    
    def evaluate_model(self, model: Any, X_test: np.ndarray, y_test: np.ndarray,
                      model_name: str, dataset_name: str) -> Dict[str, Any]:
        """
        Evaluate a trained model
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test targets
            model_name: Name of the model
            dataset_name: Name of the dataset
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'Model': model_name,
            'Dataset': dataset_name,
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1 Score': f1_score(y_test, y_pred),
            'AUC-ROC': roc_auc_score(y_test, y_pred_proba),
            'AUC-PR': average_precision_score(y_test, y_pred_proba)
        }
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        metrics['True Negatives'] = tn
        metrics['False Positives'] = fp
        metrics['False Negatives'] = fn
        metrics['True Positives'] = tp
        
        return metrics
    
    def evaluate_all_models(self, X_test: np.ndarray, y_test: np.ndarray,
                           dataset_name: str) -> pd.DataFrame:
        """
        Evaluate all trained models
        
        Args:
            X_test: Test features
            y_test: Test targets
            dataset_name: Name of the dataset
            
        Returns:
            DataFrame with evaluation results
        """
        results = []
        for name, model in self.trained_models.items():
            metrics = self.evaluate_model(model, X_test, y_test, name, dataset_name)
            results.append(metrics)
        
        self.results[dataset_name] = pd.DataFrame(results)
        return self.results[dataset_name]
    
    def cross_validate(self, model: Any, X: np.ndarray, y: np.ndarray,
                      model_name: str, dataset_name: str, cv: int = 5) -> Dict[str, float]:
        """
        Perform cross-validation for a model
        
        Args:
            model: Model to cross-validate
            X: Features
            y: Targets
            model_name: Name of the model
            dataset_name: Name of the dataset
            cv: Number of folds
            
        Returns:
            Dictionary with cross-validation scores
        """
        cv_strat = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        
        scores = {
            'Model': model_name,
            'Dataset': dataset_name,
            'AUC-ROC_mean': cross_val_score(model, X, y, cv=cv_strat, scoring='roc_auc').mean(),
            'AUC-ROC_std': cross_val_score(model, X, y, cv=cv_strat, scoring='roc_auc').std(),
            'AUC-PR_mean': cross_val_score(model, X, y, cv=cv_strat, scoring='average_precision').mean(),
            'AUC-PR_std': cross_val_score(model, X, y, cv=cv_strat, scoring='average_precision').std(),
            'F1_mean': cross_val_score(model, X, y, cv=cv_strat, scoring='f1').mean(),
            'F1_std': cross_val_score(model, X, y, cv=cv_strat, scoring='f1').std()
        }
        
        return scores
    
    def cross_validate_all(self, X: np.ndarray, y: np.ndarray,
                          dataset_name: str, cv: int = 5) -> pd.DataFrame:
        """
        Cross-validate all trained models
        
        Args:
            X: Features
            y: Targets
            dataset_name: Name of the dataset
            cv: Number of folds
            
        Returns:
            DataFrame with cross-validation results
        """
        results = []
        for name, model in self.trained_models.items():
            scores = self.cross_validate(model, X, y, name, dataset_name, cv)
            results.append(scores)
        
        cv_results = pd.DataFrame(results)
        return cv_results
    
    def select_best_model(self, dataset_name: str, metric: str = 'F1 Score') -> Dict[str, Any]:
        """
        Select the best model based on a metric
        
        Args:
            dataset_name: Name of the dataset
            metric: Metric to use for selection
            
        Returns:
            Dictionary with best model information
        """
        if dataset_name not in self.results:
            raise ValueError(f"Results for {dataset_name} not found")
        
        df_results = self.results[dataset_name]
        best_idx = df_results[metric].idxmax()
        best_model_name = df_results.loc[best_idx, 'Model']
        best_model = self.trained_models[best_model_name]
        
        return {
            'model_name': best_model_name,
            'model': best_model,
            'metrics': df_results.loc[best_idx].to_dict()
        }
    
    def save_models(self, path: str = 'models/'):
        """
        Save all trained models
        
        Args:
            path: Directory to save models
        """
        import os
        os.makedirs(path, exist_ok=True)
        
        for name, model in self.trained_models.items():
            filename = f"{path}/{name.replace(' ', '_').lower()}.pkl"
            joblib.dump(model, filename)
            print(f"Saved {name} to {filename}")
    
    def load_models(self, path: str = 'models/'):
        """
        Load trained models
        
        Args:
            path: Directory containing models
        """
        import glob
        import os
        
        model_files = glob.glob(f"{path}/*.pkl")
        for file in model_files:
            name = os.path.basename(file).replace('.pkl', '').replace('_', ' ').title()
            self.trained_models[name] = joblib.load(file)


def create_model_summary(results_df: pd.DataFrame) -> str:
    """
    Create a summary string from model results
    
    Args:
        results_df: DataFrame with model results
        
    Returns:
        Formatted summary string
    """
    summary = "=" * 60 + "\n"
    summary += "MODEL PERFORMANCE SUMMARY\n"
    summary += "=" * 60 + "\n\n"
    
    for _, row in results_df.iterrows():
        summary += f"{row['Model']}:\n"
        summary += f"  F1 Score: {row['F1 Score']:.4f}\n"
        summary += f"  AUC-PR:   {row['AUC-PR']:.4f}\n"
        summary += f"  Recall:   {row['Recall']:.4f}\n"
        summary += f"  Precision:{row['Precision']:.4f}\n"
        summary += "\n"
    
    return summary