"""
Complete Pipeline Script
Runs the entire fraud detection pipeline from data loading to model evaluation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from src.data_loader import DataLoader, get_data_info
from src.preprocessor import FraudDataPreprocessor, CreditDataPreprocessor
from src.model_trainer import ModelTrainer, create_model_summary


def run_fraud_pipeline():
    """
    Run the complete fraud detection pipeline
    """
    print("="*60)
    print("FRAUD DETECTION PIPELINE")
    print("="*60)
    
    # Step 1: Load Data
    print("\n" + "="*60)
    print("STEP 1: LOADING DATA")
    print("="*60)
    
    loader = DataLoader()
    data = loader.load_all_data()
    
    fraud_df = data['fraud']
    ip_df = data['ip']
    credit_df = data['credit']
    
    print(f"\nFraud Data: {fraud_df.shape}")
    print(f"IP Data: {ip_df.shape}")
    print(f"Credit Data: {credit_df.shape}")
    
    # Step 2: Preprocess Fraud Data
    print("\n" + "="*60)
    print("STEP 2: PREPROCESSING FRAUD DATA")
    print("="*60)
    
    fraud_preprocessor = FraudDataPreprocessor()
    
    # Clean data
    fraud_clean = fraud_preprocessor.clean_data(fraud_df)
    print(f"Cleaned fraud data: {fraud_clean.shape}")
    
    # Add IP features
    fraud_with_ip = fraud_preprocessor.add_ip_features(fraud_clean)
    print(f"Added IP features: {fraud_with_ip.shape}")
    
    # Add country feature
    fraud_with_country = fraud_preprocessor.add_country_feature(fraud_with_ip, ip_df)
    print(f"Added country feature: {fraud_with_country.shape}")
    
    # Engineer features
    fraud_engineered = fraud_preprocessor.engineer_features(fraud_with_country)
    print(f"Engineered features: {fraud_engineered.shape}")
    
    # Prepare for modeling
    X_fraud, y_fraud = fraud_preprocessor.prepare_for_modeling(fraud_engineered)
    print(f"Features shape: {X_fraud.shape}, Target shape: {y_fraud.shape}")
    
    # Step 3: Preprocess Credit Data
    print("\n" + "="*60)
    print("STEP 3: PREPROCESSING CREDIT DATA")
    print("="*60)
    
    credit_preprocessor = CreditDataPreprocessor()
    
    # Clean data
    credit_clean = credit_preprocessor.clean_data(credit_df)
    print(f"Cleaned credit data: {credit_clean.shape}")
    
    # Prepare for modeling
    X_credit, y_credit = credit_preprocessor.prepare_for_modeling(credit_clean)
    print(f"Features shape: {X_credit.shape}, Target shape: {y_credit.shape}")
    
    # Step 4: Train-Test Split
    print("\n" + "="*60)
    print("STEP 4: TRAIN-TEST SPLIT")
    print("="*60)
    
    # Fraud data split
    X_fraud_train, X_fraud_test, y_fraud_train, y_fraud_test = train_test_split(
        X_fraud, y_fraud, test_size=0.2, random_state=42, stratify=y_fraud
    )
    print(f"Fraud Data - Train: {X_fraud_train.shape}, Test: {X_fraud_test.shape}")
    print(f"Fraud rate - Train: {y_fraud_train.mean()*100:.2f}%, Test: {y_fraud_test.mean()*100:.2f}%")
    
    # Credit data split
    X_credit_train, X_credit_test, y_credit_train, y_credit_test = train_test_split(
        X_credit, y_credit, test_size=0.2, random_state=42, stratify=y_credit
    )
    print(f"Credit Data - Train: {X_credit_train.shape}, Test: {X_credit_test.shape}")
    print(f"Fraud rate - Train: {y_credit_train.mean()*100:.2f}%, Test: {y_credit_test.mean()*100:.2f}%")
    
    # Step 5: Handle Class Imbalance (Oversampling)
    print("\n" + "="*60)
    print("STEP 5: CLASS IMBALANCE HANDLING")
    print("="*60)
    
    def random_oversample(X, y, target_ratio=0.2):
        """Simple random oversampling"""
        # Convert to numpy arrays
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        # Separate majority and minority
        majority_idx = np.where(y == 0)[0]
        minority_idx = np.where(y == 1)[0]
        
        X_majority = X[majority_idx]
        X_minority = X[minority_idx]
        
        n_majority = len(X_majority)
        n_minority = len(X_minority)
        n_target = int(n_majority * target_ratio)
        
        if n_target <= n_minority:
            return X, y
        
        n_to_add = n_target - n_minority
        
        # Sample with replacement
        additional_idx = np.random.choice(n_minority, n_to_add, replace=True)
        X_additional = X_minority[additional_idx]
        y_additional = np.ones(n_to_add)
        
        # Combine
        X_resampled = np.vstack([X, X_additional])
        y_resampled = np.hstack([y, y_additional])
        
        return X_resampled, y_resampled
    
    # Scale features
    scaler_fraud = StandardScaler()
    X_fraud_train_scaled = scaler_fraud.fit_transform(X_fraud_train)
    
    scaler_credit = StandardScaler()
    X_credit_train_scaled = scaler_credit.fit_transform(X_credit_train)
    
    # Apply oversampling
    X_fraud_train_resampled, y_fraud_train_resampled = random_oversample(
        X_fraud_train_scaled, y_fraud_train, target_ratio=0.2
    )
    print(f"Fraud Data - Before: {y_fraud_train.mean()*100:.2f}%")
    print(f"Fraud Data - After: {y_fraud_train_resampled.mean()*100:.2f}%")
    
    X_credit_train_resampled, y_credit_train_resampled = random_oversample(
        X_credit_train_scaled, y_credit_train, target_ratio=0.2
    )
    print(f"Credit Data - Before: {y_credit_train.mean()*100:.2f}%")
    print(f"Credit Data - After: {y_credit_train_resampled.mean()*100:.2f}%")
    
    # Step 6: Train Models on Fraud Data
    print("\n" + "="*60)
    print("STEP 6: TRAINING MODELS ON FRAUD DATA")
    print("="*60)
    
    trainer_fraud = ModelTrainer()
    trainer_fraud.train_all_models(X_fraud_train_resampled, y_fraud_train_resampled)
    
    # Evaluate on test set
    X_fraud_test_scaled = scaler_fraud.transform(X_fraud_test)
    fraud_results = trainer_fraud.evaluate_all_models(
        X_fraud_test_scaled, y_fraud_test, 'Fraud Data'
    )
    print("\nFraud Data Results:")
    print(fraud_results[['Model', 'F1 Score', 'AUC-PR', 'Recall', 'Precision']].to_string(index=False))
    
    # Cross-validation
    fraud_cv = trainer_fraud.cross_validate_all(
        X_fraud_train_resampled, y_fraud_train_resampled, 'Fraud Data', cv=5
    )
    print("\nFraud Data Cross-Validation:")
    print(fraud_cv[['Model', 'F1_mean', 'F1_std', 'AUC-PR_mean', 'AUC-PR_std']].to_string(index=False))
    
    # Step 7: Train Models on Credit Data
    print("\n" + "="*60)
    print("STEP 7: TRAINING MODELS ON CREDIT DATA")
    print("="*60)
    
    trainer_credit = ModelTrainer()
    trainer_credit.train_all_models(X_credit_train_resampled, y_credit_train_resampled)
    
    # Evaluate on test set
    X_credit_test_scaled = scaler_credit.transform(X_credit_test)
    credit_results = trainer_credit.evaluate_all_models(
        X_credit_test_scaled, y_credit_test, 'Credit Data'
    )
    print("\nCredit Data Results:")
    print(credit_results[['Model', 'F1 Score', 'AUC-PR', 'Recall', 'Precision']].to_string(index=False))
    
    # Cross-validation
    credit_cv = trainer_credit.cross_validate_all(
        X_credit_train_resampled, y_credit_train_resampled, 'Credit Data', cv=5
    )
    print("\nCredit Data Cross-Validation:")
    print(credit_cv[['Model', 'F1_mean', 'F1_std', 'AUC-PR_mean', 'AUC-PR_std']].to_string(index=False))
    
    # Step 8: Select Best Models
    print("\n" + "="*60)
    print("STEP 8: SELECTING BEST MODELS")
    print("="*60)
    
    best_fraud = trainer_fraud.select_best_model('Fraud Data', 'F1 Score')
    print(f"Best Model for Fraud Data: {best_fraud['model_name']}")
    print(f"  F1 Score: {best_fraud['metrics']['F1 Score']:.4f}")
    print(f"  AUC-PR: {best_fraud['metrics']['AUC-PR']:.4f}")
    print(f"  Recall: {best_fraud['metrics']['Recall']:.4f}")
    print(f"  Precision: {best_fraud['metrics']['Precision']:.4f}")
    
    best_credit = trainer_credit.select_best_model('Credit Data', 'F1 Score')
    print(f"\nBest Model for Credit Data: {best_credit['model_name']}")
    print(f"  F1 Score: {best_credit['metrics']['F1 Score']:.4f}")
    print(f"  AUC-PR: {best_credit['metrics']['AUC-PR']:.4f}")
    print(f"  Recall: {best_credit['metrics']['Recall']:.4f}")
    print(f"  Precision: {best_credit['metrics']['Precision']:.4f}")
    
    # Step 9: Save Models and Results
    print("\n" + "="*60)
    print("STEP 9: SAVING MODELS AND RESULTS")
    print("="*60)
    
    # Save models
    trainer_fraud.save_models('models/fraud/')
    trainer_credit.save_models('models/credit/')
    
    # Save scalers
    joblib.dump(scaler_fraud, 'models/fraud_scaler.pkl')
    joblib.dump(scaler_credit, 'models/credit_scaler.pkl')
    print("✓ Scalers saved")
    
    # Save results
    fraud_results.to_csv('data/processed/fraud_model_results.csv', index=False)
    credit_results.to_csv('data/processed/credit_model_results.csv', index=False)
    print("✓ Results saved")
    
    # Step 10: Final Summary
    print("\n" + "="*60)
    print("PIPELINE COMPLETE!")
    print("="*60)
    
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    
    print(f"""
    ┌─────────────────────────────────────────────────────────────────────┐
    │                    PIPELINE EXECUTION SUMMARY                      │
    ├─────────────────────────────────────────────────────────────────────┤
    │                                                                     │
    │  FRAUD DATA (E-commerce):                                          │
    │    Best Model: {best_fraud['model_name']}                               │
    │    F1 Score:  {best_fraud['metrics']['F1 Score']:.4f}                    │
    │    AUC-PR:    {best_fraud['metrics']['AUC-PR']:.4f}                    │
    │    Recall:    {best_fraud['metrics']['Recall']:.4f}                    │
    │    Precision: {best_fraud['metrics']['Precision']:.4f}                    │
    │                                                                     │
    │  CREDIT DATA (Bank Transactions):                                  │
    │    Best Model: {best_credit['model_name']}                               │
    │    F1 Score:  {best_credit['metrics']['F1 Score']:.4f}                    │
    │    AUC-PR:    {best_credit['metrics']['AUC-PR']:.4f}                    │
    │    Recall:    {best_credit['metrics']['Recall']:.4f}                    │
    │    Precision: {best_credit['metrics']['Precision']:.4f}                    │
    │                                                                     │
    │  FILES SAVED:                                                      │
    │    • models/fraud/ - Fraud data models                             │
    │    • models/credit/ - Credit data models                           │
    │    • models/*_scaler.pkl - Feature scalers                         │
    │    • data/processed/*_model_results.csv - Results                  │
    │                                                                     │
    └─────────────────────────────────────────────────────────────────────┘
    """)


if __name__ == "__main__":
    run_fraud_pipeline()