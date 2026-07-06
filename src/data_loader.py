"""
Data Loader Module
Handles loading and initial data validation
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any


class DataLoader:
    """Class for loading and validating fraud detection datasets"""
    
    def __init__(self, data_path: str = "data/raw/"):
        """
        Initialize DataLoader with path to raw data
        
        Args:
            data_path: Path to raw data directory
        """
        self.data_path = Path(data_path)
        self.data_path.mkdir(parents=True, exist_ok=True)
        
    def load_fraud_data(self) -> pd.DataFrame:
        """
        Load e-commerce fraud dataset
        
        Returns:
            DataFrame containing fraud data
        """
        file_path = self.data_path / "Fraud_Data.csv"
        if not file_path.exists():
            raise FileNotFoundError(f"Fraud_Data.csv not found in {self.data_path}")
        
        df = pd.read_csv(file_path)
        print(f"✅ Loaded Fraud_Data.csv: {df.shape[0]:,} rows, {df.shape[1]} columns")
        return df
    
    def load_ip_data(self) -> pd.DataFrame:
        """
        Load IP to country mapping dataset
        
        Returns:
            DataFrame containing IP to country mapping
        """
        file_path = self.data_path / "IpAddress_to_Country.csv"
        if not file_path.exists():
            raise FileNotFoundError(f"IpAddress_to_Country.csv not found in {self.data_path}")
        
        df = pd.read_csv(file_path)
        print(f"✅ Loaded IpAddress_to_Country.csv: {df.shape[0]:,} rows, {df.shape[1]} columns")
        return df
    
    def load_credit_data(self) -> pd.DataFrame:
        """
        Load credit card fraud dataset
        
        Returns:
            DataFrame containing credit card data
        """
        file_path = self.data_path / "creditcard.csv"
        if not file_path.exists():
            raise FileNotFoundError(f"creditcard.csv not found in {self.data_path}")
        
        df = pd.read_csv(file_path)
        print(f"✅ Loaded creditcard.csv: {df.shape[0]:,} rows, {df.shape[1]} columns")
        return df
    
    def validate_data(self, df: pd.DataFrame, expected_columns: list) -> bool:
        """
        Validate that DataFrame has expected columns
        
        Args:
            df: DataFrame to validate
            expected_columns: List of expected column names
            
        Returns:
            True if valid, raises ValueError otherwise
        """
        missing_cols = set(expected_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")
        return True
    
    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load all datasets
        
        Returns:
            Dictionary containing all datasets
        """
        return {
            'fraud': self.load_fraud_data(),
            'ip': self.load_ip_data(),
            'credit': self.load_credit_data()
        }


def get_data_info(df: pd.DataFrame, name: str = "Dataset") -> Dict[str, Any]:
    """
    Get comprehensive information about a dataset
    
    Args:
        df: DataFrame to analyze
        name: Name of the dataset
        
    Returns:
        Dictionary with dataset information
    """
    info = {
        'name': name,
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.to_dict(),
        'missing': df.isnull().sum().to_dict(),
        'memory_usage': df.memory_usage(deep=True).sum() / 1024**2,  # MB
        'duplicates': df.duplicated().sum(),
        'numeric_cols': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_cols': df.select_dtypes(include=['object']).columns.tolist()
    }
    return info


if __name__ == "__main__":
    # Test data loading
    loader = DataLoader()
    data = loader.load_all_data()
    
    for name, df in data.items():
        info = get_data_info(df, name)
        print(f"\n{name.upper()} DATA INFO:")
        print(f"  Shape: {info['shape']}")
        print(f"  Missing: {sum(info['missing'].values())}")
        print(f"  Memory: {info['memory_usage']:.2f} MB")