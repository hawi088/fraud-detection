"""
Preprocessor Module
Handles data cleaning, preprocessing, and feature engineering
"""

import pandas as pd
import numpy as np
import ipaddress
from datetime import datetime
from typing import Optional, Tuple, List
from sklearn.preprocessing import StandardScaler, LabelEncoder


class FraudDataPreprocessor:
    """Preprocessor for e-commerce fraud data"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.categorical_cols = ['source', 'browser', 'sex']
        self.fitted = False
        
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the fraud dataset
        
        Args:
            df: Raw fraud data DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        df_clean = df.copy()
        
        # Convert timestamps
        df_clean['signup_time'] = pd.to_datetime(df_clean['signup_time'])
        df_clean['purchase_time'] = pd.to_datetime(df_clean['purchase_time'])
        
        # Convert categorical columns
        for col in self.categorical_cols:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].astype('category')
        
        # Ensure numeric columns are correct type
        df_clean['purchase_value'] = df_clean['purchase_value'].astype(float)
        df_clean['age'] = df_clean['age'].astype(int)
        df_clean['class'] = df_clean['class'].astype(int)
        
        # Remove duplicates
        df_clean = df_clean.drop_duplicates()
        
        return df_clean
    
    def ip_to_int(self, ip_address) -> Optional[int]:
        """
        Convert IP address string to integer
        
        Args:
            ip_address: IP address as string or float
            
        Returns:
            Integer representation of IP or None
        """
        try:
            if isinstance(ip_address, float):
                ip_address = str(int(ip_address))
            elif isinstance(ip_address, int):
                ip_address = str(ip_address)
            return int(ipaddress.IPv4Address(ip_address))
        except (ValueError, ipaddress.AddressValueError, TypeError):
            return None
    
    def add_ip_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add IP-related features to DataFrame
        
        Args:
            df: DataFrame with ip_address column
            
        Returns:
            DataFrame with IP features added
        """
        df_fe = df.copy()
        df_fe['ip_int'] = df_fe['ip_address'].apply(self.ip_to_int)
        return df_fe
    
    def add_country_feature(self, df: pd.DataFrame, ip_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add country feature using IP range lookup
        
        Args:
            df: DataFrame with ip_int column
            ip_df: IP to country mapping DataFrame
            
        Returns:
            DataFrame with country column added
        """
        df_fe = df.copy()
        
        # Preprocess IP data
        ip_processed = ip_df.copy()
        ip_processed['lower_bound_ip_int'] = ip_processed['lower_bound_ip_address'].apply(
            lambda x: self.ip_to_int(x) if x is not None else None
        )
        ip_processed['upper_bound_ip_int'] = ip_processed['upper_bound_ip_address'].apply(
            lambda x: self.ip_to_int(x) if x is not None else None
        )
        ip_processed = ip_processed.dropna(subset=['lower_bound_ip_int', 'upper_bound_ip_int'])
        ip_processed = ip_processed.sort_values('lower_bound_ip_int')
        
        def get_country(ip_int):
            if pd.isna(ip_int):
                return 'Unknown'
            try:
                # Use merge_asof for efficient range lookup
                temp_df = pd.DataFrame({'ip_int': [ip_int]})
                temp_df = temp_df.sort_values('ip_int')
                result = pd.merge_asof(
                    temp_df,
                    ip_processed,
                    left_on='ip_int',
                    right_on='lower_bound_ip_int',
                    direction='backward'
                )
                if not result.empty and len(result) > 0:
                    row = result.iloc[0]
                    if row['ip_int'] <= row['upper_bound_ip_int']:
                        return row['country']
                return 'Unknown'
            except:
                return 'Unknown'
        
        df_fe['country'] = df_fe['ip_int'].apply(get_country)
        return df_fe
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features for fraud detection
        
        Args:
            df: DataFrame with signup_time and purchase_time
            
        Returns:
            DataFrame with engineered features
        """
        df_fe = df.copy()
        
        # Sort by user and time for velocity features
        df_fe = df_fe.sort_values(['user_id', 'purchase_time'])
        
        # Time-based features
        df_fe['purchase_hour'] = df_fe['purchase_time'].dt.hour
        df_fe['purchase_dayofweek'] = df_fe['purchase_time'].dt.dayofweek
        df_fe['purchase_month'] = df_fe['purchase_time'].dt.month
        df_fe['purchase_quarter'] = df_fe['purchase_time'].dt.quarter
        
        # Time since signup
        df_fe['time_since_signup_hours'] = (
            df_fe['purchase_time'] - df_fe['signup_time']
        ).dt.total_seconds() / 3600
        df_fe['time_since_signup_days'] = df_fe['time_since_signup_hours'] / 24
        
        # Transaction velocity
        df_fe['user_transaction_count'] = df_fe.groupby('user_id').cumcount() + 1
        df_fe['time_since_last_transaction_hours'] = (
            df_fe.groupby('user_id')['purchase_time']
            .diff()
            .dt.total_seconds() / 3600
        ).fillna(-1)
        
        # Transaction rate
        df_fe['avg_transaction_rate_per_hour'] = (
            df_fe['user_transaction_count'] / (df_fe['time_since_signup_hours'] + 0.1)
        )
        df_fe['avg_transaction_rate_per_day'] = df_fe['avg_transaction_rate_per_hour'] * 24
        
        # User behavior features
        user_devices = df_fe.groupby('user_id')['device_id'].nunique()
        df_fe['user_device_count'] = df_fe['user_id'].map(user_devices)
        user_browsers = df_fe.groupby('user_id')['browser'].nunique()
        df_fe['user_browser_count'] = df_fe['user_id'].map(user_browsers)
        
        # Amount features
        df_fe['purchase_value_scaled'] = np.log1p(df_fe['purchase_value'])
        df_fe['purchase_value_squared'] = df_fe['purchase_value'] ** 2
        
        # Flag features
        df_fe['is_weekend'] = df_fe['purchase_dayofweek'].isin([5, 6]).astype(int)
        df_fe['is_night'] = df_fe['purchase_hour'].between(22, 5).astype(int)
        df_fe['is_early_morning'] = df_fe['purchase_hour'].between(0, 5).astype(int)
        df_fe['is_business_hours'] = df_fe['purchase_hour'].between(9, 17).astype(int)
        
        # Age grouping
        df_fe['age_group'] = pd.cut(
            df_fe['age'],
            bins=[0, 18, 25, 35, 50, 65, 100],
            labels=['0-18', '18-25', '25-35', '35-50', '50-65', '65+']
        )
        
        return df_fe
    
    def prepare_for_modeling(self, df: pd.DataFrame, y_col: str = 'class') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for modeling
        
        Args:
            df: Processed DataFrame
            y_col: Name of target column
            
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        # Select features for modeling
        feature_cols = [
            'purchase_value', 'age', 'time_since_signup_hours',
            'user_transaction_count', 'time_since_last_transaction_hours',
            'avg_transaction_rate_per_hour', 'user_device_count',
            'user_browser_count', 'purchase_value_scaled',
            'purchase_hour', 'purchase_dayofweek', 'purchase_month',
            'is_weekend', 'is_night', 'source', 'browser', 'sex'
        ]
        
        # Add country if available
        if 'country' in df.columns:
            feature_cols.append('country')
        
        # Get available features
        available_cols = [col for col in feature_cols if col in df.columns]
        
        X = df[available_cols].copy()
        y = df[y_col].copy()
        
        # One-hot encode categorical features
        categorical_cols = ['source', 'browser', 'sex']
        if 'country' in available_cols:
            categorical_cols.append('country')
        
        # Only encode columns that exist
        encode_cols = [col for col in categorical_cols if col in X.columns]
        if encode_cols:
            X = pd.get_dummies(X, columns=encode_cols, drop_first=True)
        
        # Handle missing values
        X = X.fillna(-1)
        
        # Ensure all columns are numeric
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X = X[numeric_cols]
        
        return X, y


class CreditDataPreprocessor:
    """Preprocessor for credit card fraud data"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.fitted = False
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the credit card dataset
        
        Args:
            df: Raw credit card data DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        df_clean = df.copy()
        
        # No missing values in this dataset
        # Scale amount separately since it's not PCA-transformed
        df_clean['Amount_scaled'] = self.scaler.fit_transform(df_clean[['Amount']])
        
        # Time-based features
        df_clean['Time_hours'] = df_clean['Time'] / 3600
        df_clean['Time_bin'] = pd.cut(df_clean['Time_hours'], bins=24, labels=range(24))
        
        return df_clean
    
    def prepare_for_modeling(self, df: pd.DataFrame, y_col: str = 'Class') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare credit card data for modeling
        
        Args:
            df: Processed DataFrame
            y_col: Name of target column
            
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        # Use V1-V28 and scaled amount
        feature_cols = [f'V{i}' for i in range(1, 29)]
        feature_cols.extend(['Amount_scaled', 'Time_hours'])
        
        X = df[feature_cols].copy()
        y = df[y_col].copy()
        
        # Handle missing values
        X = X.fillna(0)
        
        return X, y