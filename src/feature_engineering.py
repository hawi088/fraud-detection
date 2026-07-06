"""
Feature Engineering Module
Contains functions for creating and transforming features
"""

import numpy as np
import pandas as pd
from typing import List, Tuple


def create_time_features(df: pd.DataFrame, time_col: str = 'purchase_time') -> pd.DataFrame:
    """
    Create time-based features from datetime column
    
    Args:
        df: DataFrame with datetime column
        time_col: Name of datetime column
        
    Returns:
        DataFrame with time features added
    """
    df_fe = df.copy()
    df_fe[f'{time_col}_hour'] = df_fe[time_col].dt.hour
    df_fe[f'{time_col}_dayofweek'] = df_fe[time_col].dt.dayofweek
    df_fe[f'{time_col}_month'] = df_fe[time_col].dt.month
    df_fe[f'{time_col}_quarter'] = df_fe[time_col].dt.quarter
    df_fe[f'{time_col}_dayofyear'] = df_fe[time_col].dt.dayofyear
    df_fe[f'{time_col}_is_weekend'] = df_fe[time_col].dt.dayofweek.isin([5, 6]).astype(int)
    df_fe[f'{time_col}_is_night'] = df_fe[time_col].dt.hour.between(22, 5).astype(int)
    
    return df_fe


def create_velocity_features(df: pd.DataFrame, 
                             user_col: str = 'user_id',
                             time_col: str = 'purchase_time') -> pd.DataFrame:
    """
    Create transaction velocity features
    
    Args:
        df: DataFrame with user_id and timestamp
        user_col: Name of user identifier column
        time_col: Name of timestamp column
        
    Returns:
        DataFrame with velocity features
    """
    df_fe = df.copy()
    
    # Sort by user and time
    df_fe = df_fe.sort_values([user_col, time_col])
    
    # Transaction count per user
    df_fe['user_transaction_count'] = df_fe.groupby(user_col).cumcount() + 1
    
    # Time since last transaction (hours)
    df_fe['time_since_last_transaction'] = (
        df_fe.groupby(user_col)[time_col]
        .diff()
        .dt.total_seconds() / 3600
    ).fillna(-1)
    
    # Transactions in last hour
    df_fe['transactions_last_hour'] = (
        df_fe.groupby(user_col)[time_col]
        .transform(
            lambda x: x.diff()
            .dt.total_seconds()
            .fillna(0)
            .apply(lambda y: 1 if y < 3600 and y > 0 else 0)
        )
    )
    
    # Cumulative transactions in last hour
    df_fe['transactions_last_hour_cumulative'] = (
        df_fe.groupby(user_col)['transactions_last_hour']
        .transform(lambda x: x.rolling(10, min_periods=1).sum())
    )
    
    return df_fe


def create_rate_features(df: pd.DataFrame, 
                         signup_col: str = 'signup_time',
                         time_col: str = 'purchase_time') -> pd.DataFrame:
    """
    Create transaction rate features
    
    Args:
        df: DataFrame with signup and purchase times
        signup_col: Name of signup timestamp column
        time_col: Name of purchase timestamp column
        
    Returns:
        DataFrame with rate features
    """
    df_fe = df.copy()
    
    # Time since signup (hours)
    df_fe['time_since_signup_hours'] = (
        df_fe[time_col] - df_fe[signup_col]
    ).dt.total_seconds() / 3600
    
    # Transaction rate per hour
    df_fe['transaction_rate_per_hour'] = (
        df_fe['user_transaction_count'] / (df_fe['time_since_signup_hours'] + 0.1)
    )
    
    # Transaction rate per day
    df_fe['transaction_rate_per_day'] = df_fe['transaction_rate_per_hour'] * 24
    
    return df_fe


def create_user_behavior_features(df: pd.DataFrame,
                                  user_col: str = 'user_id',
                                  device_col: str = 'device_id',
                                  browser_col: str = 'browser') -> pd.DataFrame:
    """
    Create user behavior features
    
    Args:
        df: DataFrame with user metadata
        user_col: Name of user identifier column
        device_col: Name of device identifier column
        browser_col: Name of browser column
        
    Returns:
        DataFrame with user behavior features
    """
    df_fe = df.copy()
    
    # Device count per user
    user_devices = df_fe.groupby(user_col)[device_col].nunique()
    df_fe['user_device_count'] = df_fe[user_col].map(user_devices)
    
    # Browser count per user
    user_browsers = df_fe.groupby(user_col)[browser_col].nunique()
    df_fe['user_browser_count'] = df_fe[user_col].map(user_browsers)
    
    return df_fe


def create_amount_features(df: pd.DataFrame, amount_col: str = 'purchase_value') -> pd.DataFrame:
    """
    Create amount-based features
    
    Args:
        df: DataFrame with amount column
        amount_col: Name of amount column
        
    Returns:
        DataFrame with amount features
    """
    df_fe = df.copy()
    
    # Log transformation
    df_fe[f'{amount_col}_log'] = np.log1p(df_fe[amount_col])
    
    # Square transformation
    df_fe[f'{amount_col}_squared'] = df_fe[amount_col] ** 2
    
    # Binning
    df_fe[f'{amount_col}_bin'] = pd.qcut(
        df_fe[amount_col],
        q=5,
        labels=[1, 2, 3, 4, 5],
        duplicates='drop'
    )
    
    return df_fe


def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create interaction features between key columns
    
    Args:
        df: DataFrame with existing features
        
    Returns:
        DataFrame with interaction features
    """
    df_fe = df.copy()
    
    # Interactions with time since signup
    if 'time_since_signup_hours' in df.columns:
        df_fe['velocity_rate_signup_interaction'] = (
            df_fe.get('avg_transaction_rate_per_hour', 0) * 
            np.exp(-df_fe['time_since_signup_hours'] / 168)
        )  # 168 hours = 1 week
    
    # Age and purchase value interaction
    if 'age' in df.columns and 'purchase_value' in df.columns:
        df_fe['age_value_interaction'] = df_fe['age'] * np.log1p(df_fe['purchase_value'])
    
    return df_fe


def get_feature_importance_names() -> Dict[str, List[str]]:
    """
    Get categorized feature names
    
    Returns:
        Dictionary mapping feature categories to feature names
    """
    return {
        'time_features': [
            'purchase_hour', 'purchase_dayofweek', 'purchase_month',
            'purchase_quarter', 'purchase_dayofyear'
        ],
        'velocity_features': [
            'user_transaction_count', 'time_since_last_transaction',
            'transactions_last_hour', 'transactions_last_hour_cumulative'
        ],
        'rate_features': [
            'time_since_signup_hours', 'transaction_rate_per_hour',
            'transaction_rate_per_day'
        ],
        'user_behavior_features': [
            'user_device_count', 'user_browser_count'
        ],
        'amount_features': [
            'purchase_value_log', 'purchase_value_squared'
        ],
        'flag_features': [
            'is_weekend', 'is_night', 'is_early_morning', 'is_business_hours'
        ]
    }


def select_features_for_modeling(df: pd.DataFrame, 
                                 include_categorical: bool = True) -> List[str]:
    """
    Select features suitable for modeling
    
    Args:
        df: DataFrame with all features
        include_categorical: Whether to include categorical features
        
    Returns:
        List of selected feature names
    """
    # Get all feature categories
    all_features = get_feature_importance_names()
    selected_features = []
    
    for category, features in all_features.items():
        # Check if features exist in DataFrame
        available = [f for f in features if f in df.columns]
        selected_features.extend(available)
    
    # Add base numeric features
    base_features = ['purchase_value', 'age']
    selected_features.extend([f for f in base_features if f in df.columns])
    
    # Add categorical features
    if include_categorical:
        categorical = ['source', 'browser', 'sex']
        if 'country' in df.columns:
            categorical.append('country')
        selected_features.extend([f for f in categorical if f in df.columns])
    
    # Remove duplicates while preserving order
    seen = set()
    unique_features = []
    for f in selected_features:
        if f not in seen:
            seen.add(f)
            unique_features.append(f)
    
    return unique_features