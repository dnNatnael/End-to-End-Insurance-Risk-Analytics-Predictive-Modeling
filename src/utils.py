"""
Utility functions for insurance data analysis.
"""

import logging
from typing import List, Dict, Any
import pandas as pd
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)


def get_numeric_columns(df: pd.DataFrame, candidate_cols: List[str]) -> List[str]:
    """
    Get list of numeric columns from candidate list that exist in DataFrame.
    
    Args:
        df: DataFrame to check
        candidate_cols: List of candidate column names
        
    Returns:
        list: List of existing numeric columns
    """
    existing_cols = [c for c in candidate_cols if c in df.columns]
    return existing_cols


def get_categorical_columns(df: pd.DataFrame, candidate_cols: List[str]) -> List[str]:
    """
    Get list of categorical columns from candidate list that exist in DataFrame.
    
    Args:
        df: DataFrame to check
        candidate_cols: List of candidate column names
        
    Returns:
        list: List of existing categorical columns
    """
    existing_cols = [c for c in candidate_cols if c in df.columns]
    return existing_cols


def calculate_variability_metrics(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Calculate variability metrics (mean, std, var, coefficient of variation) for numeric columns.
    
    Args:
        df: DataFrame with numeric data
        columns: List of column names to analyze
        
    Returns:
        pd.DataFrame: DataFrame with variability metrics
        
    Raises:
        ValueError: If no valid columns found
    """
    existing_cols = get_numeric_columns(df, columns)
    
    if not existing_cols:
        raise ValueError(f"None of the specified columns exist: {columns}")
    
    try:
        variability = df[existing_cols].agg(["mean", "std", "var"]).T
        variability["coef_var"] = variability["std"] / variability["mean"].replace(0, np.nan)
        
        return variability
        
    except Exception as e:
        logger.error(f"Error calculating variability metrics: {e}")
        raise


def get_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get comprehensive data summary including dtypes and column types.
    
    Args:
        df: DataFrame to summarize
        
    Returns:
        dict: Dictionary with summary information
    """
    summary = {}
    
    try:
        summary["shape"] = df.shape
        summary["dtypes"] = df.dtypes.to_dict()
        
        # Identify column types
        summary["categorical_columns"] = [
            c for c in df.columns if df[c].dtype == "object"
        ]
        
        summary["date_columns"] = [
            c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])
        ]
        
        summary["numeric_columns"] = [
            c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
        ]
        
        summary["missing_values"] = df.isna().sum().to_dict()
        
    except Exception as e:
        logger.error(f"Error generating data summary: {e}")
        raise
    
    return summary


def calculate_portfolio_loss_ratio(
    df: pd.DataFrame,
    claims_col: str = "TotalClaims",
    premium_col: str = "TotalPremium"
) -> float:
    """
    Calculate portfolio-level loss ratio.
    
    Args:
        df: DataFrame with claims and premium data
        claims_col: Name of claims column
        premium_col: Name of premium column
        
    Returns:
        float: Portfolio loss ratio
        
    Raises:
        ValueError: If required columns don't exist or calculation fails
    """
    required_cols = {claims_col, premium_col}
    missing_cols = required_cols - set(df.columns)
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    try:
        total_claims = df[claims_col].sum()
        total_premium = df[premium_col].sum()
        
        if total_premium == 0:
            raise ValueError("Total premium is zero, cannot calculate loss ratio")
        
        loss_ratio = total_claims / total_premium
        logger.info(f"Portfolio loss ratio: {loss_ratio:.4f}")
        
        return loss_ratio
        
    except Exception as e:
        logger.error(f"Error calculating portfolio loss ratio: {e}")
        raise

