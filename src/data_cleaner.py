"""
Data cleaning module with error handling for data parsing and transformation.
"""

import logging
from typing import Optional, Dict, List
import pandas as pd
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)


def convert_to_numeric(
    df: pd.DataFrame,
    columns: List[str],
    errors: str = "coerce",
    fill_na: Optional[float] = None
) -> pd.DataFrame:
    """
    Convert specified columns to numeric with error handling.
    
    Args:
        df: DataFrame to process
        columns: List of column names to convert
        errors: How to handle errors ('coerce', 'raise', 'ignore')
        fill_na: Optional value to fill NaN after conversion
        
    Returns:
        pd.DataFrame: DataFrame with converted columns
        
    Raises:
        ValueError: If columns don't exist or conversion fails
    """
    df = df.copy()
    
    for col in columns:
        if col not in df.columns:
            logger.warning(f"Column '{col}' not found in DataFrame, skipping")
            continue
        
        try:
            df[col] = pd.to_numeric(df[col], errors=errors)
            
            if fill_na is not None:
                df[col] = df[col].fillna(fill_na)
            
            logger.info(f"Converted column '{col}' to numeric")
            
        except Exception as e:
            logger.error(f"Error converting column '{col}' to numeric: {e}")
            if errors == "raise":
                raise ValueError(f"Failed to convert column '{col}' to numeric: {e}")
    
    return df


def calculate_loss_ratio(
    df: pd.DataFrame,
    claims_col: str = "TotalClaims",
    premium_col: str = "TotalPremium",
    output_col: str = "LossRatio"
) -> pd.DataFrame:
    """
    Calculate loss ratio (claims / premium) with error handling.
    
    Args:
        df: DataFrame containing claims and premium data
        claims_col: Name of claims column
        premium_col: Name of premium column
        output_col: Name of output column for loss ratio
        
    Returns:
        pd.DataFrame: DataFrame with loss ratio column added
        
    Raises:
        ValueError: If required columns are missing
    """
    df = df.copy()
    
    # Validate required columns exist
    required_cols = {claims_col, premium_col}
    missing_cols = required_cols - set(df.columns)
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    try:
        # Ensure columns are numeric
        df = convert_to_numeric(df, [claims_col, premium_col])
        
        # Calculate loss ratio, handling division by zero
        df[output_col] = df[claims_col] / df[premium_col].replace(0, np.nan)
        
        logger.info(f"Calculated loss ratio in column '{output_col}'")
        
    except Exception as e:
        logger.error(f"Error calculating loss ratio: {e}")
        df[output_col] = np.nan
        raise
    
    return df


def create_month_column(
    df: pd.DataFrame,
    date_col: str = "TransactionMonth",
    output_col: str = "Month"
) -> pd.DataFrame:
    """
    Create a month column from a date column.
    
    Args:
        df: DataFrame with date column
        date_col: Name of date column
        output_col: Name of output month column
        
    Returns:
        pd.DataFrame: DataFrame with month column added
        
    Raises:
        ValueError: If date column doesn't exist or isn't datetime
    """
    df = df.copy()
    
    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not found")
    
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        try:
            df[date_col] = pd.to_datetime(df[date_col])
        except Exception as e:
            raise ValueError(f"Cannot convert '{date_col}' to datetime: {e}")
    
    try:
        df[output_col] = df[date_col].dt.to_period("M").dt.to_timestamp()
        logger.info(f"Created month column '{output_col}' from '{date_col}'")
    except Exception as e:
        logger.error(f"Error creating month column: {e}")
        raise
    
    return df


def generate_quality_report(df: pd.DataFrame) -> Dict[str, int]:
    """
    Generate a data quality report with error handling.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        dict: Dictionary with quality metrics
    """
    report = {}
    
    try:
        report["rows"] = len(df)
        report["columns"] = df.shape[1]
        report["duplicate_rows"] = df.duplicated().sum()
        
        # Check for negative or zero values in key columns
        numeric_cols = ["TotalPremium", "TotalClaims", "SumInsured"]
        
        for col in numeric_cols:
            if col in df.columns:
                try:
                    numeric_series = pd.to_numeric(df[col], errors="coerce")
                    report[f"{col}_negative"] = (numeric_series < 0).sum()
                    report[f"{col}_zero"] = (numeric_series == 0).sum()
                except Exception as e:
                    logger.warning(f"Error checking {col}: {e}")
                    report[f"{col}_negative"] = -1  # Error indicator
                    report[f"{col}_zero"] = -1
        
        logger.info("Generated quality report successfully")
        
    except Exception as e:
        logger.error(f"Error generating quality report: {e}")
        raise
    
    return report


def clean_insurance_data(
    df: pd.DataFrame,
    numeric_columns: Optional[List[str]] = None,
    calculate_loss_ratio_flag: bool = True,
    create_month_flag: bool = True
) -> pd.DataFrame:
    """
    Comprehensive data cleaning pipeline for insurance data.
    
    Args:
        df: Raw DataFrame to clean
        numeric_columns: List of columns to convert to numeric
        calculate_loss_ratio_flag: Whether to calculate loss ratio
        create_month_flag: Whether to create month column
        
    Returns:
        pd.DataFrame: Cleaned DataFrame
        
    Raises:
        ValueError: If cleaning fails
    """
    if df.empty:
        raise ValueError("Cannot clean empty DataFrame")
    
    df_cleaned = df.copy()
    
    try:
        # Convert numeric columns
        if numeric_columns is None:
            numeric_columns = ["TotalPremium", "TotalClaims", "SumInsured", "CustomValueEstimate"]
        
        df_cleaned = convert_to_numeric(df_cleaned, numeric_columns)
        
        # Calculate loss ratio
        if calculate_loss_ratio_flag:
            df_cleaned = calculate_loss_ratio(df_cleaned)
        
        # Create month column
        if create_month_flag and "TransactionMonth" in df_cleaned.columns:
            df_cleaned = create_month_column(df_cleaned)
        
        logger.info("Data cleaning completed successfully")
        
    except Exception as e:
        logger.error(f"Error in data cleaning pipeline: {e}")
        raise
    
    return df_cleaned

