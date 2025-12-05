"""
Data loading module with error handling for file access and parsing.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)


def get_project_root() -> Path:
    """
    Resolve project root directory robustly.
    
    Returns:
        Path: Project root directory path
        
    Raises:
        FileNotFoundError: If project root cannot be determined
    """
    project_root = Path.cwd().resolve()
    
    # Check if we're in the project root
    if (project_root / "data").exists() and (project_root / "src").exists():
        return project_root
    
    # If we're inside notebooks/ or scripts/, step one level up
    if (project_root.parent / "data").exists():
        return project_root.parent
    
    raise FileNotFoundError(
        "Cannot determine project root. Please run from project root or notebooks/ directory."
    )


def get_data_path(filename: str = "MachineLearningRating_v3.txt") -> Path:
    """
    Get the path to the data file.
    
    Args:
        filename: Name of the data file
        
    Returns:
        Path: Full path to the data file
        
    Raises:
        FileNotFoundError: If data directory or file doesn't exist
    """
    project_root = get_project_root()
    data_path = project_root / "data" / filename
    
    if not data_path.exists():
        raise FileNotFoundError(
            f"Data file not found: {data_path}. "
            f"Please ensure the file exists in the data/ directory."
        )
    
    return data_path


def peek_file(filepath: Path, n_lines: int = 10, encoding: str = "utf-8") -> list:
    """
    Peek at the first n lines of a file to understand its structure.
    
    Args:
        filepath: Path to the file
        n_lines: Number of lines to read
        encoding: File encoding
        
    Returns:
        list: List of first n lines
        
    Raises:
        FileNotFoundError: If file doesn't exist
        PermissionError: If file cannot be read
        UnicodeDecodeError: If encoding issues occur
    """
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    if not filepath.is_file():
        raise ValueError(f"Path is not a file: {filepath}")
    
    try:
        with open(filepath, "r", encoding=encoding, errors="ignore") as f:
            lines = [next(f).rstrip("\n") for _ in range(n_lines)]
        return lines
    except PermissionError as e:
        logger.error(f"Permission denied reading file: {filepath}")
        raise
    except UnicodeDecodeError as e:
        logger.warning(f"Encoding issue with {encoding}, trying with errors='ignore'")
        with open(filepath, "r", encoding=encoding, errors="ignore") as f:
            lines = [next(f).rstrip("\n") for _ in range(n_lines)]
        return lines


def load_insurance_data(
    filepath: Optional[Path] = None,
    filename: str = "MachineLearningRating_v3.txt",
    sep: str = "|",
    dtype: Optional[Dict[str, Any]] = None,
    parse_dates: Optional[list] = None,
    low_memory: bool = False,
    **kwargs
) -> pd.DataFrame:
    """
    Load insurance dataset with comprehensive error handling.
    
    Args:
        filepath: Explicit path to data file. If None, will use get_data_path()
        filename: Name of the data file (used if filepath is None)
        sep: Delimiter for CSV file
        dtype: Dictionary of column dtypes
        parse_dates: List of columns to parse as dates
        low_memory: Whether to use low_memory mode for pandas
        **kwargs: Additional arguments passed to pd.read_csv
        
    Returns:
        pd.DataFrame: Loaded dataset
        
    Raises:
        FileNotFoundError: If data file doesn't exist
        PermissionError: If file cannot be read
        pd.errors.EmptyDataError: If file is empty
        pd.errors.ParserError: If file cannot be parsed
        ValueError: If invalid arguments provided
    """
    # Determine file path
    if filepath is None:
        filepath = get_data_path(filename)
    
    # Validate file exists
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    # Set default dtype if not provided
    if dtype is None:
        dtype = {
            "UnderwrittenCoverID": "int64",
            "PolicyID": "int64",
            "TransactionMonth": "string",
        }
    
    # Set default parse_dates if not provided
    if parse_dates is None:
        parse_dates = ["TransactionMonth"]
    
    logger.info(f"Loading data from: {filepath}")
    
    try:
        # Attempt to load the data
        df = pd.read_csv(
            filepath,
            sep=sep,
            dtype=dtype,
            parse_dates=parse_dates,
            low_memory=low_memory,
            **kwargs
        )
        
        logger.info(f"Successfully loaded {len(df)} rows and {len(df.columns)} columns")
        
        return df
        
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        raise
    except PermissionError:
        logger.error(f"Permission denied reading file: {filepath}")
        raise
    except pd.errors.EmptyDataError:
        logger.error(f"File is empty: {filepath}")
        raise
    except pd.errors.ParserError as e:
        logger.error(f"Error parsing file {filepath}: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading data: {e}")
        raise


def validate_dataframe(df: pd.DataFrame, required_columns: Optional[list] = None) -> bool:
    """
    Validate that a DataFrame meets basic requirements.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        
    Returns:
        bool: True if validation passes
        
    Raises:
        ValueError: If validation fails
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")
    
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    if required_columns:
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    return True

