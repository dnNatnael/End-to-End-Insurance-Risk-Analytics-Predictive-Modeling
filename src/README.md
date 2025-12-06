# Source Code Modules

This directory contains refactored, modular code for the insurance risk analytics project. The code has been organized into functional modules with comprehensive error handling for improved maintainability, testability, and resilience.

## Module Structure

### `data_loader.py`
**Purpose**: Data loading with comprehensive error handling for file access and parsing.

**Key Functions**:
- `get_project_root()`: Resolves project root directory robustly
- `get_data_path()`: Gets path to data file with validation
- `peek_file()`: Safely peeks at file structure
- `load_insurance_data()`: Loads dataset with error handling for:
  - FileNotFoundError
  - PermissionError
  - EmptyDataError
  - ParserError
- `validate_dataframe()`: Validates loaded DataFrame structure

**Error Handling**:
- File existence checks
- Permission validation
- Encoding error handling
- Parser error catching with informative messages
- Logging for debugging

### `data_cleaner.py`
**Purpose**: Data cleaning and transformation with error handling for data parsing.

**Key Functions**:
- `convert_to_numeric()`: Converts columns to numeric with error handling
- `calculate_loss_ratio()`: Calculates loss ratio with division-by-zero protection
- `create_month_column()`: Creates month column from date with validation
- `generate_quality_report()`: Generates comprehensive data quality report
- `clean_insurance_data()`: Complete cleaning pipeline

**Error Handling**:
- Type conversion errors
- Missing column validation
- Division by zero protection
- Data type validation
- Comprehensive logging

### `visualizations.py`
**Purpose**: Plotting functions for data visualization.

**Key Functions**:
- `setup_plot_style()`: Configures matplotlib/seaborn styles
- `plot_univariate_distributions()`: Histograms and boxplots
- `plot_categorical_frequencies()`: Categorical frequency plots
- `plot_bivariate_scatter()`: Scatter plots with optional log scaling
- `plot_loss_ratio_by_dimension()`: Loss ratio analysis by category
- `plot_temporal_trends()`: Time series visualizations
- `plot_correlation_matrix()`: Correlation heatmaps

**Error Handling**:
- Empty DataFrame checks
- Missing column validation
- Plotting error catching
- Graceful degradation with warnings

### `utils.py`
**Purpose**: Utility functions for data analysis.

**Key Functions**:
- `get_numeric_columns()`: Filters existing numeric columns
- `get_categorical_columns()`: Filters existing categorical columns
- `calculate_variability_metrics()`: Computes variability statistics
- `get_data_summary()`: Comprehensive data summary
- `calculate_portfolio_loss_ratio()`: Portfolio-level metrics

**Error Handling**:
- Column existence validation
- Empty result handling
- Calculation error catching

## Benefits of Refactoring

### 1. **Maintainability**
- Code is organized into logical modules
- Functions have single responsibilities
- Easy to locate and modify specific functionality

### 2. **Testability**
- Functions are isolated and can be unit tested
- Error paths are explicit and testable
- Mock data can be easily injected

### 3. **Resilience**
- Comprehensive error handling at every step
- Graceful degradation when errors occur
- Informative error messages for debugging
- Logging for troubleshooting

### 4. **Reusability**
- Functions can be imported and reused across notebooks/scripts
- Consistent interfaces across the codebase
- Easy to extend with new functionality

## Usage Example

```python
from src.data_loader import load_insurance_data, get_data_path
from src.data_cleaner import clean_insurance_data
from src.visualizations import plot_univariate_distributions
from src.utils import get_numeric_columns

# Load data with error handling
try:
    data_path = get_data_path()
    df = load_insurance_data(filepath=data_path)
except FileNotFoundError as e:
    print(f"Data file not found: {e}")
    raise

# Clean data
df = clean_insurance_data(df)

# Visualize
num_cols = get_numeric_columns(df, ["TotalPremium", "TotalClaims"])
plot_univariate_distributions(df, num_cols)
```

## Error Handling Strategy

All modules follow a consistent error handling strategy:

1. **Validation First**: Check inputs before processing
2. **Specific Exceptions**: Raise appropriate exception types
3. **Logging**: Log errors with context for debugging
4. **User-Friendly Messages**: Provide clear error messages
5. **Graceful Degradation**: Continue when possible, fail fast when necessary

## Logging

All modules use Python's `logging` module for consistent logging:
- INFO: Normal operations
- WARNING: Recoverable issues
- ERROR: Errors that prevent operation

Configure logging in your notebook/script:
```python
import logging
logging.basicConfig(level=logging.INFO)
```


