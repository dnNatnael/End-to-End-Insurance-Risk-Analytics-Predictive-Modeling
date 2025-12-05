"""
Visualization module for insurance data analysis plots.
"""

import logging
from typing import Optional, List
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)


def setup_plot_style(style: str = "seaborn-v0_8-whitegrid", palette: str = "tab10"):
    """
    Set up matplotlib and seaborn plotting style.
    
    Args:
        style: Matplotlib style name
        palette: Seaborn color palette name
    """
    try:
        plt.style.use(style)
        sns.set_palette(palette)
        logger.info(f"Plot style set to '{style}' with palette '{palette}'")
    except Exception as e:
        logger.warning(f"Error setting plot style: {e}, using defaults")
        plt.style.use("default")


def plot_univariate_distributions(
    df: pd.DataFrame,
    columns: List[str],
    figsize: tuple = (12, 4),
    save_path: Optional[str] = None
) -> None:
    """
    Plot univariate distributions (histogram and boxplot) for numeric columns.
    
    Args:
        df: DataFrame with data to plot
        columns: List of column names to plot
        figsize: Figure size tuple
        save_path: Optional path to save figure
        
    Raises:
        ValueError: If columns don't exist or DataFrame is empty
    """
    if df.empty:
        raise ValueError("Cannot plot empty DataFrame")
    
    # Filter to existing columns
    existing_cols = [c for c in columns if c in df.columns]
    if not existing_cols:
        raise ValueError(f"None of the specified columns exist: {columns}")
    
    try:
        n_cols = len(existing_cols)
        fig, axes = plt.subplots(n_cols, 2, figsize=(figsize[0], figsize[1] * n_cols))
        
        if n_cols == 1:
            axes = np.array([axes])
        
        for i, col in enumerate(existing_cols):
            # Histogram
            try:
                sns.histplot(data=df, x=col, kde=True, ax=axes[i, 0])
                axes[i, 0].set_title(f"Distribution of {col}")
                axes[i, 0].set_xlabel(col)
            except Exception as e:
                logger.warning(f"Error plotting histogram for {col}: {e}")
            
            # Boxplot
            try:
                sns.boxplot(data=df, x=col, ax=axes[i, 1])
                axes[i, 1].set_title(f"Boxplot of {col}")
            except Exception as e:
                logger.warning(f"Error plotting boxplot for {col}: {e}")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Figure saved to {save_path}")
        
        plt.show()
        
    except Exception as e:
        logger.error(f"Error in plot_univariate_distributions: {e}")
        raise


def plot_categorical_frequencies(
    df: pd.DataFrame,
    columns: List[str],
    top_n: int = 15,
    figsize: tuple = (10, 4),
    save_path: Optional[str] = None
) -> None:
    """
    Plot frequency distributions for categorical columns.
    
    Args:
        df: DataFrame with data to plot
        columns: List of categorical column names
        top_n: Number of top categories to show
        figsize: Figure size tuple
        save_path: Optional path to save figure
        
    Raises:
        ValueError: If columns don't exist
    """
    existing_cols = [c for c in columns if c in df.columns]
    if not existing_cols:
        logger.warning(f"None of the specified columns exist: {columns}")
        return
    
    for col in existing_cols:
        try:
            plt.figure(figsize=figsize)
            counts = df[col].value_counts(dropna=False).head(top_n)
            sns.barplot(x=counts.index.astype(str), y=counts.values)
            plt.xticks(rotation=45, ha="right")
            plt.title(f"Top {top_n} categories for {col}")
            plt.ylabel("Count")
            plt.tight_layout()
            
            if save_path:
                plt.savefig(f"{save_path}_{col}.png", dpi=300, bbox_inches="tight")
            
            plt.show()
            
        except Exception as e:
            logger.warning(f"Error plotting categorical frequencies for {col}: {e}")


def plot_bivariate_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    sample_size: Optional[int] = 50000,
    log_x: bool = False,
    log_y: bool = False,
    figsize: tuple = (6, 6),
    alpha: float = 0.3,
    save_path: Optional[str] = None
) -> None:
    """
    Plot bivariate scatter plot with optional log scaling.
    
    Args:
        df: DataFrame with data to plot
        x_col: Name of x-axis column
        y_col: Name of y-axis column
        sample_size: Maximum number of points to sample
        log_x: Whether to use log scale on x-axis
        log_y: Whether to use log scale on y-axis
        figsize: Figure size tuple
        alpha: Transparency level
        save_path: Optional path to save figure
        
    Raises:
        ValueError: If columns don't exist
    """
    required_cols = {x_col, y_col}
    missing_cols = required_cols - set(df.columns)
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    try:
        # Sample data if needed
        plot_df = df.sample(min(sample_size, len(df)), random_state=42) if len(df) > sample_size else df
        
        plt.figure(figsize=figsize)
        sns.scatterplot(data=plot_df, x=x_col, y=y_col, alpha=alpha)
        
        if log_x:
            plt.xscale("log")
        if log_y:
            plt.yscale("log")
        
        plt.title(f"{y_col} vs {x_col}")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Figure saved to {save_path}")
        
        plt.show()
        
    except Exception as e:
        logger.error(f"Error in plot_bivariate_scatter: {e}")
        raise


def plot_loss_ratio_by_dimension(
    df: pd.DataFrame,
    dimension: str,
    min_premium: float = 0.0,
    top_n: int = 15,
    figsize: tuple = (10, 4),
    save_path: Optional[str] = None
) -> None:
    """
    Plot loss ratio by a categorical dimension (e.g., Province, Gender).
    
    Args:
        df: DataFrame with loss ratio and dimension data
        dimension: Name of categorical dimension column
        min_premium: Minimum total premium to include in plot
        top_n: Number of top categories to show
        figsize: Figure size tuple
        save_path: Optional path to save figure
        
    Raises:
        ValueError: If required columns don't exist
    """
    required_cols = {dimension, "TotalPremium", "TotalClaims"}
    missing_cols = required_cols - set(df.columns)
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    try:
        # Calculate loss ratio by dimension
        grouped = (
            df.groupby(dimension)
            .agg(
                TotalPremium=("TotalPremium", "sum"),
                TotalClaims=("TotalClaims", "sum"),
                Exposure=("UnderwrittenCoverID", "count"),
            )
        )
        grouped["LossRatio"] = grouped["TotalClaims"] / grouped["TotalPremium"].replace(0, np.nan)
        
        # Filter and sort
        grouped = (
            grouped[grouped["TotalPremium"] > min_premium]
            .sort_values("LossRatio", ascending=False)
            .head(top_n)
        )
        
        if grouped.empty:
            logger.warning(f"No data to plot for dimension '{dimension}'")
            return
        
        plt.figure(figsize=figsize)
        sns.barplot(x=grouped.index.astype(str), y=grouped["LossRatio"])
        plt.xticks(rotation=45, ha="right")
        plt.title(f"Loss Ratio by {dimension}")
        plt.ylabel("Loss Ratio")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Figure saved to {save_path}")
        
        plt.show()
        
    except Exception as e:
        logger.error(f"Error plotting loss ratio by {dimension}: {e}")
        raise


def plot_temporal_trends(
    df: pd.DataFrame,
    date_col: str = "Month",
    figsize: tuple = (12, 10),
    save_path: Optional[str] = None
) -> None:
    """
    Plot temporal trends for premium, claims, and loss ratio.
    
    Args:
        df: DataFrame with temporal data
        date_col: Name of date/month column
        figsize: Figure size tuple
        save_path: Optional path to save figure
        
    Raises:
        ValueError: If required columns don't exist
    """
    required_cols = {date_col, "TotalPremium", "TotalClaims"}
    missing_cols = required_cols - set(df.columns)
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    try:
        # Aggregate by date
        monthly = (
            df.groupby(date_col)
            .agg(
                TotalPremium=("TotalPremium", "sum"),
                TotalClaims=("TotalClaims", "sum"),
                Policies=("PolicyID", "nunique"),
                Covers=("UnderwrittenCoverID", "count"),
            )
        )
        monthly["LossRatio"] = monthly["TotalClaims"] / monthly["TotalPremium"].replace(0, np.nan)
        
        fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
        
        # Premium and Claims
        monthly[["TotalPremium", "TotalClaims"]].plot(ax=axes[0])
        axes[0].set_title("Monthly Total Premium and Total Claims")
        axes[0].legend()
        
        # Loss Ratio
        monthly["LossRatio"].plot(ax=axes[1], color="crimson")
        axes[1].set_title("Monthly Loss Ratio")
        axes[1].set_ylabel("Loss Ratio")
        
        # Policy and Cover Counts
        monthly[["Policies", "Covers"]].plot(ax=axes[2])
        axes[2].set_title("Monthly Policy and Cover Counts")
        axes[2].set_ylabel("Count")
        axes[2].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Figure saved to {save_path}")
        
        plt.show()
        
    except Exception as e:
        logger.error(f"Error plotting temporal trends: {e}")
        raise


def plot_correlation_matrix(
    df: pd.DataFrame,
    columns: List[str],
    figsize: tuple = (8, 6),
    save_path: Optional[str] = None
) -> None:
    """
    Plot correlation matrix heatmap for numeric columns.
    
    Args:
        df: DataFrame with numeric data
        columns: List of column names to include
        figsize: Figure size tuple
        save_path: Optional path to save figure
        
    Raises:
        ValueError: If columns don't exist
    """
    existing_cols = [c for c in columns if c in df.columns]
    if not existing_cols:
        raise ValueError(f"None of the specified columns exist: {columns}")
    
    try:
        corr = df[existing_cols].corr()
        
        plt.figure(figsize=figsize)
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
        plt.title("Correlation matrix of key numeric variables")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Figure saved to {save_path}")
        
        plt.show()
        
    except Exception as e:
        logger.error(f"Error plotting correlation matrix: {e}")
        raise

