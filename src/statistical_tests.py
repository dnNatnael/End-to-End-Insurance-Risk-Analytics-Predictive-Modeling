"""
Statistical hypothesis testing module for insurance risk analysis.
Provides functions for A/B testing and hypothesis validation.
"""

import logging
from typing import Dict, Tuple, Optional, List
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency, ttest_ind, f_oneway, normaltest

# Configure logging
logger = logging.getLogger(__name__)


def calculate_claim_frequency(df: pd.DataFrame, group_col: str, claims_col: str = "TotalClaims") -> pd.DataFrame:
    """
    Calculate claim frequency (proportion of policies with ≥1 claim) by group.
    
    Args:
        df: DataFrame with insurance data
        group_col: Column name to group by
        claims_col: Column name for claims amount
        
    Returns:
        pd.DataFrame: Summary with claim frequency metrics
    """
    try:
        # Create binary claim indicator
        df = df.copy()
        df['HasClaim'] = (df[claims_col] > 0).astype(int)
        
        # Calculate metrics by group
        summary = df.groupby(group_col).agg({
            'HasClaim': ['sum', 'count', 'mean'],
            claims_col: ['sum', 'mean'],
            'TotalPremium': ['sum', 'mean']
        }).round(4)
        
        # Flatten column names
        summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
        summary = summary.rename(columns={
            'HasClaim_sum': 'NumClaims',
            'HasClaim_count': 'TotalPolicies',
            'HasClaim_mean': 'ClaimFrequency',
            f'{claims_col}_sum': 'TotalClaims',
            f'{claims_col}_mean': 'AvgClaimAmount',
            'TotalPremium_sum': 'TotalPremium',
            'TotalPremium_mean': 'AvgPremium'
        })
        
        # Calculate loss ratio
        summary['LossRatio'] = summary['TotalClaims'] / summary['TotalPremium'].replace(0, np.nan)
        
        # Calculate margin
        summary['Margin'] = summary['TotalPremium'] - summary['TotalClaims']
        summary['MarginPerPolicy'] = summary['Margin'] / summary['TotalPolicies']
        
        return summary.reset_index()
        
    except Exception as e:
        logger.error(f"Error calculating claim frequency: {e}")
        raise


def calculate_claim_severity(df: pd.DataFrame, group_col: str, claims_col: str = "TotalClaims") -> pd.DataFrame:
    """
    Calculate claim severity (average claim amount among claimants) by group.
    
    Args:
        df: DataFrame with insurance data
        group_col: Column name to group by
        claims_col: Column name for claims amount
        
    Returns:
        pd.DataFrame: Summary with claim severity metrics
    """
    try:
        # Filter to only claimants
        claimants = df[df[claims_col] > 0].copy()
        
        if len(claimants) == 0:
            logger.warning("No claimants found in dataset")
            return pd.DataFrame()
        
        # Calculate severity by group
        summary = claimants.groupby(group_col).agg({
            claims_col: ['mean', 'median', 'std', 'count'],
            'TotalPremium': 'mean'
        }).round(4)
        
        # Flatten column names
        summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
        summary = summary.rename(columns={
            f'{claims_col}_mean': 'AvgSeverity',
            f'{claims_col}_median': 'MedianSeverity',
            f'{claims_col}_std': 'StdSeverity',
            f'{claims_col}_count': 'NumClaimants',
            'TotalPremium_mean': 'AvgPremium'
        })
        
        return summary.reset_index()
        
    except Exception as e:
        logger.error(f"Error calculating claim severity: {e}")
        raise


def create_ab_groups(
    df: pd.DataFrame,
    group_col: str,
    group_a_value: str,
    group_b_value: str,
    balance_features: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create A/B testing groups with optional feature balancing.
    
    Args:
        df: DataFrame with insurance data
        group_col: Column name to segment by
        group_a_value: Value for Group A (control)
        group_b_value: Value for Group B (test)
        balance_features: Optional list of features to balance on
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Group A and Group B DataFrames
    """
    try:
        group_a = df[df[group_col] == group_a_value].copy()
        group_b = df[df[group_col] == group_b_value].copy()
        
        if len(group_a) == 0:
            raise ValueError(f"No data found for Group A: {group_a_value}")
        if len(group_b) == 0:
            raise ValueError(f"No data found for Group B: {group_b_value}")
        
        logger.info(f"Group A ({group_a_value}): {len(group_a)} records")
        logger.info(f"Group B ({group_b_value}): {len(group_b)} records")
        
        # Optional: Balance on other features (simple random sampling)
        if balance_features and len(group_a) != len(group_b):
            min_size = min(len(group_a), len(group_b))
            if len(group_a) > min_size:
                group_a = group_a.sample(n=min_size, random_state=42)
            if len(group_b) > min_size:
                group_b = group_b.sample(n=min_size, random_state=42)
            logger.info(f"Balanced groups to {min_size} records each")
        
        return group_a, group_b
        
    except Exception as e:
        logger.error(f"Error creating A/B groups: {e}")
        raise


def test_proportion_difference(
    group_a: pd.DataFrame,
    group_b: pd.DataFrame,
    success_col: str = "HasClaim",
    alpha: float = 0.05
) -> Dict[str, any]:
    """
    Perform two-proportion Z-test to compare claim frequencies.
    
    Args:
        group_a: Group A DataFrame
        group_b: Group B DataFrame
        success_col: Column name for binary success indicator
        alpha: Significance level
        
    Returns:
        dict: Test results including p-value, statistic, and interpretation
    """
    try:
        # Create success indicator if not exists
        if success_col not in group_a.columns:
            group_a = group_a.copy()
            group_a[success_col] = (group_a['TotalClaims'] > 0).astype(int)
        if success_col not in group_b.columns:
            group_b = group_b.copy()
            group_b[success_col] = (group_b['TotalClaims'] > 0).astype(int)
        
        # Calculate proportions
        n_a = len(group_a)
        n_b = len(group_b)
        x_a = group_a[success_col].sum()
        x_b = group_b[success_col].sum()
        p_a = x_a / n_a if n_a > 0 else 0
        p_b = x_b / n_b if n_b > 0 else 0
        
        # Pooled proportion
        p_pool = (x_a + x_b) / (n_a + n_b) if (n_a + n_b) > 0 else 0
        
        # Standard error
        se = np.sqrt(p_pool * (1 - p_pool) * (1/n_a + 1/n_b))
        
        if se == 0:
            z_stat = 0
            p_value = 1.0
        else:
            # Z-statistic
            z_stat = (p_a - p_b) / se
            
            # Two-tailed p-value
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        
        # Effect size (Cohen's h)
        h = 2 * (np.arcsin(np.sqrt(p_a)) - np.arcsin(np.sqrt(p_b)))
        
        # Decision
        reject_h0 = p_value < alpha
        
        results = {
            'test_name': 'Two-Proportion Z-Test',
            'group_a_proportion': p_a,
            'group_b_proportion': p_b,
            'difference': p_a - p_b,
            'z_statistic': z_stat,
            'p_value': p_value,
            'alpha': alpha,
            'reject_h0': reject_h0,
            'effect_size_cohens_h': h,
            'group_a_n': n_a,
            'group_b_n': n_b,
            'group_a_successes': x_a,
            'group_b_successes': x_b,
            'interpretation': 'Reject H₀' if reject_h0 else 'Fail to reject H₀'
        }
        
        return results
        
    except Exception as e:
        logger.error(f"Error in proportion difference test: {e}")
        raise


def test_chi_squared(
    group_a: pd.DataFrame,
    group_b: pd.DataFrame,
    success_col: str = "HasClaim"
) -> Dict[str, any]:
    """
    Perform Chi-squared test for independence.
    
    Args:
        group_a: Group A DataFrame
        group_b: Group B DataFrame
        success_col: Column name for binary success indicator
        
    Returns:
        dict: Test results
    """
    try:
        # Create success indicator if not exists
        if success_col not in group_a.columns:
            group_a = group_a.copy()
            group_a[success_col] = (group_a['TotalClaims'] > 0).astype(int)
        if success_col not in group_b.columns:
            group_b = group_b.copy()
            group_b[success_col] = (group_b['TotalClaims'] > 0).astype(int)
        
        # Create contingency table
        contingency = pd.DataFrame({
            'Group_A': [
                (group_a[success_col] == 0).sum(),  # No claim
                (group_a[success_col] == 1).sum()   # Has claim
            ],
            'Group_B': [
                (group_b[success_col] == 0).sum(),  # No claim
                (group_b[success_col] == 1).sum()   # Has claim
            ]
        }, index=['No_Claim', 'Has_Claim'])
        
        # Perform chi-squared test
        chi2, p_value, dof, expected = chi2_contingency(contingency)
        
        # Decision
        reject_h0 = p_value < 0.05
        
        results = {
            'test_name': 'Chi-Squared Test',
            'chi2_statistic': chi2,
            'p_value': p_value,
            'degrees_of_freedom': dof,
            'contingency_table': contingency,
            'expected_frequencies': pd.DataFrame(expected, index=contingency.index, columns=contingency.columns),
            'reject_h0': reject_h0,
            'interpretation': 'Reject H₀' if reject_h0 else 'Fail to reject H₀'
        }
        
        return results
        
    except Exception as e:
        logger.error(f"Error in chi-squared test: {e}")
        raise


def test_mean_difference(
    group_a: pd.DataFrame,
    group_b: pd.DataFrame,
    metric_col: str,
    alpha: float = 0.05,
    equal_var: bool = False
) -> Dict[str, any]:
    """
    Perform t-test to compare means between two groups.
    
    Args:
        group_a: Group A DataFrame
        group_b: Group B DataFrame
        metric_col: Column name for metric to compare
        alpha: Significance level
        equal_var: Whether to assume equal variances
        
    Returns:
        dict: Test results
    """
    try:
        # Extract metric values
        values_a = group_a[metric_col].dropna()
        values_b = group_b[metric_col].dropna()
        
        if len(values_a) == 0 or len(values_b) == 0:
            raise ValueError("One or both groups have no valid values")
        
        # Perform t-test
        t_stat, p_value = ttest_ind(values_a, values_b, equal_var=equal_var)
        
        # Calculate effect size (Cohen's d)
        mean_a = values_a.mean()
        mean_b = values_b.mean()
        std_a = values_a.std()
        std_b = values_b.std()
        pooled_std = np.sqrt(((len(values_a) - 1) * std_a**2 + (len(values_b) - 1) * std_b**2) / 
                            (len(values_a) + len(values_b) - 2))
        cohens_d = (mean_a - mean_b) / pooled_std if pooled_std > 0 else 0
        
        # Decision
        reject_h0 = p_value < alpha
        
        results = {
            'test_name': 'Independent Samples t-Test',
            'group_a_mean': mean_a,
            'group_b_mean': mean_b,
            'group_a_std': std_a,
            'group_b_std': std_b,
            'difference': mean_a - mean_b,
            't_statistic': t_stat,
            'p_value': p_value,
            'alpha': alpha,
            'reject_h0': reject_h0,
            'effect_size_cohens_d': cohens_d,
            'group_a_n': len(values_a),
            'group_b_n': len(values_b),
            'interpretation': 'Reject H₀' if reject_h0 else 'Fail to reject H₀'
        }
        
        return results
        
    except Exception as e:
        logger.error(f"Error in mean difference test: {e}")
        raise


def test_anova(
    groups: Dict[str, pd.DataFrame],
    metric_col: str,
    alpha: float = 0.05
) -> Dict[str, any]:
    """
    Perform one-way ANOVA to compare means across multiple groups.
    
    Args:
        groups: Dictionary of group names to DataFrames
        metric_col: Column name for metric to compare
        alpha: Significance level
        
    Returns:
        dict: Test results
    """
    try:
        # Extract metric values for each group
        group_values = {}
        for name, df in groups.items():
            values = df[metric_col].dropna()
            if len(values) > 0:
                group_values[name] = values
        
        if len(group_values) < 2:
            raise ValueError("Need at least 2 groups with valid data for ANOVA")
        
        # Perform ANOVA
        f_stat, p_value = f_oneway(*group_values.values())
        
        # Calculate group means
        group_means = {name: values.mean() for name, values in group_values.items()}
        group_stds = {name: values.std() for name, values in group_values.items()}
        group_ns = {name: len(values) for name, values in group_values.items()}
        
        # Decision
        reject_h0 = p_value < alpha
        
        results = {
            'test_name': 'One-Way ANOVA',
            'f_statistic': f_stat,
            'p_value': p_value,
            'alpha': alpha,
            'reject_h0': reject_h0,
            'group_means': group_means,
            'group_stds': group_stds,
            'group_ns': group_ns,
            'num_groups': len(group_values),
            'interpretation': 'Reject H₀' if reject_h0 else 'Fail to reject H₀'
        }
        
        return results
        
    except Exception as e:
        logger.error(f"Error in ANOVA test: {e}")
        raise


def calculate_margin_metrics(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """
    Calculate margin (TotalPremium - TotalClaims) metrics by group.
    
    Args:
        df: DataFrame with insurance data
        group_col: Column name to group by
        
    Returns:
        pd.DataFrame: Summary with margin metrics
    """
    try:
        df = df.copy()
        df['Margin'] = df['TotalPremium'] - df['TotalClaims']
        
        summary = df.groupby(group_col).agg({
            'Margin': ['sum', 'mean', 'std', 'count'],
            'TotalPremium': 'sum',
            'TotalClaims': 'sum'
        }).round(4)
        
        # Flatten column names
        summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
        summary = summary.rename(columns={
            'Margin_sum': 'TotalMargin',
            'Margin_mean': 'AvgMargin',
            'Margin_std': 'StdMargin',
            'Margin_count': 'NumPolicies',
            'TotalPremium_sum': 'TotalPremium',
            'TotalClaims_sum': 'TotalClaims'
        })
        
        # Calculate margin ratio
        summary['MarginRatio'] = summary['TotalMargin'] / summary['TotalPremium'].replace(0, np.nan)
        
        return summary.reset_index()
        
    except Exception as e:
        logger.error(f"Error calculating margin metrics: {e}")
        raise

