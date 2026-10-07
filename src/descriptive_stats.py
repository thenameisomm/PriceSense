"""
PriceSense — Statistical Analysis Module

This module implements comprehensive statistical analysis for the Amazon product dataset.
All functions return structured pandas DataFrames for reusability in Streamlit GUI and notebooks.

Statistical concepts covered (DSML syllabus):
- Central tendency: Mean, Median, Mode
- Dispersion: Range, Variance, Standard Deviation
- Distribution shape: Skewness, Kurtosis
- Additional: Min, Max, Percentiles

Focus variables:
- actual_price
- discounted_price
- discount_percentage
- rating
- rating_count
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Union
from scipy import stats


# ============================================================================
# CONFIGURATION
# ============================================================================

CLEANED_DATA_PATH = Path("outputs/cleaned_data.csv")

# Key numerical variables for analysis (must exist in cleaned data)
KEY_NUMERIC_COLUMNS = [
    "actual_price",
    "discounted_price",
    "discount_percentage",
    "rating",
    "rating_count"
]

# Extended numerical variables for additional analysis
EXTENDED_NUMERIC_COLUMNS = [
    "price_savings",
    "discount_depth",
    "price_ratio",
    "category_depth",
    "log_rating_count",
    "log_discounted_price",
    "log_actual_price",
    "review_count_parsed",
    "avg_review_length"
]

ALL_NUMERIC_COLUMNS = KEY_NUMERIC_COLUMNS + EXTENDED_NUMERIC_COLUMNS


# ============================================================================
# CORE STATISTICAL FUNCTIONS
# ============================================================================

def load_cleaned_data(filepath: Path = CLEANED_DATA_PATH) -> pd.DataFrame:
    """
    Load the cleaned dataset.

    Parameters
    ----------
    filepath : Path
        Path to cleaned CSV file.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe.
    """
    df = pd.read_csv(filepath)
    print(f"[LOAD] Loaded cleaned dataset: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


def get_numeric_columns(df: pd.DataFrame, columns: Optional[List[str]] = None) -> List[str]:
    """
    Get valid numeric columns that exist in the dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    columns : list, optional
        List of column names to check. If None, uses ALL_NUMERIC_COLUMNS.

    Returns
    -------
    list
        List of valid numeric column names present in dataframe.
    """
    if columns is None:
        columns = ALL_NUMERIC_COLUMNS

    valid_cols = [c for c in columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    return valid_cols


def calculate_central_tendency(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Calculate central tendency statistics: Mean, Median, Mode.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    columns : list, optional
        Columns to analyze. Defaults to key numeric columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns as index and Mean, Median, Mode as columns.
    """
    valid_cols = get_numeric_columns(df, columns)

    results = {}
    for col in valid_cols:
        series = df[col].dropna()
        if len(series) == 0:
            results[col] = {"Mean": np.nan, "Median": np.nan, "Mode": np.nan}
            continue

        mean_val = series.mean()
        median_val = series.median()

        # Mode - handle multiple modes by taking the first
        mode_result = series.mode()
        mode_val = mode_result.iloc[0] if len(mode_result) > 0 else np.nan

        results[col] = {
            "Mean": round(mean_val, 4),
            "Median": round(median_val, 4),
            "Mode": round(mode_val, 4) if not np.isnan(mode_val) else np.nan
        }

    return pd.DataFrame(results).T


def calculate_dispersion(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Calculate dispersion statistics: Min, Max, Range, Variance, Standard Deviation.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    columns : list, optional
        Columns to analyze. Defaults to key numeric columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns as index and dispersion statistics as columns.
    """
    valid_cols = get_numeric_columns(df, columns)

    results = {}
    for col in valid_cols:
        series = df[col].dropna()
        if len(series) < 2:
            results[col] = {
                "Minimum": np.nan,
                "Maximum": np.nan,
                "Range": np.nan,
                "Variance": np.nan,
                "Std_Deviation": np.nan
            }
            continue

        min_val = series.min()
        max_val = series.max()
        range_val = max_val - min_val
        var_val = series.var(ddof=1)  # Sample variance
        std_val = series.std(ddof=1)  # Sample standard deviation

        results[col] = {
            "Minimum": round(min_val, 4),
            "Maximum": round(max_val, 4),
            "Range": round(range_val, 4),
            "Variance": round(var_val, 4),
            "Std_Deviation": round(std_val, 4)
        }

    return pd.DataFrame(results).T


def calculate_distribution_shape(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Calculate distribution shape statistics: Skewness, Kurtosis.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    columns : list, optional
        Columns to analyze. Defaults to key numeric columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns as index and Skewness, Kurtosis as columns.
    """
    valid_cols = get_numeric_columns(df, columns)

    results = {}
    for col in valid_cols:
        series = df[col].dropna()
        if len(series) < 3:
            results[col] = {"Skewness": np.nan, "Kurtosis": np.nan}
            continue

        # Use scipy.stats for unbiased estimators
        skew_val = stats.skew(series, bias=False)
        kurt_val = stats.kurtosis(series, bias=False)  # Fisher's kurtosis (excess kurtosis)

        results[col] = {
            "Skewness": round(skew_val, 4),
            "Kurtosis": round(kurt_val, 4)
        }

    return pd.DataFrame(results).T


def calculate_percentiles(df: pd.DataFrame, columns: Optional[List[str]] = None,
                          percentiles: List[float] = None) -> pd.DataFrame:
    """
    Calculate percentiles for numeric columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    columns : list, optional
        Columns to analyze. Defaults to key numeric columns.
    percentiles : list, optional
        Percentiles to calculate. Defaults to [10, 25, 50, 75, 90, 95, 99].

    Returns
    -------
    pd.DataFrame
        DataFrame with columns as index and percentiles as columns.
    """
    if percentiles is None:
        percentiles = [10, 25, 50, 75, 90, 95, 99]

    valid_cols = get_numeric_columns(df, columns)

    results = {}
    for col in valid_cols:
        series = df[col].dropna()
        if len(series) == 0:
            results[col] = {f"P{p}": np.nan for p in percentiles}
            continue

        perc_values = series.quantile([p/100 for p in percentiles])
        results[col] = {f"P{p}": round(perc_values.get(p/100, np.nan), 4) for p in percentiles}

    return pd.DataFrame(results).T


def calculate_summary_statistics(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Calculate comprehensive summary statistics combining all measures.

    This is the main function that provides a complete statistical summary
    for each numeric variable.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    columns : list, optional
        Columns to analyze. Defaults to key numeric columns.

    Returns
    -------
    pd.DataFrame
        Comprehensive summary statistics DataFrame.
    """
    valid_cols = get_numeric_columns(df, columns)

    # Calculate all components
    central = calculate_central_tendency(df, valid_cols)
    dispersion = calculate_dispersion(df, valid_cols)
    shape = calculate_distribution_shape(df, valid_cols)
    percentiles = calculate_percentiles(df, valid_cols)

    # Combine all statistics
    summary = pd.concat([central, dispersion, shape, percentiles], axis=1)

    # Reorder columns for logical presentation
    column_order = [
        "Mean", "Median", "Mode",
        "Minimum", "Maximum", "Range",
        "Variance", "Std_Deviation",
        "Skewness", "Kurtosis",
        "P10", "P25", "P50", "P75", "P90", "P95", "P99"
    ]

    # Only keep columns that exist
    existing_cols = [c for c in column_order if c in summary.columns]
    summary = summary[existing_cols]

    return summary


def calculate_correlation_matrix(df: pd.DataFrame, columns: Optional[List[str]] = None,
                                 method: str = "pearson") -> pd.DataFrame:
    """
    Calculate correlation matrix for numeric columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    columns : list, optional
        Columns to include. Defaults to key numeric columns.
    method : str
        Correlation method: 'pearson', 'spearman', or 'kendall'.

    Returns
    -------
    pd.DataFrame
        Correlation matrix.
    """
    valid_cols = get_numeric_columns(df, columns)
    corr_matrix = df[valid_cols].corr(method=method).round(4)
    return corr_matrix


def calculate_covariance_matrix(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Calculate covariance matrix for numeric columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    columns : list, optional
        Columns to include. Defaults to key numeric columns.

    Returns
    -------
    pd.DataFrame
        Covariance matrix.
    """
    valid_cols = get_numeric_columns(df, columns)
    cov_matrix = df[valid_cols].cov().round(4)
    return cov_matrix


def calculate_group_statistics(df: pd.DataFrame,
                               group_column: str,
                               value_columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Calculate grouped statistics (mean, median, std, count) by a categorical column.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    group_column : str
        Categorical column to group by.
    value_columns : list, optional
        Numeric columns to aggregate. Defaults to key numeric columns.

    Returns
    -------
    pd.DataFrame
        Grouped statistics with multi-level columns.
    """
    if group_column not in df.columns:
        raise ValueError(f"Group column '{group_column}' not found in dataframe")

    if value_columns is None:
        value_columns = KEY_NUMERIC_COLUMNS

    valid_value_cols = get_numeric_columns(df, value_columns)

    # Perform aggregation
    agg_funcs = ["mean", "median", "std", "count", "min", "max"]
    grouped = df.groupby(group_column)[valid_value_cols].agg(agg_funcs).round(4)

    # Flatten column multi-index for readability
    grouped.columns = [f"{col[0]}_{col[1]}" for col in grouped.columns]

    return grouped.reset_index()


def create_pivot_table(df: pd.DataFrame,
                       index: str,
                       columns: str,
                       values: str,
                       aggfunc: str = "mean") -> pd.DataFrame:
    """
    Create a pivot table for cross-tabulation analysis.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    index : str
        Column to use as row index.
    columns : str
        Column to use as column headers.
    values : str
        Numeric column to aggregate.
    aggfunc : str
        Aggregation function: 'mean', 'median', 'sum', 'count', 'std', 'min', 'max'.

    Returns
    -------
    pd.DataFrame
        Pivot table.
    """
    for col in [index, columns, values]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in dataframe")

    pivot = pd.pivot_table(
        df,
        index=index,
        columns=columns,
        values=values,
        aggfunc=aggfunc,
        fill_value=np.nan
    ).round(4)

    return pivot


def interpret_skewness(skew_val: float) -> str:
    """
    Interpret skewness value.

    Parameters
    ----------
    skew_val : float
        Skewness value.

    Returns
    -------
    str
        Interpretation string.
    """
    if np.isnan(skew_val):
        return "Undefined"
    elif abs(skew_val) < 0.5:
        return "Approximately symmetric"
    elif skew_val > 0:
        return "Right-skewed (positive skew)"
    else:
        return "Left-skewed (negative skew)"


def interpret_kurtosis(kurt_val: float) -> str:
    """
    Interpret kurtosis value (Fisher's excess kurtosis).

    Parameters
    ----------
    kurt_val : float
        Kurtosis value.

    Returns
    -------
    str
        Interpretation string.
    """
    if np.isnan(kurt_val):
        return "Undefined"
    elif abs(kurt_val) < 0.5:
        return "Mesokurtic (normal-like)"
    elif kurt_val > 0:
        return "Leptokurtic (heavy-tailed, peaked)"
    else:
        return "Platykurtic (light-tailed, flat)"


def generate_statistical_report(df: pd.DataFrame, columns: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
    """
    Generate a complete statistical report as a dictionary of DataFrames.

    This is the main entry point for getting all statistical outputs at once.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    columns : list, optional
        Columns to analyze. Defaults to key numeric columns.

    Returns
    -------
    dict
        Dictionary with keys: 'summary', 'correlation', 'covariance', 'group_by_category'
    """
    valid_cols = get_numeric_columns(df, columns)

    report = {}

    # 1. Comprehensive summary
    print("Calculating summary statistics...")
    report["summary"] = calculate_summary_statistics(df, valid_cols)

    # 2. Correlation matrix
    print("Calculating correlation matrix...")
    report["correlation"] = calculate_correlation_matrix(df, valid_cols)

    # 3. Covariance matrix
    print("Calculating covariance matrix...")
    report["covariance"] = calculate_covariance_matrix(df, valid_cols)

    # 4. Group by main_category (if available)
    if "main_category" in df.columns:
        print("Calculating group statistics by main_category...")
        report["group_by_category"] = calculate_group_statistics(df, "main_category", valid_cols)

    # 5. Group by discount bands and create discount_band column for pivot tables
    df_with_bands = df.copy()
    if "discount_percentage" in df.columns:
        print("Calculating group statistics by discount bands...")
        df_with_bands["discount_band"] = pd.cut(
            df_with_bands["discount_percentage"],
            bins=[-1, 10, 30, 50, 70, 100],
            labels=["0-10%", "10-30%", "30-50%", "50-70%", "70-100%"]
        )
        report["group_by_discount_band"] = calculate_group_statistics(
            df_with_bands, "discount_band", valid_cols
        )

    # 6. Pivot table: main_category vs discount_band (count)
    if "main_category" in df.columns and "discount_percentage" in df.columns:
        print("Creating pivot tables...")
        report["pivot_category_discount"] = create_pivot_table(
            df_with_bands, "main_category", "discount_band", "product_id", "count"
        )
        report["pivot_category_rating"] = create_pivot_table(
            df_with_bands, "main_category", "discount_band", "rating", "mean"
        )

    return report


def save_statistical_report(report: Dict[str, pd.DataFrame], output_dir: Path = Path("outputs/statistics")) -> None:
    """
    Save all statistical report DataFrames to CSV files.

    Parameters
    ----------
    report : dict
        Dictionary of DataFrames from generate_statistical_report.
    output_dir : Path
        Output directory.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, df in report.items():
        filepath = output_dir / f"{name}.csv"
        df.to_csv(filepath)
        print(f"[SAVE] {name} -> {filepath}")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    print("\n" + "#"*60)
    print("# PRICESENSE STATISTICAL ANALYSIS")
    print("#"*60)

    # Load cleaned data
    df = load_cleaned_data()

    # Generate full statistical report
    report = generate_statistical_report(df, KEY_NUMERIC_COLUMNS)

    # Display key results
    print("\n" + "="*60)
    print("SUMMARY STATISTICS (Key Variables)")
    print("="*60)
    print(report["summary"].to_string())

    print("\n" + "="*60)
    print("CORRELATION MATRIX (Pearson)")
    print("="*60)
    print(report["correlation"].to_string())

    print("\n" + "="*60)
    print("COVARIANCE MATRIX")
    print("="*60)
    print(report["covariance"].to_string())

    if "group_by_category" in report:
        print("\n" + "="*60)
        print("GROUP STATISTICS BY MAIN CATEGORY (Mean)")
        print("="*60)
        # Show only mean columns for readability
        mean_cols = [c for c in report["group_by_category"].columns if c.endswith("_mean")]
        print(report["group_by_category"][["main_category"] + mean_cols].to_string())

    # Save reports
    print("\n" + "="*60)
    print("SAVING STATISTICAL REPORTS")
    print("="*60)
    save_statistical_report(report)

    print("\n" + "#"*60)
    print("# STATISTICAL ANALYSIS COMPLETE")
    print("#"*60)