"""
PriceSense — Exploratory Data Analysis & Visualization Module

This module implements comprehensive EDA visualizations for the Amazon product dataset.
All visualizations follow DSML syllabus requirements and are saved as PNG files.

Visualizations include:
1. Distribution Analysis - Histograms with KDE
2. Box Plots - For outlier detection and distribution comparison
3. Correlation Analysis - Heatmaps
4. Category-wise Analysis - Grouped bar charts
5. Pivot Table Visualizations - Heatmaps
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Optional, Dict, Any
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURATION
# ============================================================================

CLEANED_DATA_PATH = Path("outputs/cleaned_data.csv")
CHARTS_DIR = Path("outputs/charts")

# Ensure charts directory exists
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

# Style configuration
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Color palette for consistency
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'accent': '#F18F01',
    'success': '#C73E1D',
    'neutral': '#6C757D'
}

# Key variables for analysis
KEY_VARIABLES = [
    "actual_price",
    "discounted_price",
    "discount_percentage",
    "rating",
    "rating_count"
]

CATEGORY_COL = "main_category"


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_cleaned_data(filepath: Path = CLEANED_DATA_PATH) -> pd.DataFrame:
    """Load the cleaned dataset."""
    df = pd.read_csv(filepath)
    print(f"[LOAD] Loaded cleaned dataset: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


def save_figure(fig: plt.Figure, filename: str, dpi: int = 300) -> Path:
    """Save figure to charts directory with consistent settings."""
    filepath = CHARTS_DIR / filename
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[SAVE] Chart saved: {filepath}")
    return filepath


def setup_plot(title: str, xlabel: str, ylabel: str, figsize: tuple = (10, 6)) -> tuple:
    """Create a figure and axis with consistent styling."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, alpha=0.3)
    return fig, ax


# ============================================================================
# 1. DISTRIBUTION ANALYSIS - HISTOGRAMS WITH KDE
# ============================================================================

def plot_distribution_histogram(df: pd.DataFrame, column: str, bins: int = 50,
                                 kde: bool = True, color: str = None) -> Path:
    """
    Plot histogram with optional KDE for a single variable.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    column : str
        Column name to plot.
    bins : int
        Number of histogram bins.
    kde : bool
        Whether to overlay KDE curve.
    color : str
        Bar color.

    Returns
    -------
    Path
        Saved chart filepath.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")

    color = color or COLORS['primary']
    data = df[column].dropna()

    fig, ax = setup_plot(
        title=f'Distribution of {column.replace("_", " ").title()}',
        xlabel=column.replace("_", " ").title(),
        ylabel='Frequency'
    )

    # Histogram
    n, bins_edges, patches = ax.hist(data, bins=bins, density=True, alpha=0.7,
                                      color=color, edgecolor='white', linewidth=0.5)

    # KDE overlay
    if kde and len(data) > 1:
        from scipy.stats import gaussian_kde
        kde_func = gaussian_kde(data)
        x_range = np.linspace(data.min(), data.max(), 300)
        ax.plot(x_range, kde_func(x_range), color=COLORS['secondary'],
                linewidth=2, label='KDE')

    # Mean and median lines
    ax.axvline(data.mean(), color=COLORS['success'], linestyle='--',
               linewidth=2, label=f'Mean: {data.mean():.2f}')
    ax.axvline(data.median(), color=COLORS['accent'], linestyle='--',
               linewidth=2, label=f'Median: {data.median():.2f}')

    ax.legend(fontsize=10)
    plt.tight_layout()

    filename = f"{column}_distribution.png"
    return save_figure(fig, filename)


def plot_all_distributions(df: pd.DataFrame, columns: List[str] = None) -> List[Path]:
    """Plot distribution histograms for multiple columns."""
    if columns is None:
        columns = KEY_VARIABLES

    saved_files = []
    for col in columns:
        if col in df.columns:
            try:
                filepath = plot_distribution_histogram(df, col)
                saved_files.append(filepath)
            except Exception as e:
                print(f"[WARN] Could not plot {col}: {e}")

    return saved_files


# ============================================================================
# 2. BOX PLOTS
# ============================================================================

def plot_boxplot_single(df: pd.DataFrame, column: str, color: str = None) -> Path:
    """
    Plot box plot for a single variable.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    column : str
        Column name to plot.
    color : str
        Box color.

    Returns
    -------
    Path
        Saved chart filepath.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")

    color = color or COLORS['primary']
    data = df[column].dropna()

    fig, ax = setup_plot(
        title=f'Box Plot of {column.replace("_", " ").title()}',
        xlabel='',
        ylabel=column.replace("_", " ").title()
    )

    # Box plot
    bp = ax.boxplot(data, vert=True, patch_artist=True, widths=0.5)
    bp['boxes'][0].set_facecolor(color)
    bp['boxes'][0].set_alpha(0.7)
    bp['medians'][0].set_color(COLORS['success'])
    bp['medians'][0].set_linewidth(2)

    # Add statistics text
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = len(data[(data < lower_bound) | (data > upper_bound)])

    stats_text = (f"Min: {data.min():.2f}\n"
                  f"Q1: {q1:.2f}\n"
                  f"Median: {data.median():.2f}\n"
                  f"Q3: {q3:.2f}\n"
                  f"Max: {data.max():.2f}\n"
                  f"Outliers: {outliers}")

    ax.text(1.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    filename = f"{column}_boxplot.png"
    return save_figure(fig, filename)


def plot_boxplot_by_category(df: pd.DataFrame, value_column: str,
                              category_column: str = CATEGORY_COL,
                              color: str = None) -> Path:
    """
    Plot box plot grouped by category.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    value_column : str
        Numeric column to plot.
    category_column : str
        Categorical column to group by.
    color : str
        Box color.

    Returns
    -------
    Path
        Saved chart filepath.
    """
    if value_column not in df.columns or category_column not in df.columns:
        raise ValueError("Required columns not found in dataframe")

    color = color or COLORS['primary']

    fig, ax = setup_plot(
        title=f'{value_column.replace("_", " ").title()} by {category_column.replace("_", " ").title()}',
        xlabel=category_column.replace("_", " ").title(),
        ylabel=value_column.replace("_", " ").title(),
        figsize=(14, 7)
    )

    # Seaborn boxplot for better category handling
    sns.boxplot(data=df, x=category_column, y=value_column, ax=ax,
                color=color, width=0.6, fliersize=3)

    ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()

    filename = f"{value_column}_by_{category_column}_boxplot.png"
    return save_figure(fig, filename)


def plot_all_boxplots(df: pd.DataFrame, columns: List[str] = None,
                       category_column: str = CATEGORY_COL) -> List[Path]:
    """Plot box plots for multiple columns."""
    if columns is None:
        columns = KEY_VARIABLES

    saved_files = []
    for col in columns:
        if col in df.columns:
            try:
                # Single variable boxplot
                filepath = plot_boxplot_single(df, col)
                saved_files.append(filepath)

                # By category boxplot
                if category_column in df.columns:
                    filepath2 = plot_boxplot_by_category(df, col, category_column)
                    saved_files.append(filepath2)
            except Exception as e:
                print(f"[WARN] Could not plot boxplot for {col}: {e}")

    return saved_files


# ============================================================================
# 3. CORRELATION HEATMAP
# ============================================================================

def plot_correlation_heatmap(df: pd.DataFrame, columns: List[str] = None,
                              method: str = 'pearson',
                              cmap: str = 'RdBu_r',
                              annot: bool = True) -> Path:
    """
    Plot correlation matrix heatmap.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    columns : list
        Columns to include in correlation matrix.
    method : str
        Correlation method ('pearson', 'spearman', 'kendall').
    cmap : str
        Colormap for heatmap.
    annot : bool
        Whether to annotate cells with values.

    Returns
    -------
    Path
        Saved chart filepath.
    """
    if columns is None:
        columns = KEY_VARIABLES

    valid_cols = [c for c in columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    corr_matrix = df[valid_cols].corr(method=method)

    fig, ax = setup_plot(
        title=f'{method.capitalize()} Correlation Matrix',
        xlabel='',
        ylabel='',
        figsize=(10, 8)
    )

    # Create heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Upper triangle mask
    sns.heatmap(corr_matrix, mask=mask, annot=annot, fmt='.3f', cmap=cmap,
                center=0, square=True, linewidths=0.5, cbar_kws={'label': 'Correlation Coefficient'},
                ax=ax)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    plt.tight_layout()

    filename = f"correlation_heatmap_{method}.png"
    return save_figure(fig, filename)


def plot_correlation_with_target(df: pd.DataFrame, target: str,
                                  features: List[str] = None,
                                  method: str = 'pearson') -> Path:
    """
    Plot correlation of features with a target variable.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    target : str
        Target column name.
    features : list
        Feature columns to correlate with target.
    method : str
        Correlation method.

    Returns
    -------
    Path
        Saved chart filepath.
    """
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found")

    if features is None:
        features = [c for c in KEY_VARIABLES if c != target and c in df.columns]

    valid_features = [f for f in features if f in df.columns and pd.api.types.is_numeric_dtype(df[f])]

    correlations = {}
    for feat in valid_features:
        corr = df[[target, feat]].corr(method=method).iloc[0, 1]
        correlations[feat] = corr

    # Sort by absolute correlation
    sorted_corr = dict(sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True))

    fig, ax = setup_plot(
        title=f'Correlation with {target.replace("_", " ").title()} ({method.capitalize()})',
        xlabel='Correlation Coefficient',
        ylabel='Features',
        figsize=(10, 6)
    )

    colors = [COLORS['success'] if v > 0 else COLORS['secondary'] for v in sorted_corr.values()]
    bars = ax.barh(list(sorted_corr.keys()), list(sorted_corr.values()), color=colors, alpha=0.7, edgecolor='white')

    # Add value labels
    for bar, val in zip(bars, sorted_corr.values()):
        ax.text(val + (0.01 if val >= 0 else -0.01), bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', ha='left' if val >= 0 else 'right', fontsize=10)

    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_xlim(-1, 1)

    plt.tight_layout()

    filename = f"correlation_with_{target}_{method}.png"
    return save_figure(fig, filename)


# ============================================================================
# 4. CATEGORY-WISE ANALYSIS - BAR CHARTS
# ============================================================================

def plot_category_means(df: pd.DataFrame, value_columns: List[str] = None,
                         category_column: str = CATEGORY_COL) -> List[Path]:
    """
    Plot grouped bar chart of mean values by category.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    value_columns : list
        Numeric columns to plot means for.
    category_column : str
        Categorical column to group by.

    Returns
    -------
    list of Path
        Saved chart filepaths.
    """
    if category_column not in df.columns:
        raise ValueError(f"Category column '{category_column}' not found")

    if value_columns is None:
        value_columns = KEY_VARIABLES

    valid_cols = [c for c in value_columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]

    # Calculate means by category
    means_df = df.groupby(category_column)[valid_cols].mean().reset_index()

    saved_files = []

    for col in valid_cols:
        fig, ax = setup_plot(
            title=f'Mean {col.replace("_", " ").title()} by {category_column.replace("_", " ").title()}',
            xlabel=category_column.replace("_", " ").title(),
            ylabel=f'Mean {col.replace("_", " ").title()}',
            figsize=(12, 6)
        )

        # Sort by mean value
        sorted_df = means_df.sort_values(col, ascending=True)

        bars = ax.barh(sorted_df[category_column], sorted_df[col],
                       color=COLORS['primary'], alpha=0.8, edgecolor='white')

        # Add value labels
        for bar, val in zip(bars, sorted_df[col]):
            ax.text(val + val*0.01, bar.get_y() + bar.get_height()/2,
                    f'{val:.1f}', va='center', fontsize=9)

        ax.tick_params(axis='x', rotation=0)
        plt.tight_layout()

        filename = f"mean_{col}_by_{category_column}.png"
        saved_files.append(save_figure(fig, filename))

    return saved_files


def plot_category_counts(df: pd.DataFrame, category_column: str = CATEGORY_COL) -> Path:
    """
    Plot count of products by category.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    category_column : str
        Categorical column.

    Returns
    -------
    Path
        Saved chart filepath.
    """
    if category_column not in df.columns:
        raise ValueError(f"Category column '{category_column}' not found")

    counts = df[category_column].value_counts().sort_values(ascending=True)

    fig, ax = setup_plot(
        title=f'Product Count by {category_column.replace("_", " ").title()}',
        xlabel='Number of Products',
        ylabel=category_column.replace("_", " ").title(),
        figsize=(10, 6)
    )

    bars = ax.barh(counts.index, counts.values, color=COLORS['primary'], alpha=0.8, edgecolor='white')

    for bar, val in zip(bars, counts.values):
        ax.text(val + val*0.01, bar.get_y() + bar.get_height()/2,
                f'{val}', va='center', fontsize=10)

    plt.tight_layout()

    filename = f"product_count_by_{category_column}.png"
    return save_figure(fig, filename)


# ============================================================================
# 5. PIVOT TABLE VISUALIZATIONS
# ============================================================================

def create_discount_bands(df: pd.DataFrame, column: str = 'discount_percentage') -> pd.DataFrame:
    """Add discount band column to dataframe."""
    df_copy = df.copy()
    df_copy['discount_band'] = pd.cut(
        df_copy[column],
        bins=[-1, 10, 30, 50, 70, 100],
        labels=['0-10%', '10-30%', '30-50%', '50-70%', '70-100%']
    )
    return df_copy


def plot_pivot_heatmap(df: pd.DataFrame, index: str, columns: str, values: str,
                        aggfunc: str = 'mean', cmap: str = 'YlOrRd') -> Path:
    """
    Plot pivot table as heatmap.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    index : str
        Row index column.
    columns : str
        Column header column.
    values : str
        Value column to aggregate.
    aggfunc : str
        Aggregation function.
    cmap : str
        Colormap.

    Returns
    -------
    Path
        Saved chart filepath.
    """
    for col in [index, columns, values]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in dataframe")

    pivot = pd.pivot_table(df, index=index, columns=columns, values=values,
                           aggfunc=aggfunc, fill_value=np.nan)

    fig, ax = setup_plot(
        title=f'{aggfunc.capitalize()} {values.replace("_", " ").title()} by {index.replace("_", " ").title()} × {columns.replace("_", " ").title()}',
        xlabel=columns.replace("_", " ").title(),
        ylabel=index.replace("_", " ").title(),
        figsize=(12, 8)
    )

    sns.heatmap(pivot, annot=True, fmt='.1f', cmap=cmap, linewidths=0.5,
                cbar_kws={'label': f'{aggfunc.capitalize()} {values.replace("_", " ").title()}'},
                ax=ax)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    plt.tight_layout()

    filename = f"pivot_{index}_{columns}_{values}_{aggfunc}.png"
    return save_figure(fig, filename)


def plot_pivot_count_heatmap(df: pd.DataFrame, index: str, columns: str) -> Path:
    """
    Plot pivot table of counts as heatmap.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    index : str
        Row index column.
    columns : str
        Column header column.

    Returns
    -------
    Path
        Saved chart filepath.
    """
    for col in [index, columns]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in dataframe")

    pivot = pd.pivot_table(df, index=index, columns=columns, values='product_id',
                           aggfunc='count', fill_value=0)

    fig, ax = setup_plot(
        title=f'Product Count by {index.replace("_", " ").title()} × {columns.replace("_", " ").title()}',
        xlabel=columns.replace("_", " ").title(),
        ylabel=index.replace("_", " ").title(),
        figsize=(12, 8)
    )

    sns.heatmap(pivot, annot=True, fmt='d', cmap='Blues', linewidths=0.5,
                cbar_kws={'label': 'Product Count'}, ax=ax)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    plt.tight_layout()

    filename = f"pivot_{index}_{columns}_count.png"
    return save_figure(fig, filename)


# ============================================================================
# 6. SCATTER PLOTS - RELATIONSHIPS
# ============================================================================

def plot_scatter_relationship(df: pd.DataFrame, x_col: str, y_col: str,
                               hue_col: str = None, size_col: str = None,
                               alpha: float = 0.6) -> Path:
    """
    Plot scatter plot for two variables.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    x_col : str
        X-axis column.
    y_col : str
        Y-axis column.
    hue_col : str
        Column for color encoding.
    size_col : str
        Column for size encoding.
    alpha : float
        Point transparency.

    Returns
    -------
    Path
        Saved chart filepath.
    """
    for col in [x_col, y_col]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in dataframe")

    fig, ax = setup_plot(
        title=f'{y_col.replace("_", " ").title()} vs {x_col.replace("_", " ").title()}',
        xlabel=x_col.replace("_", " ").title(),
        ylabel=y_col.replace("_", " ").title(),
        figsize=(10, 7)
    )

    if hue_col and hue_col in df.columns:
        scatter = ax.scatter(df[x_col], df[y_col], c=df[hue_col],
                             cmap='viridis', alpha=alpha, s=30, edgecolors='white', linewidth=0.3)
        plt.colorbar(scatter, ax=ax, label=hue_col.replace("_", " ").title())
    else:
        ax.scatter(df[x_col], df[y_col], color=COLORS['primary'], alpha=alpha,
                   s=30, edgecolors='white', linewidth=0.3)

    # Add correlation annotation
    valid_data = df[[x_col, y_col]].dropna()
    if len(valid_data) > 2:
        from scipy.stats import pearsonr
        corr, p_val = pearsonr(valid_data[x_col], valid_data[y_col])
        ax.text(0.05, 0.95, f'Pearson r = {corr:.3f}\np = {p_val:.3e}',
                transform=ax.transAxes, fontsize=11,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                verticalalignment='top')

    plt.tight_layout()

    filename = f"scatter_{y_col}_vs_{x_col}.png"
    return save_figure(fig, filename)


def plot_pairplot(df: pd.DataFrame, columns: List[str] = None,
                   hue_col: str = None) -> Path:
    """
    Plot pairplot for multiple variables.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    columns : list
        Columns to include in pairplot.
    hue_col : str
        Column for color encoding.

    Returns
    -------
    Path
        Saved chart filepath.
    """
    if columns is None:
        columns = KEY_VARIABLES

    valid_cols = [c for c in columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]

    # Use seaborn pairplot
    if hue_col and hue_col in df.columns:
        g = sns.pairplot(df[valid_cols + [hue_col]], hue=hue_col,
                         diag_kind='hist', plot_kws={'alpha': 0.6, 's': 20},
                         height=2.5)
    else:
        g = sns.pairplot(df[valid_cols], diag_kind='hist',
                         plot_kws={'alpha': 0.6, 's': 20}, height=2.5)

    g.fig.suptitle('Pairplot of Key Numerical Variables', y=1.02, fontsize=16, fontweight='bold')

    filename = "pairplot_key_variables.png"
    return save_figure(g.fig, filename)


# ============================================================================
# 7. ADDITIONAL EDA PLOTS
# ============================================================================

def plot_price_vs_rating_scatter(df: pd.DataFrame) -> Path:
    """Plot price vs rating with discount as color."""
    return plot_scatter_relationship(df, 'discounted_price', 'rating',
                                      hue_col='discount_percentage')


def plot_price_vs_rating_count(df: pd.DataFrame) -> Path:
    """Plot price vs rating count."""
    return plot_scatter_relationship(df, 'discounted_price', 'rating_count')


def plot_discount_vs_rating(df: pd.DataFrame) -> Path:
    """Plot discount percentage vs rating."""
    return plot_scatter_relationship(df, 'discount_percentage', 'rating')


def plot_rating_distribution_by_category(df: pd.DataFrame,
                                          category_column: str = CATEGORY_COL) -> Path:
    """
    Plot rating distribution by category using violin plot.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    category_column : str
        Categorical column.

    Returns
    -------
    Path
        Saved chart filepath.
    """
    if 'rating' not in df.columns or category_column not in df.columns:
        raise ValueError("Required columns not found")

    fig, ax = setup_plot(
        title=f'Rating Distribution by {category_column.replace("_", " ").title()}',
        xlabel=category_column.replace("_", " ").title(),
        ylabel='Rating',
        figsize=(14, 7)
    )

    sns.violinplot(data=df, x=category_column, y='rating', ax=ax,
                   color=COLORS['primary'], alpha=0.7, inner='box')

    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()

    filename = f"rating_distribution_by_{category_column}.png"
    return save_figure(fig, filename)


def plot_cumulative_discount_distribution(df: pd.DataFrame) -> Path:
    """Plot cumulative distribution of discount percentage."""
    if 'discount_percentage' not in df.columns:
        raise ValueError("discount_percentage column not found")

    data = df['discount_percentage'].dropna().sort_values()
    cumulative = np.arange(1, len(data) + 1) / len(data) * 100

    fig, ax = setup_plot(
        title='Cumulative Distribution of Discount Percentage',
        xlabel='Discount Percentage (%)',
        ylabel='Cumulative Percentage of Products (%)',
        figsize=(10, 6)
    )

    ax.plot(data, cumulative, color=COLORS['primary'], linewidth=2)
    ax.fill_between(data, cumulative, alpha=0.3, color=COLORS['primary'])

    # Add reference lines
    for pct in [25, 50, 75]:
        idx = np.searchsorted(cumulative, pct)
        if idx < len(data):
            ax.axvline(data.iloc[idx], color=COLORS['secondary'], linestyle='--', alpha=0.7)
            ax.axhline(pct, color=COLORS['secondary'], linestyle='--', alpha=0.7)

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    plt.tight_layout()

    filename = "cumulative_discount_distribution.png"
    return save_figure(fig, filename)


def plot_price_savings_distribution(df: pd.DataFrame) -> Path:
    """Plot distribution of price savings (actual - discounted)."""
    if 'price_savings' not in df.columns:
        raise ValueError("price_savings column not found")

    data = df['price_savings'].dropna()

    fig, ax = setup_plot(
        title='Distribution of Price Savings (Actual - Discounted Price)',
        xlabel='Price Savings (₹)',
        ylabel='Frequency',
        figsize=(10, 6)
    )

    ax.hist(data, bins=50, density=True, alpha=0.7, color=COLORS['primary'],
            edgecolor='white', linewidth=0.5)

    # KDE
    from scipy.stats import gaussian_kde
    kde_func = gaussian_kde(data)
    x_range = np.linspace(data.min(), data.max(), 300)
    ax.plot(x_range, kde_func(x_range), color=COLORS['secondary'], linewidth=2, label='KDE')

    ax.axvline(data.mean(), color=COLORS['success'], linestyle='--', linewidth=2,
               label=f'Mean: ₹{data.mean():.0f}')
    ax.axvline(data.median(), color=COLORS['accent'], linestyle='--', linewidth=2,
               label=f'Median: ₹{data.median():.0f}')

    ax.legend()
    plt.tight_layout()

    filename = "price_savings_distribution.png"
    return save_figure(fig, filename)


# ============================================================================
# MAIN EDA PIPELINE
# ============================================================================

def run_full_eda(df: pd.DataFrame = None) -> Dict[str, Any]:
    """
    Run complete EDA pipeline and generate all visualizations.

    Parameters
    ----------
    df : pd.DataFrame, optional
        Input dataframe. If None, loads from default path.

    Returns
    -------
    dict
        Dictionary with generated chart paths and summary statistics.
    """
    if df is None:
        df = load_cleaned_data()

    print("\n" + "#"*60)
    print("# PRICESENSE EXPLORATORY DATA ANALYSIS")
    print("#"*60)

    results = {
        'charts': [],
        'summary': {}
    }

    # Add discount bands for pivot tables
    df_with_bands = create_discount_bands(df)

    # 1. Distribution Histograms
    print("\n[1/7] Generating distribution histograms...")
    dist_files = plot_all_distributions(df, KEY_VARIABLES)
    results['charts'].extend(dist_files)

    # 2. Box Plots (single and by category)
    print("\n[2/7] Generating box plots...")
    box_files = plot_all_boxplots(df, KEY_VARIABLES, CATEGORY_COL)
    results['charts'].extend(box_files)

    # 3. Correlation Heatmaps
    print("\n[3/7] Generating correlation heatmaps...")
    corr_file = plot_correlation_heatmap(df, KEY_VARIABLES, method='pearson')
    results['charts'].append(corr_file)
    corr_file2 = plot_correlation_heatmap(df, KEY_VARIABLES, method='spearman')
    results['charts'].append(corr_file2)

    # Correlation with target variables
    corr_target1 = plot_correlation_with_target(df, 'rating', KEY_VARIABLES)
    results['charts'].append(corr_target1)
    corr_target2 = plot_correlation_with_target(df, 'discounted_price', KEY_VARIABLES)
    results['charts'].append(corr_target2)

    # 4. Category-wise Analysis
    print("\n[4/7] Generating category-wise analysis...")
    cat_count = plot_category_counts(df, CATEGORY_COL)
    results['charts'].append(cat_count)

    cat_means = plot_category_means(df, KEY_VARIABLES, CATEGORY_COL)
    results['charts'].extend(cat_means)

    # 5. Pivot Table Visualizations
    print("\n[5/7] Generating pivot table visualizations...")
    pivot_count = plot_pivot_count_heatmap(df_with_bands, CATEGORY_COL, 'discount_band')
    results['charts'].append(pivot_count)

    pivot_rating = plot_pivot_heatmap(df_with_bands, CATEGORY_COL, 'discount_band', 'rating', 'mean')
    results['charts'].append(pivot_rating)

    pivot_price = plot_pivot_heatmap(df_with_bands, CATEGORY_COL, 'discount_band', 'discounted_price', 'mean')
    results['charts'].append(pivot_price)

    pivot_discount = plot_pivot_heatmap(df_with_bands, CATEGORY_COL, 'discount_band', 'discount_percentage', 'mean')
    results['charts'].append(pivot_discount)

    # 6. Scatter Plots - Key Relationships
    print("\n[6/7] Generating scatter plots...")
    scatter1 = plot_price_vs_rating_scatter(df)
    results['charts'].append(scatter1)

    scatter2 = plot_price_vs_rating_count(df)
    results['charts'].append(scatter2)

    scatter3 = plot_discount_vs_rating(df)
    results['charts'].append(scatter3)

    # Pairplot
    pairplot = plot_pairplot(df, KEY_VARIABLES)
    results['charts'].append(pairplot)

    # 7. Additional Plots
    print("\n[7/7] Generating additional plots...")
    violin = plot_rating_distribution_by_category(df, CATEGORY_COL)
    results['charts'].append(violin)

    cum_disc = plot_cumulative_discount_distribution(df)
    results['charts'].append(cum_disc)

    savings = plot_price_savings_distribution(df)
    results['charts'].append(savings)

    # Summary
    print("\n" + "="*60)
    print("EDA COMPLETE - GENERATED CHARTS")
    print("="*60)
    for chart in results['charts']:
        print(f"  ✓ {chart.name}")

    results['summary'] = {
        'total_charts': len(results['charts']),
        'charts_dir': str(CHARTS_DIR),
        'variables_analyzed': KEY_VARIABLES
    }

    print(f"\nTotal charts generated: {len(results['charts'])}")
    print(f"Saved to: {CHARTS_DIR}")

    return results


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Run full EDA
    results = run_full_eda()

    print("\n" + "#"*60)
    print("# EDA PIPELINE COMPLETE")
    print("#"*60)