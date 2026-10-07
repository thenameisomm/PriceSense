"""
PriceSense — Data Preprocessing Module

This module implements a complete preprocessing pipeline for the Amazon product dataset.
Each transformation is documented and explainable for academic purposes.

Pipeline Steps:
1. Dataset loading
2. Dataset inspection
3. Missing-value analysis
4. Duplicate analysis (exact duplicates vs repeated product IDs)
5. Duplicate handling
6. Data-type correction
7. Price cleaning (₹1,899 → 1899)
8. Discount cleaning (64% → 64.0)
9. Rating cleaning (validation)
10. Rating-count cleaning (24,269 → 24269)
11. Category processing (hierarchical parsing)
12. Feature engineering
13. Final cleaned dataset creation
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional


# ============================================================================
# CONFIGURATION
# ============================================================================

RAW_DATA_PATH = Path("data/amazon.csv")
CLEANED_DATA_PATH = Path("outputs/cleaned_data.csv")

# Ensure output directory exists
CLEANED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)


# ============================================================================
# STEP 1: DATASET LOADING
# ============================================================================

def load_dataset(filepath: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Load the raw Amazon dataset from CSV.

    Parameters
    ----------
    filepath : Path
        Path to the raw CSV file.

    Returns
    -------
    pd.DataFrame
        Raw dataframe with original string columns.
    """
    df = pd.read_csv(filepath, dtype=str)  # Load everything as string initially
    print(f"[LOAD] Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


# ============================================================================
# STEP 2: DATASET INSPECTION
# ============================================================================

def inspect_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform comprehensive inspection of the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    Returns
    -------
    dict
        Inspection results including shape, dtypes, missing values, sample data.
    """
    inspection = {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "missing_percentage": (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
        "sample_head": df.head(3).to_dict(orient="records"),
        "sample_tail": df.tail(2).to_dict(orient="records"),
    }

    print("\n" + "="*60)
    print("DATASET INSPECTION")
    print("="*60)
    print(f"Shape: {inspection['shape']}")
    print(f"Columns: {inspection['columns']}")
    print("\nMissing Values:")
    for col, miss in inspection["missing_values"].items():
        if miss > 0:
            print(f"  {col}: {miss} ({inspection['missing_percentage'][col]}%)")
    print("\nFirst 3 rows:")
    for i, row in enumerate(inspection["sample_head"]):
        print(f"  Row {i}: {row}")

    return inspection


# ============================================================================
# STEP 3: MISSING VALUE ANALYSIS
# ============================================================================

def analyze_missing_values(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze missing values in detail.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    Returns
    -------
    dict
        Detailed missing value analysis.
    """
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)

    analysis = {
        "total_missing": int(missing.sum()),
        "columns_with_missing": missing[missing > 0].to_dict(),
        "missing_percentage": missing_pct[missing_pct > 0].to_dict(),
    }

    print("\n" + "="*60)
    print("MISSING VALUE ANALYSIS")
    print("="*60)
    print(f"Total missing values: {analysis['total_missing']}")
    for col, count in analysis["columns_with_missing"].items():
        print(f"  {col}: {count} missing ({analysis['missing_percentage'][col]}%)")

    return analysis


# ============================================================================
# STEP 4: DUPLICATE ANALYSIS
# ============================================================================

def analyze_duplicates(df: pd.DataFrame, id_column: str = "product_id") -> Dict[str, Any]:
    """
    Analyze duplicates at two levels:
    1. Exact duplicate rows (all columns identical)
    2. Repeated product IDs (same product_id, potentially different reviews/data)

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    id_column : str
        Column name for product identifier.

    Returns
    -------
    dict
        Duplicate analysis results.
    """
    # Exact duplicate rows
    exact_duplicates = df.duplicated(keep=False)
    exact_dup_count = exact_duplicates.sum()
    exact_dup_rows = df[exact_duplicates].copy()

    # Repeated product IDs
    id_duplicates = df.duplicated(subset=[id_column], keep=False)
    id_dup_count = id_duplicates.sum()
    unique_ids = df[id_column].nunique()
    total_rows = len(df)

    # Group by product_id to see how many rows per ID
    id_counts = df.groupby(id_column).size()
    repeated_ids = id_counts[id_counts > 1]
    max_rows_per_id = id_counts.max() if len(id_counts) > 0 else 0

    analysis = {
        "exact_duplicate_rows": int(exact_dup_count),
        "exact_duplicate_groups": int(exact_dup_count / 2) if exact_dup_count > 0 else 0,
        "repeated_product_ids": int(repeated_ids.shape[0]),
        "rows_with_repeated_ids": int(id_dup_count),
        "unique_product_ids": int(unique_ids),
        "total_rows": int(total_rows),
        "max_rows_per_product_id": int(max_rows_per_id),
        "id_count_distribution": id_counts.value_counts().to_dict(),
    }

    print("\n" + "="*60)
    print("DUPLICATE ANALYSIS")
    print("="*60)
    print(f"Exact duplicate rows: {analysis['exact_duplicate_rows']}")
    print(f"Repeated product IDs: {analysis['repeated_product_ids']} unique IDs")
    print(f"Rows with repeated IDs: {analysis['rows_with_repeated_ids']}")
    print(f"Unique product IDs: {analysis['unique_product_ids']} / {analysis['total_rows']} rows")
    print(f"Max rows per product ID: {analysis['max_rows_per_product_id']}")
    print(f"ID count distribution: {analysis['id_count_distribution']}")

    return analysis


# ============================================================================
# STEP 5: DUPLICATE HANDLING
# ============================================================================

def handle_duplicates(df: pd.DataFrame, id_column: str = "product_id") -> pd.DataFrame:
    """
    Handle duplicates with a clear strategy:
    - Remove EXACT duplicate rows (keep first occurrence)
    - For repeated product_ids: keep the first row (assuming it's the primary record)
      since review data is aggregated in comma-separated strings

    Rationale:
    - Exact duplicates add no information and waste computation
    - Repeated product_ids likely represent the same product with aggregated reviews
      in a single row; keeping first avoids data loss from arbitrary selection

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    id_column : str
        Column name for product identifier.

    Returns
    -------
    pd.DataFrame
        Deduplicated dataframe.
    """
    original_shape = df.shape

    # Remove exact duplicates (keep first)
    df = df.drop_duplicates(keep="first")
    after_exact = df.shape[0]

    # Remove repeated product_ids (keep first)
    df = df.drop_duplicates(subset=[id_column], keep="first")
    after_id_dedup = df.shape[0]

    removed_exact = original_shape[0] - after_exact
    removed_id = after_exact - after_id_dedup

    print("\n" + "="*60)
    print("DUPLICATE HANDLING")
    print("="*60)
    print(f"Original rows: {original_shape[0]}")
    print(f"After removing exact duplicates: {after_exact} (removed {removed_exact})")
    print(f"After removing repeated product_ids: {after_id_dedup} (removed {removed_id})")
    print(f"Final rows: {after_id_dedup}")

    return df.reset_index(drop=True)


# ============================================================================
# STEP 6-10: DATA TYPE CORRECTION & CLEANING FUNCTIONS
# ============================================================================

def clean_price(price_str: str) -> Optional[float]:
    """
    Convert price string to numeric float.

    Handles:
    - ₹ symbol: "₹1,899" → 1899.0
    - Commas: "1,099" → 1099.0
    - Already numeric strings: "399" → 399.0
    - Empty/invalid: returns NaN

    Why this approach:
    - Indian currency format uses ₹ symbol and comma as thousand separator
    - Removing non-numeric chars except decimal point handles all cases
    - Returns NaN for invalid so we can detect and handle separately

    Parameters
    ----------
    price_str : str
        Price string like "₹1,899" or "₹399".

    Returns
    -------
    float or NaN
        Clean numeric price.
    """
    if pd.isna(price_str) or price_str == "":
        return np.nan

    # Remove ₹ symbol and commas
    cleaned = str(price_str).replace("₹", "").replace(",", "").strip()

    try:
        return float(cleaned)
    except ValueError:
        return np.nan


def clean_discount(discount_str: str) -> Optional[float]:
    """
    Convert discount percentage string to numeric float.

    Handles:
    - Percentage symbol: "64%" → 64.0
    - Already numeric: "50" → 50.0
    - Decimal percentages: "12.5%" → 12.5
    - Empty/invalid: returns NaN

    Why this approach:
    - Discount column stores percentage as string with % symbol
    - Stripping % and converting to float gives 0-100 scale
    - NaN for invalid allows proper missing value handling

    Parameters
    ----------
    discount_str : str
        Discount string like "64%" or "0%".

    Returns
    -------
    float or NaN
        Clean discount percentage (0-100).
    """
    if pd.isna(discount_str) or discount_str == "":
        return np.nan

    # Remove % symbol
    cleaned = str(discount_str).replace("%", "").strip()

    try:
        return float(cleaned)
    except ValueError:
        return np.nan


def clean_rating(rating_str: str) -> Optional[float]:
    """
    Convert rating to numeric float with validation.

    Handles:
    - Valid ratings: "4.2" → 4.2
    - Invalid/out of range: returns NaN
    - Expected range: 0.0 to 5.0 (Amazon ratings)

    Why this approach:
    - Ratings should be numeric 0-5
    - Validation catches data entry errors
    - NaN for invalid allows imputation if needed

    Parameters
    ----------
    rating_str : str
        Rating string like "4.2".

    Returns
    -------
    float or NaN
        Clean rating (0-5 scale).
    """
    if pd.isna(rating_str) or rating_str == "":
        return np.nan

    try:
        rating = float(str(rating_str).strip())
        # Validate range (Amazon ratings are 0-5)
        if 0 <= rating <= 5:
            return rating
        else:
            return np.nan
    except ValueError:
        return np.nan


def clean_rating_count(count_str: str) -> Optional[float]:
    """
    Convert rating count string to numeric integer.

    Handles:
    - Comma separators: "24,269" → 24269
    - Already numeric: "100" → 100
    - Empty/invalid: returns NaN

    Why this approach:
    - Rating counts use comma as thousand separator (Indian format)
    - Removing commas and converting to int gives actual count
    - NaN for missing allows median imputation (robust to outliers)

    Parameters
    ----------
    count_str : str
        Rating count string like "24,269".

    Returns
    -------
    float or NaN
        Clean rating count as float (NaN for missing).
    """
    if pd.isna(count_str) or count_str == "":
        return np.nan

    # Remove commas
    cleaned = str(count_str).replace(",", "").strip()

    try:
        return float(cleaned)
    except ValueError:
        return np.nan


# ============================================================================
# STEP 11: CATEGORY PROCESSING
# ============================================================================

def parse_category_hierarchy(category_str: str) -> Dict[str, Optional[str]]:
    """
    Parse hierarchical category string into levels.

    Input format: "Computers&Accessories|Accessories&Peripherals|Cables&Accessories|Cables|USBCables"
    Output: Dict with main_category, sub_category, sub_sub_category, etc.

    Why this approach:
    - Categories use | as delimiter (not standard > or /)
    - Extracting levels enables grouping at different granularities
    - Main category (level 1) is most useful for aggregation

    Parameters
    ----------
    category_str : str
        Pipe-separated category hierarchy.

    Returns
    -------
    dict
        Dictionary with category levels.
    """
    if pd.isna(category_str) or category_str == "":
        return {
            "main_category": None,
            "sub_category": None,
            "level_3": None,
            "level_4": None,
            "level_5": None,
            "category_depth": 0
        }

    levels = str(category_str).split("|")

    return {
        "main_category": levels[0] if len(levels) > 0 else None,
        "sub_category": levels[1] if len(levels) > 1 else None,
        "level_3": levels[2] if len(levels) > 2 else None,
        "level_4": levels[3] if len(levels) > 3 else None,
        "level_5": levels[4] if len(levels) > 4 else None,
        "category_depth": len(levels)
    }


def process_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process category column into structured hierarchy.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with 'category' column.

    Returns
    -------
    pd.DataFrame
        Dataframe with added category hierarchy columns.
    """
    print("\n" + "="*60)
    print("CATEGORY PROCESSING")
    print("="*60)

    # Parse each category
    parsed = df["category"].apply(parse_category_hierarchy)
    cat_df = pd.DataFrame(parsed.tolist(), index=df.index)

    # Add to dataframe
    df = pd.concat([df, cat_df], axis=1)

    # Print category stats
    print(f"Unique main categories: {df['main_category'].nunique()}")
    print(f"Category depth distribution:\n{df['category_depth'].value_counts().sort_index()}")

    # Show top main categories
    top_cats = df["main_category"].value_counts().head(10)
    print(f"\nTop 10 main categories:")
    for cat, count in top_cats.items():
        print(f"  {cat}: {count}")

    return df


# ============================================================================
# STEP 12: FEATURE ENGINEERING
# ============================================================================

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features for analysis and modeling.

    Features created:
    1. price_savings = actual_price - discounted_price (absolute savings in ₹)
    2. discount_depth = discount_percentage / 100 (proportion 0-1)
    3. price_ratio = discounted_price / actual_price (proportion paid)
    4. has_high_rating = rating >= 4.0 (boolean)
    5. is_heavily_discounted = discount_percentage >= 50 (boolean)
    6. is_budget = discounted_price <= 500 (boolean, affordable)
    7. is_premium = discounted_price >= 10000 (boolean, premium)
    8. log_rating_count = log1p(rating_count) (for skewed distribution)
    9. log_discounted_price = log1p(discounted_price)
    10. log_actual_price = log1p(actual_price)

    Why these features:
    - price_savings: absolute value of discount in rupees
    - discount_depth: normalized discount for ML algorithms
    - price_ratio: alternative discount representation
    - Boolean flags: enable rule-based filtering and segmentation
    - Log transforms: normalize highly skewed price/rating_count distributions

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with cleaned numeric columns.

    Returns
    -------
    pd.DataFrame
        Dataframe with engineered features added.
    """
    print("\n" + "="*60)
    print("FEATURE ENGINEERING")
    print("="*60)

    # Price savings (absolute discount in rupees)
    df["price_savings"] = df["actual_price"] - df["discounted_price"]

    # Discount depth (proportion 0-1)
    df["discount_depth"] = df["discount_percentage"] / 100.0

    # Price ratio (what fraction of original price is paid)
    df["price_ratio"] = df["discounted_price"] / df["actual_price"]
    # Handle division by zero (shouldn't happen but safety)
    df["price_ratio"] = df["price_ratio"].replace([np.inf, -np.inf], np.nan)

    # Boolean flags
    df["has_high_rating"] = df["rating"] >= 4.0
    df["is_heavily_discounted"] = df["discount_percentage"] >= 50.0
    df["is_budget"] = df["discounted_price"] <= 500
    df["is_premium"] = df["discounted_price"] >= 10000

    # Log transforms for skewed distributions
    df["log_rating_count"] = np.log1p(df["rating_count"])
    df["log_discounted_price"] = np.log1p(df["discounted_price"])
    df["log_actual_price"] = np.log1p(df["actual_price"])

    # Review count parsed (from comma-separated review_ids)
    def count_reviews(review_id_str):
        if pd.isna(review_id_str) or review_id_str == "":
            return 0
        return len(str(review_id_str).split(","))

    df["review_count_parsed"] = df["review_id"].apply(count_reviews)

    # Average review content length (sample from first review)
    def avg_review_length(content_str):
        if pd.isna(content_str) or content_str == "":
            return 0
        reviews = str(content_str).split(",")
        lengths = [len(r.strip()) for r in reviews if r.strip()]
        return np.mean(lengths) if lengths else 0

    df["avg_review_length"] = df["review_content"].apply(avg_review_length)

    print("Engineered features:")
    feature_cols = [
        "price_savings", "discount_depth", "price_ratio",
        "has_high_rating", "is_heavily_discounted", "is_budget", "is_premium",
        "log_rating_count", "log_discounted_price", "log_actual_price",
        "review_count_parsed", "avg_review_length"
    ]
    for col in feature_cols:
        if col in df.columns:
            dtype = df[col].dtype
            if dtype == bool:
                print(f"  {col}: {df[col].sum()} True, {(~df[col]).sum()} False")
            else:
                print(f"  {col}: mean={df[col].mean():.2f}, std={df[col].std():.2f}")

    return df


# ============================================================================
# MAIN PREPROCESSING PIPELINE
# ============================================================================

def preprocess_pipeline(filepath: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Run the complete preprocessing pipeline.

    Parameters
    ----------
    filepath : Path
        Path to raw CSV file.

    Returns
    -------
    pd.DataFrame
        Fully cleaned and feature-engineered dataframe.
    """
    print("\n" + "#"*60)
    print("# PRICESENSE PREPROCESSING PIPELINE")
    print("#"*60)

    # Step 1: Load
    df = load_dataset(filepath)
    original_shape = df.shape

    # Step 2: Inspect
    inspect_dataset(df)

    # Step 3: Missing value analysis
    analyze_missing_values(df)

    # Step 4: Duplicate analysis
    analyze_duplicates(df)

    # Step 5: Handle duplicates
    df = handle_duplicates(df)

    # Step 6-10: Data type correction & cleaning
    print("\n" + "="*60)
    print("DATA TYPE CORRECTION & CLEANING")
    print("="*60)

    # Price columns
    print("Cleaning discounted_price...")
    df["discounted_price"] = df["discounted_price"].apply(clean_price)

    print("Cleaning actual_price...")
    df["actual_price"] = df["actual_price"].apply(clean_price)

    # Discount percentage
    print("Cleaning discount_percentage...")
    df["discount_percentage"] = df["discount_percentage"].apply(clean_discount)

    # Rating
    print("Cleaning rating...")
    df["rating"] = df["rating"].apply(clean_rating)

    # Rating count
    print("Cleaning rating_count...")
    df["rating_count"] = df["rating_count"].apply(clean_rating_count)

    # Check missing after cleaning
    print("\nMissing values after cleaning:")
    for col in ["discounted_price", "actual_price", "discount_percentage", "rating", "rating_count"]:
        miss = df[col].isna().sum()
        if miss > 0:
            print(f"  {col}: {miss} missing")

    # Step 11: Category processing
    df = process_categories(df)

    # Step 12: Handle missing numerical values
    print("\n" + "="*60)
    print("MISSING VALUE HANDLING")
    print("="*60)

    # For rating_count: use median (robust to outliers like 426973)
    if df["rating_count"].isna().any():
        median_rc = df["rating_count"].median()
        df["rating_count"] = df["rating_count"].fillna(median_rc)
        print(f"Filled {df['rating_count'].isna().sum()} missing rating_count with median: {median_rc:.0f}")

    # For rating: use median (robust)
    if df["rating"].isna().any():
        median_rating = df["rating"].median()
        df["rating"] = df["rating"].fillna(median_rating)
        print(f"Filled missing rating with median: {median_rating:.2f}")

    # For prices: should not be missing, but if so use median
    for price_col in ["discounted_price", "actual_price"]:
        if df[price_col].isna().any():
            median_price = df[price_col].median()
            df[price_col] = df[price_col].fillna(median_price)
            print(f"Filled missing {price_col} with median: {median_price:.2f}")

    # For discount: use median
    if df["discount_percentage"].isna().any():
        median_disc = df["discount_percentage"].median()
        df["discount_percentage"] = df["discount_percentage"].fillna(median_disc)
        print(f"Filled missing discount_percentage with median: {median_disc:.2f}")

    # Step 13: Feature engineering
    df = engineer_features(df)

    # Final verification
    print("\n" + "="*60)
    print("FINAL VERIFICATION")
    print("="*60)
    print(f"Original shape: {original_shape}")
    print(f"Cleaned shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nRemaining missing values:")
    missing_final = df.isnull().sum()
    print(missing_final[missing_final > 0].to_dict() if missing_final.sum() > 0 else "None")

    print(f"\nDuplicate check:")
    exact_dups = df.duplicated().sum()
    id_dups = df.duplicated(subset=["product_id"]).sum()
    print(f"  Exact duplicate rows: {exact_dups}")
    print(f"  Repeated product_ids: {id_dups}")

    print(f"\nSample cleaned records:")
    sample_cols = ["product_id", "product_name", "main_category", "discounted_price",
                   "actual_price", "discount_percentage", "rating", "rating_count",
                   "price_savings", "discount_depth", "has_high_rating"]
    print(df[sample_cols].head(5).to_string())

    return df


def save_cleaned_data(df: pd.DataFrame, filepath: Path = CLEANED_DATA_PATH) -> None:
    """
    Save cleaned dataset to CSV.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe.
    filepath : Path
        Output path.
    """
    df.to_csv(filepath, index=False)
    print(f"\n[SAVE] Cleaned dataset saved to: {filepath}")
    print(f"       Shape: {df.shape}")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Run the full pipeline
    cleaned_df = preprocess_pipeline()

    # Save cleaned data
    save_cleaned_data(cleaned_df)

    print("\n" + "#"*60)
    print("# PREPROCESSING COMPLETE")
    print("#"*60)