"""
PriceSense — Rule-Based Recommendation Module

This module implements a rule-based recommendation system for products.
No ML-based collaborative filtering or advanced recommendation algorithms
(per DSML syllabus constraints).

Recommendation Strategies:
1. Cluster-based: Recommend products from the same cluster
2. Price-based: Recommend similar-priced products
3. Rating-based: Recommend higher-rated alternatives
4. Category-based: Recommend within same category
5. Hybrid: Combined approach
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURATION
# ============================================================================

CLEANED_DATA_PATH = Path("outputs/cleaned_data.csv")
CLUSTER_RESULTS_PATH = Path("outputs/cluster_results.csv")


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load cleaned data and cluster results."""
    df = pd.read_csv(CLEANED_DATA_PATH)
    cluster_df = pd.read_csv(CLUSTER_RESULTS_PATH)

    # Merge cluster info into main dataframe
    if 'cluster' not in df.columns:
        cluster_map = cluster_df.set_index('product_id')['cluster']
        df['cluster'] = df['product_id'].map(cluster_map)

    return df, cluster_df


# ============================================================================
# RECOMMENDATION RULES
# ============================================================================

def recommend_by_cluster(df: pd.DataFrame, product_id: str,
                         n_recommendations: int = 5,
                         exclude_same: bool = True) -> pd.DataFrame:
    """
    Recommend products from the same cluster.

    Parameters
    ----------
    df : pd.DataFrame
        Product dataframe with cluster column.
    product_id : str
        Target product ID.
    n_recommendations : int
        Number of recommendations to return.
    exclude_same : bool
        Whether to exclude the target product itself.

    Returns
    -------
    pd.DataFrame
        Recommended products.
    """
    if product_id not in df['product_id'].values:
        return pd.DataFrame()

    target_cluster = df.loc[df['product_id'] == product_id, 'cluster'].values[0]

    candidates = df[df['cluster'] == target_cluster].copy()

    if exclude_same:
        candidates = candidates[candidates['product_id'] != product_id]

    # Sort by rating (descending) then by review count (descending)
    candidates = candidates.sort_values(['rating', 'rating_count'], ascending=[False, False])

    return candidates.head(n_recommendations)


def recommend_by_price_range(df: pd.DataFrame, product_id: str,
                             price_tolerance: float = 0.3,
                             n_recommendations: int = 5) -> pd.DataFrame:
    """
    Recommend products within similar price range.

    Parameters
    ----------
    df : pd.DataFrame
        Product dataframe.
    product_id : str
        Target product ID.
    price_tolerance : float
        Tolerance as fraction of price (e.g., 0.3 = ±30%).
    n_recommendations : int
        Number of recommendations.

    Returns
    -------
    pd.DataFrame
        Recommended products.
    """
    if product_id not in df['product_id'].values:
        return pd.DataFrame()

    target_price = df.loc[df['product_id'] == product_id, 'discounted_price'].values[0]

    min_price = target_price * (1 - price_tolerance)
    max_price = target_price * (1 + price_tolerance)

    candidates = df[
        (df['discounted_price'] >= min_price) &
        (df['discounted_price'] <= max_price) &
        (df['product_id'] != product_id)
    ].copy()

    # Sort by rating descending
    candidates = candidates.sort_values('rating', ascending=False)

    return candidates.head(n_recommendations)


def recommend_by_category(df: pd.DataFrame, product_id: str,
                          n_recommendations: int = 5) -> pd.DataFrame:
    """
    Recommend products from the same main category.

    Parameters
    ----------
    df : pd.DataFrame
        Product dataframe.
    product_id : str
        Target product ID.
    n_recommendations : int
        Number of recommendations.

    Returns
    -------
    pd.DataFrame
        Recommended products.
    """
    if product_id not in df['product_id'].values:
        return pd.DataFrame()

    target_category = df.loc[df['product_id'] == product_id, 'main_category'].values[0]

    candidates = df[
        (df['main_category'] == target_category) &
        (df['product_id'] != product_id)
    ].copy()

    # Sort by rating then review count
    candidates = candidates.sort_values(['rating', 'rating_count'], ascending=[False, False])

    return candidates.head(n_recommendations)


def recommend_better_rated(df: pd.DataFrame, product_id: str,
                           min_rating_diff: float = 0.2,
                           n_recommendations: int = 5) -> pd.DataFrame:
    """
    Recommend higher-rated alternatives in similar price range.

    Parameters
    ----------
    df : pd.DataFrame
        Product dataframe.
    product_id : str
        Target product ID.
    min_rating_diff : float
        Minimum rating improvement to recommend.
    n_recommendations : int
        Number of recommendations.

    Returns
    -------
    pd.DataFrame
        Recommended products.
    """
    if product_id not in df['product_id'].values:
        return pd.DataFrame()

    target = df[df['product_id'] == product_id].iloc[0]
    target_rating = target['rating']
    target_price = target['discounted_price']
    target_category = target['main_category']

    # Find products with better rating in similar price range (±50%)
    candidates = df[
        (df['rating'] >= target_rating + min_rating_diff) &
        (df['discounted_price'] >= target_price * 0.5) &
        (df['discounted_price'] <= target_price * 1.5) &
        (df['main_category'] == target_category) &
        (df['product_id'] != product_id)
    ].copy()

    # Sort by rating improvement potential
    candidates['rating_improvement'] = candidates['rating'] - target_rating
    candidates = candidates.sort_values(['rating_improvement', 'rating_count'], ascending=[False, False])

    return candidates.head(n_recommendations)


def recommend_budget_alternative(df: pd.DataFrame, product_id: str,
                                  max_price_ratio: float = 0.7,
                                  n_recommendations: int = 5) -> pd.DataFrame:
    """
    Recommend cheaper alternatives with decent ratings.

    Parameters
    ----------
    df : pd.DataFrame
        Product dataframe.
    product_id : str
        Target product ID.
    max_price_ratio : float
        Maximum price as fraction of target price.
    n_recommendations : int
        Number of recommendations.

    Returns
    -------
    pd.DataFrame
        Recommended products.
    """
    if product_id not in df['product_id'].values:
        return pd.DataFrame()

    target_price = df.loc[df['product_id'] == product_id, 'discounted_price'].values[0]
    target_category = df.loc[df['product_id'] == product_id, 'main_category'].values[0]

    candidates = df[
        (df['discounted_price'] <= target_price * max_price_ratio) &
        (df['rating'] >= 3.5) &  # Minimum quality threshold
        (df['main_category'] == target_category) &
        (df['product_id'] != product_id)
    ].copy()

    # Sort by best value: high rating, low price
    candidates['value_score'] = candidates['rating'] / (candidates['discounted_price'] + 1)
    candidates = candidates.sort_values('value_score', ascending=False)

    return candidates.head(n_recommendations)


def recommend_premium_upgrade(df: pd.DataFrame, product_id: str,
                               min_price_ratio: float = 1.5,
                               min_rating: float = 4.0,
                               n_recommendations: int = 5) -> pd.DataFrame:
    """
    Recommend premium upgrades (higher price, better rating).

    Parameters
    ----------
    df : pd.DataFrame
        Product dataframe.
    product_id : str
        Target product ID.
    min_price_ratio : float
        Minimum price as multiple of target price.
    min_rating : float
        Minimum rating for premium products.
    n_recommendations : int
        Number of recommendations.

    Returns
    -------
    pd.DataFrame
        Recommended products.
    """
    if product_id not in df['product_id'].values:
        return pd.DataFrame()

    target_price = df.loc[df['product_id'] == product_id, 'discounted_price'].values[0]
    target_category = df.loc[df['product_id'] == product_id, 'main_category'].values[0]

    candidates = df[
        (df['discounted_price'] >= target_price * min_price_ratio) &
        (df['rating'] >= min_rating) &
        (df['main_category'] == target_category) &
        (df['product_id'] != product_id)
    ].copy()

    # Sort by rating then price
    candidates = candidates.sort_values(['rating', 'discounted_price'], ascending=[False, True])

    return candidates.head(n_recommendations)


# ============================================================================
# USER PREFERENCE-BASED RECOMMENDATION (Phase 5)
# ============================================================================

def recommend_products(
    df: pd.DataFrame,
    category: Optional[str] = None,
    max_price: Optional[float] = None,
    min_rating: Optional[float] = None,
    min_discount: Optional[float] = None,
    n_recommendations: int = 10,
    use_cluster_boost: bool = True
) -> pd.DataFrame:
    """
    Recommend products based on user preferences.

    This is a simple, transparent, rule-based recommendation system that:
    1. Validates user inputs
    2. Filters the cleaned dataset by category, price, rating, discount
    3. Uses K-Means cluster information to boost relevant products
    4. Ranks remaining products using interpretable criteria
    5. Returns top N products with key fields

    Parameters
    ----------
    df : pd.DataFrame
        Product dataframe with cluster column.
    category : str, optional
        Main category to filter by (e.g., 'Electronics', 'Home&Kitchen').
    max_price : float, optional
        Maximum discounted price (₹).
    min_rating : float, optional
        Minimum rating (0-5 scale).
    min_discount : float, optional
        Minimum discount percentage (0-100).
    n_recommendations : int, default 10
        Number of products to return.
    use_cluster_boost : bool, default True
        Whether to prioritize products from clusters with better value profiles.

    Returns
    -------
    pd.DataFrame
        Recommended products with columns:
        product_id, product_name, main_category, actual_price,
        discounted_price, discount_percentage, rating, rating_count, cluster
    """
    # ---- Ensure cluster column exists ----
    if 'cluster' not in df.columns:
        # Try to load from cluster results
        try:
            cluster_df = pd.read_csv(CLUSTER_RESULTS_PATH)
            cluster_map = cluster_df.set_index('product_id')['cluster']
            df = df.copy()
            df['cluster'] = df['product_id'].map(cluster_map)
        except Exception:
            pass  # Continue without cluster if not available

    # ---- Input Validation ----
    valid_categories = df['main_category'].unique().tolist()

    if category is not None:
        if category not in valid_categories:
            raise ValueError(f"Invalid category: '{category}'. Valid categories: {valid_categories}")

    if max_price is not None:
        if not isinstance(max_price, (int, float)) or max_price <= 0:
            raise ValueError("max_price must be a positive number")
        if max_price > df['discounted_price'].max():
            print(f"Note: max_price (₹{max_price}) exceeds maximum in dataset (₹{df['discounted_price'].max():.0f})")

    if min_rating is not None:
        if not isinstance(min_rating, (int, float)) or min_rating < 0 or min_rating > 5:
            raise ValueError("min_rating must be between 0 and 5")

    if min_discount is not None:
        if not isinstance(min_discount, (int, float)) or min_discount < 0 or min_discount > 100:
            raise ValueError("min_discount must be between 0 and 100")

    if not isinstance(n_recommendations, int) or n_recommendations <= 0:
        raise ValueError("n_recommendations must be a positive integer")

    # ---- Filtering ----
    candidates = df.copy()
    filters_applied = []

    # 1. Category filter
    if category is not None:
        candidates = candidates[candidates['main_category'] == category]
        filters_applied.append(f"Category = {category}")

    # 2. Maximum price filter
    if max_price is not None:
        candidates = candidates[candidates['discounted_price'] <= max_price]
        filters_applied.append(f"Price ≤ ₹{max_price:,.0f}")

    # 3. Minimum rating filter
    if min_rating is not None:
        candidates = candidates[candidates['rating'] >= min_rating]
        filters_applied.append(f"Rating ≥ {min_rating}")

    # 4. Minimum discount filter
    if min_discount is not None:
        candidates = candidates[candidates['discount_percentage'] >= min_discount]
        filters_applied.append(f"Discount ≥ {min_discount}%")

    print(f"\nFilters applied: {'; '.join(filters_applied) if filters_applied else 'None'}")
    print(f"Products after filtering: {len(candidates)}")

    if len(candidates) == 0:
        print("No products match the criteria. Try relaxing filters.")
        return pd.DataFrame()

    # ---- Ranking ----
    # Primary: Rating (higher is better)
    # Secondary: Review count (more reviews = more trustworthy)
    # Tertiary: Discount percentage (higher discount = better value)
    # Quaternary: Cluster boost (prefer clusters with good value)

    candidates = candidates.copy()

    # Calculate a composite score for ranking
    # Normalize each component to 0-1 range
    rating_norm = (candidates['rating'] - candidates['rating'].min()) / (candidates['rating'].max() - candidates['rating'].min() + 1e-6)

    # Log-normalize review count (handles extreme skew)
    log_reviews = np.log1p(candidates['rating_count'])
    review_norm = (log_reviews - log_reviews.min()) / (log_reviews.max() - log_reviews.min() + 1e-6)

    # Normalize discount percentage
    discount_norm = (candidates['discount_percentage'] - candidates['discount_percentage'].min()) / (candidates['discount_percentage'].max() - candidates['discount_percentage'].min() + 1e-6)

    # Cluster boost: prefer clusters with better value profile
    # Cluster 0 (Mainstream): good ratings, high reviews, moderate price
    # Cluster 1 (Budget): low price, high discount, lower ratings
    # Cluster 2 (Premium): high price, low discount, good ratings
    cluster_boost_map = {0: 0.1, 1: 0.05, 2: 0.0}  # Small boost for value-oriented clusters

    if use_cluster_boost and 'cluster' in candidates.columns:
        cluster_boost = candidates['cluster'].map(cluster_boost_map).fillna(0)
    else:
        cluster_boost = 0

    # Composite score: weighted sum
    # Weights: rating=0.4, reviews=0.3, discount=0.2, cluster_boost=0.1
    candidates['composite_score'] = (
        0.4 * rating_norm +
        0.3 * review_norm +
        0.2 * discount_norm +
        0.1 * cluster_boost
    )

    # Sort by composite score descending
    candidates = candidates.sort_values('composite_score', ascending=False)

    # Select output columns
    output_columns = [
        'product_id', 'product_name', 'main_category',
        'actual_price', 'discounted_price', 'discount_percentage',
        'rating', 'rating_count', 'cluster'
    ]
    available_columns = [c for c in output_columns if c in candidates.columns]

    result = candidates[available_columns].head(n_recommendations).copy()
    result = result.reset_index(drop=True)
    result.index = result.index + 1  # 1-based ranking

    print(f"Returning top {len(result)} recommendations")

    return result


def recommend_products_interactive(
    df: pd.DataFrame,
    category: Optional[str] = None,
    max_price: Optional[float] = None,
    min_rating: Optional[float] = None,
    min_discount: Optional[float] = None,
    n_recommendations: int = 10
) -> None:
    """
    Interactive version that prints formatted recommendations.
    """
    print("\n" + "="*70)
    print("PRODUCT RECOMMENDATIONS BASED ON USER PREFERENCES")
    print("="*70)

    # Show what filters are being applied
    print("\nUser Preferences:")
    if category: print(f"  Category: {category}")
    if max_price: print(f"  Max Price: ₹{max_price:,.0f}")
    if min_rating: print(f"  Min Rating: {min_rating}")
    if min_discount: print(f"  Min Discount: {min_discount}%")
    if not any([category, max_price, min_rating, min_discount]):
        print("  (No filters - showing top rated products)")

    results = recommend_products(
        df, category, max_price, min_rating, min_discount, n_recommendations
    )

    if len(results) == 0:
        print("\nNo products found matching criteria.")
        return

    print(f"\n{'Rank':<5} {'Product Name':<45} {'Category':<20} {'Price':<10} {'Disc%':<6} {'Rating':<7} {'Reviews':<10} {'Cluster'}")
    print("-" * 120)

    for idx, row in results.iterrows():
        name = row['product_name'][:42] + '..' if len(row['product_name']) > 44 else row['product_name']
        cat = row['main_category'][:18] + '..' if len(row['main_category']) > 20 else row['main_category']
        print(f"{idx:<5} {name:<45} {cat:<20} ₹{row['discounted_price']:<9,.0f} {row['discount_percentage']:<6.1f} "
              f"{row['rating']:<7.1f} {row['rating_count']:<10,.0f} {row.get('cluster', 'N/A')}")


# ============================================================================
# HYBRID RECOMMENDATION ENGINE (Product-to-Product)
# ============================================================================

def get_hybrid_recommendations(df: pd.DataFrame, product_id: str,
                                n_recommendations: int = 5) -> Dict[str, pd.DataFrame]:
    """
    Generate recommendations using multiple strategies.

    Returns
    -------
    dict
        Strategy name -> recommended products DataFrame.
    """
    recommendations = {}

    # Strategy 1: Same cluster (similar products)
    recommendations['cluster_based'] = recommend_by_cluster(df, product_id, n_recommendations)

    # Strategy 2: Same category, better rated
    recommendations['category_better_rated'] = recommend_better_rated(df, product_id, n_recommendations=n_recommendations)

    # Strategy 3: Budget alternatives
    recommendations['budget_alternative'] = recommend_budget_alternative(df, product_id, n_recommendations=n_recommendations)

    # Strategy 4: Premium upgrades
    recommendations['premium_upgrade'] = recommend_premium_upgrade(df, product_id, n_recommendations=n_recommendations)

    # Strategy 5: Similar price range
    recommendations['price_range'] = recommend_by_price_range(df, product_id, n_recommendations=n_recommendations)

    return recommendations


def format_recommendations(recommendations: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Combine recommendations from different strategies into a single DataFrame.

    Parameters
    ----------
    recommendations : dict
        Strategy name -> DataFrame mapping.

    Returns
    -------
    pd.DataFrame
        Combined recommendations with strategy column.
    """
    combined = []

    for strategy, recs in recommendations.items():
        if len(recs) > 0:
            recs = recs.copy()
            recs['recommendation_strategy'] = strategy
            combined.append(recs)

    if combined:
        result = pd.concat(combined, ignore_index=True)
        # Remove duplicates, keep first occurrence (highest priority strategy)
        result = result.drop_duplicates(subset='product_id', keep='first')
        return result
    else:
        return pd.DataFrame()


def get_product_details(df: pd.DataFrame, product_id: str) -> Dict:
    """Get details of a product for display."""
    if product_id not in df['product_id'].values:
        return {}

    row = df[df['product_id'] == product_id].iloc[0]
    return {
        'product_id': row['product_id'],
        'product_name': row['product_name'][:80] + '...' if len(row['product_name']) > 80 else row['product_name'],
        'main_category': row['main_category'],
        'discounted_price': row['discounted_price'],
        'actual_price': row['actual_price'],
        'discount_percentage': row['discount_percentage'],
        'rating': row['rating'],
        'rating_count': row['rating_count'],
        'cluster': row.get('cluster', 'N/A')
    }


def run_recommendation_demo(df: pd.DataFrame, n_samples: int = 3) -> None:
    """Run demonstration of recommendation system."""
    print("\n" + "="*60)
    print("RULE-BASED RECOMMENDATION DEMO")
    print("="*60)

    # Get sample products from each cluster
    for cluster_id in sorted(df['cluster'].unique()):
        cluster_products = df[df['cluster'] == cluster_id]
        if len(cluster_products) == 0:
            continue

        # Pick a random product from this cluster
        sample = cluster_products.sample(1, random_state=42)
        product_id = sample['product_id'].values[0]

        print(f"\n{'='*60}")
        print(f"TARGET PRODUCT (Cluster {cluster_id})")
        print(f"{'='*60}")

        details = get_product_details(df, product_id)
        for k, v in details.items():
            print(f"  {k}: {v}")

        # Get hybrid recommendations
        recs = get_hybrid_recommendations(df, product_id, n_recommendations=3)

        for strategy, rec_df in recs.items():
            if len(rec_df) > 0:
                print(f"\n  [{strategy.replace('_', ' ').title()}]")
                for _, r in rec_df.iterrows():
                    print(f"    • {r['product_name'][:60]}... | ₹{r['discounted_price']:.0f} | ⭐{r['rating']:.1f} | {r['main_category']}")


# ============================================================================
# SAVE RECOMMENDATIONS FOR ALL PRODUCTS
# ============================================================================

def generate_all_recommendations(df: pd.DataFrame, n_per_strategy: int = 3) -> pd.DataFrame:
    """
    Generate top recommendations for all products.
    Useful for pre-computing recommendations for the GUI.

    Returns
    -------
    pd.DataFrame
        All recommendations with product_id, recommended_product_id, strategy.
    """
    all_recs = []

    for _, row in df.iterrows():
        product_id = row['product_id']
        recs = get_hybrid_recommendations(df, product_id, n_recommendations=n_per_strategy)

        for strategy, rec_df in recs.items():
            for _, r in rec_df.iterrows():
                all_recs.append({
                    'product_id': product_id,
                    'recommended_product_id': r['product_id'],
                    'strategy': strategy,
                    'recommended_price': r['discounted_price'],
                    'recommended_rating': r['rating'],
                    'recommended_category': r['main_category']
                })

    return pd.DataFrame(all_recs)


def save_recommendations(df: pd.DataFrame, output_path: Path = Path("outputs/recommendations.csv")) -> None:
    """Save all recommendations to CSV."""
    recs_df = generate_all_recommendations(df)
    recs_df.to_csv(output_path, index=False)
    print(f"[SAVE] All recommendations: {output_path} ({len(recs_df)} rows)")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    df, _ = load_data()

    # Run demo
    run_recommendation_demo(df)

    # Save all recommendations
    save_recommendations(df)

    print("\nRecommendation module ready for use.")