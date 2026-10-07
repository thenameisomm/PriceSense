"""
PriceSense — K-Means Clustering Module

This module implements K-Means clustering for product segmentation.
Only K-Means is used as the ML algorithm (per DSML syllabus).

Pipeline:
1. Feature selection and justification
2. Missing value handling
3. Feature scaling (StandardScaler)
4. Elbow Method (K=2 to 10)
5. K-Means training with optimal K
6. Cluster assignment
7. Cluster analysis and interpretation
8. Visualization with PCA
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURATION
# ============================================================================

CLEANED_DATA_PATH = Path("outputs/cleaned_data.csv")
CLUSTER_RESULTS_PATH = Path("outputs/cluster_results.csv")
CHARTS_DIR = Path("outputs/charts")
MODELS_DIR = Path("models")

# Ensure directories exist
CHARTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# FEATURE SELECTION AND JUSTIFICATION
# ============================================================================

def get_clustering_features() -> Dict[str, str]:
    """
    Define features for clustering with justification.

    Returns
    -------
    dict
        Feature name -> justification mapping.
    """
    return {
        "discounted_price": "Price customer actually pays - primary purchase decision factor",
        "discount_percentage": "Discount depth - indicates pricing strategy and value perception",
        "rating": "Product quality/satisfaction signal - key trust indicator",
        "rating_count": "Review volume - proxy for popularity and market presence",
        "log_rating_count": "Log-transformed rating count - handles extreme skew (max 426K)",
        "log_discounted_price": "Log-transformed price - handles right-skew (max ₹78K)"
    }


def select_features_for_clustering(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Select and prepare features for clustering.

    Feature Selection Rationale:
    - discounted_price: What customer pays (more relevant than actual_price)
    - discount_percentage: Independent pricing strategy dimension
    - rating: Quality dimension
    - log_rating_count: Popularity dimension (log-transformed for skew)
    - log_discounted_price: Alternative price representation (log scale)

    We use log transforms for heavily skewed variables instead of raw values.
    We exclude actual_price (highly correlated with discounted_price, r=0.96).
    We exclude price_savings, discount_depth, price_ratio (derived from above).
    We exclude review_count_parsed, avg_review_length (redundant with rating/rating_count).
    We exclude category_depth (not a product attribute per se).

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe.

    Returns
    -------
    tuple
        (feature_matrix, feature_names)
    """
    feature_justifications = get_clustering_features()
    selected_features = list(feature_justifications.keys())

    print("FEATURE SELECTION FOR CLUSTERING")
    print("=" * 60)
    for feat, justification in feature_justifications.items():
        print(f"  ✓ {feat}: {justification}")

    # Verify all features exist
    missing = [f for f in selected_features if f not in df.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")

    X = df[selected_features].copy()

    # Check for missing values
    missing_counts = X.isna().sum()
    if missing_counts.any():
        print(f"\nMissing values found: {missing_counts[missing_counts > 0].to_dict()}")
        # Fill with median (robust)
        X = X.fillna(X.median())
        print("Filled with median values.")
    else:
        print("\nNo missing values in selected features.")

    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Features: {selected_features}")

    return X, selected_features


# ============================================================================
# SCALING
# ============================================================================

def scale_features(X: pd.DataFrame) -> Tuple[np.ndarray, StandardScaler]:
    """
    Scale features using StandardScaler (zero mean, unit variance).

    Why StandardScaler:
    - K-Means uses Euclidean distance, which is scale-sensitive
    - Features have vastly different scales:
      * discounted_price: 39 - 77,990
      * discount_percentage: 0 - 94
      * rating: 2 - 5
      * log_rating_count: 1.1 - 12.96
      * log_discounted_price: 3.7 - 11.3
    - StandardScaler makes all features contribute equally

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.

    Returns
    -------
    tuple
        (scaled_array, fitted_scaler)
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("\nFEATURE SCALING (StandardScaler)")
    print("=" * 60)
    print("Feature means (after scaling):", X_scaled.mean(axis=0).round(4))
    print("Feature stds (after scaling):", X_scaled.std(axis=0).round(4))

    return X_scaled, scaler


# ============================================================================
# ELBOW METHOD
# ============================================================================

def compute_elbow_method(X_scaled: np.ndarray, k_range: range = range(2, 11),
                          random_state: int = 42) -> Dict[int, float]:
    """
    Compute inertia (within-cluster sum of squares) for each K.

    Parameters
    ----------
    X_scaled : np.ndarray
        Scaled feature matrix.
    k_range : range
        Range of K values to test.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        K -> inertia mapping.
    """
    print("\nELBOW METHOD")
    print("=" * 60)

    inertias = {}
    silhouette_scores = {}

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        kmeans.fit(X_scaled)
        inertias[k] = kmeans.inertia_
        silhouette_scores[k] = silhouette_score(X_scaled, kmeans.labels_)
        print(f"  K={k}: Inertia={inertias[k]:.2f}, Silhouette={silhouette_scores[k]:.4f}")

    return inertias, silhouette_scores


def plot_elbow_method(inertias: Dict[int, float], silhouette_scores: Dict[int, float] = None) -> Path:
    """
    Plot Elbow Method chart with inertia and silhouette scores.

    Parameters
    ----------
    inertias : dict
        K -> inertia mapping.
    silhouette_scores : dict, optional
        K -> silhouette score mapping.

    Returns
    -------
    Path
        Saved chart filepath.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Inertia plot (Elbow)
    ks = list(inertias.keys())
    inertia_vals = list(inertias.values())

    ax1.plot(ks, inertia_vals, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Clusters (K)', fontsize=12)
    ax1.set_ylabel('Inertia (Within-Cluster SSE)', fontsize=12)
    ax1.set_title('Elbow Method: Inertia vs K', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(ks)

    # Mark the "elbow" point if detectable
    # Calculate rate of change
    if len(ks) >= 3:
        diffs = np.diff(inertia_vals)
        diffs2 = np.diff(diffs)
        # Elbow often where second derivative is maximum (rate of decrease slows)
        elbow_idx = np.argmax(diffs2) + 1 if len(diffs2) > 0 else 1
        ax1.annotate(f'Suggested K={ks[elbow_idx]}',
                     xy=(ks[elbow_idx], inertia_vals[elbow_idx]),
                     xytext=(ks[elbow_idx] + 1, inertia_vals[elbow_idx] + (max(inertia_vals) - min(inertia_vals)) * 0.1),
                     arrowprops=dict(arrowstyle='->', color='red'),
                     fontsize=11, color='red', fontweight='bold')

    # Silhouette plot
    if silhouette_scores:
        sil_vals = [silhouette_scores[k] for k in ks]
        ax2.plot(ks, sil_vals, 'ro-', linewidth=2, markersize=8)
        ax2.set_xlabel('Number of Clusters (K)', fontsize=12)
        ax2.set_ylabel('Silhouette Score', fontsize=12)
        ax2.set_title('Silhouette Analysis', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(ks)

        # Mark best silhouette
        best_k = max(silhouette_scores, key=silhouette_scores.get)
        ax2.annotate(f'Best K={best_k} (score={silhouette_scores[best_k]:.3f})',
                     xy=(best_k, silhouette_scores[best_k]),
                     xytext=(best_k + 1, silhouette_scores[best_k] - 0.02),
                     arrowprops=dict(arrowstyle='->', color='red'),
                     fontsize=11, color='red', fontweight='bold')

    plt.suptitle('K-Means Cluster Selection', fontsize=16, fontweight='bold')
    plt.tight_layout()

    filename = "elbow_method.png"
    filepath = CHARTS_DIR / filename
    fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[SAVE] Elbow method chart: {filepath}")

    return filepath


def select_optimal_k(inertias: Dict[int, float], silhouette_scores: Dict[int, float] = None) -> int:
    """
    Select optimal K based on elbow method and silhouette analysis.

    Strategy:
    1. Primary: Elbow point (where inertia decrease slows)
    2. Secondary: Highest silhouette score
    3. Practical: Prefer simpler models (lower K) if similar performance

    Parameters
    ----------
    inertias : dict
        K -> inertia mapping.
    silhouette_scores : dict, optional
        K -> silhouette score mapping.

    Returns
    -------
    int
        Optimal K value.
    """
    ks = list(inertias.keys())
    inertia_vals = list(inertias.values())

    # Method 1: Elbow detection using knee/elbow heuristic
    # Find point where marginal improvement drops significantly
    if len(ks) >= 3:
        diffs = np.diff(inertia_vals)
        diffs2 = np.diff(diffs)
        elbow_idx = np.argmax(diffs2) + 1
        elbow_k = ks[elbow_idx]
    else:
        elbow_k = ks[0]

    # Method 2: Best silhouette
    best_sil_k = max(silhouette_scores, key=silhouette_scores.get) if silhouette_scores else elbow_k

    # Decision logic
    print("\nOPTIMAL K SELECTION")
    print("=" * 60)
    print(f"Elbow method suggests: K={elbow_k}")
    if silhouette_scores:
        print(f"Silhouette analysis suggests: K={best_sil_k} (score={silhouette_scores[best_sil_k]:.4f})")

    # If they agree, use that. If not, prefer elbow (more interpretable) but consider silhouette
    if elbow_k == best_sil_k:
        optimal_k = elbow_k
        print(f"Both methods agree: K={optimal_k}")
    else:
        # Prefer elbow but check if silhouette is much better at another K
        sil_diff = silhouette_scores[best_sil_k] - silhouette_scores.get(elbow_k, 0)
        if sil_diff > 0.05:  # Significant improvement
            optimal_k = best_sil_k
            print(f"Silhouette significantly better at K={best_sil_k}, choosing it.")
        else:
            optimal_k = elbow_k
            print(f"Elbow method preferred (simpler model), choosing K={optimal_k}")

    print(f"\n✓ SELECTED OPTIMAL K = {optimal_k}")

    return optimal_k


# ============================================================================
# K-MEANS TRAINING
# ============================================================================

def train_kmeans(X_scaled: np.ndarray, k: int, random_state: int = 42) -> KMeans:
    """
    Train K-Means with specified K.

    Parameters
    ----------
    X_scaled : np.ndarray
        Scaled feature matrix.
    k : int
        Number of clusters.
    random_state : int
        Random seed.

    Returns
    -------
    KMeans
        Fitted KMeans model.
    """
    print(f"\nTRAINING K-MEANS (K={k})")
    print("=" * 60)

    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=20, max_iter=300)
    kmeans.fit(X_scaled)

    print(f"Inertia: {kmeans.inertia_:.2f}")
    print(f"Silhouette Score: {silhouette_score(X_scaled, kmeans.labels_):.4f}")
    print(f"Cluster sizes: {np.bincount(kmeans.labels_)}")

    return kmeans


def assign_clusters(df: pd.DataFrame, kmeans: KMeans, feature_names: List[str]) -> pd.DataFrame:
    """
    Assign cluster labels to dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Original dataframe.
    kmeans : KMeans
        Fitted KMeans model.
    feature_names : list
        Names of features used for clustering.

    Returns
    -------
    pd.DataFrame
        Dataframe with cluster column added.
    """
    df_result = df.copy()
    df_result['cluster'] = kmeans.labels_

    print("\nCLUSTER ASSIGNMENT")
    print("=" * 60)
    cluster_counts = df_result['cluster'].value_counts().sort_index()
    for cluster_id, count in cluster_counts.items():
        print(f"  Cluster {cluster_id}: {count} products ({count/len(df_result)*100:.1f}%)")

    return df_result


# ============================================================================
# CLUSTER ANALYSIS
# ============================================================================

def analyze_clusters(df: pd.DataFrame, key_vars: List[str] = None) -> pd.DataFrame:
    """
    Calculate cluster-wise statistics.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with cluster column.
    key_vars : list, optional
        Variables to analyze. Defaults to key numerical variables.

    Returns
    -------
    pd.DataFrame
        Cluster statistics.
    """
    if key_vars is None:
        key_vars = [
            "actual_price", "discounted_price", "discount_percentage",
            "rating", "rating_count", "price_savings"
        ]

    valid_vars = [v for v in key_vars if v in df.columns]

    print("\nCLUSTER ANALYSIS")
    print("=" * 60)

    # Overall stats
    cluster_stats = df.groupby('cluster')[valid_vars].agg([
        'count', 'mean', 'median', 'std', 'min', 'max'
    ]).round(2)

    # Print summary
    for cluster_id in sorted(df['cluster'].unique()):
        subset = df[df['cluster'] == cluster_id]
        print(f"\nCluster {cluster_id} (n={len(subset)}):")
        for var in valid_vars:
            mean_val = subset[var].mean()
            median_val = subset[var].median()
            print(f"  {var}: mean={mean_val:.2f}, median={median_val:.2f}")

    return cluster_stats


def interpret_clusters(df: pd.DataFrame, feature_names: List[str] = None) -> Dict[int, Dict]:
    """
    Interpret clusters based on statistics.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with cluster column.
    feature_names : list
        Features used for clustering.

    Returns
    -------
    dict
        Cluster interpretations.
    """
    if feature_names is None:
        feature_names = ["discounted_price", "discount_percentage", "rating", "log_rating_count", "log_discounted_price"]

    valid_features = [f for f in feature_names if f in df.columns]

    print("\nCLUSTER INTERPRETATION")
    print("=" * 60)

    interpretations = {}

    for cluster_id in sorted(df['cluster'].unique()):
        subset = df[df['cluster'] == cluster_id]
        n = len(subset)

        # Calculate mean for each feature
        means = {f: subset[f].mean() for f in valid_features}

        # Compare to global means
        global_means = {f: df[f].mean() for f in valid_features}

        # Determine characteristics
        characteristics = []
        for f in valid_features:
            diff_pct = (means[f] - global_means[f]) / global_means[f] * 100
            if abs(diff_pct) > 20:  # Significant difference
                direction = "higher" if diff_pct > 0 else "lower"
                characteristics.append(f"{f}: {direction} than average ({diff_pct:+.1f}%)")

        # Generate label based on characteristics
        label = generate_cluster_label(means, global_means)

        interpretations[cluster_id] = {
            'size': n,
            'percentage': n / len(df) * 100,
            'means': means,
            'characteristics': characteristics,
            'label': label
        }

        print(f"\nCluster {cluster_id} ({label}) - {n} products ({n/len(df)*100:.1f}%):")
        for f in valid_features:
            print(f"  {f}: {means[f]:.2f} (global: {global_means[f]:.2f})")
        if characteristics:
            for c in characteristics:
                print(f"  → {c}")
        else:
            print("  → Similar to overall average")

    return interpretations


def generate_cluster_label(cluster_means: Dict, global_means: Dict) -> str:
    """
    Generate descriptive label for cluster based on feature means.

    Parameters
    ----------
    cluster_means : dict
        Mean values for this cluster.
    global_means : dict
        Global mean values.

    Returns
    -------
    str
        Descriptive label.
    """
    price_level = ""
    discount_level = ""
    quality_level = ""
    popularity_level = ""

    # Price level (using log_discounted_price or discounted_price)
    if 'log_discounted_price' in cluster_means:
        price_diff = (cluster_means['log_discounted_price'] - global_means['log_discounted_price']) / global_means['log_discounted_price']
    else:
        price_diff = (cluster_means['discounted_price'] - global_means['discounted_price']) / global_means['discounted_price']

    if price_diff > 0.3:
        price_level = "High-Price"
    elif price_diff < -0.3:
        price_level = "Low-Price"
    else:
        price_level = "Mid-Price"

    # Discount level
    if 'discount_percentage' in cluster_means:
        disc_diff = (cluster_means['discount_percentage'] - global_means['discount_percentage']) / global_means['discount_percentage']
        if disc_diff > 0.3:
            discount_level = "High-Discount"
        elif disc_diff < -0.3:
            discount_level = "Low-Discount"
        else:
            discount_level = "Mid-Discount"

    # Quality level
    if 'rating' in cluster_means:
        rating_diff = cluster_means['rating'] - global_means['rating']
        if rating_diff > 0.15:
            quality_level = "High-Rating"
        elif rating_diff < -0.15:
            quality_level = "Low-Rating"
        else:
            quality_level = "Mid-Rating"

    # Popularity level
    if 'log_rating_count' in cluster_means:
        pop_diff = (cluster_means['log_rating_count'] - global_means['log_rating_count']) / global_means['log_rating_count']
        if pop_diff > 0.3:
            popularity_level = "Popular"
        elif pop_diff < -0.3:
            popularity_level = "Niche"
        else:
            popularity_level = "Moderate"

    # Combine into label
    parts = [price_level, discount_level, quality_level, popularity_level]
    # Only include distinctive ones
    distinctive = [p for p in parts if not p.startswith("Mid-") and not p.startswith("Moderate")]
    if distinctive:
        return " / ".join(distinctive)
    else:
        return "Average"


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_clusters_pca(X_scaled: np.ndarray, labels: np.ndarray, df: pd.DataFrame = None,
                       feature_names: List[str] = None) -> Path:
    """
    Visualize clusters using PCA (2D).

    PCA is used ONLY for visualization/dimensionality reduction,
    not as the primary ML algorithm.

    Parameters
    ----------
    X_scaled : np.ndarray
        Scaled feature matrix.
    labels : np.ndarray
        Cluster labels.
    df : pd.DataFrame, optional
        Original dataframe for hover info.
    feature_names : list
        Feature names for PCA component interpretation.

    Returns
    -------
    Path
        Saved chart filepath.
    """
    print("\nPCA VISUALIZATION")
    print("=" * 60)

    # Apply PCA
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    explained_var = pca.explained_variance_ratio_
    print(f"PCA Explained Variance: PC1={explained_var[0]:.3f}, PC2={explained_var[1]:.3f}")
    print(f"Total: {explained_var.sum():.3f}")

    # Component interpretation
    if feature_names:
        components_df = pd.DataFrame(
            pca.components_.T,
            index=feature_names,
            columns=['PC1', 'PC2']
        )
        print("\nPCA Component Loadings:")
        print(components_df.round(3))

    # Plot
    fig, ax = plt.subplots(figsize=(12, 9))

    unique_labels = np.unique(labels)
    colors = plt.cm.Set2(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=[colors[i]], label=f'Cluster {label} ({mask.sum()})',
                   alpha=0.7, s=50, edgecolors='white', linewidth=0.5)

    ax.set_xlabel(f'Principal Component 1 ({explained_var[0]*100:.1f}% variance)', fontsize=12)
    ax.set_ylabel(f'Principal Component 2 ({explained_var[1]*100:.1f}% variance)', fontsize=12)
    ax.set_title('K-Means Clusters Visualized via PCA (2D)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    filename = "clusters_pca.png"
    filepath = CHARTS_DIR / filename
    fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[SAVE] PCA cluster visualization: {filepath}")

    return filepath


def plot_cluster_profiles(df: pd.DataFrame, feature_names: List[str] = None) -> Path:
    """
    Plot cluster profiles as radar/spider chart or grouped bar chart.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with cluster column.
    feature_names : list
        Features to plot.

    Returns
    -------
    Path
        Saved chart filepath.
    """
    if feature_names is None:
        feature_names = ["discounted_price", "discount_percentage", "rating", "log_rating_count"]

    valid_features = [f for f in feature_names if f in df.columns]

    # Normalize features to 0-1 for comparison
    profile_data = {}
    for f in valid_features:
        col_data = df[f]
        min_val, max_val = col_data.min(), col_data.max()
        if max_val > min_val:
            df[f'{f}_norm'] = (col_data - min_val) / (max_val - min_val)
        else:
            df[f'{f}_norm'] = 0.5

    norm_features = [f'{f}_norm' for f in valid_features]

    # Calculate mean normalized values per cluster
    cluster_profiles = df.groupby('cluster')[norm_features].mean()

    # Plot as grouped bar chart
    fig, ax = plt.subplots(figsize=(12, 7))

    x = np.arange(len(valid_features))
    width = 0.8 / len(cluster_profiles)
    colors = plt.cm.Set2(np.linspace(0, 1, len(cluster_profiles)))

    for i, (cluster_id, row) in enumerate(cluster_profiles.iterrows()):
        offset = (i - len(cluster_profiles)/2 + 0.5) * width
        bars = ax.bar(x + offset, row.values, width, label=f'Cluster {cluster_id}',
                      color=colors[i], alpha=0.8, edgecolor='white')

        # Add value labels
        for j, (bar, val) in enumerate(zip(bars, row.values)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Features (Normalized 0-1)', fontsize=12)
    ax.set_ylabel('Normalized Mean Value', fontsize=12)
    ax.set_title('Cluster Profiles (Normalized Feature Means)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f.replace('_norm', '').replace('_', ' ').title() for f in norm_features], fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.15)

    plt.tight_layout()

    filename = "cluster_profiles.png"
    filepath = CHARTS_DIR / filename
    fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[SAVE] Cluster profiles chart: {filepath}")

    return filepath


def plot_cluster_sizes(df: pd.DataFrame) -> Path:
    """Plot cluster size distribution."""
    cluster_counts = df['cluster'].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, len(cluster_counts)))
    bars = ax.bar(cluster_counts.index.astype(str), cluster_counts.values,
                  color=colors, alpha=0.8, edgecolor='white')

    for bar, val in zip(bars, cluster_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{val}\n({val/len(df)*100:.1f}%)', ha='center', va='bottom', fontsize=11)

    ax.set_xlabel('Cluster', fontsize=12)
    ax.set_ylabel('Number of Products', fontsize=12)
    ax.set_title('Cluster Size Distribution', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    filename = "cluster_sizes.png"
    filepath = CHARTS_DIR / filename
    fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[SAVE] Cluster sizes chart: {filepath}")

    return filepath


# ============================================================================
# MAIN CLUSTERING PIPELINE
# ============================================================================

def run_clustering_pipeline(filepath: Path = CLEANED_DATA_PATH) -> Dict:
    """
    Run complete K-Means clustering pipeline.

    Parameters
    ----------
    filepath : Path
        Path to cleaned data CSV.

    Returns
    -------
    dict
        Results including model, cluster assignments, statistics, visualizations.
    """
    print("\n" + "#"*60)
    print("# PRICESENSE K-MEANS CLUSTERING PIPELINE")
    print("#"*60)

    # 1. Load data
    df = pd.read_csv(filepath)
    print(f"[LOAD] Loaded {df.shape[0]} products with {df.shape[1]} features")

    # 2. Feature selection
    X, feature_names = select_features_for_clustering(df)

    # 3. Scale features
    X_scaled, scaler = scale_features(X)

    # 4. Elbow Method
    inertias, silhouette_scores = compute_elbow_method(X_scaled, range(2, 11))
    plot_elbow_method(inertias, silhouette_scores)

    # 5. Select optimal K
    optimal_k = select_optimal_k(inertias, silhouette_scores)

    # 6. Train K-Means
    kmeans = train_kmeans(X_scaled, optimal_k)

    # 7. Assign clusters
    df_clustered = assign_clusters(df, kmeans, feature_names)

    # 8. Cluster analysis
    cluster_stats = analyze_clusters(df_clustered)
    interpretations = interpret_clusters(df_clustered, feature_names)

    # 9. Visualizations
    print("\nGENERATING VISUALIZATIONS")
    print("=" * 60)
    pca_chart = plot_clusters_pca(X_scaled, kmeans.labels_, df_clustered, feature_names)
    profile_chart = plot_cluster_profiles(df_clustered, feature_names)
    size_chart = plot_cluster_sizes(df_clustered)

    # 10. Save results
    print("\nSAVING RESULTS")
    print("=" * 60)

    # Save cluster assignments
    output_cols = ['product_id', 'product_name', 'main_category', 'discounted_price',
                   'actual_price', 'discount_percentage', 'rating', 'rating_count', 'cluster']
    available_cols = [c for c in output_cols if c in df_clustered.columns]
    df_clustered[available_cols].to_csv(CLUSTER_RESULTS_PATH, index=False)
    print(f"[SAVE] Cluster results: {CLUSTER_RESULTS_PATH}")

    # Save cluster statistics
    stats_path = Path("outputs/cluster_statistics.csv")
    cluster_stats.to_csv(stats_path)
    print(f"[SAVE] Cluster statistics: {stats_path}")

    # Save interpretations
    interp_df = pd.DataFrame([
        {
            'cluster': k,
            'label': v['label'],
            'size': v['size'],
            'percentage': round(v['percentage'], 2),
            **{f'mean_{feat}': round(val, 2) for feat, val in v['means'].items()}
        }
        for k, v in interpretations.items()
    ])
    interp_path = Path("outputs/cluster_interpretations.csv")
    interp_df.to_csv(interp_path, index=False)
    print(f"[SAVE] Cluster interpretations: {interp_path}")

    # Save model artifacts
    import joblib
    joblib.dump(kmeans, MODELS_DIR / "kmeans_model.pkl")
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    joblib.dump(feature_names, MODELS_DIR / "feature_names.pkl")
    print(f"[SAVE] Model artifacts: {MODELS_DIR}")

    # Compile results
    results = {
        'model': kmeans,
        'scaler': scaler,
        'feature_names': feature_names,
        'optimal_k': optimal_k,
        'inertias': inertias,
        'silhouette_scores': silhouette_scores,
        'df_clustered': df_clustered,
        'cluster_stats': cluster_stats,
        'interpretations': interpretations,
        'charts': [pca_chart, profile_chart, size_chart, CHARTS_DIR / "elbow_method.png"]
    }

    print("\n" + "#"*60)
    print("# CLUSTERING PIPELINE COMPLETE")
    print("#"*60)

    return results


# ============================================================================
# ENTRY POINT
# ============================================================================

# ============================================================================
# CLUSTER ANALYSIS (Phase 5)
# ============================================================================

def analyze_cluster_category_composition(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Analyze cluster composition by main_category.

    Returns
    -------
    tuple
        (count_matrix, percentage_matrix)
    """
    print("\nCLUSTER COMPOSITION BY CATEGORY")
    print("=" * 60)

    # Count matrix
    count_matrix = df.groupby(['cluster', 'main_category']).size().unstack(fill_value=0)
    print("Count matrix:")
    print(count_matrix.to_string())

    # Percentage matrix (row-wise: within each cluster)
    pct_matrix = count_matrix.div(count_matrix.sum(axis=1), axis=0).round(4) * 100
    print("\nPercentage matrix (within cluster):")
    print(pct_matrix.to_string())

    # Also show column-wise: within each category, cluster distribution
    col_pct_matrix = count_matrix.div(count_matrix.sum(axis=0), axis=1).round(4) * 100
    print("\nPercentage matrix (within category):")
    print(col_pct_matrix.to_string())

    return count_matrix, pct_matrix


def analyze_cluster_discount_band_composition(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Analyze cluster composition by discount bands.

    Returns
    -------
    tuple
        (count_matrix, percentage_matrix)
    """
    print("\nCLUSTER COMPOSITION BY DISCOUNT BAND")
    print("=" * 60)

    # Create discount bands if not present
    if 'discount_band' not in df.columns:
        df['discount_band'] = pd.cut(
            df['discount_percentage'],
            bins=[-1, 10, 30, 50, 70, 100],
            labels=['0-10%', '10-30%', '30-50%', '50-70%', '70-100%']
        )

    # Count matrix
    count_matrix = df.groupby(['cluster', 'discount_band']).size().unstack(fill_value=0)
    print("Count matrix:")
    print(count_matrix.to_string())

    # Percentage matrix (within cluster)
    pct_matrix = count_matrix.div(count_matrix.sum(axis=1), axis=0).round(4) * 100
    print("\nPercentage matrix (within cluster):")
    print(pct_matrix.to_string())

    return count_matrix, pct_matrix


def plot_cluster_category_heatmap(count_matrix: pd.DataFrame, pct_matrix: pd.DataFrame) -> Path:
    """Plot cluster × category composition heatmaps."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    # Count heatmap
    sns.heatmap(count_matrix, annot=True, fmt='d', cmap='Blues', ax=ax1, cbar_kws={'label': 'Count'})
    ax1.set_title('Cluster × Category: Product Count', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Main Category', fontsize=12)
    ax1.set_ylabel('Cluster', fontsize=12)

    # Percentage heatmap
    sns.heatmap(pct_matrix, annot=True, fmt='.1f', cmap='RdYlBu_r', ax=ax2,
                cbar_kws={'label': 'Percentage (%)'}, center=50)
    ax2.set_title('Cluster × Category: % Within Cluster', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Main Category', fontsize=12)
    ax2.set_ylabel('Cluster', fontsize=12)

    plt.suptitle('Cluster Composition by Main Category', fontsize=16, fontweight='bold')
    plt.tight_layout()

    filename = "cluster_category_composition.png"
    filepath = CHARTS_DIR / filename
    fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[SAVE] Cluster × Category composition: {filepath}")

    return filepath


def plot_cluster_discount_band_heatmap(count_matrix: pd.DataFrame, pct_matrix: pd.DataFrame) -> Path:
    """Plot cluster × discount band composition heatmaps."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Count heatmap
    sns.heatmap(count_matrix, annot=True, fmt='d', cmap='Greens', ax=ax1, cbar_kws={'label': 'Count'})
    ax1.set_title('Cluster × Discount Band: Product Count', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Discount Band', fontsize=12)
    ax1.set_ylabel('Cluster', fontsize=12)

    # Percentage heatmap
    sns.heatmap(pct_matrix, annot=True, fmt='.1f', cmap='RdYlBu_r', ax=ax2,
                cbar_kws={'label': 'Percentage (%)'}, center=50)
    ax2.set_title('Cluster × Discount Band: % Within Cluster', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Discount Band', fontsize=12)
    ax2.set_ylabel('Cluster', fontsize=12)

    plt.suptitle('Cluster Composition by Discount Band', fontsize=16, fontweight='bold')
    plt.tight_layout()

    filename = "cluster_discount_band_composition.png"
    filepath = CHARTS_DIR / filename
    fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[SAVE] Cluster × Discount Band composition: {filepath}")

    return filepath


def generate_cluster_business_interpretation(df: pd.DataFrame, interpretations: Dict) -> str:
    """
    Generate comprehensive business interpretation of clusters.

    Parameters
    ----------
    df : pd.DataFrame
        Clustered dataframe.
    interpretations : dict
        Cluster interpretations from interpret_clusters().

    Returns
    -------
    str
        Formatted business interpretation.
    """
    print("\nBUSINESS INTERPRETATION")
    print("=" * 60)

    # Analyze category composition
    cat_count, cat_pct = analyze_cluster_category_composition(df)
    disc_count, disc_pct = analyze_cluster_discount_band_composition(df)

    interpretation_text = []
    interpretation_text.append("=" * 80)
    interpretation_text.append("CLUSTER BUSINESS INTERPRETATION")
    interpretation_text.append("=" * 80)

    for cluster_id in sorted(df['cluster'].unique()):
        subset = df[df['cluster'] == cluster_id]
        n = len(subset)
        pct = n / len(df) * 100
        label = interpretations[cluster_id]['label']
        means = interpretations[cluster_id]['means']

        interpretation_text.append(f"\n{'='*80}")
        interpretation_text.append(f"CLUSTER {cluster_id}: {label.upper()}")
        interpretation_text.append(f"Size: {n} products ({pct:.1f}% of dataset)")
        interpretation_text.append(f"{'='*80}")

        # Key metrics
        interpretation_text.append(f"\nKey Metrics (mean):")
        interpretation_text.append(f"  • Discounted Price: ₹{means.get('discounted_price', 0):.0f}")
        interpretation_text.append(f"  • Discount Percentage: {means.get('discount_percentage', 0):.1f}%")
        interpretation_text.append(f"  • Rating: {means.get('rating', 0):.2f}/5.0")
        interpretation_text.append(f"  • Review Count: {means.get('rating_count', 0):.0f}")

        # Category composition
        if cluster_id in cat_pct.index:
            cat_row = cat_pct.loc[cluster_id]
            top_cats = cat_row[cat_row > 5].sort_values(ascending=False)
            interpretation_text.append(f"\nCategory Composition (top):")
            for cat, val in top_cats.items():
                interpretation_text.append(f"  • {cat}: {val:.1f}%")

        # Discount band composition
        if cluster_id in disc_pct.index:
            disc_row = disc_pct.loc[cluster_id]
            top_discs = disc_row[disc_row > 5].sort_values(ascending=False)
            interpretation_text.append(f"\nDiscount Band Composition:")
            for band, val in top_discs.items():
                interpretation_text.append(f"  • {band}: {val:.1f}%")

        # Business interpretation
        interpretation_text.append(f"\nBusiness Interpretation:")
        if 'Average' in label or 'Mid' in label:
            interpretation_text.append("  This cluster represents the MAINSTREAM market segment.")
            interpretation_text.append("  Products have moderate prices, moderate discounts, and good ratings.")
            interpretation_text.append("  High review counts indicate established, popular products.")
            interpretation_text.append("  TARGET: Mass-market consumers seeking value for money.")
        elif 'Low-Rating' in label or 'Low-Price' in label:
            interpretation_text.append("  This cluster represents the BUDGET/DEEP-DISCOUNT segment.")
            interpretation_text.append("  Deep discounts (59.8% avg) but lower ratings (3.94) and fewer reviews.")
            interpretation_text.append("  May indicate lower quality, newer products, or niche items.")
            interpretation_text.append("  TARGET: Price-sensitive consumers willing to trade quality for savings.")
        elif 'High-Price' in label or 'Low-Discount' in label:
            interpretation_text.append("  This cluster represents the PREMIUM segment.")
            interpretation_text.append("  High prices (₹19,687 avg) with low discounts (31.6% avg).")
            interpretation_text.append("  Good ratings (4.18) with moderate review counts.")
            interpretation_text.append("  Brand strength allows pricing power without deep discounts.")
            interpretation_text.append("  TARGET: Quality-conscious consumers, brand loyalists.")

    interpretation_text.append(f"\n{'='*80}")
    interpretation_text.append("SUMMARY")
    interpretation_text.append(f"{'='*80}")
    interpretation_text.append("Three distinct product segments identified:")
    interpretation_text.append("  1. MAINSTREAM (Cluster 0, 47.8%): Moderate price/discount, high reviews, good ratings")
    interpretation_text.append("  2. BUDGET/DISCOUNT (Cluster 1, 41.1%): Low price, deep discounts, lower ratings, fewer reviews")
    interpretation_text.append("  3. PREMIUM (Cluster 2, 11.1%): High price, low discounts, good ratings, moderate reviews")
    interpretation_text.append("\nStrategic Implications:")
    interpretation_text.append("  • Mainstream: Volume play, competitive pricing, review management")
    interpretation_text.append("  • Budget: Cost optimization, quality improvement to raise ratings")
    interpretation_text.append("  • Premium: Brand investment, maintain pricing power, loyalty programs")

    result = "\n".join(interpretation_text)
    print(result)

    return result


def save_cluster_analysis_results(cat_count: pd.DataFrame, cat_pct: pd.DataFrame,
                                   disc_count: pd.DataFrame, disc_pct: pd.DataFrame,
                                   business_interpretation: str) -> None:
    """Save cluster analysis results to CSV files."""
    # Save category composition
    cat_count.to_csv("outputs/cluster_category_count.csv")
    cat_pct.to_csv("outputs/cluster_category_pct.csv")
    print("[SAVE] Cluster category composition: outputs/cluster_category_count.csv, cluster_category_pct.csv")

    # Save discount band composition
    disc_count.to_csv("outputs/cluster_discount_band_count.csv")
    disc_pct.to_csv("outputs/cluster_discount_band_pct.csv")
    print("[SAVE] Cluster discount band composition: outputs/cluster_discount_band_count.csv, cluster_discount_band_pct.csv")

    # Save business interpretation
    with open("outputs/cluster_business_interpretation.txt", "w") as f:
        f.write(business_interpretation)
    print("[SAVE] Business interpretation: outputs/cluster_business_interpretation.txt")


def run_cluster_analysis(df: pd.DataFrame, interpretations: Dict) -> Dict:
    """
    Run complete cluster analysis pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with cluster column.
    interpretations : dict
        Cluster interpretations from interpret_clusters().

    Returns
    -------
    dict
        Analysis results.
    """
    print("\n" + "#"*60)
    print("# CLUSTER ANALYSIS PIPELINE (Phase 5)")
    print("#"*60)

    # 1. Category composition
    cat_count, cat_pct = analyze_cluster_category_composition(df)
    cat_chart = plot_cluster_category_heatmap(cat_count, cat_pct)

    # 2. Discount band composition
    disc_count, disc_pct = analyze_cluster_discount_band_composition(df)
    disc_chart = plot_cluster_discount_band_heatmap(disc_count, disc_pct)

    # 3. Business interpretation
    business_interpretation = generate_cluster_business_interpretation(df, interpretations)

    # 4. Save results
    save_cluster_analysis_results(cat_count, cat_pct, disc_count, disc_pct, business_interpretation)

    print("\n" + "#"*60)
    print("# CLUSTER ANALYSIS COMPLETE")
    print("#"*60)

    return {
        'category_count': cat_count,
        'category_pct': cat_pct,
        'discount_band_count': disc_count,
        'discount_band_pct': disc_pct,
        'business_interpretation': business_interpretation,
        'charts': [cat_chart, disc_chart]
    }


if __name__ == "__main__":
    results = run_clustering_pipeline()

    # Print final summary
    print("\n" + "="*60)
    print("FINAL CLUSTER SUMMARY")
    print("="*60)

    # Run cluster analysis
    if 'df_clustered' in results and 'interpretations' in results:
        analysis_results = run_cluster_analysis(results['df_clustered'], results['interpretations'])
    for cluster_id, interp in results['interpretations'].items():
        print(f"Cluster {cluster_id} ({interp['label']}): {interp['size']} products ({interp['percentage']:.1f}%)")