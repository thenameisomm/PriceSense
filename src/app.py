"""
PriceSense — Streamlit Interactive Dashboard

5-page dashboard:
1. Home — Project overview, key metrics, quick insights
2. EDA — Interactive exploratory data analysis
3. Clustering — K-Means cluster exploration
4. Recommendation — Product recommendation engine
5. About — Project details, methodology, syllabus compliance
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

# Add src to path
sys.path.append('src')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="PriceSense — Amazon Product Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# DATA LOADING (Cached)
# ============================================================================

@st.cache_data
def load_cleaned_data():
    """Load cleaned dataset."""
    return pd.read_csv("outputs/cleaned_data.csv")


@st.cache_data
def load_cluster_results():
    """Load cluster results."""
    return pd.read_csv("outputs/cluster_results.csv")


@st.cache_data
def load_statistics():
    """Load statistical outputs."""
    stats = {}
    stat_files = {
        'summary': 'outputs/statistics/summary.csv',
        'correlation': 'outputs/statistics/correlation.csv',
        'covariance': 'outputs/statistics/covariance.csv',
        'group_category': 'outputs/statistics/group_by_category.csv',
        'group_discount': 'outputs/statistics/group_by_discount_band.csv',
        'pivot_count': 'outputs/statistics/pivot_category_discount.csv',
        'pivot_rating': 'outputs/statistics/pivot_category_rating.csv',
    }
    for key, path in stat_files.items():
        try:
            stats[key] = pd.read_csv(path, index_col=0)
        except:
            stats[key] = pd.DataFrame()
    return stats


@st.cache_data
def load_recommendations():
    """Load precomputed recommendations."""
    return pd.read_csv("outputs/recommendations.csv")


@st.cache_data
def load_business_interpretation():
    """Load cluster business interpretation."""
    try:
        with open("outputs/cluster_business_interpretation.txt", "r") as f:
            return f.read()
    except:
        return ""


# Load data
df = load_cleaned_data()
cluster_df = load_cluster_results()
stats = load_statistics()
recs_df = load_recommendations()
biz_interpretation = load_business_interpretation()

# Merge cluster info
if 'cluster' not in df.columns:
    cluster_map = cluster_df.set_index('product_id')['cluster']
    df['cluster'] = df['product_id'].map(cluster_map)

# Add discount bands
df['discount_band'] = pd.cut(
    df['discount_percentage'],
    bins=[-1, 10, 30, 50, 70, 100],
    labels=['0-10%', '10-30%', '30-50%', '50-70%', '70-100%']
)


# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown("## 📊 PriceSense")
    st.markdown("**Amazon Product Analysis & Recommendation**")
    st.markdown("---")

    # Navigation
    page = st.radio(
        "Navigate",
        ["🏠 Home", "📈 EDA", "🎯 Clustering", "🎁 Recommendation", "ℹ️ About"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("### Dataset Info")
    st.metric("Products", f"{len(df):,}")
    st.metric("Categories", f"{df['main_category'].nunique()}")
    st.metric("Clusters", f"{df['cluster'].nunique()}")

    st.markdown("---")
    st.markdown("*Academic DSML Project*")
    st.markdown("*K-Means Clustering*")


# ============================================================================
# PAGE 1: HOME
# ============================================================================

if page == "🏠 Home":
    st.markdown('<div class="main-header">PriceSense</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Amazon Product Analysis and Recommendation System</div>', unsafe_allow_html=True)

    st.markdown("---")

    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Products", f"{len(df):,}", "After cleaning")
    with col2:
        st.metric("Avg Discounted Price", f"₹{df['discounted_price'].mean():,.0f}", f"Median: ₹{df['discounted_price'].median():,.0f}")
    with col3:
        st.metric("Avg Discount", f"{df['discount_percentage'].mean():.1f}%", f"Median: {df['discount_percentage'].median():.1f}%")
    with col4:
        st.metric("Avg Rating", f"{df['rating'].mean():.2f}/5.0", f"Median: {df['rating'].median():.1f}")

    st.markdown("---")

    # Project Overview
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 📋 Project Overview")
        st.markdown("""
        **PriceSense** analyzes 1,351 Amazon India products across 9 categories to:
        - 📊 Perform comprehensive **Exploratory Data Analysis**
        - 🎯 Segment products using **K-Means Clustering** (3 segments)
        - 🎁 Provide **Rule-Based Recommendations** (5 strategies)
        - 📱 Present insights via **Interactive Dashboard**

        **Key Features:**
        - Indian currency format handling (₹, commas)
        - Median imputation for skewed review counts
        - 12 engineered features (log transforms, boolean flags)
        - 36 statistical visualizations
        - Primary ML algorithm: K-Means Clustering
        """)

    with col2:
        st.markdown("### 🎯 Cluster Summary")
        cluster_labels = {0: "Mainstream", 1: "Budget/Deep-Discount", 2: "Premium"}
        for cid in sorted(df['cluster'].unique()):
            subset = df[df['cluster'] == cid]
            label = cluster_labels.get(cid, f"Cluster {cid}")
            st.markdown(f"""
            **Cluster {cid}: {label}** ({len(subset)} products, {len(subset)/len(df)*100:.1f}%)
            - Avg Price: ₹{subset['discounted_price'].mean():,.0f}
            - Avg Discount: {subset['discount_percentage'].mean():.1f}%
            - Avg Rating: {subset['rating'].mean():.2f}
            """)

    st.markdown("---")

    # Quick Visualizations
    st.markdown("### 📊 Quick Insights")

    tab1, tab2, tab3 = st.tabs(["Price Distribution", "Category Analysis", "Discount vs Rating"])

    with tab1:
        fig = px.histogram(df, x='discounted_price', nbins=50,
                          title='Discounted Price Distribution',
                          labels={'discounted_price': 'Price (₹)', 'count': 'Products'})
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        cat_stats = df.groupby('main_category').agg(
            count=('product_id', 'count'),
            avg_price=('discounted_price', 'mean'),
            avg_rating=('rating', 'mean')
        ).reset_index().sort_values('count', ascending=False)

        fig = px.bar(cat_stats, x='main_category', y='count',
                    title='Product Count by Category',
                    labels={'main_category': 'Category', 'count': 'Products'})
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        fig = px.scatter(df, x='discount_percentage', y='rating',
                        color='main_category', size='rating_count',
                        title='Discount % vs Rating (size = review count)',
                        labels={'discount_percentage': 'Discount %', 'rating': 'Rating'})
        st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# PAGE 2: EDA
# ============================================================================

elif page == "📈 EDA":
    st.markdown('<div class="main-header">Exploratory Data Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Interactive visualizations of product attributes</div>', unsafe_allow_html=True)

    # Sidebar filters for EDA
    with st.sidebar:
        st.markdown("### EDA Filters")
        selected_categories = st.multiselect(
            "Categories",
            options=sorted(df['main_category'].unique()),
            default=sorted(df['main_category'].unique())
        )

        price_range = st.slider(
            "Price Range (₹)",
            int(df['discounted_price'].min()),
            int(df['discounted_price'].max()),
            (int(df['discounted_price'].min()), int(df['discounted_price'].max()))
        )

        discount_range = st.slider(
            "Discount % Range",
            0, 100, (0, 100)
        )

    # Filter data
    filtered_df = df[
        (df['main_category'].isin(selected_categories)) &
        (df['discounted_price'] >= price_range[0]) &
        (df['discounted_price'] <= price_range[1]) &
        (df['discount_percentage'] >= discount_range[0]) &
        (df['discount_percentage'] <= discount_range[1])
    ]

    st.info(f"Showing {len(filtered_df):,} of {len(df):,} products")

    # Tabs for different analyses
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Distributions", "📦 Box Plots", "🔗 Correlations",
        "📈 Category Analysis", "📋 Pivot Tables"
    ])

    with tab1:
        st.markdown("#### Distribution Analysis")

        col1, col2 = st.columns(2)

        with col1:
            var = st.selectbox("Variable",
                              ['discounted_price', 'actual_price', 'discount_percentage', 'rating', 'rating_count'],
                              format_func=lambda x: x.replace('_', ' ').title())

            fig = px.histogram(filtered_df, x=var, nbins=50, marginal='box',
                              title=f'{var.replace("_", " ").title()} Distribution')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # KDE overlay option
            if st.checkbox("Show KDE", value=True):
                from scipy.stats import gaussian_kde
                data = filtered_df[var].dropna()
                if len(data) > 1:
                    kde = gaussian_kde(data)
                    x_range = np.linspace(data.min(), data.max(), 200)
                    fig2 = go.Figure()
                    fig2.add_trace(go.Histogram(x=data, nbinsx=50, histnorm='probability density', name='Histogram', opacity=0.6))
                    fig2.add_trace(go.Scatter(x=x_range, y=kde(x_range), mode='lines', name='KDE', line=dict(color='red', width=2)))
                    fig2.update_layout(title=f'{var.replace("_", " ").title()} with KDE')
                    st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        st.markdown("#### Box Plots by Category")

        box_var = st.selectbox("Variable for Box Plot",
                              ['discounted_price', 'actual_price', 'discount_percentage', 'rating'],
                              format_func=lambda x: x.replace('_', ' ').title())

        fig = px.box(filtered_df, x='main_category', y=box_var,
                    title=f'{box_var.replace("_", " ").title()} by Category',
                    labels={'main_category': 'Category', box_var: box_var.replace('_', ' ').title()})
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown("#### Correlation Analysis")

        # Numerical variables for correlation
        num_vars = ['actual_price', 'discounted_price', 'discount_percentage', 'rating', 'rating_count', 'price_savings']
        available_vars = [v for v in num_vars if v in filtered_df.columns]

        corr_method = st.radio("Method", ["Pearson", "Spearman"], horizontal=True)

        corr_matrix = filtered_df[available_vars].corr(method=corr_method.lower())

        fig = px.imshow(corr_matrix, text_auto='.2f', aspect='auto',
                       color_continuous_scale='RdBu_r', zmin=-1, zmax=1,
                       title=f'{corr_method} Correlation Matrix')
        st.plotly_chart(fig, use_container_width=True)

        # Target correlations
        if 'rating' in available_vars:
            st.markdown("**Correlation with Rating:**")
            target_corr = corr_matrix['rating'].drop('rating').sort_values(ascending=False)
            st.bar_chart(target_corr)

    with tab4:
        st.markdown("#### Category-wise Analysis")

        metric = st.selectbox("Metric",
                             ['count', 'mean_discounted_price', 'mean_actual_price',
                              'mean_discount_percentage', 'mean_rating'],
                             format_func=lambda x: x.replace('mean_', '').replace('_', ' ').title())

        if metric == 'count':
            cat_data = filtered_df.groupby('main_category').size().reset_index(name='count')
            y_col = 'count'
        else:
            cat_data = filtered_df.groupby('main_category').agg(
                **{metric: (metric.replace('mean_', ''), 'mean')}
            ).reset_index()
            y_col = metric

        fig = px.bar(cat_data, x='main_category', y=y_col,
                    title=f'{metric.replace("mean_", "").replace("_", " ").title()} by Category',
                    labels={'main_category': 'Category', y_col: metric.replace('mean_', '').replace('_', ' ').title()})
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

    with tab5:
        st.markdown("#### Pivot Tables: Category × Discount Band")

        pivot_metric = st.selectbox("Pivot Metric",
                                   ['count', 'rating', 'discounted_price', 'discount_percentage'],
                                   format_func=lambda x: x.replace('_', ' ').title())

        pivot_df = filtered_df.pivot_table(
            values=pivot_metric if pivot_metric != 'count' else 'product_id',
            index='main_category',
            columns='discount_band',
            aggfunc='count' if pivot_metric == 'count' else 'mean'
        )

        fig = px.imshow(pivot_df, text_auto='.1f' if pivot_metric != 'count' else 'd',
                       aspect='auto', color_continuous_scale='Blues',
                       title=f'{pivot_metric.replace("_", " ").title()} by Category × Discount Band')
        st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# PAGE 3: CLUSTERING
# ============================================================================

elif page == "🎯 Clustering":
    st.markdown('<div class="main-header">K-Means Clustering Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Product segmentation with 3 clusters (K=3)</div>', unsafe_allow_html=True)

    st.markdown("""
    **Feature Selection (6 features):**
    - `discounted_price` — Primary purchase factor
    - `discount_percentage` — Pricing strategy
    - `rating` — Quality signal
    - `log_rating_count` — Popularity (log-transformed)
    - `log_discounted_price` — Price (log-transformed)
    - `rating` — Quality signal

    **Scaling:** StandardScaler (zero mean, unit variance)
    **K Selection:** Elbow Method (K=3) preferred over Silhouette (K=7) for interpretability
    """)

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Cluster Overview", "🎯 Cluster Profiles", "📈 PCA Visualization", "📋 Business Interpretation"])

    with tab1:
        st.markdown("#### Cluster Sizes & Key Metrics")

        cluster_summary = df.groupby('cluster').agg(
            count=('product_id', 'count'),
            avg_price=('discounted_price', 'mean'),
            median_price=('discounted_price', 'median'),
            avg_discount=('discount_percentage', 'mean'),
            avg_rating=('rating', 'mean'),
            avg_reviews=('rating_count', 'mean')
        ).round(2)

        cluster_summary['percentage'] = (cluster_summary['count'] / len(df) * 100).round(1)
        cluster_summary.index = ['Mainstream', 'Budget/Deep-Discount', 'Premium']

        st.dataframe(cluster_summary, use_container_width=True)

        # Cluster size pie chart
        fig = px.pie(values=cluster_summary['count'], names=cluster_summary.index,
                    title='Cluster Size Distribution')
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown("#### Cluster Profiles (Normalized Feature Means)")

        # Normalize features for comparison
        profile_features = ['discounted_price', 'discount_percentage', 'rating', 'log_rating_count']
        profile_df = df[profile_features + ['cluster']].copy()

        for f in profile_features:
            min_val, max_val = profile_df[f].min(), profile_df[f].max()
            if max_val > min_val:
                profile_df[f'{f}_norm'] = (profile_df[f] - min_val) / (max_val - min_val)
            else:
                profile_df[f'{f}_norm'] = 0.5

        norm_features = [f'{f}_norm' for f in profile_features]
        cluster_profiles = profile_df.groupby('cluster')[norm_features].mean()
        cluster_profiles.index = ['Mainstream', 'Budget/Deep-Discount', 'Premium']

        # Radar chart
        fig = go.Figure()
        categories = [f.replace('_norm', '').replace('_', ' ').title() for f in norm_features]

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        for i, (cluster_name, row) in enumerate(cluster_profiles.iterrows()):
            fig.add_trace(go.Scatterpolar(
                r=row.values.tolist() + [row.values[0]],
                theta=categories + [categories[0]],
                fill='toself',
                name=cluster_name,
                line_color=colors[i]
            ))

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            title="Cluster Profiles (Normalized 0-1)"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Bar chart alternative
        st.markdown("**Detailed Statistics:**")
        st.dataframe(cluster_profiles, use_container_width=True)

    with tab3:
        st.markdown("#### PCA Visualization (2D)")
        st.markdown("*PCA used only for visualization, not as primary ML algorithm*")

        # Load PCA chart if available, or compute on the fly
        try:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler

            features = ['discounted_price', 'discount_percentage', 'rating', 'rating_count', 'log_rating_count', 'log_discounted_price']
            X = df[features].fillna(df[features].median())
            X_scaled = StandardScaler().fit_transform(X)

            pca = PCA(n_components=2, random_state=42)
            X_pca = pca.fit_transform(X_scaled)

            pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
            pca_df['cluster'] = df['cluster'].values
            pca_df['cluster_name'] = pca_df['cluster'].map({0: 'Mainstream', 1: 'Budget/Deep-Discount', 2: 'Premium'})

            fig = px.scatter(pca_df, x='PC1', y='PC2', color='cluster_name',
                           title=f'K-Means Clusters via PCA (Explained Variance: {pca.explained_variance_ratio_.sum():.1%})',
                           labels={'PC1': f'PC1 ({pca.explained_variance_ratio_[0]:.1%})',
                                  'PC2': f'PC2 ({pca.explained_variance_ratio_[1]:.1%})'})
            st.plotly_chart(fig, use_container_width=True)

            # Component loadings
            st.markdown("**PCA Component Loadings:**")
            loadings = pd.DataFrame(pca.components_.T, index=features, columns=['PC1', 'PC2']).round(3)
            st.dataframe(loadings, use_container_width=True)

        except Exception as e:
            st.error(f"Could not generate PCA: {e}")

    with tab4:
        st.markdown("#### Business Interpretation")

        if biz_interpretation:
            st.text(biz_interpretation)
        else:
            st.markdown("""
            **Cluster 0: Mainstream (47.8%)**
            - Moderate price (₹1,461), moderate discount (38.9%)
            - High rating (4.20), very high reviews (31,196)
            - Categories: Home&Kitchen, Computers, Electronics
            - *Strategy: Volume play, competitive pricing*

            **Cluster 1: Budget/Deep-Discount (41.1%)**
            - Low price (₹1,023), deep discount (59.8%)
            - Lower rating (3.94), low reviews (2,780)
            - Categories: Home&Kitchen, Electronics, Computers
            - *Strategy: Cost optimization, quality improvement*

            **Cluster 2: Premium (11.1%)**
            - High price (₹19,687), low discount (31.6%)
            - Good rating (4.18), moderate reviews (14,108)
            - Categories: Electronics (81%)
            - *Strategy: Brand investment, maintain pricing power*
            """)


# ============================================================================
# PAGE 4: RECOMMENDATION
# ============================================================================

elif page == "🎁 Recommendation":
    st.markdown('<div class="main-header">Product Recommendation Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Rule-based recommendations (5 strategies)</div>', unsafe_allow_html=True)

    st.markdown("""
    **Recommendation Strategies:**
    1. **Cluster-Based** — Products from same K-Means cluster
    2. **Category Better Rated** — Higher-rated alternatives in same category & price range
    3. **Budget Alternative** — Cheaper options (≥3.5 rating) in same category
    4. **Premium Upgrade** — Higher-priced, highly-rated products in same category
    5. **Price Range** — Products within ±30% price range
    """)

    # Product selector
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("#### Select Product")

        # Category filter
        rec_category = st.selectbox("Filter by Category",
                                   ['All'] + sorted(df['main_category'].unique()))

        if rec_category != 'All':
            filtered_products = df[df['main_category'] == rec_category]
        else:
            filtered_products = df

        # Product search
        search_term = st.text_input("Search Product", placeholder="Type product name...")

        if search_term:
            filtered_products = filtered_products[
                filtered_products['product_name'].str.contains(search_term, case=False, na=False)
            ]

        # Product selector
        if len(filtered_products) > 0:
            product_options = filtered_products.apply(
                lambda r: f"{r['product_name'][:60]}... (₹{r['discounted_price']:.0f}, ⭐{r['rating']:.1f})", axis=1
            ).tolist()

            selected_idx = st.selectbox("Choose Product", range(len(product_options)),
                                       format_func=lambda i: product_options[i])

            selected_product = filtered_products.iloc[selected_idx]
            product_id = selected_product['product_id']
        else:
            st.warning("No products found")
            product_id = None

    with col2:
        if product_id:
            # Show target product details
            details = selected_product
            st.markdown("#### Target Product")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Price", f"₹{details['discounted_price']:,.0f}")
            c2.metric("Discount", f"{details['discount_percentage']:.1f}%")
            c3.metric("Rating", f"{details['rating']:.1f}/5.0")
            c4.metric("Reviews", f"{details['rating_count']:,.0f}")

            st.caption(f"Category: {details['main_category']} | Cluster: {details.get('cluster', 'N/A')}")
            st.caption(f"{details['product_name'][:100]}...")

            # Get recommendations using imported functions
            from recommendation import get_hybrid_recommendations, get_product_details

            recs = get_hybrid_recommendations(df, product_id, n_recommendations=5)

            st.markdown("#### Recommendations")

            strategy_names = {
                'cluster_based': '🎯 Cluster-Based (Similar Products)',
                'category_better_rated': '⭐ Better Rated in Category',
                'budget_alternative': '💰 Budget Alternatives',
                'premium_upgrade': '💎 Premium Upgrades',
                'price_range': '💵 Similar Price Range'
            }

            for strategy, rec_df in recs.items():
                if len(rec_df) > 0:
                    with st.expander(strategy_names.get(strategy, strategy), expanded=True):
                        for _, r in rec_df.iterrows():
                            col_a, col_b, col_c, col_d = st.columns([3, 1, 1, 1])
                            col_a.write(f"**{r['product_name'][:50]}...**")
                            col_b.write(f"₹{r['discounted_price']:,.0f}")
                            col_c.write(f"⭐{r['rating']:.1f}")
                            col_d.write(r['main_category'])
        else:
            st.info("Select a product to see recommendations")


# ============================================================================
# PAGE 5: ABOUT
# ============================================================================

elif page == "ℹ️ About":
    st.markdown('<div class="main-header">About PriceSense</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Amazon Product Analysis and Recommendation System</div>', unsafe_allow_html=True)

    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs(["📚 Project Details", "🔬 Methodology", "🎓 Academic Context", "📁 File Structure"])

    with tab1:
        st.markdown("""
        ### Project Overview

        **PriceSense** analyzes Amazon India product data using comprehensive data science techniques
        and provides explainable, rule-based product recommendations through an interactive Streamlit dashboard.

        **Dataset:** 1,351 products × 34 features (after cleaning)
        **Source:** `data/amazon.csv` (1,465 raw rows × 16 columns)
        **Categories:** 9 main categories (Computers&Accessories, Electronics, Home&Kitchen, etc.)

        **Pipeline:**
        1. **Data Preprocessing** — Currency cleaning, missing value imputation, duplicate handling, feature engineering
        2. **Statistical Analysis** — Central tendency, dispersion, distribution shape, correlation, covariance, grouped stats, pivot tables
        3. **Exploratory Data Analysis** — 36 visualizations (histograms, box plots, heatmaps, scatter plots, pivot tables)
        4. **K-Means Clustering** — 3 segments (Mainstream, Budget, Premium) using 6 features
        5. **Cluster Analysis** — Business interpretation, category/discount band composition
        6. **Rule-Based Recommendation** — 5 strategies, 19,397 precomputed recommendations
        7. **Interactive Dashboard** — 5-page Streamlit application

        *Originally developed as an academic Data Science and Machine Learning project.*
        """)

    with tab2:
        st.markdown("""
        ### Methodology Details

        #### Data Preprocessing
        - **Price Cleaning:** ₹1,899 → 1899.0 (remove ₹ and commas)
        - **Discount Cleaning:** 64% → 64.0 (remove %)
        - **Rating Count:** 24,269 → 24269 (remove commas), median imputation for 2 missing
        - **Duplicate Handling:** 114 rows with repeated product_ids removed (kept first)
        - **Category Parsing:** Pipe-separated hierarchy → main_category, sub_category, levels 3-5
        - **Feature Engineering:** 12 derived features (price_savings, log transforms, boolean flags)

        #### Statistical Analysis
        - Central Tendency: Mean, Median, Mode
        - Dispersion: Min, Max, Range, Variance, Std Dev
        - Distribution: Skewness, Kurtosis (Fisher's excess)
        - Percentiles: P10, P25, P50, P75, P90, P95, P99
        - Correlation: Pearson & Spearman matrices
        - Covariance: Matrix for key variables
        - Grouped Stats: By main_category (9) and discount bands (5)
        - Pivot Tables: Category × Discount Band (count, mean rating)

        #### K-Means Clustering
        - **Features:** 6 selected with justification (price, discount, rating, popularity)
        - **Scaling:** StandardScaler (required for Euclidean distance)
        - **K Selection:** Elbow Method (K=2-10) → K=3 chosen
        - **Training:** n_init=20, max_iter=300, random_state=42
        - **Visualization:** PCA (60.3% variance) for 2D plotting only
        - **Interpretation:** Based on actual statistics, not pre-assumed labels

        #### Rule-Based Recommendation
        - 5 distinct strategies (cluster, category, budget, premium, price)
        - No collaborative filtering, no user tracking
        - All rules explainable and category-constrained
        - Precomputed for all products (19,397 recommendations)
        """)

    with tab3:
        st.markdown("""
        ### Academic Context

        This project was originally developed as an academic **Data Science and Machine Learning (DSML) Mini-Project**
        with 18 required components.

        **Implemented (Required):**
        ✅ Data loading, inspection, preprocessing, cleaning
        ✅ Missing value handling (median for skewed rating_count)
        ✅ Duplicate handling (exact vs repeated product_ids)
        ✅ Feature transformation & engineering (12 features)
        ✅ Statistical analysis (central tendency, dispersion, distribution, percentiles)
        ✅ Correlation & covariance analysis
        ✅ Grouping, aggregation, pivot tables
        ✅ EDA visualizations (histograms, box plots, heatmaps, scatter, pairplot)
        ✅ **K-Means Clustering** (primary ML algorithm)
        ✅ Elbow Method & Silhouette analysis for K selection
        ✅ Cluster analysis & business interpretation
        ✅ Rule-based recommendation system
        ✅ Streamlit GUI with 5 pages
        ✅ Jupyter Notebook analysis narrative

        **Intentionally Excluded:**
        ❌ Future price prediction / time series forecasting
        ❌ LSTM, GRU, XGBoost, LightGBM, CatBoost
        ❌ ARIMA, SARIMA, Prophet
        ❌ Deep learning / neural networks
        ❌ NLP / sentiment analysis
        ❌ Collaborative filtering / matrix factorization
        ❌ Web scraping / live APIs
        ❌ Advanced recommendation systems
        """)

    with tab4:
        st.markdown("""
        ### Project File Structure
        ```
        PriceSense/
        ├── data/
        │   └── amazon.csv                    # Raw dataset (1,465 × 16)
        ├── outputs/
        │   ├── cleaned_data.csv              # Cleaned data (1,351 × 34)
        │   ├── cluster_results.csv           # Product IDs + cluster labels
        │   ├── cluster_statistics.csv        # Detailed cluster stats
        │   ├── cluster_interpretations.csv   # Cluster labels & metrics
        │   ├── recommendations.csv           # All recommendations (19,397 rows)
        │   ├── cluster_business_interpretation.txt
        │   ├── statistics/                   # 7 statistical CSV files
        │   └── charts/                       # 40+ PNG visualizations
        ├── models/
        │   ├── kmeans_model.pkl
        │   ├── scaler.pkl
        │   └── feature_names.pkl
        ├── src/
        │   ├── preprocessing.py              # Data cleaning pipeline
        │   ├── descriptive_stats.py          # Statistical analysis
        │   ├── eda.py                        # Visualization functions
        │   ├── clustering.py                 # K-Means + Cluster Analysis
        │   ├── recommendation.py             # Rule-based recommender
        │   └── app.py                        # Streamlit dashboard
        ├── notebooks/
        │   └── PriceSense_Analysis.ipynb     # Complete analysis narrative
        ├── PROJECT_RULES.md                  # Academic constraints
        ├── PROJECT_PLAN.md                   # Implementation plan
        ├── README.md                         # Project documentation
        └── requirements.txt                  # Dependencies
        """)

    st.markdown("---")
    st.markdown("*Amazon Product Analysis and Recommendation System • K-Means Clustering • Rule-Based Recommendations*")


# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #888; font-size: 0.8rem;'>"
    "PriceSense — Amazon Product Analysis and Recommendation System | K-Means Clustering | Rule-Based Recommendations"
    "</div>",
    unsafe_allow_html=True
)