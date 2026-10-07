# Project Plan - PriceSense

## Dataset Overview
- **File**: `data/amazon.csv`
- **Rows**: 1,465 products
- **Columns**: 16
- **Source**: Amazon India product data with aggregated reviews

## Column Details & Transformation Plan

| # | Column | Raw Type | Target Type | Transformation |
|---|--------|----------|-------------|----------------|
| 1 | product_id | string | string | Keep as-is (primary key) |
| 2 | product_name | string | string | Keep as-is |
| 3 | category | string | string + extracted | Split by `\|`, extract `main_category` (level 1), `sub_category` (level 2) |
| 4 | discounted_price | string (₹, commas) | float | Remove ₹, commas → numeric |
| 5 | actual_price | string (₹, commas) | float | Remove ₹, commas → numeric |
| 6 | discount_percentage | string (%) | float | Remove % → numeric (0-100) |
| 7 | rating | float | float | Keep as-is (validate 2.0-5.0) |
| 8 | rating_count | string (commas) | int | Remove commas → numeric, fill 2 missing |
| 9 | about_product | string (pipe-separated) | string | Keep as-is, optionally extract bullet count |
| 10 | user_id | string (comma-separated) | list/string | Parse to list for review analysis |
| 11 | user_name | string (comma-separated) | list/string | Parse to list |
| 12 | review_id | string (comma-separated) | list/string | Parse to list |
| 13 | review_title | string (comma-separated) | list/string | Parse to list |
| 14 | review_content | string (comma-separated) | list/string | Parse to list |
| 15 | img_link | string | string | Keep as-is |
| 16 | product_link | string | string | Keep as-is |

## Derived Features (Feature Engineering)
1. `price_savings` = actual_price - discounted_price
2. `discount_depth` = discount_percentage / 100
3. `main_category` = first level of category hierarchy
4. `category_depth` = number of levels in category path
5. `review_count_parsed` = count of reviews per product (from review_id)
6. `avg_review_length` = average character length of review_content
7. `has_high_rating` = rating >= 4.0 (boolean)
8. `is_heavily_discounted` = discount_percentage >= 50 (boolean)

## Implementation Steps

### Phase 1: Data Loading & Inspection (src/preprocessing.py)
- [ ] Load CSV with pandas
- [ ] Display shape, dtypes, head, info
- [ ] Check for duplicates (full row & product_id)
- [ ] Profile each column (unique counts, missing, stats)

### Phase 2: Data Cleaning & Preprocessing (src/preprocessing.py)
- [ ] Convert price columns to numeric
- [ ] Convert discount_percentage to numeric
- [ ] Convert rating_count to numeric, handle 2 missing values (median imputation)
- [ ] Parse category hierarchy → main_category, sub_category
- [ ] Parse review columns into lists (for potential future use)
- [ ] Remove exact duplicates (if any)
- [ ] Validate rating range (2.0-5.0)
- [ ] Create cleaned dataframe

### Phase 3: Feature Engineering (src/preprocessing.py)
- [ ] Create derived features listed above
- [ ] Encode categorical variables (main_category → label/one-hot)
- [ ] Scale numeric features for clustering (StandardScaler)

### Phase 4: Statistical Analysis (src/statistics.py)
- [ ] Descriptive statistics (mean, median, std, quartiles) for all numeric columns
- [ ] Correlation matrix (Pearson) for numeric features
- [ ] Covariance matrix for numeric features
- [ ] Grouping & aggregation:
  - By main_category: count, avg price, avg rating, avg discount
  - By rating bands: count, avg price
  - By discount bands: count, avg rating
- [ ] Pivot tables:
  - main_category vs discount bands (count, avg rating)
  - rating bands vs price bands (count)

### Phase 5: Exploratory Data Analysis & Visualization (src/eda.py)
- [ ] Distribution plots: rating, price, discount, rating_count
- [ ] Box plots: price by category, rating by discount band
- [ ] Scatter plots: rating vs price, discount vs rating, price vs rating_count
- [ ] Bar charts: top categories by count, avg rating by category
- [ ] Heatmap: correlation matrix
- [ ] Pair plot for key numeric features
- [ ] Category hierarchy treemap/sunburst

### Phase 6: K-Means Clustering (src/clustering.py)
- [ ] Prepare feature matrix (scaled numeric features)
- [ ] Elbow method to determine optimal k (test k=2 to 10)
- [ ] Silhouette analysis for validation
- [ ] Fit K-Means with optimal k
- [ ] Assign cluster labels to products
- [ ] Cluster profiling: describe each cluster (size, avg features)
- [ ] Visualize clusters (PCA 2D scatter colored by cluster)

### Phase 7: Cluster Analysis (src/clustering.py)
- [ ] Interpret clusters in business terms
- [ ] Map clusters to product segments (e.g., "Budget High-Rated", "Premium Low-Discount")
- [ ] Analyze cluster composition by category
- [ ] Identify representative products per cluster

### Phase 8: Rule-Based Recommendation (src/recommendation.py)
- [ ] Define recommendation rules:
  - Same cluster, higher rating
  - Same main_category, similar price range (±20%), higher rating
  - High discount (>50%) + high rating (>4.0) in same category
  - "Value picks": high rating, low price, moderate discount
- [ ] Implement `recommend(product_id, n=5)` function
- [ ] Test with sample products

### Phase 9: Streamlit GUI (app/app.py)
- [ ] Page 1: Dataset Overview (stats, sample data)
- [ ] Page 2: EDA Visualizations (interactive plots)
- [ ] Page 3: Statistical Analysis (correlation, pivot tables)
- [ ] Page 4: Clustering Results (cluster viz, profiles)
- [ ] Page 5: Product Recommender (input product_id → recommendations)
- [ ] Sidebar: filters (category, price range, rating range)

### Phase 10: Jupyter Notebook (notebooks/PriceSense_Analysis.ipynb)
- [ ] Complete end-to-end analysis narrative
- [ ] All visualizations with explanations
- [ ] Clustering interpretation
- [ ] Recommendation examples

## Deliverables
1. Cleaned dataset (CSV/Parquet)
2. Python modules in `src/`
3. Streamlit app in `app/app.py`
4. Jupyter notebook in `notebooks/`
5. Visualizations saved to `outputs/`
6. Model artifacts (K-Means, scaler) in `models/`

## Timeline Estimate
- Phase 1-2: 2-3 hours
- Phase 3: 1-2 hours
- Phase 4: 2-3 hours
- Phase 5: 3-4 hours
- Phase 6-7: 2-3 hours
- Phase 8: 1-2 hours
- Phase 9: 3-4 hours
- Phase 10: 2-3 hours
**Total**: ~16-24 hours