# PriceSense

## Amazon Product Analysis and Recommendation System

PriceSense analyzes Amazon India product data using comprehensive data science techniques and provides explainable, rule-based product recommendations through an interactive Streamlit dashboard.

Originally developed as an academic Data Science and Machine Learning project.

---

## 1. Overview

### Problem Statement
E-commerce platforms like Amazon host millions of products with varying prices, discounts, ratings, and review counts. Consumers struggle to navigate this vast catalog to find products that match their preferences for quality, value, and price range.

### Purpose
PriceSense addresses this by:
- **Analyzing** 1,351 Amazon India products across 9 categories using statistical and visual exploratory data analysis
- **Segmenting** products into 3 meaningful clusters using K-Means clustering
- **Recommending** products through a transparent, rule-based system with 5 distinct strategies
- **Presenting** all insights through an interactive Streamlit dashboard

### Why Recommendations Are Explainable
Every recommendation is traceable to explicit, human-readable rules:
- Filter by category, price ceiling, minimum rating, minimum discount
- Rank by composite score with documented weights (Rating 40%, Review Count 30%, Discount 20%, Cluster Boost 10%)
- No collaborative filtering, no embeddings, no black-box models
- Fully transparent rule-based logic — no hidden ML models

---

## 2. Features

| Feature | Description |
|---------|-------------|
| **Data Preprocessing** | Handles Indian currency (₹, commas), percentage strings, missing values (median imputation), duplicate product_ids |
| **Feature Engineering** | 12 derived features: price_savings, discount_depth, price_ratio, boolean flags (high_rating, heavily_discounted, budget, premium), log transforms, review parsing |
| **Statistical Analysis** | Central tendency, dispersion, distribution shape (skew/kurtosis), percentiles, Pearson & Spearman correlation, covariance, grouped statistics, pivot tables |
| **Exploratory Data Analysis** | 36 visualizations: histograms with KDE, box plots (single & by category), correlation heatmaps, category bar charts, pivot heatmaps, scatter plots, pairplot, violin plot, cumulative distribution, price savings |
| **K-Means Clustering** | 6 features, StandardScaler, Elbow Method + Silhouette for K selection (K=3), PCA for 2D visualization (60.3% variance) |
| **Cluster Analysis** | Category composition, discount band composition, business interpretation with strategic implications |
| **Rule-Based Recommendation** | 4 filter parameters, 5 strategies, 19,397 precomputed recommendations |
| **Streamlit Dashboard** | 5 pages: Home, EDA, Clustering, Recommendation, About |

---

## 3. DSML Concepts Demonstrated

| Concept | Implementation |
|---------|----------------|
| Data Loading & Inspection | `src/preprocessing.py: load_dataset(), inspect_dataset()` |
| Data Preprocessing | Complete pipeline: load → clean → transform → engineer |
| Data Cleaning | Price (₹, commas), discount (%), rating, rating_count |
| Missing Value Handling | Median imputation for 2 missing `rating_count` (robust to extreme skew) |
| Duplicate Handling | 0 exact duplicates; 114 rows with repeated product_ids removed (kept first) |
| Data Transformation | String → numeric, category hierarchy parsing (5 levels) |
| Feature Engineering | 12 derived features with business justifications |
| Descriptive Statistics | Mean, median, mode, std, variance, range, skew, kurtosis, percentiles |
| Exploratory Data Analysis | 36 charts covering all required visualization types |
| Correlation Analysis | Pearson & Spearman matrices + target correlations |
| Covariance Analysis | Covariance matrix for key numeric variables |
| Grouping & Aggregation | By main_category (9), discount bands (5) |
| Pivot Tables | Category × Discount Band (count, mean rating/price/discount) |
| **K-Means Clustering** | 6 features, StandardScaler, K=3 |
| Elbow Method & Silhouette | Tested K=2-10, inertia & silhouette computed |
| Cluster Analysis | Category/discount composition heatmaps, business interpretation |
| Rule-Based Recommendation | 5 strategies, composite scoring, bulk generation |
| Streamlit GUI | 5-page interactive dashboard |

---

## 4. Dataset

**File**: `data/amazon.csv`  
**Raw Size**: 1,465 rows × 16 columns  
**Cleaned Size**: 1,351 rows × 34 columns  
**Source**: Amazon India product data with aggregated reviews

### Key Raw Columns
| Column | Type | Description |
|--------|------|-------------|
| `product_id` | string | Unique ASIN identifier |
| `product_name` | string | Product title |
| `category` | string | Hierarchical (pipe-separated, e.g., `Electronics\|Audio\|Headphones`) |
| `discounted_price` | string | Price after discount (₹ symbol, commas) |
| `actual_price` | string | Original price (₹ symbol, commas) |
| `discount_percentage` | string | Discount depth (with % symbol) |
| `rating` | float | Average rating (2.0-5.0) |
| `rating_count` | string | Number of ratings (comma-separated) |
| `about_product` | string | Description bullets (pipe-separated) |
| `user_id` / `user_name` / `review_id` / `review_title` / `review_content` | string | Comma-separated review data |
| `img_link`, `product_link` | string | Amazon URLs |

### Preprocessing Performed
1. **Price columns**: `₹1,899` → `1899.0` (remove ₹ and commas)
2. **Discount**: `64%` → `64.0` (remove %)
3. **Rating**: Validate 0-5 range, convert to float
4. **Rating count**: `24,269` → `24269.0`, median imputation (4,740) for 2 missing
5. **Categories**: Split by `|` → `main_category` (9), `sub_category`, levels 3-5, `category_depth`
6. **Duplicates**: 114 rows with repeated product_ids removed (kept first occurrence)
7. **Feature engineering**: 12 derived features created

---

## 5. Project Architecture

```
PriceSense/
├── data/
│   └── amazon.csv                    # Raw dataset (1,465 × 16)
├── outputs/
│   ├── cleaned_data.csv              # Cleaned data (1,351 × 34)
│   ├── cluster_results.csv           # Product IDs + cluster labels (0,1,2)
│   ├── cluster_statistics.csv        # Detailed cluster stats
│   ├── cluster_interpretations.csv   # Cluster labels & metrics
│   ├── recommendations.csv           # All 19,397 recommendations (5 strategies)
│   ├── cluster_business_interpretation.txt
│   ├── statistics/                   # 7 statistical CSV files
│   │   ├── summary.csv
│   │   ├── correlation.csv
│   │   ├── covariance.csv
│   │   ├── group_by_category.csv
│   │   ├── group_by_discount_band.csv
│   │   ├── pivot_category_discount.csv
│   │   └── pivot_category_rating.csv
│   └── charts/                       # 36 PNG visualizations
├── models/
│   ├── kmeans_model.pkl              # Trained K-Means model
│   ├── scaler.pkl                    # StandardScaler
│   └── feature_names.pkl             # Feature names for clustering
├── src/
│   ├── __init__.py
│   ├── preprocessing.py              # Data loading, cleaning, feature engineering
│   ├── descriptive_stats.py          # Statistical analysis functions
│   ├── eda.py                        # Visualization functions
│   ├── clustering.py                 # K-Means + cluster analysis
│   ├── recommendation.py             # Rule-based recommendation engine
│   └── app.py                        # Streamlit dashboard (ENTRY POINT)
├── notebooks/
│   └── PriceSense_Analysis.ipynb     # Complete analysis narrative (60 cells)
├── PROJECT_RULES.md                  # Syllabus constraints & scope
├── PROJECT_PLAN.md                   # Implementation plan
├── requirements.txt                  # Python dependencies
└── .gitignore
```

---

## 6. How PriceSense Works

### Pipeline Flow
```
Amazon Dataset (data/amazon.csv)
         ↓
    Preprocessing (src/preprocessing.py)
    - Currency cleaning, missing value imputation
    - Duplicate handling, category parsing
    - 12 feature engineering steps
         ↓
    Statistical Analysis (src/descriptive_stats.py)
    - Summary stats, correlation, covariance
    - Grouped stats, pivot tables
         ↓
    EDA & Visualization (src/eda.py)
    - 36 charts saved to outputs/charts/
         ↓
    Feature Engineering for Clustering
    - 6 features selected with justification
    - StandardScaler (zero mean, unit variance)
         ↓
    K-Means Clustering (src/clustering.py)
    - 6 features selected with justification
    - StandardScaler (zero mean, unit variance)
    - Elbow Method + Silhouette → K=3
    - Training: n_init=20, max_iter=300, random_state=42
         ↓
    Cluster Analysis (src/clustering.py)
    - Category & discount band composition
    - Business interpretation
    - PCA visualization (60.3% variance)
         ↓
    Rule-Based Recommendation (src/recommendation.py)
    - 5 strategies, 4 filters, composite scoring
    - 19,397 precomputed recommendations
         ↓
    Streamlit Dashboard (src/app.py)
    - 5 interactive pages
```

---

## 7. K-Means Clustering

### Selected Features (6) with Justification
| Feature | Rationale |
|---------|-----------|
| `discounted_price` | Primary purchase decision factor |
| `discount_percentage` | Pricing strategy & value perception |
| `rating` | Product quality/satisfaction signal |
| `rating_count` | Popularity/market presence proxy |
| `log_rating_count` | Handles extreme skew (max 426,973) |
| `log_discounted_price` | Handles right-skew (max ₹77,990) |

**Excluded**: `actual_price` (r=0.96 with discounted_price), derived features (redundant)

### Scaling
**StandardScaler** (zero mean, unit variance) — required because K-Means uses Euclidean distance and features have vastly different scales (price: 39-77,990 vs rating: 2-5).

### K Selection
| K | Inertia | Silhouette |
|---|---------|------------|
| 2 | 6,484 | 0.2118 |
| **3** | **5,243** | **0.2188** |
| 4 | 4,318 | 0.2317 |
| 5 | 3,693 | 0.2286 |
| 6 | 3,300 | 0.2225 |
| 7 | 2,943 | **0.2349** |
| 8 | 2,725 | 0.2305 |
| 9 | 2,513 | 0.2280 |
| 10 | 2,350 | 0.2220 |

**Decision**: K=3 chosen — Elbow method primary criterion (inertia decrease slows at K=3); silhouette at K=3 (0.2188) close to K=7 (0.2349); simpler model preferred for interpretability.

### Cluster Results (K=3)
| Cluster | Label | Size | % | Avg Price | Avg Discount | Avg Rating | Avg Reviews |
|---------|-------|------|---|-----------|--------------|------------|-------------|
| 0 | Mainstream | 646 | 47.8% | ₹1,461 | 38.9% | 4.20 | 31,196 |
| 1 | Budget/Deep-Discount | 555 | 41.1% | ₹1,023 | 59.8% | 3.94 | 2,780 |
| 2 | Premium | 150 | 11.1% | ₹19,687 | 31.6% | 4.18 | 14,108 |

### PCA Visualization
- **Explained Variance**: 60.3% (PC1=34.3%, PC2=26.0%)
- PC1 primarily loads on price features
- PC2 primarily loads on rating_count
- Clusters well-separated in 2D space

---

## 8. Recommendation System

**Design Philosophy**: Transparent, rule-based, and fully explainable. No collaborative filtering, embeddings, cosine similarity, or user behavior tracking.

### User Input Parameters
| Parameter | Type | Description |
|-----------|------|-------------|
| Category | Optional | Filter by main_category (9 categories) |
| Max Price | Optional | Maximum discounted price in ₹ |
| Min Rating | Optional | Minimum rating (0-5 scale) |
| Min Discount | Optional | Minimum discount percentage (0-100) |

### Filtering Pipeline
Sequential: Category → Price → Rating → Discount

### Ranking: Composite Score
```
Score = 0.40 × normalized_rating
      + 0.30 × normalized_log_review_count
      + 0.20 × normalized_discount_percentage
      + 0.10 × cluster_boost (Cluster 0 preferred)
```

### 5 Recommendation Strategies
| Strategy | Description |
|----------|-------------|
| **Cluster-Based** | Products from same K-Means cluster |
| **Category Better Rated** | Higher-rated alternatives in same category & price range (±20%) |
| **Budget Alternative** | Cheaper options (≥3.5 rating) in same category |
| **Premium Upgrade** | Higher-priced, highly-rated (≥4.2) products in same category |
| **Price Range** | Products within ±30% price range |

### Output
19,397 total recommendations precomputed and saved to `outputs/recommendations.csv`

---

## 9. Technologies

| Category | Technologies |
|----------|--------------|
| **Language** | Python 3.8+ |
| **Data Science** | pandas, NumPy, scikit-learn |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **Web App** | Streamlit |
| **Notebook** | Jupyter, ipykernel |
| **Serialization** | joblib (model persistence) |

---

## 10. Local Installation

### Prerequisites
- Python 3.8+
- Git

### Setup
```bash
# Clone repository
git clone <repository-url>
cd PriceSense

# Create virtual environment (recommended for PEP 668 compliance)
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip and install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

## 11. Running PriceSense

### Streamlit Dashboard (Main GUI)
```bash
streamlit run src/app.py
```
Opens at **http://localhost:8501** with 5 pages:
- **🏠 Home** — Project overview, key metrics, quick insights
- **📈 EDA** — Interactive visualizations with sidebar filters
- **🎯 Clustering** — K-Means results, PCA, cluster profiles, business interpretation
- **🎁 Recommendation** — Product-to-product recommendations (5 strategies)
- **ℹ️ About** — Project details, methodology, syllabus compliance

### Jupyter Notebook
```bash
cd notebooks
jupyter notebook PriceSense_Analysis.ipynb
```
Run from `notebooks/` directory. Complete end-to-end analysis with all 21 required sections.

### Individual Modules
```bash
python -m src.preprocessing      # Full preprocessing pipeline
python -m src.descriptive_stats  # Statistical analysis (7 CSVs)
python -m src.eda                # EDA visualizations (36 charts)
python -m src.clustering         # K-Means + cluster analysis
python -m src.recommendation     # Recommendation demo + bulk generation
```

---

## 12. Generated Outputs

| Output | Location | Description |
|--------|----------|-------------|
| Cleaned dataset | `outputs/cleaned_data.csv` | 1,351 rows × 34 columns |
| Statistical reports | `outputs/statistics/*.csv` | 7 CSV files (summary, correlation, covariance, grouped, pivots) |
| Visualizations | `outputs/charts/*.png` | 36 PNG charts |
| Cluster assignments | `outputs/cluster_results.csv` | Product ID + cluster (0,1,2) |
| Cluster analysis | `outputs/cluster_*.csv` | Statistics, interpretations, compositions |
| Recommendations | `outputs/recommendations.csv` | 19,397 recommendations (5 strategies) |
| Model artifacts | `models/*.pkl` | K-Means model, scaler, feature names |
| Business interpretation | `outputs/cluster_business_interpretation.txt` | Detailed segment analysis |

---

## 13. Deployment (Streamlit Community Cloud)

### Deployment Process
1. **Push** this repository to GitHub
2. **Sign in** to [Streamlit Community Cloud](https://share.streamlit.io/)
3. **Create** a new app
4. **Select** your repository and branch (`main`)
5. **Enter** the entry point path: `src/app.py`
6. **Deploy** — Streamlit installs requirements from `requirements.txt`
7. **Verify** the app loads all 5 pages correctly

### Files Required in GitHub
| File/Directory | Required? | Reason |
|----------------|-----------|--------|
| `data/amazon.csv` | ✅ Yes | Raw dataset (1.5 MB) |
| `outputs/` | ✅ Yes | All generated artifacts (cleaned data, clusters, stats, charts, recommendations) |
| `models/` | ✅ Yes | Pre-trained K-Means, scaler, feature names |
| `src/` | ✅ Yes | All Python modules |
| `notebooks/` | ✅ Yes | Analysis notebook |
| `requirements.txt` | ✅ Yes | Dependencies |
| `README.md` | ✅ Yes | Documentation |
| `PROJECT_RULES.md` | Optional | Academic constraints reference |
| `PROJECT_PLAN.md` | Optional | Architecture reference |
| `.gitignore` | ✅ Yes | Excludes venv, cache, IDE files |

### Files NOT to Commit
| Pattern | Reason |
|---------|--------|
| `.venv/`, `venv/`, `env/` | Virtual environments |
| `__pycache__/`, `*.pyc` | Python bytecode cache |
| `.DS_Store` | macOS metadata |
| `*.ipynb_checkpoints/` | Jupyter checkpoints |
| `.vscode/`, `.idea/` | IDE configuration |
| `.env`, `.env.*` | Environment variables/secrets |
| `*.log` | Log files |

### Deployment Strategy
- **Dataset**: Committed to GitHub (small enough at ~1.5 MB)
- **Outputs**: Committed to GitHub — the Streamlit app loads pre-computed CSVs and charts directly (no dynamic regeneration at runtime)
- **Models**: Committed to GitHub — `src/app.py` uses CSV outputs, but models are available for reference
- **No secrets**: No API keys, tokens, or environment variables required
- **No external dependencies**: Fully self-contained

---

## 14. Limitations

| Limitation | Details |
|------------|---------|
| **Static dataset** | No live Amazon integration, no web scraping, no API access |
| **Rule-based only** | No collaborative filtering, matrix factorization, embeddings, or user behavior modeling |
| **Single ML algorithm** | K-Means only (per DSML syllabus constraints) |
| **Aggregated reviews** | Review data stored as comma-separated strings, not individual review rows |
| **Category granularity** | Only main_category (9 values) used for grouping; sub-categories not fully utilized |
| **Clustering features** | Limited to 6 numerical features; categorical features not encoded for clustering |
| **No temporal analysis** | Dataset lacks timestamps; no trend analysis possible |

---

## 15. Future Scope (NOT Implemented)

> **Note**: The following are potential improvements and are NOT implemented in this project.

- Live price tracking via Amazon Product Advertising API
- Collaborative filtering using user review history
- NLP-based sentiment analysis on review content
- Time-series forecasting for price trends (ARIMA, Prophet)
- Advanced recommendation: matrix factorization, neural collaborative filtering
- A/B testing framework for recommendation strategies
- Mobile-responsive UI enhancements
- Multi-language support

---

## 16. Academic Context

This project was originally developed as an academic **Data Science and Machine Learning (DSML) Mini-Project** with 18 required components:

1. **All 18 requirements implemented and verified**
2. **K-Means clustering** used as the primary ML algorithm
3. **Prohibited techniques explicitly excluded** (future-price prediction, XGBoost, LSTM, ARIMA, deep learning, NLP, collaborative filtering, web scraping, advanced recommenders)
4. **Explainability emphasized** — every recommendation traceable to explicit rules
5. **Code modularity** — reusable functions in `src/`, suitable for demonstration and review
6. **Academic documentation** — `PROJECT_RULES.md` and `PROJECT_PLAN.md` capture constraints and design decisions

---

## 17. Author

**PriceSense** — Amazon Product Analysis and Recommendation System  
Originally developed as an academic Data Science and Machine Learning project

---

## License

Educational Use