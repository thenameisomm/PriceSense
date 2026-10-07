# Project Rules - PriceSense

## Project Scope
PriceSense — Amazon Product Analysis and Recommendation System

## Syllabus Requirements (DSML Mini-Project)
The project must implement the following in order:
1. Data loading and inspection
2. Data preprocessing
3. Data cleaning
4. Missing-value handling
5. Duplicate handling
6. Data transformation
7. Feature engineering
8. Statistical analysis
9. Exploratory Data Analysis
10. Visualization
11. Correlation
12. Covariance
13. Grouping and aggregation
14. Pivot tables
15. K-Means clustering
16. Cluster analysis
17. Simple rule-based product recommendation
18. Streamlit GUI

## Prohibited Techniques
Do NOT implement:
- Future-price prediction
- LSTM, GRU, XGBoost, LightGBM, CatBoost
- ARIMA, SARIMA, Prophet
- Deep learning, neural networks, transformers
- NLP, sentiment analysis
- Collaborative filtering, embeddings, cosine similarity
- Web scraping, Amazon APIs, live price tracking
- Advanced recommendation systems

## Only Allowed ML Algorithm
- K-Means clustering (for product segmentation)

## Dataset
- Source: `data/amazon.csv`
- 1465 rows × 16 columns
- Amazon India product data with reviews

## Code Structure
- `src/preprocessing.py` — Data loading, cleaning, transformation, feature engineering
- `src/statistics.py` — Statistical analysis, correlation, covariance, grouping, pivot tables
- `src/eda.py` — Exploratory Data Analysis, visualization
- `src/clustering.py` — K-Means clustering, cluster analysis
- `src/recommendation.py` — Rule-based product recommendation
- `app/app.py` — Streamlit GUI
- `notebooks/PriceSense_Analysis.ipynb` — Jupyter notebook for analysis

## Data Schema (Actual)
| Column | Type | Description |
|--------|------|-------------|
| product_id | string | Unique product identifier (ASIN) |
| product_name | string | Product title |
| category | string | Hierarchical category (pipe-separated) |
| discounted_price | string | Price after discount (₹ symbol, commas) |
| actual_price | string | Original price (₹ symbol, commas) |
| discount_percentage | string | Discount % (with % symbol) |
| rating | float | Average rating (2.0-5.0) |
| rating_count | string | Number of ratings (with commas) |
| about_product | string | Product description bullets (pipe-separated) |
| user_id | string | Comma-separated reviewer IDs |
| user_name | string | Comma-separated reviewer names |
| review_id | string | Comma-separated review IDs |
| review_title | string | Comma-separated review titles |
| review_content | string | Comma-separated review contents |
| img_link | string | Product image URL |
| product_link | string | Amazon product page URL |

## Key Data Quality Issues
1. Price columns have ₹ symbol and commas — need numeric conversion
2. discount_percentage has % symbol — need numeric conversion
3. rating_count has commas — need numeric conversion
4. category is hierarchical (pipe-separated) — can extract main category
5. Review columns are comma-separated multi-value — need parsing for analysis
6. 2 missing values in rating_count
7. 211 unique category paths — can group by top-level category