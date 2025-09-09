## Week_2_mini_assignment_first_data_analysis

## ☕️ Coffee Sales Analysis & Product Clustering

# Overview

This project loads a Point of sale dataset `Coffe_sales.csv` from Kaggle (https://www.kaggle.com/datasets/navjotkaushal/coffee-sales-dataset/data), runs quick exploratory data analysis (top drinks, weekday/hour trends, daily revenue), and clusters products with K-Means using: avg_price, popularity (transaction count), revenue, and Time_of_Day shares (Morning/Afternoon/Night).

# About the Dataset

This dataset contains coffee shop transaction records, including details about sales, payment type, time of purchase, and customer preferences.

With attributes covering time of day, weekdays, months, coffee types, and revenue, this dataset provides a strong foundation for analyzing customer behavior, sales patterns, and business performance trends.

Format: CSV (`Coffe_sales.csv`)

| Column name | feature |
| ------------- | ------------- |
| `hour_of_day` | Hour of purchase (0–23) |
| `cash_type` | Mode of payment (cash / card) |
| `money` | Transaction amount (local currency) |
| `coffee_name` | Product name (e.g., Latte, Americano, Hot Chocolate) |
| `Time_of_Day` | Categorized time of purchase (Morning, Afternoon, Night) |
| `Weekday` | Day of the week (Mon, Tue, …) |
| `Month_name` | Month of purchase (Jan, Feb, …) |
| `Weekdaysort` | Numeric weekday order (1 = Mon … 7 = Sun) |
| `Monthsort` | Numeric month order (1 = Jan … 12 = Dec) |
| `Date` | Date of transaction (YYYY-MM-DD) |
| `Time` | Exact time (HH:MM:SS) |

Note: The analysis script merges Date + the clock portion of Time into a proper DateTime for time-aware charts and features.

# Why this analysis? (Objectives)

- Operational visibility: see which products drive volume vs. margin.

- Menu structure: group items into practical buckets (e.g., bestsellers, premium treats, value staples).

- Staffing & inventory: align stock and labor with when items actually sell (dayparts).

- Promo strategy: pick the right items to discount/bundle or to upsell from.

## Quickstart

1) Create a Python environment
```bash
python3 -m venv ~/.week_2_mini_assign
source ~/.week_2_mini_assign/bin/activate
```

2) Install dependencies
Create `requirements.txt` and run:

```text
pandas
numpy
matplotlib
scikit-learn
```

```bash
pip install -r requirements.txt
```

3) Run
As a script:
```bash
coffee_analysis.py
```
Or run the notebook cells in order.


## Process

# 1. Load & Inspect

- Read `Coffe_sales.csv`.
- Quickly check the first rows (shape, obvious issues).
- Inspect dtypes and non-null counts.
- Summarize numeric columns (min/median/max).
- Count missing values and duplicate rows.
- List unique categories for `cash_type`, `coffee_name`, and `Time_of_Day`

# 2. Clean & Combine Time

- Parse `Date` (YYYY-MM-DD) and `Time` (HH:MM:SS) into datetimes.
- Strip the calendar from `Time` so only the clock portion remains.
- Build a true timestamp `DateTime`.
- Drop the original Time column once DateTime exists and create a clean dataframe `coffee_new` to use in future analysis.

Combining the real calendar date with clock time gives a proper timestamp for time-aware analysis (and future hourly models).

# 3. Exploratory data analysis (filter and grouping)

- By product `coffee_name`: compute transactions (row count) and revenue (sum of money), then sort to see top sellers.
- By weekday: aggregate transactions and revenue and reindex to Mon - Sun to keep charts in calendar order.
- By month: aggregate by Monthsort to preserve Jan - Dec order.

# Visuals:
- Top coffees by revenue: a bar chart ranking products.
- Transactions by hour of day: a bar chart using `hour_of_day`.
- Daily revenue trend: a time series chart summing revenue by `Date`.
- Correlation heatmap: a quick look at relationships among numeric fields.

# 4. Machine learning model: Product Clustering (K-Means)

- Per-product features:
`avg_price`, `popularity`, `revenue`, `Time_of_Day shares` (fraction of sales in Morning/Afternoon/Night)

- Standardize features, sweep k=2..9 (bounded by #items)

- Pick k by silhouette (elbow printed for context)

- Fit K-Means and print:
- Cluster sizes
- Cluster feature means
- Top 5 coffees per cluster by revenue

# 5. PCA Scatter plot

- Re-standardize the same features used for clustering.

- Project them into 2D using PCA.

- Plot the products colored by their cluster; annotate with product names.

## Why K-Means here?

- We want simple, explainable groups of products without labels.

- Price–volume–revenue and daypart behavior naturally separate items into buckets that map to business actions (stocking, placement, promos).

- K-Means is fast, repeatable, and works well with standardized numeric features.

## Findings and Insights

# Product mix

Top-earning drinks are Latte and Americano with Milk (clear leaders on the revenue bar chart). Cappuccino, Hot Chocolate, Cocoa follow; Espresso is lowest among the listed items. 

# When customers buy

Traffic by hour peaks in the late morning (around 9–11) and has another lift mid/late afternoon; evenings taper. 

# Revenue over time

Daily revenue is volatile with recurring spikes; no obvious long-term decline is visible in the plotted period. Use rolling averages for staffing and target setting. 

# Correlations

Numeric features show weak direct correlations (e.g., hour vs. spend), implying patterns are more categorical/time-bucket driven than linear. 

# K-Means clustering (chosen k=3)

Model selection: Elbow bends around 3–4; best silhouette at k=3. Good separation with a compact number of clusters. 

Cluster sizes: 2, 3, 3 products. 

Cluster profiles (means):

1. Cluster 0: “Bestseller workhorses”: highest popularity & revenue, balanced dayparts (~0.35/0.33/0.33). Latte, Americano with Milk. 

2. Cluster 1: “Premium evening treats”: highest avg_price, night-heavy (≈0.46). Cappuccino, Hot Chocolate, Cocoa. 

3. Cluster 2: “Value daytime staples”: lowest price, day-leaning (Morning/Afternoon ≈0.41/0.38). Americano, Cortado, Espresso. 

## What this means (business implications)

1. Menu & merchandising

Put Cluster 0 front-and-center on the menu (high visibility, minimal discounting).

Feature Cluster 1 as premium/night items (dessert pairings, seasonal creatives).

Use Cluster 2 as value anchors with explicit upsells (“+ milk → Latte”).

2. Inventory & purchasing

Milk/espresso supplies should be protected for Cluster 0 to avoid stockouts.

Cocoa/whipped toppings and evening disposables align with Cluster 1.

Beans & small-format cups align with Cluster 2 (morning/afternoon rush).
(Base this on the daypart shares in each cluster.) 

3. Staffing & scheduling

Staff heavier late morning and mid/late afternoon (traffic peaks). 

For evenings, ensure at least one barista proficient with premium foam/latte art for Cluster-1 orders.

4. Promotions & pricing

Cluster 2 (value daytime): run morning bundles (coffee + pastry) to drive traffic; use them as upsell feeders to Cluster-0 lattes.

Cluster 1 (evening): push dessert combos; limited-time flavors.

Test small price increases (+3–5%) on Cluster 0 (inelastic high-demand items). Monitor attach rate and wait time.
