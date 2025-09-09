import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

## Load the dataset
coffee = pd.read_csv("Coffe_sales.csv")

## Inspect the data
# display the first few rows of the dataset
coffee.head()
# understand the structure and data types of the dataset
coffee.info()
# get summary statistics of the dataset
coffee.describe()
# Check for missing values
coffee.isnull().sum()
# Check for duplicate rows
coffee.duplicated().sum()
# Check for unique values in 'cash_type' column
coffee["cash_type"].unique()
# Check for unique values in 'coffee_name' column
coffee["coffee_name"].unique()
## Check for unique values in 'Time_of_Day' column
coffee["Time_of_Day"].unique()

## Data Cleaning
# change the data types of 'Date' and 'Time' columns to datetime
coffee["Date"] = pd.to_datetime(coffee["Date"], errors="coerce")
coffee["Time"] = pd.to_datetime(
    coffee["Time"], errors="coerce"
)  # has wrong date, but correct clock time

# Extract just the time-of-day as a timedelta, then add to the real Date
time_of_day = coffee["Time"] - coffee["Time"].dt.normalize()
coffee["DateTime"] = coffee["Date"].dt.normalize() + time_of_day

# drop the old time column
coffee_new = coffee.drop("Time", axis=1)
# check updated dataset coffee_new after data time conversion
coffee_new.info()
coffee_new.sample(n=5)

## Basic filtering and grouping
# By coffee type revenue and transactions number
by_coffee = coffee_new.groupby("coffee_name").agg(
    transactions=("coffee_name", "size"), revenue=("money", "sum")
)
by_coffee.sort_values(["transactions", "revenue"], ascending=False)

# By weekday revenue and transactions number
coffee_new["Weekday"].unique()
Weekday_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
by_weekday = (
    coffee_new.groupby("Weekday")
    .agg(transactions=("Weekday", "size"), revenue=("money", "sum"))
    .reindex(Weekday_order)
)
by_weekday

# By month revenue and transactions number
by_month = coffee_new.groupby("Monthsort").agg(
    transactions=("Monthsort", "size"), revenue=("money", "sum")
)
by_month

## visualization of revenue by coffee type
by_coffee_rev = (
    coffee_new.groupby("coffee_name")["money"].sum().sort_values(ascending=False)
)
by_coffee_rev

by_coffee_rev.plot(kind="bar")
plt.title("Top Coffees by Revenue")
plt.xlabel("Coffee Name")
plt.ylabel("Revenue")
plt.show()

## hour_of_day transaction count plot
by_hour = coffee_new.groupby("hour_of_day").size()
by_hour.plot(kind="bar")
plt.title("Transactions by Hour of Day")
plt.xlabel("Hour of Day")
plt.ylabel("Number of Transactions")
plt.show()

## daily revenue time series plot
daily_revenue = coffee_new.groupby("Date")["money"].sum()
daily_revenue.plot()
plt.title("Daily Revenue Over Time")
plt.xlabel("Date")
plt.ylabel("Revenue")
plt.show()

## correlation heatmap
num = coffee_new.select_dtypes(include=[np.number]).copy()
corr = num.corr(method="pearson")
plt.imshow(corr, interpolation="nearest")
plt.title("Numeric feature correlation")
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.colorbar()
plt.tight_layout()
plt.show()

## Machine learning model
# Product Clustering (KMeans)
# Goal: cluster coffee products by behavior to inform menu layout/promos.
# Features per coffee:
#   - avg_price (mean of `money`)
#   - popularity (transaction count)
#   - revenue (sum money)
#   - daypart shares (Morning/Afternoon/Night)


# 1) Per-product base features
per_coffee = coffee_new.groupby("coffee_name", as_index=False).agg(
    avg_price=("money", "mean"),
    popularity=("coffee_name", "size"),
    revenue=("money", "sum"),
)

# 2) Daypart shares (auto-detect categories from Time_of_Day)
cats = coffee_new["Time_of_Day"].astype(str).str.title().str.strip().unique().tolist()

dp_counts = (
    coffee_new.assign(
        Time_of_Day=lambda d: d["Time_of_Day"].astype(str).str.title().str.strip()
    )
    .pivot_table(
        index="coffee_name",
        columns="Time_of_Day",
        values="money",
        aggfunc="size",
        fill_value=0,
    )
    .reindex(columns=cats, fill_value=0)
)

dp_shares = dp_counts.div(dp_counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
per_coffee = per_coffee.merge(
    dp_shares.add_suffix("_share"), left_on="coffee_name", right_index=True, how="left"
).fillna(0)

# 3) Features, scale, choose k, fit
share_cols = [f"{c}_share" for c in cats]
feat_cols = ["avg_price", "popularity", "revenue"] + share_cols
X = per_coffee[feat_cols].values
Xz = StandardScaler().fit_transform(X)

n_items = len(per_coffee)
ks = list(range(2, min(10, n_items))) or [2]

inertias, sils = [], []
for k in ks:
    km = KMeans(n_clusters=k, n_init=20, random_state=42)
    labels = km.fit_predict(Xz)
    inertias.append(km.inertia_)
    sils.append(silhouette_score(Xz, labels) if n_items > k else np.nan)

best_k = ks[int(np.nanargmax(sils))] if any(not np.isnan(s) for s in sils) else 2
print("Elbow (inertia):", {k: round(i, 1) for k, i in zip(ks, inertias)})
print(
    "Silhouette:", {k: (None if np.isnan(s) else round(s, 3)) for k, s in zip(ks, sils)}
)
print("Chosen k:", best_k)

kmeans = KMeans(n_clusters=best_k, n_init=50, random_state=42)
per_coffee["cluster"] = kmeans.fit_predict(Xz)

# 4) Inspect results
print("\nCluster sizes:")
print(per_coffee["cluster"].value_counts().sort_index())

cluster_profile = (
    per_coffee.groupby("cluster")[feat_cols]
    .mean()
    .round(2)
    .sort_values("revenue", ascending=False)
)
print("\nCluster profile (feature means):")
print(cluster_profile)

print("\nTop 5 coffees per cluster (by revenue):")
for c in sorted(per_coffee["cluster"].unique()):
    cols = ["coffee_name", "revenue", "popularity", "avg_price"] + share_cols
    top = (
        per_coffee[per_coffee["cluster"] == c]
        .sort_values("revenue", ascending=False)
        .head(5)[cols]
    )
    print(f"\nCluster {c}:")
    print(top.to_string(index=False))

## PCA scatter of items colored by cluster
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Use the same features you clustered on
share_cols = [c for c in per_coffee.columns if c.endswith("_share")]
feat_cols = ["avg_price", "popularity", "revenue"] + share_cols

X = per_coffee[feat_cols].values
Xz = StandardScaler().fit_transform(X)

coords = PCA(n_components=2, random_state=42).fit_transform(Xz)
per_coffee["pc1"], per_coffee["pc2"] = coords[:, 0], coords[:, 1]

plt.figure(figsize=(7, 5))
for c in sorted(per_coffee["cluster"].unique()):
    sub = per_coffee[per_coffee["cluster"] == c]
    plt.scatter(sub["pc1"], sub["pc2"], s=50, label=f"Cluster {c}")

# annotate each point with product name
annotate_names = True
if annotate_names:
    for _, r in per_coffee.iterrows():
        plt.text(r["pc1"], r["pc2"], r["coffee_name"], fontsize=8)

plt.title("KMeans clusters — PCA 2D")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(title="Clusters")
plt.tight_layout()
plt.savefig("cluster_pca_scatter_simple.png", dpi=200, bbox_inches="tight")
plt.show()
