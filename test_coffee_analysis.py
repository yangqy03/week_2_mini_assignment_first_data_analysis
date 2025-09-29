import os
import sys
import importlib
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

MODULE_NAME = "coffee_analysis"


# Fixtures: temp repo & import


@pytest.fixture(scope="session")
def tmp_repo(tmp_path_factory):
    """
    Create an isolated temp repo-like directory structure.
    Steps:
      - Makes a temporary /data folder
      - Writes a small but realistic CSV that matches the columns your script uses
      - Ensures the tests don't depend on your real dataset

    Columns covered:
      Date, Time, Time_of_Day, Weekday, Monthsort, money,
      coffee_name, hour_of_day, cash_type
    """
    root = tmp_path_factory.mktemp("repo")
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Minimal yet representative dataset with multiple coffees, dayparts, weekdays
    df = pd.DataFrame(
        [
            # Morning purchases
            {
                "Date": "2025-01-10",
                "Time": "1900-01-01 09:05:00",
                "Time_of_Day": "Morning",
                "Weekday": "Fri",
                "Monthsort": 1,
                "money": 4.50,
                "coffee_name": "Latte",
                "hour_of_day": 9,
                "cash_type": "Card",
            },
            {
                "Date": "2025-01-10",
                "Time": "1900-01-01 09:15:00",
                "Time_of_Day": "Morning",
                "Weekday": "Fri",
                "Monthsort": 1,
                "money": 3.00,
                "coffee_name": "Americano",
                "hour_of_day": 9,
                "cash_type": "Cash",
            },
            {
                "Date": "2025-01-11",
                "Time": "1900-01-01 09:20:00",
                "Time_of_Day": "Morning",
                "Weekday": "Sat",
                "Monthsort": 1,
                "money": 5.00,
                "coffee_name": "Cappuccino",
                "hour_of_day": 9,
                "cash_type": "Card",
            },
            # Afternoon purchases
            {
                "Date": "2025-01-11",
                "Time": "1900-01-01 14:00:00",
                "Time_of_Day": "Afternoon",
                "Weekday": "Sat",
                "Monthsort": 1,
                "money": 4.00,
                "coffee_name": "Latte",
                "hour_of_day": 14,
                "cash_type": "Cash",
            },
            {
                "Date": "2025-01-12",
                "Time": "1900-01-01 14:30:00",
                "Time_of_Day": "Afternoon",
                "Weekday": "Sun",
                "Monthsort": 1,
                "money": 2.75,
                "coffee_name": "Americano",
                "hour_of_day": 14,
                "cash_type": "Card",
            },
            # Night purchases
            {
                "Date": "2025-01-12",
                "Time": "1900-01-01 20:10:00",
                "Time_of_Day": "Night",
                "Weekday": "Sun",
                "Monthsort": 1,
                "money": 6.50,
                "coffee_name": "Latte",
                "hour_of_day": 20,
                "cash_type": "Card",
            },
        ]
    )
    (data_dir / "Coffe_sales.csv").write_text(df.to_csv(index=False))
    return root


@pytest.fixture()
def coffee_module(tmp_repo, monkeypatch):
    """
    Prepare the runtime environment and import the script.

    Steps:
      1) Ensure the original project root is importable (sys.path)
      2) chdir into the temporary repo so the script reads the synthetic CSV
      3) Set matplotlib to a non-GUI backend (Agg) for saving plots in CI
      4) Import script once, and return it for tests

    Why:
      - The script runs at import time, creating variables & plots we can assert on.
    """
    # Add original project root to sys.path so one can import the module by name
    project_root = Path.cwd()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # Switch working dir to the temp repo so the script reads temp CSV & writes plots there
    monkeypatch.chdir(tmp_repo)

    # Headless plotting (no display needed)
    os.environ["MPLBACKEND"] = "Agg"

    # Import the user's script as a module
    mod = importlib.import_module(MODULE_NAME)
    return mod


# Unit tests: loading & preprocessing


def test_data_loading_and_datetime(coffee_module):
    """
    Verify data loading and core datetime preprocessing.

    Checks:
      - Raw DataFrame exists and is non-empty
      - Cleaned DataFrame has DateTime column
      - Date and DateTime are proper datetime dtypes
      - Original Time column is dropped after transformation
    """
    assert isinstance(coffee_module.coffee, pd.DataFrame)
    assert not coffee_module.coffee.empty

    cn = coffee_module.coffee_new
    assert "DateTime" in cn.columns
    assert pd.api.types.is_datetime64_any_dtype(cn["Date"])
    assert pd.api.types.is_datetime64_any_dtype(cn["DateTime"])
    assert "Time" not in cn.columns  # dropped


def test_no_missing_core_columns_after_cleaning(coffee_module):
    """
    Validate cleaned dataset has essential columns and no missing key fields.

    Checks:
      - Required columns are present
      - 'money' and 'coffee_name' have no missing values (used later in grouping/ML)
    """
    cn = coffee_module.coffee_new
    required = [
        "Date",
        "DateTime",
        "Weekday",
        "Monthsort",
        "money",
        "coffee_name",
        "hour_of_day",
    ]
    for col in required:
        assert col in cn.columns, f"Missing required column: {col}"
    assert cn["money"].notna().all()
    assert cn["coffee_name"].notna().all()


# Unit tests: grouping & aggregation


def test_group_by_coffee_matches_manual_sum(coffee_module):
    """
    Cross-check grouped results against a manual recomputation.

    Validates:
      - 'by_coffee' transactions and revenue match a fresh groupby on coffee_new
      - Ensures aggregation logic is correct
    """
    cn = coffee_module.coffee_new
    by = coffee_module.by_coffee

    exp = cn.groupby("coffee_name").agg(
        transactions=("coffee_name", "size"), revenue=("money", "sum")
    )
    by_sorted = by.sort_index()
    exp_sorted = exp.sort_index()

    pd.testing.assert_index_equal(by_sorted.index, exp_sorted.index)
    assert np.allclose(by_sorted["revenue"].values, exp_sorted["revenue"].values)
    assert (by_sorted["transactions"].values == exp_sorted["transactions"].values).all()


def test_weekday_index_order_is_mon_to_sun(coffee_module):
    """
    Ensure weekday aggregation respects the specified order.

    Checks:
      - The code reindexes weekday results to Mon..Sun.
      - This test enforces the exact index order required by the logic.
    """
    by_weekday = coffee_module.by_weekday
    assert list(by_weekday.index) == ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


# Unit tests: explicit filtering checks


def test_filter_by_coffee_name_matches_grouped_revenue(coffee_module):
    """
    Filtering test 1:
    For each coffee_name, filter rows and sum 'money', and verify it equals
    the grouped 'revenue' value for that coffee in by_coffee.

    Purpose:
      - Confirms boolean filtering logic yields the same result as groupby.
    """
    cn = coffee_module.coffee_new
    by = coffee_module.by_coffee

    for coffee_name in cn["coffee_name"].unique():
        filtered_rev = cn.loc[cn["coffee_name"] == coffee_name, "money"].sum()
        grouped_rev = by.loc[coffee_name, "revenue"]
        assert np.isclose(filtered_rev, grouped_rev), f"Mismatch for {coffee_name}"


def test_filter_by_weekday_counts_are_correct(coffee_module):
    """
    Filtering test 2:
    For each present weekday, count rows via boolean filter and ensure the
    result equals the 'transactions' value for that weekday in by_weekday.

    Notes:
      - by_weekday includes all Mon..Sun, so missing days can be NaN; coerce to 0.
      - We only check weekdays actually present in synthetic data to avoid false failures.
    """
    cn = coffee_module.coffee_new
    by_wd = coffee_module.by_weekday

    present_wd = cn["Weekday"].unique().tolist()
    for wd in present_wd:
        filtered_count = (cn["Weekday"] == wd).sum()
        grouped_tx = by_wd.loc[wd, "transactions"]
        grouped_tx = 0 if pd.isna(grouped_tx) else int(grouped_tx)
        assert filtered_count == grouped_tx, f"Weekday count mismatch for {wd}"


# System & ML behavior tests


def test_kmeans_cluster_assignments_exist_and_are_valid(coffee_module):
    """
    End-to-end/ML behavior:
      - Confirms KMeans has been trained (kmeans exists on module)
      - 'per_coffee' has a 'cluster' column containing integer labels
      - Validates chosen n_clusters falls within a sensible range relative to item count
    """
    per = coffee_module.per_coffee
    assert "cluster" in per.columns, "Expected 'cluster' column in per_coffee"
    assert hasattr(coffee_module, "kmeans"), "Expected trained kmeans on module"

    n_clusters = coffee_module.kmeans.n_clusters
    n_items = len(per)
    assert n_items >= 2, "Need at least 2 items to cluster"
    assert 2 <= n_clusters <= max(2, n_items - 1)

    # Labels should be integer-like
    labels = per["cluster"].unique().tolist()
    for c in labels:
        assert isinstance(c, (int, int.__class__)), "Cluster labels should be ints"


def test_system_images_emitted(coffee_module):
    """
    End-to-end/artifacts:
      - Validates that all expected plot files were saved by the script.
      - Ensures the full analysis pipeline executed successfully.

    Files checked:
      by_coffee.png, hour.png, daily.png, correlation.png, cluster_pca_scatter_simple.png
    """
    expected_files = [
        "by_coffee.png",
        "hour.png",
        "daily.png",
        "correlation.png",
        "cluster_pca_scatter_simple.png",
    ]
    for f in expected_files:
        assert Path(f).exists(), f"Expected plot file not found: {f}"
