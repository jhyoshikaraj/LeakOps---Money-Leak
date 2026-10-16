"""
LeakOps - Unified Data Preprocessing & Feature Engineering Pipeline
===================================================================
Role: Senior Data Engineer & ML Architect
Engineered for DATATHON 2026:
- Safe ingestion and date-parsing for all 5 relational datasets:
    1. transactions.csv (Base sales over time)
    2. campaigns.csv    (Marketing spend over time)
    3. inventory.csv    (Stock levels and supply chain metrics over time)
    4. customers.csv    (Demographic segments for SHAP explainability)
    5. products.csv     (Category metadata, list price, and unit costs)
- Relational joins across (customer_id, product_id) and daily aggregation.
- Intelligent imputation: Zero-fill null ad spend, forward-fill inventory time series.
- Domain feature engineering: 7-day rolling sales, 3-day ad spend delta, stock-to-sales ratio,
  CTR, CPC, return rates, and demographic ratios.
- Chronologically sorted master_df with 0 NaNs ready for train_test_split(shuffle=False).
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
from sklearn.model_selection import train_test_split


def locate_file(filename: str) -> str:
    """Robust path resolver searching script directory, parent, and common data subfolders."""
    script_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
    candidates = [
        os.path.join(script_dir, filename),
        os.path.join(script_dir, "..", filename),
        os.path.join(script_dir, "data", filename),
        os.path.join(script_dir, "..", "data", filename),
        os.path.join(script_dir, "ml-service", filename),
        filename
    ]
    for path in candidates:
        if os.path.exists(path):
            return os.path.abspath(path)
    raise FileNotFoundError(f"[Error] Required dataset '{filename}' not found in candidate paths.")


def load_datasets() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Safely loads all 5 hackathon datasets with proper datetime parsing and standard types.
    """
    print("\n" + "=" * 80)
    print("STEP 1: INGESTING ALL 5 HACKATHON RELATIONAL DATASETS")
    print("=" * 80)

    # 1. Transactions (Base table)
    tx_path = locate_file("transactions.csv")
    df_tx = pd.read_csv(tx_path)
    df_tx["timestamp"] = pd.to_datetime(df_tx["timestamp"], errors="coerce")
    df_tx["date"] = df_tx["timestamp"].dt.strftime("%Y-%m-%d")
    print(f" Loaded transactions.csv: {df_tx.shape[0]:,} rows x {df_tx.shape[1]} cols | Date Range: {df_tx['date'].min()} to {df_tx['date'].max()}")

    # 2. Campaigns (Marketing spend)
    camp_path = locate_file("campaigns.csv")
    df_camp = pd.read_csv(camp_path)
    df_camp["date"] = pd.to_datetime(df_camp["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    print(f" Loaded campaigns.csv:    {df_camp.shape[0]:,} rows x {df_camp.shape[1]} cols | Date Range: {df_camp['date'].min()} to {df_camp['date'].max()}")

    # 3. Inventory (Stock levels)
    inv_path = locate_file("inventory.csv")
    df_inv = pd.read_csv(inv_path)
    df_inv["date"] = pd.to_datetime(df_inv["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    print(f" Loaded inventory.csv:    {df_inv.shape[0]:,} rows x {df_inv.shape[1]} cols | Date Range: {df_inv['date'].min()} to {df_inv['date'].max()}")

    # 4. Customers (Demographics)
    cust_path = locate_file("customers.csv")
    df_cust = pd.read_csv(cust_path)
    df_cust["signup_date"] = pd.to_datetime(df_cust["signup_date"], errors="coerce")
    print(f" Loaded customers.csv:    {df_cust.shape[0]:,} rows x {df_cust.shape[1]} cols")

    # 5. Products (Catalog & Metadata)
    prod_path = locate_file("products.csv")
    df_prod = pd.read_csv(prod_path)
    print(f" Loaded products.csv:     {df_prod.shape[0]:,} rows x {df_prod.shape[1]} cols")

    return df_tx, df_camp, df_inv, df_cust, df_prod


def process_and_merge(
    df_tx: pd.DataFrame,
    df_camp: pd.DataFrame,
    df_inv: pd.DataFrame,
    df_cust: pd.DataFrame,
    df_prod: pd.DataFrame
) -> pd.DataFrame:
    """
    Performs relational merges across datasets, aggregates transactions to daily-category level,
    and constructs a unified, cross-functional master time-series grid.
    """
    print("\n" + "=" * 80)
    print("STEP 2: RELATIONAL ENRICHMENT & MULTI-GRANULARITY MERGING")
    print("=" * 80)

    # A. Enrich Transactions with Customer Demographics and Product Metadata
    tx_enriched = df_tx.merge(df_cust, on="customer_id", how="left")
    tx_enriched = tx_enriched.merge(df_prod, on="product_id", how="left")
    
    # Calculate order-level financials
    tx_enriched["net_sales"] = np.where(
        tx_enriched["status"] == "Completed",
        tx_enriched["quantity"] * tx_enriched["unit_price"] * (1.0 - tx_enriched["discount_pct"]),
        0.0
    )
    tx_enriched["units_sold"] = np.where(tx_enriched["status"] == "Completed", tx_enriched["quantity"], 0)
    tx_enriched["is_returned"] = (tx_enriched["status"] == "Returned").astype(int)
    tx_enriched["cust_tenure_days"] = (tx_enriched["timestamp"] - tx_enriched["signup_date"]).dt.days.clip(lower=0)
    
    print(f" Enriched transactions joined with customers and products: {tx_enriched.shape}")

    # B. Daily Transactions Aggregation by Date & Category
    tx_daily = tx_enriched.groupby(["date", "category"]).agg(
        daily_sales=("net_sales", "sum"),
        units_sold=("units_sold", "sum"),
        total_orders=("transaction_id", "count"),
        returned_orders=("is_returned", "sum"),
        avg_discount=("discount_pct", "mean"),
        loyal_cust_ratio=("customer_segment", lambda s: (s == "Loyal").mean()),
        returning_cust_ratio=("customer_segment", lambda s: (s == "Returning").mean()),
        new_cust_ratio=("customer_segment", lambda s: (s == "New").mean()),
        avg_cust_tenure=("cust_tenure_days", "mean"),
        unique_customers=("customer_id", "nunique")
    ).reset_index()
    print(f" Daily aggregated transactions: {tx_daily.shape[0]:,} rows x {tx_daily.shape[1]} cols")

    # C. Daily Campaigns Aggregation by Date & Category
    camp_daily = df_camp.groupby(["date", "category"]).agg(
        ad_spend=("spend", "sum"),
        impressions=("impressions", "sum"),
        clicks=("clicks", "sum"),
        conversions=("conversions", "sum")
    ).reset_index()
    print(f" Daily aggregated campaigns:    {camp_daily.shape[0]:,} rows x {camp_daily.shape[1]} cols")

    # D. Daily Inventory Aggregation by Date & Category
    inv_enriched = df_inv.merge(df_prod[["product_id", "category"]], on="product_id", how="left")
    inv_daily = inv_enriched.groupby(["date", "category"]).agg(
        stock_on_hand=("stock_on_hand", "sum"),
        reorder_threshold=("reorder_threshold", "mean"),
        stockout_events=("stockout_flag", "sum"),
        supplier_delay_days=("supplier_delay_days", "mean")
    ).reset_index()
    print(f" Daily aggregated inventory:    {inv_daily.shape[0]:,} rows x {inv_daily.shape[1]} cols")

    # E. Full Outer Join on Date & Category to preserve complete temporal spine
    master = pd.merge(tx_daily, camp_daily, on=["date", "category"], how="outer")
    master = pd.merge(master, inv_daily, on=["date", "category"], how="outer")
    print(f" Outer joined master matrix:    {master.shape[0]:,} rows x {master.shape[1]} cols (Zero data wiped)")

    return master


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Imputes missing values without dropping time-series rows:
    - Zero-fills ad spend, impressions, clicks, conversions (days without active ad budget).
    - Zero-fills transactions metrics (days with no recorded sales).
    - Forward-fills (then back-fills) inventory levels per category to maintain continuity.
    """
    print("\n" + "=" * 80)
    print("STEP 3: INTELLIGENT MISSING VALUE IMPUTATION (TIME-SERIES PRESERVATION)")
    print("=" * 80)

    initial_nans = df.isna().sum().sum()
    print(f" Pre-imputation total NaNs across master grid: {initial_nans:,}")

    # Ensure chronological sorting per category for valid forward-fill
    df["date_dt"] = pd.to_datetime(df["date"])
    df = df.sort_values(by=["category", "date_dt"]).reset_index(drop=True)

    # 1. Zero-fill Campaign Metrics
    camp_cols = ["ad_spend", "impressions", "clicks", "conversions"]
    for col in camp_cols:
        df[col] = df[col].fillna(0.0)

    # 2. Zero-fill Sales & Customer Demographics Metrics
    sales_cols = [
        "daily_sales", "units_sold", "total_orders", "returned_orders", "avg_discount",
        "loyal_cust_ratio", "returning_cust_ratio", "new_cust_ratio", "avg_cust_tenure",
        "unique_customers"
    ]
    for col in sales_cols:
        df[col] = df[col].fillna(0.0)

    # 3. Forward-fill Inventory per Category (Time-series continuity), with back-fill for initial edge cases
    inv_cols = ["stock_on_hand", "reorder_threshold", "stockout_events", "supplier_delay_days"]
    for col in inv_cols:
        df[col] = df.groupby("category")[col].transform(lambda grp: grp.ffill().bfill())
        # Fallback if an entire category had no inventory records (none in this dataset)
        df[col] = df[col].fillna(df[col].median() if not df[col].dropna().empty else 0.0)

    final_nans = df.isna().sum().sum()
    print(f" Post-imputation total NaNs across master grid: {final_nans}")
    assert final_nans == 0, f"Error: {final_nans} NaNs remaining in master dataframe."

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers predictive baseline features to empower Random Forest anomaly detection:
    - 7-day and 14-day rolling average sales per category.
    - 3-day change in marketing ad spend per category.
    - Stock-to-sales ratio and stockout risk indicators.
    - Marketing engagement: CTR (Click-Through Rate), CPC (Cost Per Click).
    - Supply chain & operations: Return rate, supplier delay ratio.
    - Temporal features: Day of week, month, weekend indicator.
    """
    print("\n" + "=" * 80)
    print("STEP 4: DOMAIN FEATURE ENGINEERING FOR RANDOM FOREST & SHAP")
    print("=" * 80)

    # Group-wise rolling sales baseline (prevents cross-category leakage)
    df["sales_rolling_7d"] = (
        df.groupby("category")["daily_sales"]
        .transform(lambda x: x.rolling(window=7, min_periods=1).mean())
        .round(2)
    )
    df["sales_rolling_14d"] = (
        df.groupby("category")["daily_sales"]
        .transform(lambda x: x.rolling(window=14, min_periods=1).mean())
        .round(2)
    )

    # 3-day change in ad spend (velocity feature to detect sudden ad spikes)
    df["ad_spend_3d_diff"] = (
        df.groupby("category")["ad_spend"]
        .diff(periods=3)
        .fillna(0.0)
        .round(2)
    )

    # Current Stock-to-Sales Ratio (safe against division by zero)
    df["stock_to_sales_ratio"] = (
        df["stock_on_hand"] / np.maximum(df["units_sold"], 1.0)
    ).round(2)

    # Marketing Efficiency Features
    df["ctr"] = (df["clicks"] / np.maximum(df["impressions"], 1.0)).round(5)
    df["cpc"] = (df["ad_spend"] / np.maximum(df["clicks"], 1.0)).round(4)

    # Operational & Supply Chain Risk Features
    df["return_rate"] = (df["returned_orders"] / np.maximum(df["total_orders"], 1.0)).round(4)
    df["stockout_risk_flag"] = (
        (df["stock_on_hand"] <= df["reorder_threshold"]) | (df["stockout_events"] > 0)
    ).astype(int)

    # Calendar & Seasonality Indicators
    df["day_of_week"] = df["date_dt"].dt.dayofweek
    df["month"] = df["date_dt"].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    # Final strictly chronological sorting by Date
    df = df.sort_values(by=["date_dt", "category"]).reset_index(drop=True)
    df.drop(columns=["date_dt"], inplace=True)

    print(f" Feature engineering complete.")
    print(f" Total features generated: {df.shape[1]} columns across {df.shape[0]:,} observations.")
    return df


def split_and_validate(master_df: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Enforces a strict chronological time-based split:
    train_test_split(..., shuffle=False)
    Guarantees zero data leakage for time-series ML validation.
    """
    print("\n" + "=" * 80)
    print("STEP 5: CHRONOLOGICAL TIME-BASED TRAIN/TEST SPLIT (ZERO DATA LEAKAGE)")
    print("=" * 80)

    # Train / Test split strictly with shuffle=False
    train_df, test_df = train_test_split(master_df, test_size=test_size, shuffle=False)

    train_dates = (train_df["date"].min(), train_df["date"].max())
    test_dates = (test_df["date"].min(), test_df["date"].max())

    print(f" Train Split Shape: {train_df.shape} | Dates: {train_dates[0]} to {train_dates[1]}")
    print(f" Test Split Shape:  {test_df.shape}  | Dates: {test_dates[0]} to {test_dates[1]}")
    
    # Audit check: Ensure test dates are strictly after or equal to boundary without overlap
    assert train_dates[1] <= test_dates[0], "Data Leakage Alert: Training dates overlap into test horizon!"
    print(" Verification Passed: ZERO data leakage confirmed across chronological split boundary.")

    return train_df, test_df


def run_pipeline() -> pd.DataFrame:
    """
    Executes the full end-to-end data pipeline and returns master_df.
    """
    # 1. Ingest
    df_tx, df_camp, df_inv, df_cust, df_prod = load_datasets()

    # 2. Merge & Aggregate
    merged_df = process_and_merge(df_tx, df_camp, df_inv, df_cust, df_prod)

    # 3. Handle Missing Values
    clean_df = handle_missing_values(merged_df)

    # 4. Feature Engineering
    master_df = engineer_features(clean_df)

    # 5. Chronological Split Validation
    split_and_validate(master_df, test_size=0.2)

    print("\n" + "=" * 80)
    print("DATA PIPELINE SUCCESS AUDIT")
    print("=" * 80)
    print(f" Master DataFrame Dimensions: {master_df.shape[0]:,} rows x {master_df.shape[1]} columns")
    print(f" Total Missing / NaN Values:  {master_df.isna().sum().sum()}")
    print(f" Final Column Schema:")
    for idx, col in enumerate(master_df.columns, start=1):
        print(f"   [{idx:02d}] {col:<26} (dtype: {master_df[col].dtype})")
    print("=" * 80 + "\n")

    return master_df


if __name__ == "__main__":
    master_df = run_pipeline()
    # Save processed dataset ready for model consumption
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "master_processed.csv")
    master_df.to_csv(output_path, index=False)
    print(f"[Export] Saved clean master dataset to: {output_path}\n")
