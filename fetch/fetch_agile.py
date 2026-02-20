"""
Fetch half-hourly Octopus Agile tariff data via the official API.

Chains through product codes to get full history from 2019 to present.
Reverse-engineers wholesale day-ahead price from the Agile formula.
Calculates daily wholesale spread (max - min of half-hourly prices).

Outputs:
  data/raw/agile_halfhourly.csv          - all half-hourly Agile rates
  data/processed/daily_wholesale_spread.csv - daily spread from derived wholesale prices
"""

import requests
import pandas as pd
import time
import os
from datetime import datetime, timedelta

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

# Agile product codes in chronological order, with approximate date ranges
# Region C = South England
PRODUCT_CODES = [
    {
        "product": "AGILE-18-02-21",
        "tariff": "E-1R-AGILE-18-02-21-C",
        "from": "2019-01-01",
        "to": "2022-07-21",
    },
    {
        "product": "AGILE-22-07-22",
        "tariff": "E-1R-AGILE-22-07-22-C",
        "from": "2022-07-22",
        "to": "2022-08-30",
    },
    {
        "product": "AGILE-22-08-31",
        "tariff": "E-1R-AGILE-22-08-31-C",
        "from": "2022-08-31",
        "to": "2022-11-24",
    },
    {
        "product": "AGILE-FLEX-22-11-25",
        "tariff": "E-1R-AGILE-FLEX-22-11-25-C",
        "from": "2022-11-25",
        "to": "2024-09-30",
    },
    {
        "product": "AGILE-24-10-01",
        "tariff": "E-1R-AGILE-24-10-01-C",
        "from": "2024-10-01",
        "to": "2026-12-31",  # will just fetch up to today
    },
]

API_BASE = "https://api.octopus.energy/v1/products"


def fetch_product_rates(product, tariff, period_from, period_to):
    """Fetch all half-hourly rates for a product code within a date range."""
    url = f"{API_BASE}/{product}/electricity-tariffs/{tariff}/standard-unit-rates/"
    all_results = []
    page = 1

    while True:
        params = {
            "period_from": f"{period_from}T00:00:00Z",
            "period_to": f"{period_to}T00:00:00Z",
            "page_size": 1500,
            "page": page,
        }
        resp = requests.get(url, params=params, timeout=30)

        if resp.status_code == 404:
            break
        resp.raise_for_status()

        data = resp.json()
        results = data.get("results", [])
        if not results:
            break

        all_results.extend(results)
        print(f"  {product} page {page}: {len(results)} records (total {len(all_results)})")

        if data.get("next") is None:
            break
        page += 1
        time.sleep(0.3)

    return all_results


def fetch_all_agile():
    """Fetch complete Agile history across all product codes."""
    all_records = []

    for pc in PRODUCT_CODES:
        print(f"\nFetching {pc['product']} ({pc['from']} to {pc['to']})...")
        records = fetch_product_rates(pc["product"], pc["tariff"], pc["from"], pc["to"])
        for r in records:
            all_records.append({
                "valid_from": r["valid_from"],
                "valid_to": r["valid_to"],
                "value_exc_vat": r["value_exc_vat"],
                "value_inc_vat": r["value_inc_vat"],
                "product": pc["product"],
            })
        time.sleep(1)

    df = pd.DataFrame(all_records)
    if df.empty:
        print("WARNING: No Agile data fetched!")
        return df

    df["valid_from"] = pd.to_datetime(df["valid_from"], utc=True)
    df["valid_to"] = pd.to_datetime(df["valid_to"], utc=True)

    # Remove duplicates (overlap between product codes)
    df = df.sort_values("valid_from").drop_duplicates(subset=["valid_from"], keep="last")
    df = df.reset_index(drop=True)

    print(f"\nTotal records: {len(df)}")
    print(f"Date range: {df['valid_from'].min()} to {df['valid_from'].max()}")

    return df


def reverse_engineer_wholesale(df):
    """
    Derive wholesale day-ahead price from Agile exc-VAT price.

    The Agile formula (approximately):
      agile_exc_vat = wholesale_price * regional_multiplier + adder

    The adder and multiplier have changed over time. Rather than trying to
    reconstruct exact formula parameters for each product version, we use
    the Agile price itself as a close proxy for wholesale price movements.

    For spread analysis (max - min within a day), the additive components
    cancel out, so: agile_spread ≈ multiplier × wholesale_spread.

    We'll use value_exc_vat directly as our price signal and note that
    spread calculations from Agile rates will overstate wholesale spread
    by approximately the multiplier factor (~1.0-1.1 in recent products).
    We validate against Elexon market index data in the analysis step.
    """
    df = df.copy()
    # Convert p/kWh to £/MWh: multiply by 10
    df["price_mwh"] = df["value_exc_vat"] * 10
    return df


def calculate_daily_spread(df):
    """Calculate daily wholesale spread (max - min) from half-hourly prices."""
    df = df.copy()
    df["date"] = df["valid_from"].dt.date

    daily = df.groupby("date").agg(
        price_max=("price_mwh", "max"),
        price_min=("price_mwh", "min"),
        price_mean=("price_mwh", "mean"),
        price_p90=("price_mwh", lambda x: x.quantile(0.9)),
        price_p10=("price_mwh", lambda x: x.quantile(0.1)),
        n_periods=("price_mwh", "count"),
    ).reset_index()

    daily["spread_max_min"] = daily["price_max"] - daily["price_min"]
    daily["spread_p90_p10"] = daily["price_p90"] - daily["price_p10"]

    # Only keep days with reasonable data (at least 40 of 48 half-hours)
    daily = daily[daily["n_periods"] >= 40].copy()
    daily["date"] = pd.to_datetime(daily["date"])

    return daily


def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print("=" * 60)
    print("FETCHING OCTOPUS AGILE TARIFF DATA")
    print("=" * 60)

    # Fetch
    df = fetch_all_agile()
    if df.empty:
        return

    # Save raw
    raw_path = os.path.join(RAW_DIR, "agile_halfhourly.csv")
    df.to_csv(raw_path, index=False)
    print(f"\nSaved raw data: {raw_path} ({len(df)} rows)")

    # Reverse-engineer wholesale prices
    df = reverse_engineer_wholesale(df)

    # Calculate daily spread
    daily = calculate_daily_spread(df)

    spread_path = os.path.join(PROCESSED_DIR, "daily_wholesale_spread.csv")
    daily.to_csv(spread_path, index=False)
    print(f"Saved daily spread: {spread_path} ({len(daily)} rows)")

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"Date range: {daily['date'].min().date()} to {daily['date'].max().date()}")
    print(f"Days with data: {len(daily)}")
    print(f"Mean daily spread (max-min): £{daily['spread_max_min'].mean():.1f}/MWh")
    print(f"Mean daily spread (p90-p10): £{daily['spread_p90_p10'].mean():.1f}/MWh")
    print(f"Mean price: £{daily['price_mean'].mean():.1f}/MWh")

    # Show yearly averages
    daily["year"] = daily["date"].dt.year
    yearly = daily.groupby("year").agg(
        mean_spread=("spread_max_min", "mean"),
        mean_price=("price_mean", "mean"),
        days=("date", "count"),
    )
    print(f"\nYearly averages:")
    print(yearly.to_string())


if __name__ == "__main__":
    main()
