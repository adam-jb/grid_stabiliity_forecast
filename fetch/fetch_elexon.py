"""
Fetch generation by fuel type and wholesale market index prices from Elexon.

Uses the Elexon Insights Solution API (free, no auth needed).
- FUELHH: half-hourly generation (MW) by fuel type (7-day range queries)
- Market Index: day-ahead wholesale price (£/MWh) per settlement period (7-day range)

Note: Solar doesn't appear in FUELHH — it's embedded generation in GB
and counted as negative demand rather than positive generation.

Outputs:
  data/raw/elexon_generation.csv      - daily wind generation
  data/raw/elexon_market_index.csv    - half-hourly market index prices
  data/processed/daily_wind_solar_generation.csv
  data/processed/daily_elexon_wholesale.csv
"""

import requests
import pandas as pd
import time
import os
from datetime import datetime, timedelta

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

ELEXON_BASE = "https://data.elexon.co.uk/bmrs/api/v1"

START_DATE = "2019-01-01"
CHUNK_DAYS = 7  # API max for both endpoints


def fetch_fuelhh_range(from_date, to_date):
    """Fetch FUELHH data for a date range (max 7 days)."""
    url = f"{ELEXON_BASE}/datasets/FUELHH"
    params = {
        "settlementDateFrom": from_date,
        "settlementDateTo": to_date,
        "format": "json",
    }
    try:
        resp = requests.get(url, params=params, timeout=60)
        if resp.status_code != 200:
            return []
        return resp.json().get("data", [])
    except Exception as e:
        print(f"  ERROR: {e}")
        return []


def fetch_market_index_range(from_date, to_date):
    """Fetch market index data for a date range (max 7 days)."""
    url = f"{ELEXON_BASE}/balancing/pricing/market-index"
    params = {
        "from": f"{from_date}T00:00:00Z",
        "to": f"{to_date}T23:59:59Z",
        "format": "json",
    }
    try:
        resp = requests.get(url, params=params, timeout=60)
        if resp.status_code != 200:
            return []
        return resp.json().get("data", [])
    except Exception as e:
        print(f"  ERROR: {e}")
        return []


def fetch_all_generation(start_date, end_date):
    """Fetch wind generation in 7-day chunks."""
    all_rows = []
    current = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    total_chunks = ((end - current).days // CHUNK_DAYS) + 1
    chunk_num = 0

    while current <= end:
        chunk_end = min(current + timedelta(days=CHUNK_DAYS - 1), end)
        from_str = current.strftime("%Y-%m-%d")
        to_str = chunk_end.strftime("%Y-%m-%d")

        records = fetch_fuelhh_range(from_str, to_str)

        if records:
            # Group by settlement date and compute daily wind generation
            by_date = {}
            for r in records:
                d = r.get("settlementDate", "")
                ft = r.get("fuelType", "")
                gen = r.get("generation", 0)
                if d not in by_date:
                    by_date[d] = {"wind": [], "total_records": 0}
                by_date[d]["total_records"] += 1
                if ft == "WIND":
                    by_date[d]["wind"].append(gen)

            for d, vals in by_date.items():
                wind_gens = vals["wind"]
                all_rows.append({
                    "date": d,
                    "wind_gen_mw": sum(wind_gens) / len(wind_gens) if wind_gens else 0,
                    "solar_gen_mw": 0,
                    "wind_periods": len(wind_gens),
                    "solar_periods": 0,
                })

        chunk_num += 1
        if chunk_num % 20 == 0:
            print(f"  Generation: chunk {chunk_num}/{total_chunks} "
                  f"({from_str} to {to_str}), {len(all_rows)} days so far")

        current = chunk_end + timedelta(days=1)
        time.sleep(0.3)

    return pd.DataFrame(all_rows)


def fetch_all_market_index(start_date, end_date):
    """Fetch wholesale market index in 7-day chunks."""
    all_data = []
    current = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    total_chunks = ((end - current).days // CHUNK_DAYS) + 1
    chunk_num = 0

    while current <= end:
        chunk_end = min(current + timedelta(days=CHUNK_DAYS - 1), end)
        from_str = current.strftime("%Y-%m-%d")
        to_str = chunk_end.strftime("%Y-%m-%d")

        data = fetch_market_index_range(from_str, to_str)
        all_data.extend(data)

        chunk_num += 1
        if chunk_num % 20 == 0:
            print(f"  Market index: chunk {chunk_num}/{total_chunks} "
                  f"({from_str} to {to_str}), {len(all_data)} records so far")

        current = chunk_end + timedelta(days=1)
        time.sleep(0.3)

    return all_data


def process_generation(df):
    """Convert generation to daily averages in GW, save processed file."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    df["wind_gen_gw"] = df["wind_gen_mw"] / 1000
    df["solar_gen_gw"] = df["solar_gen_mw"] / 1000

    out = df[["date", "wind_gen_gw", "solar_gen_gw", "wind_gen_mw", "solar_gen_mw"]].copy()
    path = os.path.join(PROCESSED_DIR, "daily_wind_solar_generation.csv")
    out.to_csv(path, index=False)
    print(f"Saved processed generation: {path} ({len(out)} rows)")
    return out


def process_market_index(data):
    """Process market index data into daily wholesale prices."""
    if not data:
        print("WARNING: No market index data")
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df["startTime"] = pd.to_datetime(df["startTime"], utc=True)
    df["date"] = df["startTime"].dt.date

    # Use APXMIDP as primary provider
    apx = df[df["dataProvider"] == "APXMIDP"].copy()
    if apx.empty:
        apx = df[df["dataProvider"] == "N2EXMIDP"].copy()
    if apx.empty:
        apx = df.copy()

    daily = apx.groupby("date").agg(
        wholesale_mean=("price", "mean"),
        wholesale_max=("price", "max"),
        wholesale_min=("price", "min"),
        wholesale_spread=("price", lambda x: x.max() - x.min()),
        n_periods=("price", "count"),
    ).reset_index()
    daily["date"] = pd.to_datetime(daily["date"])

    raw_path = os.path.join(RAW_DIR, "elexon_market_index.csv")
    apx.to_csv(raw_path, index=False)
    print(f"Saved raw market index: {raw_path} ({len(apx)} rows)")

    processed_path = os.path.join(PROCESSED_DIR, "daily_elexon_wholesale.csv")
    daily.to_csv(processed_path, index=False)
    print(f"Saved daily wholesale prices: {processed_path} ({len(daily)} rows)")

    return daily


def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    end_date = datetime.now().strftime("%Y-%m-%d")

    print("=" * 60)
    print("FETCHING ELEXON DATA")
    print("=" * 60)

    # 1. Generation by fuel type
    print(f"\n--- Wind generation ({START_DATE} to {end_date}) ---")
    print("Fetching in 7-day chunks (API limit)...")
    print("Note: Solar not in FUELHH (embedded generation in GB)\n")

    gen_df = fetch_all_generation(START_DATE, end_date)
    if not gen_df.empty:
        # Save raw
        raw_path = os.path.join(RAW_DIR, "elexon_generation.csv")
        gen_df.to_csv(raw_path, index=False)
        print(f"\nSaved raw generation: {raw_path} ({len(gen_df)} rows)")

        gen_processed = process_generation(gen_df)
        print(f"\nGeneration summary:")
        print(f"  Date range: {gen_processed['date'].min().date()} to {gen_processed['date'].max().date()}")
        print(f"  Mean wind: {gen_processed['wind_gen_gw'].mean():.1f} GW")
        print(f"  Wind range: {gen_processed['wind_gen_gw'].min():.1f} to {gen_processed['wind_gen_gw'].max():.1f} GW")

    # 2. Market index wholesale prices
    print(f"\n--- Market index wholesale prices ({START_DATE} to {end_date}) ---")
    print("Fetching in 7-day chunks...\n")

    market_data = fetch_all_market_index(START_DATE, end_date)
    if market_data:
        wholesale_daily = process_market_index(market_data)
        if not wholesale_daily.empty:
            print(f"\nWholesale summary:")
            print(f"  Date range: {wholesale_daily['date'].min().date()} to {wholesale_daily['date'].max().date()}")
            print(f"  Mean price: £{wholesale_daily['wholesale_mean'].mean():.1f}/MWh")
            print(f"  Mean spread: £{wholesale_daily['wholesale_spread'].mean():.1f}/MWh")

    print(f"\n{'=' * 60}")
    print("ELEXON FETCH COMPLETE")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
