"""
Fetch ancillary frequency response market data from NESO.

Sources:
- Dynamic Containment (DC) results (2021-2023) via NESO data portal
- EAC (Enduring Auction Capability) results (2023+) via NESO data portal

These markets are key revenue streams for battery storage alongside arbitrage.
As more batteries enter, clearing prices in these markets compress too.

Outputs:
  data/raw/dc_masterdata.csv         - Dynamic Containment results
  data/raw/eac_results_summary.csv   - EAC auction results
  data/processed/ancillary_daily_prices.csv - cleaned daily clearing prices
"""

import requests
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

# NESO data portal API endpoints
DC_MASTERDATA_URL = (
    "https://api.neso.energy/dataset/aca07dcb-f807-409c-a4ec-da5dc052b8ba/"
    "resource/0b8dbc3c-e05e-44a4-b855-7dd1aa079c68/download/"
    "dynamic_containment_masterdata.csv"
)

# EAC results - we'll discover the resource ID from the dataset metadata
EAC_DATASET_URL = "https://api.neso.energy/api/3/action/datapackage_show?id=eac-auction-results"


def fetch_dc_data():
    """Download Dynamic Containment masterdata CSV."""
    path = os.path.join(RAW_DIR, "dc_masterdata.csv")

    if os.path.exists(path):
        print(f"DC masterdata already exists: {path}")
        return pd.read_csv(path)

    print(f"Downloading DC masterdata...")
    try:
        resp = requests.get(DC_MASTERDATA_URL, timeout=120)
        resp.raise_for_status()

        with open(path, "wb") as f:
            f.write(resp.content)
        print(f"Saved: {path} ({len(resp.content) / 1024:.0f} KB)")

        return pd.read_csv(path)
    except Exception as e:
        print(f"Failed to download DC data: {e}")
        return pd.DataFrame()


def fetch_eac_data():
    """Download EAC auction results summary CSV."""
    path = os.path.join(RAW_DIR, "eac_results_summary.csv")

    if os.path.exists(path):
        print(f"EAC results already exist: {path}")
        return pd.read_csv(path)

    print("Fetching EAC dataset metadata to find download URLs...")
    try:
        resp = requests.get(EAC_DATASET_URL, timeout=30)
        resp.raise_for_status()
        metadata = resp.json()

        # Find the results summary resource
        resources = metadata.get("result", {}).get("resources", [])
        summary_resource = None
        for r in resources:
            name = r.get("name", "").lower()
            if "summary" in name or "results" in name:
                summary_resource = r
                break

        if not summary_resource and resources:
            summary_resource = resources[0]

        if summary_resource:
            resource_id = summary_resource.get("id", "")
            download_url = f"https://api.neso.energy/datastore/dump/{resource_id}?format=csv"
            print(f"Downloading EAC results: {summary_resource.get('name', 'unknown')}")

            resp = requests.get(download_url, timeout=120)
            resp.raise_for_status()

            with open(path, "wb") as f:
                f.write(resp.content)
            print(f"Saved: {path} ({len(resp.content) / 1024:.0f} KB)")

            return pd.read_csv(path)
        else:
            print("WARNING: No EAC results resource found")
            return pd.DataFrame()

    except Exception as e:
        print(f"Failed to download EAC data: {e}")
        return pd.DataFrame()


def process_ancillary(dc_df, eac_df):
    """Process ancillary data into daily clearing prices."""
    rows = []

    # Process DC data
    # Columns: Market Name, Delivery Date, Availability Fee, Volume Accepted, etc.
    if not dc_df.empty:
        print(f"\nDC data: {len(dc_df)} rows")

        # Only accepted bids
        accepted = dc_df[dc_df["Accepted/Rejected"] == "Accepted"].copy()
        accepted["Delivery Date"] = pd.to_datetime(accepted["Delivery Date"], errors="coerce")
        accepted["Availability Fee"] = pd.to_numeric(accepted["Availability Fee"], errors="coerce")

        for market, mdf in accepted.groupby("Market Name"):
            daily = mdf.groupby(mdf["Delivery Date"].dt.date)["Availability Fee"].mean()
            for date, price in daily.items():
                rows.append({
                    "date": date,
                    "service": market,
                    "clearing_price_mw_h": price,
                })
        print(f"  Processed {len(accepted)} accepted DC bids across "
              f"{accepted['Market Name'].nunique()} markets")

    # Process EAC data
    # Columns: serviceType, clearingPrice, deliveryStart, deliveryEnd, executedQuantity
    if not eac_df.empty:
        print(f"\nEAC data: {len(eac_df)} rows")

        eac_df = eac_df.copy()
        eac_df["deliveryStart"] = pd.to_datetime(eac_df["deliveryStart"], errors="coerce")
        eac_df["clearingPrice"] = pd.to_numeric(eac_df["clearingPrice"], errors="coerce")

        # Only rows with positive executed quantity
        active = eac_df[eac_df["executedQuantity"] > 0].copy()

        for service, sdf in active.groupby("serviceType"):
            daily = sdf.groupby(sdf["deliveryStart"].dt.date)["clearingPrice"].mean()
            for date, price in daily.items():
                rows.append({
                    "date": date,
                    "service": f"EAC_{service}",
                    "clearing_price_mw_h": price,
                })
        print(f"  Processed {len(active)} active EAC results across "
              f"{active['serviceType'].nunique()} service types")

    if rows:
        result = pd.DataFrame(rows)
        result["date"] = pd.to_datetime(result["date"])
        result = result.sort_values("date")

        path = os.path.join(PROCESSED_DIR, "ancillary_daily_prices.csv")
        result.to_csv(path, index=False)
        print(f"\nSaved processed ancillary prices: {path} ({len(result)} rows)")
        return result

    print("\nWARNING: No ancillary data processed — column matching may need adjustment")
    return pd.DataFrame()


def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print("=" * 60)
    print("FETCHING ANCILLARY MARKET DATA")
    print("=" * 60)

    print("\n--- Dynamic Containment (2021-2023) ---")
    dc_df = fetch_dc_data()

    print("\n--- EAC Auction Results (2023+) ---")
    eac_df = fetch_eac_data()

    print("\n--- Processing ---")
    result = process_ancillary(dc_df, eac_df)

    if not result.empty:
        print(f"\n{'=' * 60}")
        print("ANCILLARY SUMMARY")
        print(f"{'=' * 60}")
        print(f"Date range: {result['date'].min().date()} to {result['date'].max().date()}")
        for service in result["service"].unique():
            sdf = result[result["service"] == service]
            print(f"  {service}: {len(sdf)} days, "
                  f"mean £{sdf['clearing_price_mw_h'].mean():.1f}/MW/h")


if __name__ == "__main__":
    main()
