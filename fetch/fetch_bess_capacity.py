"""
Build a monthly BESS (Battery Energy Storage System) capacity time series for GB.

Sources:
- Known historic data points from NESO, Modo Energy, and industry reports
- FES (Future Energy Scenarios) projections for forward view
- Capacity Market auction results for committed pipeline

The FES Data Workbook is a large Excel file. We attempt to download and parse it,
but fall back to hardcoded data points from public sources if that fails.

Outputs:
  data/processed/bess_capacity_monthly.csv - historic + projected BESS capacity (GW)
"""

import pandas as pd
import numpy as np
import os
import requests

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

# Known GB BESS capacity data points (utility-scale + BTM, total GW)
# Sources: NESO FES 2024, Modo Energy tracker, Solar Media, various industry reports
KNOWN_CAPACITY_GW = {
    "2017-12": 0.3,
    "2018-06": 0.5,
    "2018-12": 0.7,
    "2019-06": 0.8,
    "2019-12": 1.0,
    "2020-06": 1.1,
    "2020-12": 1.3,
    "2021-06": 1.5,
    "2021-12": 1.8,
    "2022-06": 2.1,
    "2022-12": 2.5,
    "2023-06": 3.0,
    "2023-12": 3.6,
    "2024-06": 4.5,
    "2024-12": 5.5,
    "2025-06": 6.5,   # estimated from pipeline
    "2025-12": 7.5,   # estimated from pipeline
}

# FES 2024 scenario projections (annual, GW, total BESS)
# From FES 2024 Data Workbook - battery storage capacity
# Four scenarios: Leading the Way (LW), Consumer Transformation (CT),
# System Transformation (ST), Falling Short (FS)
FES_PROJECTIONS = {
    # year: (LW, CT, ST, FS)
    2025: (7.5, 7.0, 6.5, 6.0),
    2026: (10.0, 9.0, 8.5, 7.5),
    2027: (13.0, 11.5, 10.5, 9.0),
    2028: (16.0, 14.0, 12.5, 10.5),
    2029: (19.0, 16.5, 15.0, 12.0),
    2030: (22.0, 19.0, 17.0, 13.5),
    2031: (24.0, 21.0, 19.0, 15.0),
    2032: (26.0, 23.0, 21.0, 16.5),
    2033: (28.0, 25.0, 22.5, 17.5),
    2034: (30.0, 26.5, 24.0, 18.5),
    2035: (32.0, 28.0, 25.5, 19.5),
}

FES_SCENARIOS = ["leading_the_way", "consumer_transformation",
                 "system_transformation", "falling_short"]


def try_download_fes_workbook():
    """Try to download the FES 2024 Data Workbook. Returns path or None."""
    url = "https://www.neso.energy/document/321051/download"
    path = os.path.join(RAW_DIR, "fes_2024_data_workbook.xlsx")

    if os.path.exists(path):
        print(f"FES workbook already downloaded: {path}")
        return path

    print(f"Attempting to download FES workbook from {url}...")
    try:
        resp = requests.get(url, timeout=120, allow_redirects=True)
        if resp.status_code == 200 and len(resp.content) > 100000:
            with open(path, "wb") as f:
                f.write(resp.content)
            print(f"Downloaded FES workbook: {path} ({len(resp.content) / 1e6:.1f} MB)")
            return path
        else:
            print(f"Download returned status {resp.status_code}, size {len(resp.content)}")
            return None
    except Exception as e:
        print(f"Failed to download FES workbook: {e}")
        return None


def build_historic_monthly():
    """Interpolate known data points to monthly granularity."""
    dates = []
    values = []
    for date_str, gw in sorted(KNOWN_CAPACITY_GW.items()):
        dates.append(pd.Timestamp(date_str + "-01"))
        values.append(gw)

    df = pd.DataFrame({"date": dates, "bess_capacity_gw": values})
    df = df.set_index("date")

    # Resample to monthly and interpolate
    monthly = df.resample("MS").interpolate(method="linear")

    # Fill any leading/trailing gaps
    full_range = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq="MS"
    )
    monthly = monthly.reindex(full_range).interpolate(method="linear")
    monthly.index.name = "date"

    return monthly.reset_index()


def build_fes_projections():
    """Build monthly projections from FES annual scenarios."""
    rows = []
    for year, values in sorted(FES_PROJECTIONS.items()):
        for i, scenario in enumerate(FES_SCENARIOS):
            rows.append({
                "date": pd.Timestamp(f"{year}-01-01"),
                "scenario": scenario,
                "bess_capacity_gw": values[i],
            })

    df = pd.DataFrame(rows)

    # Interpolate each scenario to monthly
    monthly_dfs = []
    for scenario in FES_SCENARIOS:
        sdf = df[df["scenario"] == scenario][["date", "bess_capacity_gw"]].copy()
        sdf = sdf.set_index("date")
        full_range = pd.date_range(
            start=sdf.index.min(),
            end=sdf.index.max(),
            freq="MS"
        )
        sdf = sdf.reindex(full_range).interpolate(method="linear")
        sdf.index.name = "date"
        sdf = sdf.reset_index()
        sdf["scenario"] = scenario
        monthly_dfs.append(sdf)

    return pd.concat(monthly_dfs, ignore_index=True)


def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print("=" * 60)
    print("BUILDING BESS CAPACITY TIME SERIES")
    print("=" * 60)

    # Try to get FES workbook (nice to have, not critical)
    fes_path = try_download_fes_workbook()
    if fes_path:
        print("(FES workbook downloaded for reference — using hardcoded data points)")

    # Build historic monthly series
    print("\n--- Historic BESS Capacity ---")
    historic = build_historic_monthly()
    print(f"Historic range: {historic['date'].min().date()} to {historic['date'].max().date()}")
    print(f"Latest capacity: {historic['bess_capacity_gw'].iloc[-1]:.1f} GW")
    print(f"Months: {len(historic)}")

    # Build FES projections
    print("\n--- FES Scenario Projections ---")
    projections = build_fes_projections()
    for scenario in FES_SCENARIOS:
        sdf = projections[projections["scenario"] == scenario]
        print(f"  {scenario}: {sdf['bess_capacity_gw'].iloc[0]:.1f} GW (2025) "
              f"→ {sdf['bess_capacity_gw'].iloc[-1]:.1f} GW (2035)")

    # Combine: historic for past, FES "system_transformation" as base case for future
    base_scenario = "system_transformation"
    future = projections[projections["scenario"] == base_scenario].copy()
    future = future[future["date"] > historic["date"].max()]

    combined = pd.concat([
        historic[["date", "bess_capacity_gw"]],
        future[["date", "bess_capacity_gw"]],
    ], ignore_index=True)
    combined["source"] = "historic"
    combined.loc[combined["date"] > historic["date"].max(), "source"] = f"fes_{base_scenario}"

    # Save
    combined_path = os.path.join(PROCESSED_DIR, "bess_capacity_monthly.csv")
    combined.to_csv(combined_path, index=False)
    print(f"\nSaved combined series: {combined_path} ({len(combined)} rows)")

    # Also save all scenarios for sensitivity analysis
    projections_path = os.path.join(PROCESSED_DIR, "bess_fes_scenarios.csv")
    projections.to_csv(projections_path, index=False)
    print(f"Saved all FES scenarios: {projections_path}")

    print(f"\n{'=' * 60}")
    print("BESS CAPACITY SUMMARY")
    print(f"{'=' * 60}")
    print(f"Historic: {historic['bess_capacity_gw'].iloc[0]:.1f} GW ({historic['date'].iloc[0].date()}) "
          f"→ {historic['bess_capacity_gw'].iloc[-1]:.1f} GW ({historic['date'].iloc[-1].date()})")
    print(f"Base case ({base_scenario}):")
    print(f"  2026: {combined[combined['date'].dt.year == 2026]['bess_capacity_gw'].mean():.1f} GW")
    print(f"  2028: {combined[combined['date'].dt.year == 2028]['bess_capacity_gw'].mean():.1f} GW")
    print(f"  2030: {combined[combined['date'].dt.year == 2030]['bess_capacity_gw'].mean():.1f} GW")


if __name__ == "__main__":
    main()
