"""
Component 1: BESS Capacity Forecast

Plot historic BESS capacity trajectory and project forward using FES scenarios.
Cross-reference with capacity market pipeline.

The BESS forecast is the strongest component — we have both actuals and official
projections from NESO's Future Energy Scenarios.

Inputs:
  data/processed/bess_capacity_monthly.csv
  data/processed/bess_fes_scenarios.csv

Outputs:
  data/processed/bess_forecast.csv
  data/processed/plots/02_bess_forecast.png
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
PLOT_DIR = os.path.join(PROCESSED_DIR, "plots")


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)

    print("=" * 60)
    print("COMPONENT 1: BESS CAPACITY FORECAST")
    print("=" * 60)

    # Load data
    bess_path = os.path.join(PROCESSED_DIR, "bess_capacity_monthly.csv")
    scenarios_path = os.path.join(PROCESSED_DIR, "bess_fes_scenarios.csv")

    bess = pd.read_csv(bess_path, parse_dates=["date"])
    print(f"Loaded BESS capacity: {len(bess)} months")

    scenarios = None
    if os.path.exists(scenarios_path):
        scenarios = pd.read_csv(scenarios_path, parse_dates=["date"])
        print(f"Loaded FES scenarios: {len(scenarios)} rows")

    # Split historic vs projected
    historic = bess[bess["source"] == "historic"].copy()
    projected = bess[bess["source"] != "historic"].copy()

    print(f"\nHistoric: {historic['date'].min().date()} to {historic['date'].max().date()}")
    print(f"  Start: {historic['bess_capacity_gw'].iloc[0]:.1f} GW")
    print(f"  End: {historic['bess_capacity_gw'].iloc[-1]:.1f} GW")

    if not projected.empty:
        print(f"\nProjected: {projected['date'].min().date()} to {projected['date'].max().date()}")
        print(f"  End: {projected['bess_capacity_gw'].iloc[-1]:.1f} GW")

    # Build the forecast: Jan 2026 to Dec 2030
    forecast_start = pd.Timestamp("2026-01-01")
    forecast_end = pd.Timestamp("2030-12-01")

    forecast = bess[
        (bess["date"] >= forecast_start) & (bess["date"] <= forecast_end)
    ].copy()

    if forecast.empty and scenarios is not None:
        # Use system_transformation scenario
        st = scenarios[scenarios["scenario"] == "system_transformation"]
        forecast = st[
            (st["date"] >= forecast_start) & (st["date"] <= forecast_end)
        ][["date", "bess_capacity_gw"]].copy()

    print(f"\nForecast period: {forecast['date'].min().date()} to {forecast['date'].max().date()}")
    print(f"  Months: {len(forecast)}")

    # Save forecast
    forecast_path = os.path.join(PROCESSED_DIR, "bess_forecast.csv")
    forecast.to_csv(forecast_path, index=False)
    print(f"Saved: {forecast_path}")

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(14, 6))

    # Historic
    ax.plot(historic["date"], historic["bess_capacity_gw"],
            color="purple", linewidth=2.5, label="Historic", zorder=5)

    # FES scenarios (fan)
    if scenarios is not None:
        scenario_colors = {
            "leading_the_way": "#7B2FBE",
            "consumer_transformation": "#A855F7",
            "system_transformation": "#C084FC",
            "falling_short": "#E9D5FF",
        }
        scenario_labels = {
            "leading_the_way": "Leading the Way",
            "consumer_transformation": "Consumer Transformation",
            "system_transformation": "System Transformation (base case)",
            "falling_short": "Falling Short",
        }

        for scenario_name in ["falling_short", "system_transformation",
                              "consumer_transformation", "leading_the_way"]:
            sdf = scenarios[scenarios["scenario"] == scenario_name]
            ax.plot(sdf["date"], sdf["bess_capacity_gw"],
                    color=scenario_colors[scenario_name],
                    linewidth=1.5, linestyle="--",
                    label=scenario_labels[scenario_name])

        # Fill between LW and FS for uncertainty band
        lw = scenarios[scenarios["scenario"] == "leading_the_way"].set_index("date")
        fs = scenarios[scenarios["scenario"] == "falling_short"].set_index("date")
        common = lw.index.intersection(fs.index)
        ax.fill_between(common, fs.loc[common, "bess_capacity_gw"],
                        lw.loc[common, "bess_capacity_gw"],
                        alpha=0.1, color="purple")

    # Vertical line at forecast start
    ax.axvline(forecast_start, color="gray", linestyle=":", alpha=0.5)
    ax.text(forecast_start, ax.get_ylim()[1] * 0.95, " Forecast period ",
            ha="left", va="top", fontsize=9, color="gray")

    ax.set_xlabel("Date")
    ax.set_ylabel("BESS Capacity (GW)")
    ax.set_title("GB Battery Energy Storage Capacity — Historic & FES Projections")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "02_bess_forecast.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot: {path}")
    plt.close()

    # Print forecast summary
    print(f"\n{'=' * 60}")
    print("BESS FORECAST SUMMARY (System Transformation base case)")
    print(f"{'=' * 60}")
    for year in range(2026, 2031):
        year_data = forecast[forecast["date"].dt.year == year]
        if not year_data.empty:
            mid = year_data["bess_capacity_gw"].iloc[len(year_data) // 2]
            print(f"  {year}: ~{mid:.1f} GW")


if __name__ == "__main__":
    main()
