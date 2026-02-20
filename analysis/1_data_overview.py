"""
Exploratory data analysis: load all processed datasets, plot time series,
check data quality, and validate the Agile → wholesale reverse-engineering.

Run after fetch_all.py. Produces overview plots and prints summary stats.

Outputs:
  data/processed/plots/01_all_timeseries.png
  data/processed/plots/01_wholesale_validation.png
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
PLOT_DIR = os.path.join(PROCESSED_DIR, "plots")


def load_datasets():
    """Load all processed CSVs, returning what's available."""
    datasets = {}

    files = {
        "spread": "daily_wholesale_spread.csv",
        "generation": "daily_wind_solar_generation.csv",
        "bess": "bess_capacity_monthly.csv",
        "renewables": "renewable_capacity_monthly.csv",
        "ancillary": "ancillary_daily_prices.csv",
        "elexon_wholesale": "daily_elexon_wholesale.csv",
    }

    for name, filename in files.items():
        path = os.path.join(PROCESSED_DIR, filename)
        if os.path.exists(path):
            df = pd.read_csv(path, parse_dates=["date"])
            datasets[name] = df
            print(f"  Loaded {name}: {len(df)} rows, "
                  f"{df['date'].min().date()} to {df['date'].max().date()}")
        else:
            print(f"  MISSING: {filename}")

    return datasets


def plot_all_timeseries(datasets):
    """Plot all time series on a shared x-axis."""
    n_plots = sum(1 for k in ["spread", "generation", "bess", "renewables", "ancillary"]
                  if k in datasets)
    if n_plots == 0:
        print("No data to plot!")
        return

    fig, axes = plt.subplots(n_plots, 1, figsize=(14, 4 * n_plots), sharex=True)
    if n_plots == 1:
        axes = [axes]

    ax_idx = 0

    # 1. Daily wholesale spread
    if "spread" in datasets:
        ax = axes[ax_idx]
        df = datasets["spread"]
        # Plot monthly rolling average for clarity
        df = df.set_index("date").sort_index()
        ax.plot(df.index, df["spread_max_min"], alpha=0.15, color="steelblue", linewidth=0.5)
        monthly = df["spread_max_min"].rolling(30, min_periods=7).mean()
        ax.plot(monthly.index, monthly, color="steelblue", linewidth=2, label="30-day avg")
        ax.set_ylabel("Spread (£/MWh)")
        ax.set_title("Daily Wholesale Spread (max - min, from Agile tariff)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax_idx += 1

    # 2. Wind and solar generation
    if "generation" in datasets:
        ax = axes[ax_idx]
        df = datasets["generation"].set_index("date").sort_index()
        wind_ma = df["wind_gen_gw"].rolling(30, min_periods=7).mean()
        solar_ma = df["solar_gen_gw"].rolling(30, min_periods=7).mean()
        ax.plot(wind_ma.index, wind_ma, color="teal", linewidth=2, label="Wind (30d avg)")
        ax.plot(solar_ma.index, solar_ma, color="orange", linewidth=2, label="Solar (30d avg)")
        ax.set_ylabel("Generation (GW)")
        ax.set_title("Daily Wind & Solar Generation (Elexon)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax_idx += 1

    # 3. BESS capacity
    if "bess" in datasets:
        ax = axes[ax_idx]
        df = datasets["bess"]
        historic = df[df["source"] == "historic"]
        projected = df[df["source"] != "historic"]
        ax.plot(historic["date"], historic["bess_capacity_gw"],
                color="purple", linewidth=2, label="Historic")
        if not projected.empty:
            ax.plot(projected["date"], projected["bess_capacity_gw"],
                    color="purple", linewidth=2, linestyle="--", label="FES projection")
        ax.set_ylabel("Capacity (GW)")
        ax.set_title("GB Battery Storage Capacity")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax_idx += 1

    # 4. Renewable installed capacity
    if "renewables" in datasets:
        ax = axes[ax_idx]
        df = datasets["renewables"]
        ax.plot(df["date"], df["onshore_wind_gw"], label="Onshore wind", color="teal")
        ax.plot(df["date"], df["offshore_wind_gw"], label="Offshore wind", color="darkblue")
        ax.plot(df["date"], df["solar_gw"], label="Solar PV", color="orange")
        ax.set_ylabel("Installed capacity (GW)")
        ax.set_title("UK Renewable Installed Capacity")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax_idx += 1

    # 5. Ancillary market prices
    if "ancillary" in datasets:
        ax = axes[ax_idx]
        df = datasets["ancillary"]
        for service in df["service"].unique():
            sdf = df[df["service"] == service].set_index("date").sort_index()
            ma = sdf["clearing_price_mw_h"].rolling(7, min_periods=1).mean()
            ax.plot(ma.index, ma, linewidth=1.5, label=service)
        ax.set_ylabel("Price (£/MW/h)")
        ax.set_title("Ancillary Market Clearing Prices")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax_idx += 1

    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "01_all_timeseries.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {path}")
    plt.close()


def validate_wholesale(datasets):
    """Compare Agile-derived wholesale prices against Elexon market index."""
    if "spread" not in datasets or "elexon_wholesale" not in datasets:
        print("\nSkipping wholesale validation — need both Agile spread and Elexon wholesale")
        return

    agile = datasets["spread"][["date", "price_mean", "spread_max_min"]].copy()
    agile.columns = ["date", "agile_mean_mwh", "agile_spread"]
    elexon = datasets["elexon_wholesale"][["date", "wholesale_mean", "wholesale_spread"]].copy()

    merged = pd.merge(agile, elexon, on="date", how="inner")
    if merged.empty:
        print("\nNo overlapping dates for wholesale validation")
        return

    print(f"\n--- Wholesale Validation ---")
    print(f"Overlapping days: {len(merged)}")

    # Correlation between price levels
    corr_price = merged["agile_mean_mwh"].corr(merged["wholesale_mean"])
    print(f"Price level correlation (Agile vs Elexon): {corr_price:.3f}")

    # Correlation between spreads
    corr_spread = merged["agile_spread"].corr(merged["wholesale_spread"])
    print(f"Spread correlation (Agile vs Elexon): {corr_spread:.3f}")

    # Ratio (the Agile multiplier)
    ratio = merged["agile_mean_mwh"].mean() / merged["wholesale_mean"].mean()
    print(f"Mean price ratio (Agile/Elexon): {ratio:.2f}")

    spread_ratio = merged["agile_spread"].mean() / merged["wholesale_spread"].mean()
    print(f"Mean spread ratio (Agile/Elexon): {spread_ratio:.2f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.scatter(merged["wholesale_mean"], merged["agile_mean_mwh"],
               alpha=0.3, s=5, color="steelblue")
    ax.set_xlabel("Elexon wholesale mean (£/MWh)")
    ax.set_ylabel("Agile-derived mean (£/MWh)")
    ax.set_title(f"Price level comparison (r={corr_price:.2f})")
    # Add 1:1 line
    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, "k--", alpha=0.3, label="1:1")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.scatter(merged["wholesale_spread"], merged["agile_spread"],
               alpha=0.3, s=5, color="coral")
    ax.set_xlabel("Elexon wholesale spread (£/MWh)")
    ax.set_ylabel("Agile-derived spread (£/MWh)")
    ax.set_title(f"Spread comparison (r={corr_spread:.2f})")
    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, "k--", alpha=0.3, label="1:1")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "01_wholesale_validation.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()


def print_data_quality(datasets):
    """Print data quality summary."""
    print(f"\n{'=' * 60}")
    print("DATA QUALITY SUMMARY")
    print(f"{'=' * 60}")

    for name, df in datasets.items():
        print(f"\n{name}:")
        print(f"  Rows: {len(df)}")
        print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
        nulls = df.isnull().sum()
        if nulls.any():
            print(f"  Missing values:")
            for col, n in nulls[nulls > 0].items():
                print(f"    {col}: {n} ({100 * n / len(df):.1f}%)")
        else:
            print(f"  No missing values")


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)

    print("=" * 60)
    print("DATA OVERVIEW & EXPLORATORY ANALYSIS")
    print("=" * 60)
    print("\nLoading datasets...")

    datasets = load_datasets()

    if not datasets:
        print("\nNo datasets found! Run fetch/fetch_all.py first.")
        sys.exit(1)

    print_data_quality(datasets)
    plot_all_timeseries(datasets)
    validate_wholesale(datasets)

    print(f"\n{'=' * 60}")
    print("OVERVIEW COMPLETE")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
