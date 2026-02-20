"""
Component 3: Wholesale Spread → Agile Tariff Spread

The easy one. The Agile tariff is mechanically derived from wholesale prices:
  agile = wholesale × multiplier + adder

So the spread pass-through is approximately:
  agile_spread ≈ multiplier × wholesale_spread

This script validates that empirically and extracts the pass-through coefficient.

Inputs:
  data/processed/daily_wholesale_spread.csv  (Agile-derived prices)
  data/processed/daily_elexon_wholesale.csv  (direct wholesale prices, if available)

Outputs:
  data/processed/tariff_model_params.json
  data/processed/plots/04_tariff_model.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
PLOT_DIR = os.path.join(PROCESSED_DIR, "plots")


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)

    print("=" * 60)
    print("COMPONENT 3: TARIFF PASS-THROUGH MODEL")
    print("=" * 60)

    # Load Agile-derived spread data
    agile_path = os.path.join(PROCESSED_DIR, "daily_wholesale_spread.csv")
    agile = pd.read_csv(agile_path, parse_dates=["date"])
    print(f"Agile spread data: {len(agile)} days")

    # Load Elexon wholesale if available
    elexon_path = os.path.join(PROCESSED_DIR, "daily_elexon_wholesale.csv")
    has_elexon = os.path.exists(elexon_path)

    if has_elexon:
        elexon = pd.read_csv(elexon_path, parse_dates=["date"])
        print(f"Elexon wholesale data: {len(elexon)} days")

        # Merge
        merged = pd.merge(
            agile[["date", "spread_max_min", "price_mean", "price_max", "price_min"]],
            elexon[["date", "wholesale_spread", "wholesale_mean"]],
            on="date", how="inner"
        )
        merged = merged.dropna()
        print(f"Overlapping days: {len(merged)}")

        # --- Regression: Agile spread = α + β × wholesale spread ---
        print("\n--- Spread pass-through regression ---")
        X = sm.add_constant(merged["wholesale_spread"])
        y = merged["spread_max_min"]

        model = sm.OLS(y, X).fit(cov_type="HC1")
        print(f"R-squared: {model.rsquared:.3f}")
        print(f"Intercept: {model.params['const']:.2f}")
        print(f"Slope (multiplier): {model.params['wholesale_spread']:.3f}")
        print(f"  SE: {model.bse['wholesale_spread']:.3f}")

        multiplier = model.params["wholesale_spread"]
        intercept = model.params["const"]

        # --- Price level regression ---
        print("\n--- Price level regression ---")
        X2 = sm.add_constant(merged["wholesale_mean"])
        y2 = merged["price_mean"]
        model2 = sm.OLS(y2, X2).fit(cov_type="HC1")
        print(f"R-squared: {model2.rsquared:.3f}")
        print(f"Intercept: {model2.params['const']:.2f} (£/MWh — the 'adder')")
        print(f"Slope: {model2.params['wholesale_mean']:.3f} (the regional multiplier)")

        # Save params
        params = {
            "spread_multiplier": float(multiplier),
            "spread_intercept": float(intercept),
            "spread_r_squared": float(model.rsquared),
            "price_multiplier": float(model2.params["wholesale_mean"]),
            "price_adder_mwh": float(model2.params["const"]),
            "price_r_squared": float(model2.rsquared),
            "n_obs": int(len(merged)),
        }

        # --- Plot ---
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        # Spread pass-through
        ax = axes[0]
        ax.scatter(merged["wholesale_spread"], merged["spread_max_min"],
                   alpha=0.2, s=8, color="steelblue")
        x_range = np.linspace(merged["wholesale_spread"].min(),
                              merged["wholesale_spread"].max(), 100)
        ax.plot(x_range, intercept + multiplier * x_range,
                color="coral", linewidth=2,
                label=f"y = {intercept:.1f} + {multiplier:.2f}x (R²={model.rsquared:.2f})")
        ax.set_xlabel("Elexon Wholesale Spread (£/MWh)")
        ax.set_ylabel("Agile Spread (£/MWh)")
        ax.set_title("Spread Pass-Through")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # Price level relationship
        ax = axes[1]
        ax.scatter(merged["wholesale_mean"], merged["price_mean"],
                   alpha=0.2, s=8, color="teal")
        x_range = np.linspace(merged["wholesale_mean"].min(),
                              merged["wholesale_mean"].max(), 100)
        ax.plot(x_range,
                model2.params["const"] + model2.params["wholesale_mean"] * x_range,
                color="coral", linewidth=2,
                label=f"multiplier={model2.params['wholesale_mean']:.2f}, "
                      f"adder=£{model2.params['const']:.0f}/MWh")
        ax.set_xlabel("Elexon Wholesale Mean (£/MWh)")
        ax.set_ylabel("Agile Mean Price (£/MWh)")
        ax.set_title("Price Level Relationship")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # Time series comparison
        ax = axes[2]
        merged_sorted = merged.sort_values("date")
        agile_ma = merged_sorted.set_index("date")["spread_max_min"].rolling(30, min_periods=7).mean()
        elexon_ma = merged_sorted.set_index("date")["wholesale_spread"].rolling(30, min_periods=7).mean()
        ax.plot(agile_ma.index, agile_ma, label="Agile spread", color="steelblue")
        ax.plot(elexon_ma.index, elexon_ma, label="Elexon wholesale spread", color="coral")
        ax.set_ylabel("Spread (£/MWh)")
        ax.set_title("Spread Over Time (30d rolling avg)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    else:
        print("\nNo Elexon wholesale data — using Agile data alone")
        print("The Agile tariff is mechanically linked to wholesale prices.")
        print("Without direct wholesale data, we note that spread_max_min from Agile")
        print("is approximately multiplier × wholesale spread.")
        print("Using multiplier ≈ 1.0 (conservative assumption).\n")

        # When we only have Agile data, the multiplier is embedded in the prices
        # For forecasting, we can use Agile spread directly as a proxy
        params = {
            "spread_multiplier": 1.0,
            "spread_intercept": 0.0,
            "spread_r_squared": None,
            "price_multiplier": 1.0,
            "price_adder_mwh": 0.0,
            "price_r_squared": None,
            "n_obs": len(agile),
            "note": "No Elexon cross-validation available — using Agile prices as direct proxy",
        }

        # Simple plot of Agile spread over time
        fig, ax = plt.subplots(figsize=(14, 5))
        agile_sorted = agile.sort_values("date")
        ma = agile_sorted.set_index("date")["spread_max_min"].rolling(30, min_periods=7).mean()
        ax.plot(ma.index, ma, color="steelblue", linewidth=2)
        ax.set_ylabel("Agile Spread (£/MWh)")
        ax.set_title("Daily Agile Tariff Spread (30d rolling avg)")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "04_tariff_model.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot: {path}")
    plt.close()

    # Save params
    params_path = os.path.join(PROCESSED_DIR, "tariff_model_params.json")
    with open(params_path, "w") as f:
        json.dump(params, f, indent=2)
    print(f"Saved params: {params_path}")

    print(f"\n{'=' * 60}")
    print("TARIFF MODEL SUMMARY")
    print(f"{'=' * 60}")
    print(f"Spread multiplier: {params['spread_multiplier']:.2f}")
    if params.get("spread_r_squared"):
        print(f"Spread R²: {params['spread_r_squared']:.3f}")
    if params.get("price_multiplier") and params["price_multiplier"] != 1.0:
        print(f"Price multiplier: {params['price_multiplier']:.2f}")
        print(f"Price adder: £{params['price_adder_mwh']:.0f}/MWh")
    print(f"\nInterpretation: A £1/MWh change in wholesale spread translates to a "
          f"~£{params['spread_multiplier']:.1f}/MWh change in Agile tariff spread")


if __name__ == "__main__":
    main()
