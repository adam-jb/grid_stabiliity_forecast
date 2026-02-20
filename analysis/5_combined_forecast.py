"""
Combined 5-Year Forecast: Chain all three components.

1. BESS capacity forecast (Component 1) -> monthly GW for 2026-2030
2. Spread model (Component 2) -> predict wholesale spread given BESS capacity
3. Tariff model (Component 3) -> convert wholesale spread to Agile tariff spread

The spread model uses a power-law with floor, calibrated from literature:
  spread = floor + (S0 - floor) * (C0 / bess_gw)^alpha

Seasonality is applied as a multiplicative adjustment from the OLS month dummies.

Includes sensitivity analysis: vary alpha (compression rate) and BESS growth scenario.

Inputs:
  data/processed/bess_forecast.csv
  data/processed/bess_fes_scenarios.csv
  data/processed/spread_model_params.json
  data/processed/tariff_model_params.json
  data/processed/renewable_capacity_monthly.csv

Outputs:
  data/processed/five_year_forecast.csv
  data/processed/plots/05_combined_forecast.png
  data/processed/plots/05_sensitivity.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
PLOT_DIR = os.path.join(PROCESSED_DIR, "plots")


def load_params(filename):
    path = os.path.join(PROCESSED_DIR, filename)
    with open(path) as f:
        return json.load(f)


def power_law_spread(bess_gw, floor, spread_ref, bess_ref, alpha):
    """
    Power-law spread compression with floor.

      spread = floor + (spread_ref - floor) * (bess_ref / bess_gw)^alpha

    As bess_gw grows beyond bess_ref, spread decays toward floor.
    """
    return floor + (spread_ref - floor) * (bess_ref / bess_gw) ** alpha


def seasonal_adjustment(month, month_coefs, spread_ref):
    """
    Apply month seasonality as a multiplicative factor.

    The OLS month_coefs are additive offsets from the January baseline.
    We convert these to a multiplicative factor relative to the annual mean
    so the power-law level is preserved but seasonality is overlaid.
    """
    # Month 1 (January) is the reference (no dummy), so its effect = 0
    month_key = f"month_{month}"
    month_effect = month_coefs.get(month_key, 0.0)

    # Average month effect across the year (including month 1 = 0)
    all_effects = [0.0] + [month_coefs.get(f"month_{m}", 0.0) for m in range(2, 13)]
    avg_effect = np.mean(all_effects)

    # Multiplicative adjustment: how much this month deviates from annual mean
    # Use ratio to spread_ref so it scales properly
    return 1.0 + (month_effect - avg_effect) / spread_ref


def project_wind_generation(forecast_dates):
    """Project monthly average wind generation for forecast period."""
    ren_path = os.path.join(PROCESSED_DIR, "renewable_capacity_monthly.csv")

    if os.path.exists(ren_path):
        ren = pd.read_csv(ren_path, parse_dates=["date"])
        latest = ren.iloc[-1]
        latest_date = ren["date"].max()

        wind_growth_gw_month = 1.5 / 12  # ~1.5 GW/year growth

        # Seasonal capacity factors for wind
        wind_cf = {1: 0.38, 2: 0.35, 3: 0.32, 4: 0.28, 5: 0.25,
                   6: 0.22, 7: 0.22, 8: 0.24, 9: 0.28, 10: 0.32,
                   11: 0.36, 12: 0.38}

        projections = []
        for date in forecast_dates:
            months_ahead = (date.year - latest_date.year) * 12 + (date.month - latest_date.month)
            wind_cap = latest["total_wind_gw"] + wind_growth_gw_month * months_ahead
            wind_gen = wind_cap * wind_cf.get(date.month, 0.30)
            projections.append({"date": date, "wind_gen_gw": wind_gen})

        return pd.DataFrame(projections)

    # Fallback: use 2025 average
    return pd.DataFrame({"date": forecast_dates, "wind_gen_gw": 6.8})


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)

    print("=" * 60)
    print("COMBINED 5-YEAR FORECAST")
    print("=" * 60)

    # Load model parameters
    spread_params = load_params("spread_model_params.json")
    tariff_params = load_params("tariff_model_params.json")

    floor = spread_params["floor"]
    spread_ref = spread_params["spread_ref"]
    bess_ref = spread_params["bess_ref"]
    alpha = spread_params["alpha"]
    alpha_low = spread_params["alpha_low"]
    alpha_high = spread_params["alpha_high"]
    month_coefs = spread_params.get("month_coefs", {})

    price_multiplier = tariff_params.get("price_multiplier", 1.31)

    print(f"\nSpread model: spread = {floor:.0f} + ({spread_ref:.0f} - {floor:.0f})"
          f" * ({bess_ref:.1f} / bess_gw)^{alpha}")
    print(f"  Alpha range: {alpha_low} (conservative) to {alpha_high} (aggressive)")
    print(f"Tariff model: price multiplier = {price_multiplier:.2f}, "
          f"adder = £{tariff_params.get('price_adder_mwh', 65):.0f}/MWh")

    # Load BESS forecast
    bess_path = os.path.join(PROCESSED_DIR, "bess_forecast.csv")
    bess = pd.read_csv(bess_path, parse_dates=["date"])
    print(f"\nBESS forecast: {len(bess)} months, "
          f"{bess['bess_capacity_gw'].iloc[0]:.1f} to {bess['bess_capacity_gw'].iloc[-1]:.1f} GW")

    # Load FES scenarios
    scenarios_path = os.path.join(PROCESSED_DIR, "bess_fes_scenarios.csv")
    scenarios = None
    if os.path.exists(scenarios_path):
        scenarios = pd.read_csv(scenarios_path, parse_dates=["date"])

    # Project wind generation
    forecast_dates = pd.to_datetime(bess["date"].values)
    wind_proj = project_wind_generation(forecast_dates)

    # --- Base case forecast ---
    print("\n--- Generating base case forecast ---")
    results = []
    for _, bess_row in bess.iterrows():
        date = bess_row["date"]
        bess_gw = bess_row["bess_capacity_gw"]
        month = date.month

        wind_row = wind_proj[wind_proj["date"] == date]
        wind_gen = wind_row["wind_gen_gw"].values[0] if not wind_row.empty else 6.8

        # Power-law spread (annual average level)
        ws_annual = power_law_spread(bess_gw, floor, spread_ref, bess_ref, alpha)

        # Apply seasonal adjustment
        season_mult = seasonal_adjustment(month, month_coefs, spread_ref)
        ws = max(ws_annual * season_mult, 0.0)

        # Agile tariff spread
        agile_spread = price_multiplier * ws
        agile_spread_p_kwh = agile_spread / 10

        # FES scenario spreads (vary BESS capacity)
        ws_leading = None
        ws_falling = None
        if scenarios is not None:
            ltw = scenarios[(scenarios["scenario"] == "leading_the_way") &
                            (scenarios["date"] == date)]
            if not ltw.empty:
                ltw_annual = power_law_spread(ltw["bess_capacity_gw"].values[0],
                                              floor, spread_ref, bess_ref, alpha)
                ws_leading = max(ltw_annual * season_mult, 0.0)

            fs = scenarios[(scenarios["scenario"] == "falling_short") &
                           (scenarios["date"] == date)]
            if not fs.empty:
                fs_annual = power_law_spread(fs["bess_capacity_gw"].values[0],
                                             floor, spread_ref, bess_ref, alpha)
                ws_falling = max(fs_annual * season_mult, 0.0)

        results.append({
            "date": date,
            "bess_gw": bess_gw,
            "wind_gen_gw": wind_gen,
            "wholesale_spread_mwh": ws,
            "agile_spread_mwh": agile_spread,
            "agile_spread_p_kwh": agile_spread_p_kwh,
            "spread_leading_the_way": ws_leading,
            "spread_falling_short": ws_falling,
        })

    forecast = pd.DataFrame(results)

    # Save
    forecast_path = os.path.join(PROCESSED_DIR, "five_year_forecast.csv")
    forecast.to_csv(forecast_path, index=False)
    print(f"Saved: {forecast_path}")

    # --- Load historic for context ---
    elexon_path = os.path.join(PROCESSED_DIR, "daily_elexon_wholesale.csv")
    if os.path.exists(elexon_path):
        historic = pd.read_csv(elexon_path, parse_dates=["date"])
        hist_monthly = historic.set_index("date").resample("MS").agg(
            spread_mean=("wholesale_spread", "mean"),
        ).reset_index()
    else:
        hist_monthly = pd.DataFrame()

    # --- Plot 1: Combined forecast ---
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    # Panel 1: BESS capacity
    ax = axes[0]
    ax.plot(forecast["date"], forecast["bess_gw"], color="purple", linewidth=2.5,
            label="Base case (System Transformation)")
    if scenarios is not None:
        for scenario, color, label in [
            ("leading_the_way", "#7B2FBE", "Leading the Way"),
            ("falling_short", "#E9D5FF", "Falling Short"),
        ]:
            sdf = scenarios[scenarios["scenario"] == scenario]
            filt = sdf[sdf["date"].isin(forecast["date"])]
            if not filt.empty:
                ax.plot(filt["date"], filt["bess_capacity_gw"],
                        color=color, linewidth=1.5, linestyle="--", label=label)
    ax.set_ylabel("BESS Capacity (GW)")
    ax.set_title("5-Year Forecast: BESS Growth → Spread Compression → Agile Tariff")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: Wholesale spread
    ax = axes[1]
    if not hist_monthly.empty:
        ax.plot(hist_monthly["date"], hist_monthly["spread_mean"],
                color="steelblue", linewidth=1.5, alpha=0.7, label="Historic (monthly avg)")
    ax.plot(forecast["date"], forecast["wholesale_spread_mwh"],
            color="coral", linewidth=2.5, label="Forecast (base case)")

    # FES scenario range
    if forecast["spread_leading_the_way"].notna().any():
        ax.fill_between(
            forecast["date"],
            forecast["spread_leading_the_way"].fillna(forecast["wholesale_spread_mwh"]),
            forecast["spread_falling_short"].fillna(forecast["wholesale_spread_mwh"]),
            alpha=0.15, color="coral", label="FES scenario range"
        )

    ax.set_ylabel("Wholesale Spread (£/MWh)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 3: Agile tariff spread
    ax = axes[2]
    ax.plot(forecast["date"], forecast["agile_spread_p_kwh"],
            color="teal", linewidth=2.5, label="Forecast Agile spread")
    ax.set_ylabel("Agile Tariff Spread (p/kWh)")
    ax.set_xlabel("Date")
    ax.legend()
    ax.grid(True, alpha=0.3)

    for ax in axes:
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "05_combined_forecast.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()

    # --- Plot 2: Sensitivity / fan chart (alpha sensitivity + FES scenarios) ---
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Panel 1: Alpha sensitivity (at base-case BESS)
    ax = axes[0]
    if not hist_monthly.empty:
        ax.plot(hist_monthly["date"], hist_monthly["spread_mean"],
                color="steelblue", linewidth=1.5, label="Historic")

    ax.plot(forecast["date"], forecast["wholesale_spread_mwh"],
            color="coral", linewidth=2.5, label=f"Central (α={alpha})")

    # Conservative (less compression) and aggressive (more compression)
    for a, label, ls in [
        (alpha_low, f"Conservative (α={alpha_low})", "--"),
        (alpha_high, f"Aggressive (α={alpha_high})", ":"),
    ]:
        spreads = []
        for _, row in forecast.iterrows():
            s_ann = power_law_spread(row["bess_gw"], floor, spread_ref, bess_ref, a)
            s_mult = seasonal_adjustment(row["date"].month, month_coefs, spread_ref)
            spreads.append(max(s_ann * s_mult, 0.0))
        ax.plot(forecast["date"], spreads, color="coral", linewidth=1.5,
                linestyle=ls, label=label)

    ax.axvline(pd.Timestamp("2026-01-01"), color="gray", linestyle=":", alpha=0.5)
    ax.set_ylabel("Wholesale Spread (£/MWh)")
    ax.set_title("Sensitivity: Alpha (compression rate)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: FES scenario sensitivity
    ax = axes[1]
    if not hist_monthly.empty:
        ax.plot(hist_monthly["date"], hist_monthly["spread_mean"],
                color="steelblue", linewidth=1.5, label="Historic")

    ax.plot(forecast["date"], forecast["wholesale_spread_mwh"],
            color="coral", linewidth=2.5, label="System Transformation")

    if forecast["spread_leading_the_way"].notna().any():
        ax.plot(forecast["date"], forecast["spread_leading_the_way"],
                color="#7B2FBE", linewidth=1.5, linestyle="--", label="Leading the Way")
        ax.plot(forecast["date"], forecast["spread_falling_short"],
                color="#E9D5FF", linewidth=1.5, linestyle="--", label="Falling Short")
        ax.fill_between(
            forecast["date"],
            forecast["spread_leading_the_way"].fillna(forecast["wholesale_spread_mwh"]),
            forecast["spread_falling_short"].fillna(forecast["wholesale_spread_mwh"]),
            alpha=0.1, color="coral"
        )

    ax.axvline(pd.Timestamp("2026-01-01"), color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Date")
    ax.set_ylabel("Wholesale Spread (£/MWh)")
    ax.set_title("Sensitivity: FES BESS Scenarios")
    ax.legend()
    ax.grid(True, alpha=0.3)

    for ax in axes:
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "05_sensitivity.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print("5-YEAR FORECAST SUMMARY")
    print(f"{'=' * 60}")
    print(f"\n{'Year':<6} {'BESS (GW)':<12} {'Spread (£/MWh)':<18} {'Agile Spread (p/kWh)'}")
    print("-" * 58)

    for year in range(2026, 2031):
        ydf = forecast[forecast["date"].dt.year == year]
        if ydf.empty:
            continue
        print(f"{year:<6} {ydf['bess_gw'].mean():<12.1f} {ydf['wholesale_spread_mwh'].mean():<18.1f} "
              f"{ydf['agile_spread_p_kwh'].mean():.1f}")

    print(f"\nModel: spread = {floor:.0f} + ({spread_ref:.0f} - {floor:.0f})"
          f" × ({bess_ref:.1f} / bess_gw)^{alpha}")
    print(f"\nKey assumptions:")
    print(f"  Spread compression: power-law α={alpha} (range {alpha_low}–{alpha_high})")
    print(f"  Floor: £{floor:.0f}/MWh (round-trip efficiency cost)")
    print(f"  BESS growth: System Transformation scenario")
    print(f"  Wind growth: +1.5 GW/year installed capacity")
    print(f"  Agile multiplier: {price_multiplier:.2f}× wholesale")
    print(f"\nSources: CAISO battery reports, Modo Energy, Karaduman (2023)")


if __name__ == "__main__":
    main()
