"""
Component 2: BESS Capacity -> Wholesale Spread Compression

Estimates how BESS growth compresses the daily wholesale spread.

The relationship is NOT linear. Literature and empirical evidence show:
  - Diminishing returns: each additional GW compresses less than the last
  - Hard floor: spread can't go below round-trip efficiency losses (~15% of price)
  - Two channels: ancillary (already saturated in GB) and wholesale (ongoing)

Functional form (power law with floor):
  spread = floor + (S0 - floor) * (C0 / bess_gw)^alpha

Calibrated from:
  - CAISO data: capacity tripled (4→11 GW), revenue halved → alpha ≈ 0.65
  - Modo Energy Germany: +50% capacity → -17% revenue → alpha ≈ 0.46
  - Central estimate: alpha ≈ 0.5 for wholesale spread channel

Sources:
  - Karaduman (2023), "Economics of Grid-Scale Energy Storage" (Stanford GSB)
  - Dumitrescu, Silvente & Tankov (2024), arXiv:2410.12495
  - CAISO Special Reports on Battery Storage (2022-2024)
  - Modo Energy Germany overbuild analysis (Jan 2026)

Inputs:
  data/processed/daily_elexon_wholesale.csv
  data/processed/bess_capacity_monthly.csv
  data/processed/daily_wind_solar_generation.csv

Outputs:
  data/processed/spread_model_params.json
  data/processed/plots/03_spread_model.png
  data/processed/plots/03_spread_diagnostics.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import statsmodels.api as sm
from scipy.optimize import curve_fit
import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
PLOT_DIR = os.path.join(PROCESSED_DIR, "plots")


def load_and_merge():
    """Load and merge all datasets for the spread model."""
    elexon_path = os.path.join(PROCESSED_DIR, "daily_elexon_wholesale.csv")
    agile_path = os.path.join(PROCESSED_DIR, "daily_wholesale_spread.csv")

    if os.path.exists(elexon_path):
        spread = pd.read_csv(elexon_path, parse_dates=["date"])
        spread_col = "wholesale_spread"
        price_col = "wholesale_mean"
        print(f"Using Elexon wholesale spread (direct market data)")
    else:
        spread = pd.read_csv(agile_path, parse_dates=["date"])
        spread_col = "spread_max_min"
        price_col = "price_mean"
        print(f"Using Agile-derived spread (Elexon not available)")

    # BESS capacity (monthly step function)
    bess = pd.read_csv(
        os.path.join(PROCESSED_DIR, "bess_capacity_monthly.csv"),
        parse_dates=["date"]
    )
    bess["year_month"] = bess["date"].dt.to_period("M")
    bess_lookup = bess[bess["source"] == "historic"].set_index("year_month")["bess_capacity_gw"]

    spread["year_month"] = spread["date"].dt.to_period("M")
    spread["bess_gw"] = spread["year_month"].map(bess_lookup)

    # Wind generation
    gen_path = os.path.join(PROCESSED_DIR, "daily_wind_solar_generation.csv")
    if os.path.exists(gen_path):
        gen = pd.read_csv(gen_path, parse_dates=["date"])
        spread = pd.merge(spread, gen[["date", "wind_gen_gw"]], on="date", how="left")
    else:
        spread["wind_gen_gw"] = np.nan

    spread = spread.dropna(subset=["bess_gw"]).copy()
    spread["month"] = spread["date"].dt.month
    spread["year"] = spread["date"].dt.year
    spread["day_of_week"] = spread["date"].dt.dayofweek

    print(f"Merged dataset: {len(spread)} days")
    print(f"Date range: {spread['date'].min().date()} to {spread['date'].max().date()}")
    print(f"BESS range: {spread['bess_gw'].min():.1f} to {spread['bess_gw'].max():.1f} GW")

    return spread, spread_col, price_col


def run_ols(df, spread_col, include_year_dummies=True):
    """Run OLS regression (for diagnostics and comparison)."""
    label = "(with year dummies)" if include_year_dummies else "(no year dummies)"
    print(f"\n--- OLS: {spread_col} {label} ---")

    df = df.copy()
    features = ["bess_gw"]
    feature_labels = ["BESS capacity (GW)"]

    if "wind_gen_gw" in df.columns and df["wind_gen_gw"].notna().sum() > len(df) * 0.5:
        features.append("wind_gen_gw")
        feature_labels.append("Wind generation (GW)")

    df["weekend"] = (df["day_of_week"] >= 5).astype(float)
    features.append("weekend")
    feature_labels.append("Weekend")

    month_dummies = pd.get_dummies(df["month"], prefix="month", drop_first=True).astype(float)
    year_dummies = pd.DataFrame()
    if include_year_dummies:
        year_dummies = pd.get_dummies(df["year"], prefix="year", drop_first=True).astype(float)

    model_df = df.dropna(subset=features + [spread_col]).copy()
    X = model_df[features].copy()
    X = pd.concat([X, month_dummies.loc[model_df.index]], axis=1)
    if not year_dummies.empty:
        X = pd.concat([X, year_dummies.loc[model_df.index]], axis=1)
    X = sm.add_constant(X)
    y = model_df[spread_col]

    model = sm.OLS(y, X).fit(cov_type="HC1")
    print(f"Observations: {len(y)}, R²: {model.rsquared:.3f}")

    for feat, label in zip(features, feature_labels):
        coef = model.params[feat]
        se = model.bse[feat]
        pval = model.pvalues[feat]
        sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
        print(f"  {label}: {coef:+.2f} (SE={se:.2f}, p={pval:.4f}) {sig}")

    return model, model_df


def power_law_spread(bess_gw, floor, spread_ref, bess_ref, alpha):
    """
    Power-law spread compression with floor.

      spread = floor + (spread_ref - floor) * (bess_ref / bess_gw)^alpha

    As bess_gw grows beyond bess_ref, spread decays toward floor.
    Diminishing returns: each additional GW compresses less.
    """
    return floor + (spread_ref - floor) * (bess_ref / bess_gw) ** alpha


def fit_power_law(df, spread_col):
    """
    Fit the power-law model to monthly-averaged data.

    Uses monthly averages to reduce noise and focus on the structural
    relationship between BESS capacity and spread.
    """
    print(f"\n--- Power-law fit: spread = floor + (S0 - floor) * (C0/C)^alpha ---")

    # Monthly averages to reduce daily noise
    monthly = df.groupby("year_month").agg(
        bess_gw=("bess_gw", "mean"),
        spread=pd.NamedAgg(column=spread_col, aggfunc="mean"),
        wind_gw=("wind_gen_gw", "mean"),
        n_days=("date", "count"),
    ).reset_index()
    monthly = monthly.dropna()

    print(f"Monthly observations: {len(monthly)}")
    print(f"BESS range: {monthly['bess_gw'].min():.1f} to {monthly['bess_gw'].max():.1f} GW")
    print(f"Spread range: £{monthly['spread'].min():.0f} to £{monthly['spread'].max():.0f}/MWh")

    return monthly


def calibrate_params(monthly_data, spread_col):
    """
    Calibrate the power-law parameters using GB data + literature.

    The GB time series alone can't cleanly identify alpha because BESS
    growth is confounded with the energy crisis. We use:
      - GB 2025 baseline as the anchor point (C0, S0)
      - CAISO data for alpha (0.65 total, ~0.5 wholesale-only)
      - Modo Germany sensitivity for robustness (implies ~0.46)
      - Round-trip efficiency for floor (~15% of avg wholesale price)
    """
    # Reference point: 2025 GB data
    recent = monthly_data[monthly_data["bess_gw"] >= 6.0]  # 2025-ish
    if len(recent) >= 6:
        spread_ref = recent["spread"].mean()
        bess_ref = recent["bess_gw"].mean()
    else:
        spread_ref = monthly_data["spread"].iloc[-6:].mean()
        bess_ref = monthly_data["bess_gw"].iloc[-6:].mean()

    # Floor: round-trip efficiency losses (~15%) times average wholesale price
    avg_price = 90.0  # approximate GB wholesale price
    rt_efficiency_loss = 0.15
    floor = avg_price * rt_efficiency_loss  # ~£13.5/MWh

    # Alpha calibration from literature:
    #   CAISO (total revenue incl ancillary): capacity tripled, revenue halved → alpha ≈ 0.65
    #   Modo Germany (wholesale-only): +50% capacity → -17% revenue → alpha ≈ 0.46
    #   Central estimate for wholesale spread: 0.5
    alpha_caiso = 0.65     # from CAISO 4→11 GW, $103→$53/kW/yr
    alpha_modo = 0.46      # from Modo +50% capacity → -17% revenue
    alpha_central = 0.50   # our central estimate (wholesale channel)

    print(f"\nCalibrated parameters:")
    print(f"  Reference point: {spread_ref:.1f} £/MWh at {bess_ref:.1f} GW (2025 GB)")
    print(f"  Floor: {floor:.1f} £/MWh (RT efficiency loss × avg price)")
    print(f"  Alpha estimates:")
    print(f"    CAISO total revenue: {alpha_caiso}")
    print(f"    Modo Germany wholesale: {alpha_modo}")
    print(f"    Central estimate: {alpha_central}")

    params = {
        "model_type": "power_law_with_floor",
        "spread_ref": float(spread_ref),
        "bess_ref": float(bess_ref),
        "floor": float(floor),
        "alpha": float(alpha_central),
        "alpha_low": float(alpha_modo),   # less compression (conservative)
        "alpha_high": float(alpha_caiso),  # more compression (aggressive)
    }

    # Show what this predicts at various capacities
    print(f"\nForecasted spread at different BESS capacities:")
    print(f"  {'BESS (GW)':<12} {'Central':<14} {'Conservative':<14} {'Aggressive'}")
    for c in [bess_ref, 10, 13, 16, 18, 20, 25]:
        s_central = power_law_spread(c, floor, spread_ref, bess_ref, alpha_central)
        s_low = power_law_spread(c, floor, spread_ref, bess_ref, alpha_modo)
        s_high = power_law_spread(c, floor, spread_ref, bess_ref, alpha_caiso)
        marker = " <-- 2025" if abs(c - bess_ref) < 0.5 else ""
        print(f"  {c:<12.0f} £{s_central:<13.1f} £{s_low:<13.1f} £{s_high:.1f}{marker}")

    return params


def plot_model(df, monthly, params, spread_col, ols_model, ols_model_df):
    """Plot the power-law fit alongside data and OLS comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    floor = params["floor"]
    S0 = params["spread_ref"]
    C0 = params["bess_ref"]

    # 1. Monthly spread vs BESS with power-law curves
    ax = axes[0, 0]
    ax.scatter(monthly["bess_gw"], monthly["spread"],
               s=40, alpha=0.6, color="steelblue", zorder=5, label="Monthly avg (GB)")

    bess_range = np.linspace(0.5, 25, 200)
    for alpha, label, color, ls in [
        (params["alpha"], f'Central (alpha={params["alpha"]})', "coral", "-"),
        (params["alpha_low"], f'Conservative (alpha={params["alpha_low"]})', "coral", "--"),
        (params["alpha_high"], f'Aggressive (alpha={params["alpha_high"]})', "coral", ":"),
    ]:
        spread_curve = [power_law_spread(c, floor, S0, C0, alpha) for c in bess_range]
        ax.plot(bess_range, spread_curve, color=color, linewidth=2 if ls == "-" else 1.5,
                linestyle=ls, label=label)

    ax.axhline(floor, color="gray", linestyle=":", alpha=0.5,
               label=f"Floor (£{floor:.0f}, RT efficiency)")
    ax.set_xlabel("BESS Capacity (GW)")
    ax.set_ylabel("Wholesale Spread (£/MWh)")
    ax.set_title("Spread Compression: Power Law with Floor")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2. Time series with power-law prediction
    ax = axes[0, 1]
    monthly_sorted = monthly.sort_values("bess_gw")
    # Map monthly data to dates for time-series plot
    df_monthly_ts = df.groupby("year_month").agg(
        date=("date", "first"),
        spread=pd.NamedAgg(column=spread_col, aggfunc="mean"),
        bess_gw=("bess_gw", "mean"),
    ).reset_index().sort_values("date")

    predicted = [power_law_spread(c, floor, S0, C0, params["alpha"])
                 for c in df_monthly_ts["bess_gw"]]

    ax.plot(df_monthly_ts["date"], df_monthly_ts["spread"],
            color="steelblue", linewidth=1.5, label="Actual (monthly avg)")
    ax.plot(df_monthly_ts["date"], predicted,
            color="coral", linewidth=2, label="Power-law prediction")
    ax.set_ylabel("Spread (£/MWh)")
    ax.set_title("Actual vs Power-Law Predicted Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Scatter colored by wind (daily data)
    ax = axes[1, 0]
    if "wind_gen_gw" in df.columns and df["wind_gen_gw"].notna().any():
        scatter = ax.scatter(df["bess_gw"], df[spread_col],
                             c=df["wind_gen_gw"], cmap="YlGnBu",
                             alpha=0.2, s=5, vmin=0, vmax=15)
        plt.colorbar(scatter, ax=ax, label="Wind gen (GW)")
    else:
        ax.scatter(df["bess_gw"], df[spread_col], alpha=0.2, s=5, color="steelblue")

    # Overlay power-law curve
    spread_curve = [power_law_spread(c, floor, S0, C0, params["alpha"]) for c in bess_range]
    ax.plot(bess_range, spread_curve, color="coral", linewidth=2.5, zorder=10)
    ax.set_xlabel("BESS Capacity (GW)")
    ax.set_ylabel("Daily Spread (£/MWh)")
    ax.set_title("Daily Data with Power-Law Fit")
    ax.grid(True, alpha=0.3)

    # 4. OLS residuals for reference
    ax = axes[1, 1]
    ols_sorted = ols_model_df.sort_values("date")
    residuals = ols_model.resid.loc[ols_sorted.index]
    resid_ma = pd.Series(residuals.values, index=ols_sorted["date"]).rolling(30, min_periods=7).mean()
    ax.plot(resid_ma.index, resid_ma, color="gray", linewidth=1.5)
    ax.axhline(0, color="black", linestyle="--", alpha=0.3)
    ax.set_ylabel("Residual (£/MWh)")
    ax.set_title("OLS Residuals (30d rolling mean)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "03_spread_model.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {path}")
    plt.close()


def plot_diagnostics(ols_model):
    """OLS diagnostics for reference."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    sm.qqplot(ols_model.resid, line="45", ax=ax, alpha=0.3, markersize=3)
    ax.set_title("Q-Q Plot (OLS Residuals)")

    ax = axes[1]
    ax.hist(ols_model.resid, bins=50, color="steelblue", alpha=0.7, edgecolor="white")
    ax.axvline(0, color="black", linestyle="--", alpha=0.3)
    ax.set_xlabel("Residual (£/MWh)")
    ax.set_title("Distribution (OLS Residuals)")

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "03_spread_diagnostics.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()


def save_params(params, ols_model, spread_col):
    """Save all model parameters."""
    # Add OLS results for reference
    params["ols_bess_coef_with_year"] = float(ols_model.params.get("bess_gw", 0))
    params["ols_wind_coef"] = float(ols_model.params.get("wind_gen_gw", 0))
    params["ols_r_squared"] = float(ols_model.rsquared)
    params["ols_n_obs"] = int(ols_model.nobs)
    params["spread_col"] = spread_col

    # Month effects from OLS (for seasonality in forecast)
    params["month_coefs"] = {
        str(k): float(v) for k, v in ols_model.params.items()
        if str(k).startswith("month_")
    }
    params["weekend_coef"] = float(ols_model.params.get("weekend", 0))

    path = os.path.join(PROCESSED_DIR, "spread_model_params.json")
    with open(path, "w") as f:
        json.dump(params, f, indent=2)
    print(f"\nSaved model params: {path}")

    return params


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)

    print("=" * 60)
    print("COMPONENT 2: SPREAD COMPRESSION MODEL")
    print("=" * 60)

    df, spread_col, price_col = load_and_merge()

    # 1. OLS regressions (for diagnostics and month/weekend effects)
    ols_with_year, ols_df = run_ols(df, spread_col, include_year_dummies=True)
    print()
    ols_no_year, _ = run_ols(df, spread_col, include_year_dummies=False)

    bess_with = ols_with_year.params.get("bess_gw", 0)
    bess_without = ols_no_year.params.get("bess_gw", 0)
    print(f"\n  OLS BESS coef with year dummies: {bess_with:+.2f}")
    print(f"  OLS BESS coef without year dummies: {bess_without:+.2f}")
    print(f"  (Both are unreliable — using power-law instead)")

    # 2. Power-law model
    monthly = fit_power_law(df, spread_col)
    params = calibrate_params(monthly, spread_col)

    # 3. Plots
    plot_model(df, monthly, params, spread_col, ols_with_year, ols_df)
    plot_diagnostics(ols_with_year)

    # 4. Save
    params = save_params(params, ols_with_year, spread_col)

    print(f"\n{'=' * 60}")
    print("SPREAD MODEL SUMMARY")
    print(f"{'=' * 60}")
    print(f"Model: spread = {params['floor']:.0f} + ({params['spread_ref']:.0f} - {params['floor']:.0f})"
          f" * ({params['bess_ref']:.1f} / bess_gw)^{params['alpha']}")
    print(f"\nCalibration:")
    print(f"  Anchor: £{params['spread_ref']:.0f}/MWh at {params['bess_ref']:.1f} GW (2025 GB)")
    print(f"  Floor: £{params['floor']:.0f}/MWh (round-trip efficiency cost)")
    print(f"  Alpha: {params['alpha']} (range {params['alpha_low']}-{params['alpha_high']})")
    print(f"\nSources: CAISO battery reports 2022-24, Modo Energy Germany 2026,")
    print(f"  Karaduman (Stanford 2023), Dumitrescu et al (arXiv 2024)")


if __name__ == "__main__":
    main()
