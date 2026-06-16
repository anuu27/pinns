"""Generate synthetic insurance data and run exploratory data analysis.

Created: 2026-06-08
Purpose: Produce and persist a portfolio of synthetic policies plus their
         classical reserve trajectories, then generate a comprehensive EDA
         report with plots saved to artifacts/eda/.

Usage:
    python generate_and_eda.py                      # defaults from config.yaml
    python generate_and_eda.py --n 500              # generate 500 policies
    python generate_and_eda.py --fmt parquet        # save as parquet instead of csv
    python generate_and_eda.py --no-plots           # skip plot generation

Output files (all under artifacts/eda/):
    policies.csv / .parquet         — one row per policy
    reserve_trajectories.csv / ...  — one row per (policy, time) point
    01_age_distribution.png
    02_term_distribution.png
    03_sum_assured_distribution.png
    04_interest_rate_distribution.png
    05_premium_distribution.png
    06_risk_profile_breakdown.png
    07_mortality_at_inception.png
    08_reserve_trajectories_sample.png
    09_reserve_by_age_band.png
    10_correlation_heatmap.png
    11_premium_vs_sum_assured.png
    12_reserve_surface_age_vs_term.png
    13_reserve_distribution.png
    14_reserve_vs_age.png
    15_reserve_vs_mortality.png
    16_reserve_vs_sum_assured.png
    17_reserve_vs_interest_rate.png
    18_reserve_vs_premium.png
    eda_summary.txt                 — printed summary statistics
"""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd

from src.actuarial.actuarial_solver import ThieleSolver
from src.data.simulator import PolicySimulator
from src.data.storage import (
    policies_to_dataframe,
    reserve_trajectories_to_dataframe,
    save_datasets,
)
from src.data.mortality_loader import CSVMortalityLoader
from src.utils.config import ConfigLoader, ensure_directories

# ── constants ──────────────────────────────────────────────────────────────────
CONFIG_PATH = Path("configs/config.yaml")
EDA_DIR = Path("artifacts/eda")
SAMPLE_TRAJECTORIES = 40   # how many policies to draw on the trajectory plot
FIGURE_DPI = 130
PALETTE = "#4C6EF5"        # single-hue accent for histograms


# ── helpers ────────────────────────────────────────────────────────────────────
def _save(fig: plt.Figure, name: str) -> None:
    path = EDA_DIR / name
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path}")


def _age_band(age: int) -> str:
    lo = (age // 10) * 10
    return f"{lo}–{lo + 9}"


# ── plot functions ─────────────────────────────────────────────────────────────
def plot_age_distribution(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df["age"], bins=range(df["age"].min(), df["age"].max() + 2), color=PALETTE, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Issue age (years)")
    ax.set_ylabel("Policy count")
    ax.set_title("Age at inception")
    ax.axvline(df["age"].mean(), color="#E03131", linestyle="--", linewidth=1.2, label=f"Mean {df['age'].mean():.1f}")
    ax.legend(fontsize=9)
    _save(fig, "01_age_distribution.png")


def plot_term_distribution(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df["term"], bins=range(df["term"].min(), df["term"].max() + 2), color=PALETTE, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Policy term (years)")
    ax.set_ylabel("Policy count")
    ax.set_title("Policy term distribution")
    _save(fig, "02_term_distribution.png")


def plot_sum_assured_distribution(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(df["sum_assured"] / 1_000, bins=40, color=PALETTE, edgecolor="white", linewidth=0.5)
    axes[0].set_xlabel("Sum assured (£k)")
    axes[0].set_ylabel("Policy count")
    axes[0].set_title("Sum assured (linear)")
    axes[1].hist(np.log10(df["sum_assured"]), bins=40, color="#7950F2", edgecolor="white", linewidth=0.5)
    axes[1].set_xlabel("log₁₀(Sum assured)")
    axes[1].set_title("Sum assured (log scale)")
    fig.tight_layout()
    _save(fig, "03_sum_assured_distribution.png")


def plot_interest_rate_distribution(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df["interest_rate"] * 100, bins=30, color=PALETTE, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Interest rate (%)")
    ax.set_ylabel("Policy count")
    ax.set_title("Interest rate assumption")
    _save(fig, "04_interest_rate_distribution.png")


def plot_premium_distribution(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df["premium"], bins=40, color=PALETTE, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Annual premium (£)")
    ax.set_ylabel("Policy count")
    ax.set_title("Premium distribution (loaded, continuous)")
    _save(fig, "05_premium_distribution.png")


def plot_risk_profile_breakdown(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    cats = ["smoker_status", "health_tier", "gender", "occupation_risk"]
    titles = ["Smoker status", "Health tier", "Gender", "Occupation risk"]
    colors = ["#4C6EF5", "#7950F2", "#F03E3E", "#2F9E44"]
    for ax, col, title, color in zip(axes, cats, titles, colors):
        counts = df[col].value_counts()
        ax.bar(counts.index, counts.values, color=color, edgecolor="white")
        ax.set_title(title)
        ax.set_ylabel("Count" if ax == axes[0] else "")
        for bar, val in zip(ax.patches, counts.values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + counts.max() * 0.01,
                f"{val}",
                ha="center", va="bottom", fontsize=8,
            )
    fig.suptitle("Underwriting risk profile breakdown", fontsize=12, y=1.02)
    fig.tight_layout()
    _save(fig, "06_risk_profile_breakdown.png")


def plot_mortality_at_inception(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    scatter = ax.scatter(
        df["age"],
        df["mortality_at_inception"] * 1000,
        c=df["risk_adjustment_factor"],
        cmap="plasma",
        alpha=0.55,
        s=18,
        linewidths=0,
    )
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Risk adjustment factor", fontsize=9)
    ax.set_xlabel("Issue age")
    ax.set_ylabel("Mortality intensity at t=0 (‰)")
    ax.set_title("Mortality at inception vs age, coloured by risk factor")
    _save(fig, "07_mortality_at_inception.png")


def plot_reserve_trajectories_sample(traj_df: pd.DataFrame) -> None:
    sample_ids = (
        traj_df["policy_id"]
        .drop_duplicates()
        .sample(min(SAMPLE_TRAJECTORIES, traj_df["policy_id"].nunique()), random_state=42)
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    cmap = cm.get_cmap("tab20", len(sample_ids))
    for i, pid in enumerate(sample_ids):
        sub = traj_df[traj_df["policy_id"] == pid].sort_values("time")
        ax.plot(sub["time"], sub["reserve"], color=cmap(i), linewidth=0.9, alpha=0.75)
    ax.set_xlabel("Policy time (years)")
    ax.set_ylabel("Reserve (£)")
    ax.set_title(f"Classical reserve trajectories — {len(sample_ids)} sampled policies")
    _save(fig, "08_reserve_trajectories_sample.png")


def plot_reserve_by_age_band(traj_df: pd.DataFrame) -> None:
    traj_df = traj_df.copy()
    traj_df["age_band"] = traj_df["age"].apply(_age_band)
    bands = sorted(traj_df["age_band"].unique())
    fig, ax = plt.subplots(figsize=(10, 5))
    cmap = cm.get_cmap("coolwarm", len(bands))
    for i, band in enumerate(bands):
        sub = traj_df[traj_df["age_band"] == band]
        grouped = sub.groupby("time")["reserve"].mean().reset_index()
        ax.plot(grouped["time"], grouped["reserve"], color=cmap(i), linewidth=1.8, label=band)
    ax.set_xlabel("Policy time (years)")
    ax.set_ylabel("Mean reserve (£)")
    ax.set_title("Mean reserve trajectory by issue age band")
    ax.legend(title="Age band", fontsize=8, ncol=2)
    _save(fig, "09_reserve_by_age_band.png")


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    numeric_cols = [
        "age", "term", "interest_rate", "sum_assured", "premium",
        "mortality_at_inception", "mortality_at_midterm",
        "risk_adjustment_factor", "premium_to_sum_assured_ratio",
    ]
    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_xticks(range(len(numeric_cols)))
    ax.set_yticks(range(len(numeric_cols)))
    ax.set_xticklabels([c.replace("_", "\n") for c in numeric_cols], fontsize=8)
    ax.set_yticklabels([c.replace("_", " ") for c in numeric_cols], fontsize=8)
    for i in range(len(numeric_cols)):
        for j in range(len(numeric_cols)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(corr.iloc[i, j]) > 0.5 else "black")
    ax.set_title("Correlation matrix — policy-level features")
    fig.tight_layout()
    _save(fig, "10_correlation_heatmap.png")


def plot_premium_vs_sum_assured(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    scatter = ax.scatter(
        df["sum_assured"] / 1_000,
        df["premium"],
        c=df["age"],
        cmap="viridis",
        alpha=0.5,
        s=16,
        linewidths=0,
    )
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Issue age", fontsize=9)
    ax.set_xlabel("Sum assured (£k)")
    ax.set_ylabel("Annual premium (£)")
    ax.set_title("Premium vs sum assured, coloured by age")
    _save(fig, "11_premium_vs_sum_assured.png")


def plot_reserve_surface_age_vs_term(traj_df: pd.DataFrame) -> None:
    """Pivot the midpoint reserve (t = term/2) onto an age × term grid."""
    midpoint_df = (
        traj_df
        .assign(rel_time=lambda d: d["time"] / d["term"])
        .pipe(lambda d: d[(d["rel_time"] >= 0.45) & (d["rel_time"] <= 0.55)])
        .groupby(["age", "term"])["reserve"]
        .mean()
        .reset_index()
    )
    pivot = midpoint_df.pivot_table(index="age", columns="term", values="reserve")
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(
        pivot.values,
        aspect="auto",
        origin="lower",
        cmap="YlOrRd",
        extent=[pivot.columns.min(), pivot.columns.max(),
                pivot.index.min(),  pivot.index.max()],
    )
    fig.colorbar(im, ax=ax, label="Mean mid-term reserve (£)")
    ax.set_xlabel("Policy term (years)")
    ax.set_ylabel("Issue age")
    ax.set_title("Mean mid-term reserve — age × term heat map")
    _save(fig, "12_reserve_surface_age_vs_term.png")




def plot_reserve_distribution(traj_df: pd.DataFrame) -> None:
    """Distribution of reserve values across all policies and time steps."""
    peak_df = traj_df.groupby("policy_id")["reserve"].max().reset_index()
    peak_df.columns = ["policy_id", "peak_reserve"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # All reserve values
    axes[0].hist(traj_df["reserve"], bins=60, color=PALETTE, edgecolor="white", linewidth=0.4)
    axes[0].axvline(traj_df["reserve"].mean(), color="#E03131", linewidth=1.4,
                    linestyle="--", label=f"Mean £{traj_df['reserve'].mean():,.0f}")
    axes[0].set_xlabel("Reserve (£)")
    axes[0].set_ylabel("Count (all time steps)")
    axes[0].set_title("Reserve distribution — all policy-time points")
    axes[0].legend(fontsize=9)

    # Peak (maximum) reserve per policy
    axes[1].hist(peak_df["peak_reserve"], bins=50, color="#7950F2", edgecolor="white", linewidth=0.4)
    axes[1].axvline(peak_df["peak_reserve"].mean(), color="#E03131", linewidth=1.4,
                    linestyle="--", label=f"Mean peak £{peak_df['peak_reserve'].mean():,.0f}")
    axes[1].set_xlabel("Peak reserve per policy (£)")
    axes[1].set_ylabel("Policy count")
    axes[1].set_title("Peak reserve distribution — one value per policy")
    axes[1].legend(fontsize=9)

    fig.suptitle("Reserve distributions", fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save(fig, "13_reserve_distribution.png")


def plot_reserve_vs_age(traj_df: pd.DataFrame) -> None:
    """Peak reserve vs issue age — shows how age drives reserve magnitude."""
    peak_df = (
        traj_df.groupby(["policy_id", "age", "term", "sum_assured"])["reserve"]
        .max().reset_index().rename(columns={"reserve": "peak_reserve"})
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Scatter: peak reserve vs age coloured by term
    sc = axes[0].scatter(
        peak_df["age"], peak_df["peak_reserve"],
        c=peak_df["term"], cmap="plasma", alpha=0.55, s=18, linewidths=0,
    )
    fig.colorbar(sc, ax=axes[0], label="Policy term (years)")
    axes[0].set_xlabel("Issue age")
    axes[0].set_ylabel("Peak reserve (£)")
    axes[0].set_title("Peak reserve vs issue age
(coloured by term)")

    # Box plot: peak reserve by age band
    peak_df["age_band"] = peak_df["age"].apply(_age_band)
    bands = sorted(peak_df["age_band"].unique())
    data_by_band = [peak_df.loc[peak_df["age_band"] == b, "peak_reserve"].values for b in bands]
    bp = axes[1].boxplot(data_by_band, labels=bands, patch_artist=True, widths=0.5,
                          medianprops={"color": "#E03131", "linewidth": 2})
    cmap_box = cm.get_cmap("coolwarm", len(bands))
    for patch, color in zip(bp["boxes"], [cmap_box(i / len(bands)) for i in range(len(bands))]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[1].set_xlabel("Issue age band")
    axes[1].set_ylabel("Peak reserve (£)")
    axes[1].set_title("Peak reserve distribution by age band")
    axes[1].tick_params(axis="x", rotation=30)

    fig.suptitle("Reserve vs Age", fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save(fig, "14_reserve_vs_age.png")


def plot_reserve_vs_mortality(traj_df: pd.DataFrame) -> None:
    """Reserve vs mortality intensity — higher mortality should raise reserves."""
    # Use mortality_at_t and reserve at each time step
    sample = traj_df.sample(min(3000, len(traj_df)), random_state=42)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Scatter: reserve vs mortality at that time point, coloured by time
    sc = axes[0].scatter(
        sample["mortality_at_t"] * 1000,
        sample["reserve"],
        c=sample["time"], cmap="viridis", alpha=0.35, s=10, linewidths=0,
    )
    fig.colorbar(sc, ax=axes[0], label="Policy time (years)")
    axes[0].set_xlabel("Mortality intensity at t (‰/yr)")
    axes[0].set_ylabel("Reserve (£)")
    axes[0].set_title("Reserve vs mortality intensity
(coloured by policy time)")

    # Mean reserve by mortality quintile
    sample = sample.copy()
    sample["mortality_quintile"] = pd.qcut(
        sample["mortality_at_t"], q=5,
        labels=["Q1
(lowest)", "Q2", "Q3", "Q4", "Q5
(highest)"]
    )
    grouped = sample.groupby("mortality_quintile", observed=True)["reserve"].mean()
    bars = axes[1].bar(grouped.index, grouped.values, color=PALETTE, edgecolor="white")
    axes[1].set_xlabel("Mortality quintile")
    axes[1].set_ylabel("Mean reserve (£)")
    axes[1].set_title("Mean reserve by mortality quintile
(expected: increasing)")
    for bar, val in zip(bars, grouped.values):
        axes[1].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + grouped.max() * 0.01,
                     f"£{val:,.0f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Reserve vs Mortality", fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save(fig, "15_reserve_vs_mortality.png")


def plot_reserve_vs_sum_assured(traj_df: pd.DataFrame) -> None:
    """Reserve vs sum assured — core proportionality relationship."""
    peak_df = (
        traj_df.groupby(["policy_id", "sum_assured", "age", "interest_rate"])["reserve"]
        .max().reset_index().rename(columns={"reserve": "peak_reserve"})
    )
    peak_df["v"] = peak_df["peak_reserve"] / peak_df["sum_assured"]  # normalised ratio

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    # Raw: peak reserve vs sum assured
    sc = axes[0].scatter(
        peak_df["sum_assured"] / 1_000, peak_df["peak_reserve"],
        c=peak_df["age"], cmap="plasma", alpha=0.5, s=14, linewidths=0,
    )
    fig.colorbar(sc, ax=axes[0], label="Issue age")
    # Fit line
    coeffs = np.polyfit(peak_df["sum_assured"], peak_df["peak_reserve"], 1)
    x_line = np.linspace(peak_df["sum_assured"].min(), peak_df["sum_assured"].max(), 100)
    axes[0].plot(x_line / 1_000, np.polyval(coeffs, x_line),
                 color="#E03131", linewidth=1.5, linestyle="--", label=f"Slope {coeffs[0]:.3f}")
    axes[0].set_xlabel("Sum assured (£k)")
    axes[0].set_ylabel("Peak reserve (£)")
    axes[0].set_title("Peak reserve vs sum assured
(coloured by age)")
    axes[0].legend(fontsize=8)

    # Normalised: v = V/S — should be roughly constant across S
    axes[1].scatter(
        peak_df["sum_assured"] / 1_000, peak_df["v"],
        c=peak_df["age"], cmap="plasma", alpha=0.5, s=14, linewidths=0,
    )
    axes[1].axhline(peak_df["v"].mean(), color="#E03131", linewidth=1.4,
                    linestyle="--", label=f"Mean v = {peak_df['v'].mean():.4f}")
    axes[1].set_xlabel("Sum assured (£k)")
    axes[1].set_ylabel("v = Peak reserve / Sum assured")
    axes[1].set_title("Normalised reserve v = V/S vs sum assured
(should be roughly flat — confirms V∝S)")
    axes[1].legend(fontsize=8)

    # Mean reserve by SA decile
    peak_df["sa_decile"] = pd.qcut(peak_df["sum_assured"], q=5,
                                    labels=["Q1
(£50k)", "Q2", "Q3", "Q4", "Q5
(£1M)"])
    grouped = peak_df.groupby("sa_decile", observed=True)["peak_reserve"].mean()
    bars = axes[2].bar(grouped.index, grouped.values / 1_000, color="#2F9E44", edgecolor="white")
    axes[2].set_xlabel("Sum assured quintile")
    axes[2].set_ylabel("Mean peak reserve (£k)")
    axes[2].set_title("Mean peak reserve by sum assured quintile
(expected: strongly increasing)")
    for bar, val in zip(bars, grouped.values):
        axes[2].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + grouped.max() / 1_000 * 0.01,
                     f"£{val/1000:.1f}k", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Reserve vs Sum Assured", fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save(fig, "16_reserve_vs_sum_assured.png")


def plot_reserve_vs_interest_rate(traj_df: pd.DataFrame) -> None:
    """Reserve vs interest rate — higher rate should reduce reserve (PV effect)."""
    peak_df = (
        traj_df.groupby(["policy_id", "interest_rate", "age", "term"])["reserve"]
        .max().reset_index().rename(columns={"reserve": "peak_reserve"})
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    sc = axes[0].scatter(
        peak_df["interest_rate"] * 100, peak_df["peak_reserve"],
        c=peak_df["term"], cmap="viridis", alpha=0.5, s=14, linewidths=0,
    )
    fig.colorbar(sc, ax=axes[0], label="Policy term (years)")
    coeffs = np.polyfit(peak_df["interest_rate"], peak_df["peak_reserve"], 1)
    x_line = np.linspace(peak_df["interest_rate"].min(), peak_df["interest_rate"].max(), 100)
    axes[0].plot(x_line * 100, np.polyval(coeffs, x_line),
                 color="#E03131", linewidth=1.5, linestyle="--",
                 label=f"Slope {coeffs[0]:.0f} £/unit r")
    axes[0].set_xlabel("Interest rate (%)")
    axes[0].set_ylabel("Peak reserve (£)")
    axes[0].set_title("Peak reserve vs interest rate
(expected: negative slope — higher r lowers PV)")
    axes[0].legend(fontsize=8)

    peak_df["rate_band"] = pd.cut(
        peak_df["interest_rate"] * 100, bins=5,
        labels=["1–2%", "2–3%", "3–5%", "5–6%", "6–8%"]
    )
    grouped = peak_df.groupby("rate_band", observed=True)["peak_reserve"].mean()
    colors = ["#2F9E44" if i == 0 else "#E03131" if i == len(grouped)-1 else PALETTE
              for i in range(len(grouped))]
    bars = axes[1].bar(grouped.index, grouped.values, color=colors, edgecolor="white")
    axes[1].set_xlabel("Interest rate band")
    axes[1].set_ylabel("Mean peak reserve (£)")
    axes[1].set_title("Mean peak reserve by interest rate band
(expected: decreasing)")
    for bar, val in zip(bars, grouped.values):
        axes[1].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + grouped.max() * 0.01,
                     f"£{val:,.0f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Reserve vs Interest Rate", fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save(fig, "17_reserve_vs_interest_rate.png")


def plot_reserve_vs_premium(traj_df: pd.DataFrame) -> None:
    """Reserve vs premium — higher premium means more liability."""
    peak_df = (
        traj_df.groupby(["policy_id", "premium", "age", "term"])["reserve"]
        .max().reset_index().rename(columns={"reserve": "peak_reserve"})
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    sc = axes[0].scatter(
        peak_df["premium"], peak_df["peak_reserve"],
        c=peak_df["age"], cmap="plasma", alpha=0.5, s=14, linewidths=0,
    )
    fig.colorbar(sc, ax=axes[0], label="Issue age")
    coeffs = np.polyfit(peak_df["premium"], peak_df["peak_reserve"], 1)
    x_line = np.linspace(peak_df["premium"].min(), peak_df["premium"].max(), 100)
    axes[0].plot(x_line, np.polyval(coeffs, x_line),
                 color="#E03131", linewidth=1.5, linestyle="--",
                 label=f"Slope {coeffs[0]:.1f} (≈ annuity factor)")
    axes[0].set_xlabel("Annual premium (£)")
    axes[0].set_ylabel("Peak reserve (£)")
    axes[0].set_title("Peak reserve vs premium
(slope ≈ annuity factor, expected ~8-15)")
    axes[0].legend(fontsize=8)

    peak_df["prem_quintile"] = pd.qcut(peak_df["premium"], q=5,
                                        labels=["Q1
(low P)", "Q2", "Q3", "Q4", "Q5
(high P)"])
    grouped = peak_df.groupby("prem_quintile", observed=True)["peak_reserve"].mean()
    bars = axes[1].bar(grouped.index, grouped.values, color=PALETTE, edgecolor="white")
    axes[1].set_xlabel("Premium quintile")
    axes[1].set_ylabel("Mean peak reserve (£)")
    axes[1].set_title("Mean peak reserve by premium quintile
(expected: increasing)")
    for bar, val in zip(bars, grouped.values):
        axes[1].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + grouped.max() * 0.01,
                     f"£{val:,.0f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Reserve vs Premium", fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save(fig, "18_reserve_vs_premium.png")

# ── summary statistics ─────────────────────────────────────────────────────────
def write_summary(policies_df: pd.DataFrame, traj_df: pd.DataFrame) -> None:
    lines = [
        "=" * 60,
        "  INSURANCE RESERVE PLATFORM — DATA GENERATION SUMMARY",
        "=" * 60,
        "",
        f"Policies generated : {len(policies_df):,}",
        f"Trajectory records : {len(traj_df):,}",
        "",
        "── Policy-level numeric summary ──",
        policies_df[[
            "age", "term", "interest_rate", "sum_assured", "premium",
            "mortality_at_inception", "risk_adjustment_factor",
        ]].describe().to_string(),
        "",
        "── Categorical counts ──",
        "",
        "Smoker status:",
        policies_df["smoker_status"].value_counts().to_string(),
        "",
        "Health tier:",
        policies_df["health_tier"].value_counts().to_string(),
        "",
        "Gender:",
        policies_df["gender"].value_counts().to_string(),
        "",
        "Occupation risk:",
        policies_df["occupation_risk"].value_counts().to_string(),
        "",
        "── Reserve trajectory summary ──",
        traj_df[["reserve", "mortality_at_t"]].describe().to_string(),
        "",
        "=" * 60,
    ]
    summary = "\n".join(lines)
    print(summary)
    path = EDA_DIR / "eda_summary.txt"
    path.write_text(summary)
    print(f"\n  summary → {path}")


# ── main ───────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate data and run EDA.")
    p.add_argument("--n", type=int, default=None,
                   help="Number of policies to generate (default: from config)")
    p.add_argument("--time-steps", type=int, default=None,
                   help="Reserve trajectory time steps (default: from config)")
    p.add_argument("--fmt", choices=["csv", "parquet"], default="csv",
                   help="Storage format for data files (default: csv)")
    p.add_argument("--no-plots", action="store_true",
                   help="Skip plot generation (only save data)")
    p.add_argument("--data-dir", type=str, default=str(EDA_DIR),
                   help="Output directory (default: artifacts/eda)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── load config ──
    config = ConfigLoader.load(CONFIG_PATH)
    ensure_directories(config)

    n_policies = args.n or config.data.train_size
    time_steps = args.time_steps or config.data.time_steps

    print(f"\n[1/4] Generating {n_policies} synthetic policies …")
    mortality_source = None
    candidate = Path(config.paths.data_dir) / "sample_mortality.csv"
    if candidate.exists():
        mortality_source = CSVMortalityLoader(candidate)
        print(f"      mortality source: {candidate}")

    simulator = PolicySimulator(
        age_range=(config.data.age_min, config.data.age_max),
        term_range=(config.data.term_min, config.data.term_max),
        interest_rate_range=(config.data.interest_rate_min, config.data.interest_rate_max),
        sum_assured_range=(config.data.sum_assured_min, config.data.sum_assured_max),
        mortality_source=mortality_source,
        mortality_scale=config.data.mortality_scale,
        mortality_shape=config.data.mortality_shape,
        mortality_reference_age=config.data.mortality_reference_age,
        premium_loading=config.data.premium_loading,
        max_expiry_age=config.data.max_expiry_age,
        sum_assured_rounding=config.data.sum_assured_rounding,
        sum_assured_age_decay=config.data.sum_assured_age_decay,
        seed=config.data.random_seed,
    )
    solver = ThieleSolver(
        method=config.solver.method,
        integration_step=config.solver.integration_step,
        rtol=config.solver.rtol,
        atol=config.solver.atol,
    )
    policies = simulator.generate_random_policies(n_policies)

    print(f"[2/4] Solving reserve trajectories ({time_steps} steps/policy) …")
    policies_df = policies_to_dataframe(policies)
    traj_df = reserve_trajectories_to_dataframe(policies, solver, time_steps)

    print(f"[3/4] Saving data ({args.fmt}) to {out_dir} …")
    paths = save_datasets(policies_df, traj_df, out_dir, fmt=args.fmt)
    print(f"      policies     → {paths['policies']}")
    print(f"      trajectories → {paths['trajectories']}")

    write_summary(policies_df, traj_df)

    if not args.no_plots:
        print(f"\n[4/4] Generating EDA plots …")
        # reassign EDA_DIR to the user's chosen output dir
        global EDA_DIR
        EDA_DIR = out_dir

        plot_age_distribution(policies_df)
        plot_term_distribution(policies_df)
        plot_sum_assured_distribution(policies_df)
        plot_interest_rate_distribution(policies_df)
        plot_premium_distribution(policies_df)
        plot_risk_profile_breakdown(policies_df)
        plot_mortality_at_inception(policies_df)
        plot_reserve_trajectories_sample(traj_df)
        plot_reserve_by_age_band(traj_df)
        plot_correlation_heatmap(policies_df)
        plot_premium_vs_sum_assured(policies_df)
        plot_reserve_surface_age_vs_term(traj_df)
        plot_reserve_distribution(traj_df)
        plot_reserve_vs_age(traj_df)
        plot_reserve_vs_mortality(traj_df)
        plot_reserve_vs_sum_assured(traj_df)
        plot_reserve_vs_interest_rate(traj_df)
        plot_reserve_vs_premium(traj_df)
        print(f"\nAll outputs written to: {out_dir}/")
    else:
        print("\n[4/4] Plots skipped (--no-plots).")


if __name__ == "__main__":
    main()
