"""Exploratory analysis: prints data-quality findings and saves figures to reports/figures/."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from freight_model.data import TARGET, load_all
from freight_model.features import haversine_miles

FIGURES = Path(__file__).resolve().parent / "reports" / "figures"
COLOR = "#064A56"


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    train, val, _ = load_all()

    print("== Shapes / date ranges")
    for name, f in [("train_test", train), ("validation", val)]:
        print(f"{name:11s} {f.shape}  {f['date'].min():%Y-%m-%d} -> {f['date'].max():%Y-%m-%d}")

    print("\n== Missing values")
    print(pd.DataFrame({"train_test": train.isna().sum(), "validation": val.isna().sum()}).query("train_test + validation > 0"))

    print("\n== Weight anomalies")
    for name, f in [("train_test", train), ("validation", val)]:
        print(f"{name:11s} negative={int((f['weight'] < 0).sum())}  at 47,500 cap={int((f['weight'].abs() == 47500).sum())}"
              f"  at 5,000 floor={int((f['weight'].abs() == 5000).sum())}")

    seen = set(train["pickup"]) | set(train["delivery"])
    new_cities = sorted((set(val["pickup"]) | set(val["delivery"])) - seen)
    touches = ~val["pickup"].isin(seen) | ~val["delivery"].isin(seen)
    print(f"\n== Cities only in validation ({len(new_cities)}): {', '.join(new_cities)}")
    print(f"validation loads touching one: {touches.sum()} ({touches.mean():.1%})")

    rpm = train[TARGET] / train["distance"]
    rpm_ratio = rpm / rpm.groupby([pd.cut(train["distance"], [0, 250, 500, 1000, 1500, 2000, 4000]),
                                   train["equipment"]], observed=True).transform("median")
    print("\n== Rate per mile relative to distance/equipment median")
    print(f"< 0.5x: {(rpm_ratio < 0.5).sum()}   > 2x: {(rpm_ratio > 2).sum()}   (rest within 0.8-1.25x: "
          f"{rpm_ratio.between(0.8, 1.25).mean():.1%})")

    circuity = train["distance"] / haversine_miles(train["pickup_lat"], train["pickup_lon"],
                                                   train["delivery_lat"], train["delivery_lon"])
    print(f"\n== Road/straight-line ratio: median {circuity.median():.3f}, >2x: {(circuity > 2).sum()} rows "
          f"(Shreveport-New Orleans coordinates only 7 mi apart; rate follows recorded distance)")

    monthly = pd.DataFrame({
        "market_index": pd.concat([train, val]).groupby(pd.concat([train, val])["date"].dt.to_period("M"))["market_index"].mean(),
    })
    print("\n== Monthly mean market_index (train Jan-Oct, validation Nov-Dec)")
    print(monthly.round(3).T.to_string())

    # Figure 1: rate vs distance by equipment, outliers visible.
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    for eq, c in zip(["Dry Van", "Reefer", "Flatbed"], ["#064A56", "#E07A2F", "#7A9E3B"]):
        s = train[train["equipment"] == eq]
        ax.scatter(s["distance"], s[TARGET], s=2, alpha=0.25, color=c, label=eq)
    ax.set(xlabel="Distance (mi)", ylabel="Posted rate ($)", title="Posted rate vs distance (outlier bands visible above/below)")
    ax.legend(markerscale=6)
    fig.tight_layout(); fig.savefig(FIGURES / "rate_vs_distance.png"); plt.close(fig)

    # Figure 2: rpm ratio histogram (log scale) showing the two mislabel clusters.
    fig, ax = plt.subplots(figsize=(8, 3.5), dpi=150)
    ax.hist(np.log2(rpm_ratio), bins=200, color=COLOR)
    ax.set_yscale("log")
    ax.set(xlabel="log2(rate per mile / median for distance band & equipment)", ylabel="loads (log)",
           title="Two clusters of mislabelled rates (~0.25x and ~4x)")
    fig.tight_layout(); fig.savefig(FIGURES / "rate_ratio_hist.png"); plt.close(fig)

    # Figure 3: daily market index vs daily rate level.
    level = np.log(rpm_ratio[rpm_ratio.between(0.5, 2)]).groupby(train["date"]).mean()
    mkt = pd.concat([train, val]).groupby("date")["market_index"].mean()
    fig, ax1 = plt.subplots(figsize=(9, 3.8), dpi=150)
    ax1.plot(mkt.index, mkt.values, color="#E07A2F", lw=1, label="daily mean market_index")
    ax1.set_ylabel("market_index", color="#E07A2F")
    ax2 = ax1.twinx()
    ax2.plot(level.index, level.values, color=COLOR, lw=1, label="daily rate level (log rpm vs median)")
    ax2.set_ylabel("rate level", color=COLOR)
    ax1.axvline(pd.Timestamp("2025-11-01"), color="grey", ls="--", lw=1)
    ax1.text(pd.Timestamp("2025-11-05"), mkt.max(), "validation\n(no labels)", fontsize=8, va="top")
    ax1.set_title("Daily rate level tracks the market index (plus a slow upward drift)")
    fig.tight_layout(); fig.savefig(FIGURES / "market_vs_rate.png"); plt.close(fig)
    print(f"\nSaved figures to {FIGURES}")


if __name__ == "__main__":
    main()
