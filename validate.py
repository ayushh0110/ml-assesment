"""Out-of-time validation of the freight-rate model.

The final task is to predict Nov-Dec 2025 from a model trained on Jan-Oct 2025, so the
development data is split the same way: rolling-origin folds that train on everything
before a cutoff and test on the following two months. A second check holds whole cities
out of training, because 12% of validation loads touch a city never seen in training.

A few extra checks on the same folds back up specific claims in the report: training with
the suspicious rates kept, and a diagnostic that swaps in the true daily level.

Writes reports/validation_metrics.csv, reports/city_holdout_metrics.csv and
reports/extra_checks.csv.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from freight_model.data import TARGET
from freight_model.features import LOAD_FEATURES, PER_LOAD_TEMPORAL_FEATURES, build_features
from freight_model.model import FreightRateModel, GBMOnlyModel, metrics
from freight_model.pipeline import prepare

REPORTS = Path(__file__).resolve().parent / "reports"
CUTOFFS = ["2025-05-01", "2025-06-01", "2025-07-01", "2025-08-01", "2025-09-01"]
HORIZON = pd.DateOffset(months=2)

CANDIDATES = {
    "GBM: load features only": lambda: GBMOnlyModel(LOAD_FEATURES),
    "GBM: load + market_index as tree feature": lambda: GBMOnlyModel(LOAD_FEATURES + ["market_daily"]),
    "GBM: load + market + date trend as tree features": lambda: GBMOnlyModel(LOAD_FEATURES + ["market_daily", "trend_years"]),
    "GBM + linear term, per-load inputs (no same-day averages)":
        lambda: FreightRateModel(temporal_features=PER_LOAD_TEMPORAL_FEATURES),
    "GBM + linear market/trend offset (chosen)": lambda: FreightRateModel(),
}


def windows(dates: pd.Series):
    for cutoff in map(pd.Timestamp, CUTOFFS):
        label = f"{cutoff:%b}-{(cutoff + pd.DateOffset(months=1)):%b}"
        yield label, (dates < cutoff).to_numpy(), ((dates >= cutoff) & (dates < cutoff + HORIZON)).to_numpy()


def rolling_origin(prep) -> pd.DataFrame:
    rows = []
    for window, train_mask, test_mask in windows(prep.train["date"]):
        y_test = prep.train.loc[test_mask, TARGET]
        for name, make in CANDIDATES.items():
            model = make().fit(prep.X_train[train_mask], prep.train.loc[train_mask, TARGET])
            pred = model.predict(prep.X_train[test_mask])
            rows.append({"model": name, "test_window": window, **metrics(y_test, pred)})
            print(f"{rows[-1]['test_window']:8s} {name:50s} MAPE {rows[-1]['MAPE_%']:.3f}  MAE {rows[-1]['MAE']:.2f}")
    return pd.DataFrame(rows)


def city_holdout(prep, n_cities: int = 8, repeats: int = 3, cutoff: str = "2025-09-01") -> pd.DataFrame:
    rng = np.random.default_rng(42)
    cities = np.array(sorted(set(prep.train["pickup"]) | set(prep.train["delivery"])))
    rows = []
    for rep in range(repeats):
        held = set(rng.choice(cities, n_cities, replace=False))
        touches = (prep.train["pickup"].isin(held) | prep.train["delivery"].isin(held)).to_numpy()
        before = (prep.train["date"] < cutoff).to_numpy()
        model = FreightRateModel().fit(prep.X_train[before & ~touches], prep.train.loc[before & ~touches, TARGET])
        for label, mask in [("seen cities", ~before & ~touches), ("held-out cities", ~before & touches)]:
            pred = model.predict(prep.X_train[mask])
            rows.append({"repeat": rep, "segment": label, **metrics(prep.train.loc[mask, TARGET], pred)})
    return pd.DataFrame(rows)


def extra_checks(prep) -> pd.DataFrame:
    """Same folds, always scored on clean rows only."""
    with_outliers = pd.concat([prep.train, prep.outliers], ignore_index=True)
    X_with_outliers = build_features(with_outliers)
    rows = []
    for window, train_mask, test_mask in windows(prep.train["date"]):
        y_test = prep.train.loc[test_mask, TARGET]

        # 1) suspicious rates left in the training data
        keep = (with_outliers["date"] < prep.train.loc[test_mask, "date"].min()).to_numpy()
        model = FreightRateModel().fit(X_with_outliers[keep], with_outliers.loc[keep, TARGET])
        rows.append({"check": "trained with suspicious rates kept", "test_window": window,
                     **metrics(y_test, model.predict(prep.X_train[test_mask]))})

        # 2) diagnostic only: shift each test day by its true mean log error. Uses the test
        #    labels, so it is not a usable model, just a measure of how much error is day-level.
        model = FreightRateModel().fit(prep.X_train[train_mask], prep.train.loc[train_mask, TARGET])
        log_pred = model.predict_log(prep.X_train[test_mask])
        day_error = pd.Series(np.log(y_test.to_numpy()) - log_pred).groupby(
            prep.train.loc[test_mask, "date"].to_numpy()).transform("mean").to_numpy()
        rows.append({"check": "diagnostic: true daily level (uses test labels)", "test_window": window,
                     **metrics(y_test, np.exp(log_pred + day_error))})
        print(f"{window:8s} extra checks done")
    return pd.DataFrame(rows)


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    prep = prepare()
    print(f"Dropped {len(prep.outliers)} mislabelled rows; training rows: {len(prep.train)}")

    folds = rolling_origin(prep)
    folds.to_csv(REPORTS / "validation_metrics.csv", index=False)
    summary = folds.groupby("model", sort=False)[["MAPE_%", "MAE", "RMSE", "bias_%"]].mean().round(3)
    summary.to_csv(REPORTS / "validation_summary.csv")
    print("\nMean over rolling-origin folds:\n", summary.to_string())

    cities = city_holdout(prep)
    cities.to_csv(REPORTS / "city_holdout_metrics.csv", index=False)
    print("\nCity holdout (Sep-Oct test):\n",
          cities.groupby("segment")[["MAPE_%", "MAE"]].mean().round(3).to_string())

    extra = extra_checks(prep)
    extra.to_csv(REPORTS / "extra_checks.csv", index=False)
    print("\nExtra checks (mean over folds):\n",
          extra.groupby("check")[["MAPE_%", "MAE"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
