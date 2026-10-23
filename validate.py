"""Out-of-time validation of the freight-rate model.

The final task is to predict Nov-Dec 2025 from a model trained on Jan-Oct 2025, so the
development data is split the same way: rolling-origin folds that train on everything
before a cutoff and test on the following two months. A second check holds whole cities
out of training, because 12% of validation loads touch a city never seen in training.

Writes reports/validation_metrics.csv and reports/city_holdout_metrics.csv.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from freight_model.data import TARGET
from freight_model.features import LOAD_FEATURES
from freight_model.model import FreightRateModel, GBMOnlyModel, metrics
from freight_model.pipeline import prepare

REPORTS = Path(__file__).resolve().parent / "reports"
CUTOFFS = ["2025-05-01", "2025-06-01", "2025-07-01", "2025-08-01", "2025-09-01"]
HORIZON = pd.DateOffset(months=2)

CANDIDATES = {
    "GBM: load features only": lambda: GBMOnlyModel(LOAD_FEATURES),
    "GBM: load + market_index as tree feature": lambda: GBMOnlyModel(LOAD_FEATURES + ["market_daily"]),
    "GBM: load + market + date trend as tree features": lambda: GBMOnlyModel(LOAD_FEATURES + ["market_daily", "trend_years"]),
    "GBM + linear market/trend offset (chosen)": lambda: FreightRateModel(),
}


def rolling_origin(prep) -> pd.DataFrame:
    rows = []
    dates = prep.train["date"]
    for cutoff in map(pd.Timestamp, CUTOFFS):
        train_mask = (dates < cutoff).to_numpy()
        test_mask = ((dates >= cutoff) & (dates < cutoff + HORIZON)).to_numpy()
        y_test = prep.train.loc[test_mask, TARGET]
        for name, make in CANDIDATES.items():
            model = make().fit(prep.X_train[train_mask], prep.train.loc[train_mask, TARGET])
            pred = model.predict(prep.X_train[test_mask])
            rows.append({"model": name, "test_window": f"{cutoff:%b}-{(cutoff + pd.DateOffset(months=1)):%b}",
                         **metrics(y_test, pred)})
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


if __name__ == "__main__":
    main()
