"""GBM on load features + a linear market/trend term, fit by backfitting.

    log(rate) = gbm(load features) + b0 + b1*market_daily + b2*quote_daily + b3*trend_years

The GBM handles the lane/distance/equipment/weight interactions. The market level and the
slow upward drift go in the linear term: a tree turns them into month-shaped steps and has
no way to extrapolate the trend into Nov-Dec.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression

from .features import LOAD_FEATURES, TEMPORAL_FEATURES

GBM_PARAMS = dict(
    max_iter=1500,
    learning_rate=0.05,
    max_leaf_nodes=31,
    min_samples_leaf=20,
    early_stopping=False,
    random_state=0,
)


def _gbm(**overrides) -> HistGradientBoostingRegressor:
    params = {**GBM_PARAMS, **overrides}
    return HistGradientBoostingRegressor(
        categorical_features=[LOAD_FEATURES.index("equipment_code")], **params
    )


def robust_log_baseline(X: pd.DataFrame, log_y: pd.Series) -> np.ndarray:
    """L1-loss GBM used only to spot mislabelled targets (not affected by them)."""
    model = _gbm(max_iter=400, learning_rate=0.08, max_leaf_nodes=31, loss="absolute_error")
    model.fit(X[LOAD_FEATURES], log_y)
    return model.predict(X[LOAD_FEATURES])


class FreightRateModel:
    def __init__(self, backfit_iterations: int = 3, **gbm_overrides):
        self.backfit_iterations = backfit_iterations
        self.gbm_overrides = gbm_overrides

    def fit(self, X: pd.DataFrame, rate: pd.Series) -> "FreightRateModel":
        log_y = np.log(rate.to_numpy())
        offset = np.zeros(len(X))
        for _ in range(self.backfit_iterations):
            self.gbm_ = _gbm(**self.gbm_overrides).fit(X[LOAD_FEATURES], log_y - offset)
            resid = log_y - self.gbm_.predict(X[LOAD_FEATURES])
            self.linear_ = LinearRegression().fit(X[TEMPORAL_FEATURES], resid)
            offset = self.linear_.predict(X[TEMPORAL_FEATURES])
        return self

    def predict_log(self, X: pd.DataFrame) -> np.ndarray:
        return self.gbm_.predict(X[LOAD_FEATURES]) + self.linear_.predict(X[TEMPORAL_FEATURES])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.exp(self.predict_log(X))


class GBMOnlyModel(FreightRateModel):
    """Benchmark: one GBM on load features plus market/time features as ordinary splits."""

    def __init__(self, features: list[str], **gbm_overrides):
        super().__init__(**gbm_overrides)
        self.features = features

    def fit(self, X, rate):
        cat = [self.features.index("equipment_code")]
        self.gbm_ = HistGradientBoostingRegressor(categorical_features=cat, **{**GBM_PARAMS, **self.gbm_overrides})
        self.gbm_.fit(X[self.features], np.log(rate.to_numpy()))
        return self

    def predict_log(self, X):
        return self.gbm_.predict(X[self.features])


def metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    err = y_pred - y_true
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "MAPE_%": float(np.mean(np.abs(err) / y_true) * 100),
        "bias_%": float(np.mean(np.log(y_pred / y_true)) * 100),
        "n": int(len(y_true)),
    }
