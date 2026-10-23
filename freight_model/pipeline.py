"""Shared preparation used by validate.py and train_predict.py."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import TARGET, city_coordinates, clean, daily_market_tables, flag_label_outliers, load_all
from .features import build_features
from .model import robust_log_baseline


@dataclass
class Prepared:
    train: pd.DataFrame          # cleaned development rows, label outliers removed
    X_train: pd.DataFrame
    outliers: pd.DataFrame       # rows dropped as mislabelled
    validation: pd.DataFrame
    X_validation: pd.DataFrame
    december: pd.DataFrame
    X_december: pd.DataFrame


def prepare() -> Prepared:
    raw_train, raw_val, raw_dec = load_all()
    daily = daily_market_tables(raw_train, raw_val)
    coords = city_coordinates(raw_train, raw_val)

    train = clean(raw_train, daily, coords)
    validation = clean(raw_val, daily, coords)
    # The chart inputs carry no market columns: use each December day's market level
    # (and average quote_signal) observed across the validation loads for that date.
    december = clean(raw_dec.drop(columns=["predicted_rate"]), daily, coords)

    X_all = build_features(train)
    is_outlier = flag_label_outliers(train, robust_log_baseline(X_all, np.log(train[TARGET])))
    keep = ~is_outlier.to_numpy()
    train_kept = train[keep].reset_index(drop=True)

    return Prepared(
        train=train_kept,
        X_train=build_features(train_kept),
        outliers=train[~keep],
        validation=validation,
        X_validation=build_features(validation),
        december=december,
        X_december=build_features(december),
    )
