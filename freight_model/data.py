"""Loading and cleaning of the freight load data."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TARGET = "posted_rate"


def load_csv(name: str) -> pd.DataFrame:
    frame = pd.read_csv(DATA_DIR / name)
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def load_all() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (train_test, validation, december_chart_inputs)."""
    return (
        load_csv("train_test.csv"),
        load_csv("validation.csv"),
        load_csv("december_chart_inputs.csv"),
    )


def daily_market_tables(*frames: pd.DataFrame) -> pd.DataFrame:
    """Per-date mean market_index / quote_signal over every load we have features for.

    The row-level market_index is the daily market level plus noise (and has missing
    values); the daily mean is the stable signal. Only features are used here, never
    the target, so pooling train and validation rows is leakage-free.
    """
    pooled = pd.concat([f[["date", "market_index", "quote_signal"]] for f in frames])
    return pooled.groupby("date").agg(
        market_daily=("market_index", "mean"),
        quote_daily=("quote_signal", "mean"),
    )


def city_coordinates(*frames: pd.DataFrame) -> pd.DataFrame:
    """City -> (lat, lon). Every city has exactly one coordinate pair across all files."""
    parts = []
    for f in frames:
        for end in ("pickup", "delivery"):
            parts.append(f[[end, f"{end}_lat", f"{end}_lon"]].set_axis(["city", "lat", "lon"], axis=1))
    coords = pd.concat(parts).drop_duplicates()
    assert not coords["city"].duplicated().any(), "a city has more than one coordinate pair"
    return coords.set_index("city")


def clean(frame: pd.DataFrame, daily: pd.DataFrame, coords: pd.DataFrame) -> pd.DataFrame:
    """Fix feature-level data-quality issues. Works on train, validation and chart inputs."""
    out = frame.copy()

    # The December chart inputs carry city names only: look their coordinates up.
    for end in ("pickup", "delivery"):
        if f"{end}_lat" not in out:
            out[f"{end}_lat"] = out[end].map(coords["lat"])
            out[f"{end}_lon"] = out[end].map(coords["lon"])

    # Negative weights are sign errors: |w| has the same rate relationship as positive w.
    out["weight_was_negative"] = (out["weight"] < 0).astype(int)
    out["weight"] = out["weight"].abs()
    # 5,000 / 47,500 are hard clip values (1,191 rows sit exactly at 47,500); keep but flag.
    out["weight_clipped"] = out["weight"].isin([5_000.0, 47_500.0]).astype(int)
    # Missing weight stays NaN: the gradient-boosting model routes NaN natively.

    out = out.join(daily, on="date")
    if "market_index" not in out:
        out["market_index"] = np.nan
    if "quote_signal" not in out:
        out["quote_signal"] = np.nan
    # Raw row value, used by the per-load variant that can't rely on a same-day batch.
    out["market_index_raw"] = out["market_index"]
    # Missing market_index -> that day's market level.
    out["market_index"] = out["market_index"].fillna(out["market_daily"])
    out["quote_signal"] = out["quote_signal"].fillna(out["quote_daily"])
    return out


def flag_label_outliers(frame: pd.DataFrame, log_pred: np.ndarray, threshold: float = 0.4) -> pd.Series:
    """Rows whose posted_rate is off by more than exp(0.4) ~ 1.5x from a robust model.

    These ~1.4% of rows sit in two tight clusters (~0.2-0.4x and ~3-5x the expected rate),
    which looks like unit / keying errors rather than genuine market prices.
    """
    resid = np.log(frame[TARGET]) - log_pred
    return resid.abs() > threshold
