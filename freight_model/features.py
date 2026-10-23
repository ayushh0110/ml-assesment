"""Feature construction."""
from __future__ import annotations

import numpy as np
import pandas as pd

EQUIPMENT_CODES = {"Dry Van": 0, "Reefer": 1, "Flatbed": 2}
TREND_ORIGIN = pd.Timestamp("2025-01-01")

# Load-level features for the gradient-boosting part. Lanes are described by coordinates
# rather than city IDs so that the 8 cities that only appear in validation still get
# sensible predictions from their geographic neighbours.
LOAD_FEATURES = [
    "log_distance",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "equipment_code",
    "weight",
    "quote_signal",
]
# Market/time features for the linear part: trees cannot extrapolate a trend, and the
# market level drifts outside the combinations seen in training.
TEMPORAL_FEATURES = ["market_daily", "quote_daily", "trend_years"]


def haversine_miles(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    h = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 3958.8 * 2 * np.arcsin(np.sqrt(h))


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=frame.index)
    X["log_distance"] = np.log(frame["distance"])
    for col in ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon", "weight", "quote_signal"]:
        X[col] = frame[col].astype(float)
    X["equipment_code"] = frame["equipment"].map(EQUIPMENT_CODES)
    X["market_daily"] = frame["market_daily"]
    X["quote_daily"] = frame["quote_daily"]
    X["trend_years"] = (frame["date"] - TREND_ORIGIN).dt.days / 365.0
    return X
