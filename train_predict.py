"""Fit the final model on all clean Jan-Oct loads and write the two prediction files.

Outputs:
  validation_predictions.csv          load_id,predicted_rate for the 12,000 validation loads
  data/december_chart_inputs.csv      predicted_rate column filled (input for score.py)
  models/freight_rate_model.joblib    fitted model
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from freight_model.data import DATA_DIR, TARGET
from freight_model.features import TEMPORAL_FEATURES
from freight_model.model import FreightRateModel
from freight_model.pipeline import prepare

ROOT = Path(__file__).resolve().parent


def main() -> None:
    prep = prepare()
    model = FreightRateModel().fit(prep.X_train, prep.train[TARGET])
    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump(model, ROOT / "models" / "freight_rate_model.joblib")
    coef = {k: round(float(v), 4) for k, v in zip(TEMPORAL_FEATURES, model.linear_.coef_)}
    print(f"Trained on {len(prep.train):,} loads. Linear offset coefficients: {coef}")

    # Validation loads -> template order.
    preds = pd.Series(model.predict(prep.X_validation), index=prep.validation["load_id"])
    template = pd.read_csv(DATA_DIR / "validation_predictions_template.csv")
    template["predicted_rate"] = template["load_id"].map(preds).round(2)
    assert template["predicted_rate"].notna().all() and (template["predicted_rate"] > 0).all()
    template[["load_id", "predicted_rate"]].to_csv(ROOT / "validation_predictions.csv", index=False)
    print(f"Wrote validation_predictions.csv ({len(template):,} rows, "
          f"mean {template['predicted_rate'].mean():.2f})")

    # December chart inputs: keep the original seven columns, only fill predicted_rate.
    chart_path = DATA_DIR / "december_chart_inputs.csv"
    chart = pd.read_csv(chart_path)
    chart["predicted_rate"] = np.round(model.predict(prep.X_december), 2)
    chart.to_csv(chart_path, index=False)
    print(f"Filled {chart_path.relative_to(ROOT)}: "
          f"{chart['predicted_rate'].min():.2f} - {chart['predicted_rate'].max():.2f}")


if __name__ == "__main__":
    main()
