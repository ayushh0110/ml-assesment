# Freight rate prediction

My solution to the Spotter ML Engineer assessment. It predicts `posted_rate` for truckload freight: it trains on Jan-Oct 2025 loads (`data/train_test.csv`) and predicts the 12,000 Nov-Dec 2025 loads in `data/validation.csv`.

The write-up (split and validation approach, data issues, results and the December chart) is in `reports/Freight_Rate_Report.pdf`.

## Setup and running

Python 3.10+ (I used 3.12).

```bash
python -m pip install -r requirements.txt

python eda.py              # data checks, prints findings, saves figures to reports/figures/
python validate.py         # rolling-origin + unseen-city validation -> reports/*.csv (a few minutes)
python train_predict.py    # trains the final model, writes validation_predictions.csv, fills data/december_chart_inputs.csv
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
python make_report.py      # rebuilds reports/Freight_Rate_Report.docx
```

`train_predict.py` and `score.py` are all you need to reproduce the submission files. The others are for the analysis.

## Code layout

```
freight_model/
  data.py       loading, cleaning, city coordinate lookup, label-outlier flag
  features.py   feature construction
  model.py      FreightRateModel (GBM + linear market/trend term), benchmark models, metrics
  pipeline.py   prepare(): runs load -> clean -> drop bad labels -> features for all three files
eda.py, validate.py, train_predict.py, make_report.py
score.py        provided scorer, unchanged
```

## Approach

**Split.** The task is a 1-2 month forecast, so validation is out-of-time:
- 5 expanding-window folds. Each one trains on everything before a cutoff (May 1 ... Sep 1) and tests on the next two months.
- A city holdout drops 8 random cities from training, because 8 cities in `validation.csv` never appear in training.
- A random K-fold split overstates accuracy here (about 1.8% vs 3.0% MAPE for the same model), because loads from the same day end up on both sides.

**Cleaning.**
- Drop the 1.4% of rows whose rate is about 0.25x or 4x the expected value. These are likely keying/unit errors; keeping them in training raises out-of-time MAPE from 2.08% to 3.88%.
- `abs()` on negative weights.
- Leave missing weights as NaN.
- Fill missing `market_index` with the daily mean.
- Look up coordinates and the daily market level for the December chart rows.

**Model.** `log(rate) = GBM(distance, lat/lon, equipment, weight, quote_signal) + linear(daily market_index, daily quote_signal, trend)`.
- Rates follow the market index and drift slowly upward.
- A plain GBM learns the index as month-shaped steps and can't extrapolate the trend, so those parts go in a linear term. The two parts are fitted by backfitting.

| Rolling-origin average | MAPE | MAE |
|---|---|---|
| GBM, load features only | 2.62% | $61 |
| GBM + market_index as a feature | 3.04% | $70 |
| GBM + linear term, per-load inputs only | 2.09% | $49 |
| **GBM + linear market/trend term (final)** | **2.08%** | **$48** |

MAPE is the mean of |pred - actual| / actual within each 2-month window, averaged over the 5 windows. Those windows were also used for model selection, so they're slightly optimistic. Spotter's Nov-Dec scoring is the untouched test.

**Same-day averages.** The final model uses the daily mean of `market_index` and `quote_signal` across all loads on a date. These are features only, never rates. That assumes the whole day's batch is available at prediction time, which is true for `validation.csv`. If loads had to be priced one by one, the per-load variant (each load's own `market_index`, no daily averages) is within 0.01 points.

See the report for the per-fold numbers, the city holdout and the extra checks (`reports/extra_checks.csv`).

## Data

`data/` holds the files from the assessment, renamed to the paths the provided README uses.
