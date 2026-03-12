# Energy Consumption Forecaster

Forecasting daily electricity consumption for Germany using Open Power System Data.
Built to practice forecasting methods I used at enercity, where I worked on consumption
models across a portfolio of several thousand customers.

## What it does

Send a date and recent consumption values, and the API returns a predicted consumption
in GWh. It fetches live temperature from Open-Meteo and checks German public holidays
automatically.

Also includes an automated plausibility check. At enercity we had to manually review
hundreds of customer model predictions every day. This automates that check and only
flags the ones that look suspicious.

## Dataset

Open Power System Data, daily electricity consumption for Germany 2012-2017.
Source: https://open-power-system-data.org

## Models

Three models compared on 2017 holdout data (365 days):

1. **Day-of-week baseline** - weighted average of the last 5 same weekday + same month
   values. More recent weeks get higher weight. This is the standard baseline approach
   in operational energy forecasting.
2. **KNN** - k-nearest neighbours using time, lag and weather features.
3. **MLP** - simple two-layer neural network with the same features as KNN.

## Results

| Model | MAE | RMSE | Train time | Inference |
|---|---|---|---|---|
| Day-of-week baseline | 48 GWh | 78 GWh | - | 0.11s |
| KNN | 25 GWh | 35 GWh | 0.02s | 0.006s |
| MLP | 20 GWh | 32 GWh | 1.85s | ~0s |

KNN is used in the API. Fast to train, near instant inference, and only slightly
behind MLP on accuracy. Adding temperature and holiday features improved MAE from
29 to 25 compared to using time and lag features only.

## Features used

- Day of week, month, is_weekend
- is_holiday (German public holidays)
- temperature (fetched from Open-Meteo API)
- lag_1 - yesterday's consumption
- lag_7 - same day last week
- rolling_7 - 7-day rolling average

## Data Pipeline

Built with dbt on DuckDB for local development:

- `stg_energy` - staging view for raw consumption data
- `stg_weather` - staging view for weather data
- `fct_energy_features` - mart table with all engineered features
- 16 data quality tests covering nulls, uniqueness and accepted values

Feature engineering is also implemented in PySpark (`src/spark_features.py`) to
support larger datasets and demonstrate production-scale pipeline patterns.

## Model Explainability

SHAP and LIME are used to explain individual predictions (`src/explain.py`):

- SHAP KernelExplainer — global feature importance and per-prediction waterfall plots
- LIME — local surrogate model explanations as cross-check

## API

- `GET /health` - check if service is running
- `POST /predict` - get predictions and plausibility check

### Request
```json
{
  "date": "2024-03-08",
  "lag_1": 1350.0,
  "lag_7": 1380.0,
  "special_event": false,
  "model": "knn"
}
```

`model` options: `knn`, `mlp`, `baseline`, `all`

`special_event` - set to true if the customer has notified you of unusual consumption
like a shutdown, production increase or closure. This suppresses the prediction
plausibility warning, but data pipeline issues are still flagged regardless.

### Response
```json
{
  "date": "2024-03-08",
  "model": "all",
  "predictions_gwh": {
    "knn": 1386.85,
    "mlp": 1337.34,
    "baseline": 1463.37
  },
  "temperature_c": 2.5,
  "is_holiday": 0,
  "plausibility": {
    "is_plausible": true,
    "warning": null,
    "expected_range": [1160.93, 1741.4],
    "deviation_pct": 4.4,
    "special_event_mode": false,
    "data_issue": false
  }
}
```

## Plausibility check

Two separate checks run on every prediction:

1. **Input check** - flags if lag_1 deviates more than 50% from the historical mean
   for that weekday and month. This usually means something is wrong in the data
   pipeline. Always runs, cannot be suppressed.
2. **Prediction check** - flags if the prediction deviates more than 20% from recent
   same-weekday historical values. Suppressed if special_event is true.

## How to run

**Train models:**
```bash
pip install -r requirements.txt
python3 src/train.py
```

**Run dbt pipeline:**
```bash
cd dbt_energy
dbt run
dbt test
```

**Run API locally:**
```bash
uvicorn src.api:app --reload
```

**Run with Docker:**
```bash
docker build -t energy-forecaster .
docker run -p 8000:8000 energy-forecaster
```

## Stack

Python · Scikit-learn · PySpark · dbt · DuckDB · FastAPI · Docker · SHAP · LIME · Open-Meteo API

## Next steps

- LLM email parser to extract special event details from customer notifications
  and automatically set the special_event flag
- Streamlit dashboard for visualising predictions and plausibility flags
- GitHub Actions for automated testing