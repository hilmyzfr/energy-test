
# Energy Consumption Forecaster

Forecasting daily electricity consumption for Germany using the Open Power System Data.
Built to practice time series forecasting methods I used during my time at enercity.

## Dataset

Open Power System Data — daily electricity consumption for Germany, 2006-2017.
Source: https://open-power-system-data.org

## Approach

Three models compared on 2017 holdout data:

1. **Day-of-week baseline** — weighted average of last 5 same weekday + same month 
   observations. Recent weeks weighted higher. Simple but surprisingly competitive.

2. **KNN** — k-nearest neighbours on time and lag features. Finds historically 
   similar days and averages their consumption.

3. **MLP** — simple two-layer neural network. Same features as KNN.

## Results

| Model | MAE | RMSE |
|-------|-----|------|
| Day-of-week baseline | 46 GWh | 75 GWh |
| KNN | 29 GwH | 51 GwH |
| MLP | 27 GwH | 52 GwH |

Lag features (yesterday's consumption, last week) made the biggest difference.
KNN and MLP perform similarly — the bottleneck is the feature set, not model complexity.
Weather and holiday data would be the logical next step.

## Features used

- Day of week, month, is_weekend
- lag_1 (yesterday's consumption)
- lag_7 (same day last week)
- rolling_7 (7-day rolling average)

## Stack

Python, Pandas, Scikit-learn, Matplotlib

## Next steps

- Add temperature data from Open-Meteo API
- Add German public holiday flags
- Refactor notebook into modular scripts