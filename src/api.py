import sys
import holidays
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

sys.path.append('src')
from preprocess import load_data, add_features, split_data, fetch_temperature

# ── Load and train models on startup ─────────────────────────────────────
print("Loading data and training models...")
df = load_data()
temperature = fetch_temperature('2006-01-01', '2017-12-31')
data = add_features(df, temperature)
train_data, test_data, features, target = split_data(data)

X_train = train_data[features]
y_train = train_data[target]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

knn = KNeighborsRegressor(n_neighbors=10)
knn.fit(X_train_scaled, y_train)

mlp = MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu',
                   max_iter=2000, random_state=42)
mlp.fit(X_train_scaled, y_train)

full_data = pd.concat([train_data, test_data])
de_holidays = holidays.Germany()
print("Models ready.")


# ── Baseline model ────────────────────────────────────────────────────────
def dow_average_baseline(date, full_data, n_weeks=5, decay=0.85):
    dow = date.dayofweek
    month = date.month
    same_dow_month = full_data[
        (full_data.index.dayofweek == dow) &
        (full_data.index.month == month) &
        (full_data.index < date)
    ]
    last_n = same_dow_month.iloc[-n_weeks:]
    weights = np.array([decay ** i for i in range(len(last_n)-1, -1, -1)])
    weights = weights / weights.sum()
    return float(np.average(last_n['Consumption'].values, weights=weights))


# ── Plausibility check ────────────────────────────────────────────────────
def check_plausibility(date, prediction, full_data, special_event,
                       request_lag_1, threshold=0.20, n_weeks=8):
    dow = date.dayofweek
    is_weekend = dow >= 5
    is_holiday = date in de_holidays

    # validate input lag values — always flag regardless of special_event
    historical_mean = full_data[
        (full_data.index.dayofweek == dow) &
        (full_data.index.month == date.month)
    ]['Consumption'].mean()

    lag_deviation = abs(request_lag_1 - historical_mean) / historical_mean * 100
    if lag_deviation > 50:
        return {
            "is_plausible": bool(False),
            "warning": f"data issue — lag_1 deviates {round(lag_deviation, 1)}% from historical mean. check input data pipeline",
            "expected_range": [round(historical_mean * 0.5, 2), round(historical_mean * 1.5, 2)],
            "deviation_pct": round(lag_deviation, 1),
            "special_event_mode": bool(special_event),
            "data_issue": bool(True)
        }

    # check prediction against historical same weekday + month
    same_dow = full_data[
        (full_data.index.dayofweek == dow) &
        (full_data.index.month == date.month) &
        (full_data.index < date)
    ]
    last_n = same_dow['Consumption'].iloc[-n_weeks:]

    if len(last_n) == 0:
        return {
            "is_plausible": bool(True),
            "warning": "not enough historical data to check plausibility",
            "expected_range": None,
            "deviation_pct": None,
            "special_event_mode": bool(special_event),
            "data_issue": bool(False)
        }

    mean_val = last_n.mean()
    lower = round(mean_val * (1 - threshold), 2)
    upper = round(mean_val * (1 + threshold), 2)
    deviation_pct = round(abs(prediction - mean_val) / mean_val * 100, 1)
    is_plausible = bool(lower <= prediction <= upper)

    # special_event suppresses prediction warning but not data issues
    if special_event:
        warning = "special event flagged — prediction plausibility check suppressed"
        is_plausible = bool(True)
    elif not is_plausible:
        day_type = "holiday" if is_holiday else "weekend" if is_weekend else "weekday"
        warning = (f"prediction deviates {deviation_pct}% from expected range "
                   f"for {day_type} in this month")
    else:
        warning = None

    return {
        "is_plausible": is_plausible,
        "warning": warning,
        "expected_range": [lower, upper],
        "deviation_pct": deviation_pct,
        "special_event_mode": bool(special_event),
        "data_issue": bool(False)
    }


# ── API ───────────────────────────────────────────────────────────────────
app = FastAPI()


class PredictionRequest(BaseModel):
    date: str
    lag_1: float
    lag_7: float
    special_event: bool = False
    model: str = "knn"  # options: "knn", "mlp", "baseline", "all"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictionRequest):
    date = pd.Timestamp(request.date)

    # fetch live temperature
    temp_df = fetch_temperature(request.date, request.date)
    temp = float(temp_df['temperature'].values[0])

    is_holiday = int(date in de_holidays)
    rolling_7 = (request.lag_1 + request.lag_7) / 2

    X = np.array([[
        date.dayofweek,
        date.month,
        int(date.dayofweek >= 5),
        is_holiday,
        temp,
        request.lag_1,
        request.lag_7,
        rolling_7
    ]])
    X_scaled = scaler.transform(X)

    # get predictions based on model selection
    predictions = {}
    if request.model in ("knn", "all"):
        predictions["knn"] = round(float(knn.predict(X_scaled)[0]), 2)
    if request.model in ("mlp", "all"):
        predictions["mlp"] = round(float(mlp.predict(X_scaled)[0]), 2)
    if request.model in ("baseline", "all"):
        predictions["baseline"] = round(dow_average_baseline(date, full_data), 2)

    # use selected model for plausibility, default to knn
    reference_pred = predictions.get(request.model) or predictions.get("knn")

    plausibility = check_plausibility(
        date, reference_pred, full_data, request.special_event, request.lag_1
    )

    return {
        "date": request.date,
        "model": request.model,
        "predictions_gwh": predictions,
        "temperature_c": round(temp, 1),
        "is_holiday": is_holiday,
        "plausibility": plausibility
    }
# ── API ───────────────────────────────────────────────────────────────────
app = FastAPI()


class PredictionRequest(BaseModel):
    date: str
    lag_1: float
    lag_7: float
    special_event: bool = False
    model: str = "knn"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictionRequest):
    date = pd.Timestamp(request.date)

    temp_df = fetch_temperature(request.date, request.date)
    temp = float(temp_df['temperature'].values[0])

    is_holiday = int(date in de_holidays)
    rolling_7 = (request.lag_1 + request.lag_7) / 2

    X = np.array([[
        date.dayofweek,
        date.month,
        int(date.dayofweek >= 5),
        is_holiday,
        temp,
        request.lag_1,
        request.lag_7,
        rolling_7
    ]])
    X_scaled = scaler.transform(X)

    predictions = {}
    if request.model in ("knn", "all"):
        predictions["knn"] = round(float(knn.predict(X_scaled)[0]), 2)
    if request.model in ("mlp", "all"):
        predictions["mlp"] = round(float(mlp.predict(X_scaled)[0]), 2)
    if request.model in ("baseline", "all"):
        predictions["baseline"] = round(dow_average_baseline(date, full_data), 2)

    reference_pred = predictions.get(request.model) or predictions.get("knn")

    plausibility = check_plausibility(
        date, reference_pred, full_data, request.special_event, request.lag_1
    )

    return {
        "date": request.date,
        "model": request.model,
        "predictions_gwh": predictions,
        "temperature_c": round(temp, 1),
        "is_holiday": is_holiday,
        "plausibility": plausibility
    }
