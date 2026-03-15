import json
import time
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from preprocess import load_data, add_features, split_data, fetch_temperature

# Try to import SMARD client for live data
try:
    from smard_client import fetch_consumption
    SMARD_AVAILABLE = True
except ImportError:
    SMARD_AVAILABLE = False
    print("Note: smard_client not found. Training with OPSD data only (2006-2017).")


def load_extended_data():
    """
    Load OPSD data (2006-2017) and extend it with SMARD data (2018-present).
    SMARD data is fetched live from the Bundesnetzagentur API — no manual download needed.
    """
    # Load your original OPSD dataset
    opsd = load_data()
    last_opsd_date = opsd.index.max().strftime("%Y-%m-%d")
    print(f"OPSD data: {opsd.index.min().date()} to {opsd.index.max().date()} ({len(opsd)} rows)")

    if not SMARD_AVAILABLE:
        print("Skipping SMARD fetch — using OPSD data only.")
        return opsd

    # Fetch SMARD data from where OPSD ends until now
    smard_start = (opsd.index.max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    smard_end = pd.Timestamp.now().strftime("%Y-%m-%d")

    print(f"Fetching SMARD data: {smard_start} to {smard_end}...")
    smard_df = fetch_consumption(smard_start, smard_end)

    if smard_df.empty:
        print("Warning: No SMARD data returned. Using OPSD data only.")
        return opsd

    # Format SMARD data to match OPSD structure
    smard_df = smard_df.set_index("date")
    smard_df.index = pd.to_datetime(smard_df.index)
    smard_df.index.name = opsd.index.name

    # Merge the two datasets
    combined = pd.concat([opsd, smard_df])

    # Remove any duplicates (overlapping dates)
    combined = combined[~combined.index.duplicated(keep="first")]
    combined = combined.sort_index()

    print(f"Combined data: {combined.index.min().date()} to {combined.index.max().date()} ({len(combined)} rows)")
    return combined


def split_data_extended(data):
    """
    Split with more recent test set now that we have data up to 2025+.
    Uses the last full year as test, everything before as train.
    Falls back to the original 2017 split if no extended data is available.
    """
    last_year = data.index.max().year
    if last_year <= 2017:
        # Original split
        train = data[data.index.year < 2017]
        test = data[data.index.year == 2017]
    else:
        # Use most recent full year as test
        test_year = last_year - 1 if last_year == pd.Timestamp.now().year else last_year
        train = data[data.index.year < test_year]
        test = data[data.index.year == test_year]
        print(f"Train: up to {test_year - 1}, Test: {test_year}")

    features = ['dayofweek', 'month', 'is_weekend', 'is_holiday',
                'temperature', 'lag_1', 'lag_7', 'rolling_7',
                'holiday_temp', 'weekend_temp']
    target = 'Consumption'
    return train, test, features, target


def dow_average_baseline(train, test, n_weeks=5, decay=0.85):
    predictions = []
    all_data = pd.concat([train, test])
    
    for date, row in test.iterrows():
        dow = date.dayofweek
        month = date.month
        same_dow_month = all_data[
            (all_data.index.dayofweek == dow) &
            (all_data.index.month == month) &
            (all_data.index < date)
        ]
        last_n = same_dow_month.iloc[-n_weeks:]
        weights = np.array([decay ** i for i in range(len(last_n)-1, -1, -1)])
        weights = weights / weights.sum()
        pred = np.average(last_n['Consumption'].values, weights=weights)
        predictions.append(pred)
    
    return np.array(predictions)


def evaluate(y_true, y_pred, train_time, inference_time):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {
        'MAE': round(mae, 2),
        'RMSE': round(rmse, 2),
        'train_time_sec': round(train_time, 3),
        'inference_time_sec': round(inference_time, 3)
    }


def train():
    # Load data — automatically extends with SMARD if available
    df = load_extended_data()

    # Fetch temperature for the full date range
    start_date = df.index.min().strftime("%Y-%m-%d")
    end_date = df.index.max().strftime("%Y-%m-%d")
    print(f"Fetching temperature data: {start_date} to {end_date}...")
    temperature = fetch_temperature(start_date, end_date)

    data = add_features(df, temperature)

    # Use extended split if we have data beyond 2017
    train_data, test_data, features, target = split_data_extended(data)

    print(f"Train set: {len(train_data)} rows, Test set: {len(test_data)} rows")

    X_train = train_data[features]
    y_train = train_data[target]
    X_test = test_data[features]
    y_test = test_data[target]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # baseline
    t0 = time.time()
    dow_pred = dow_average_baseline(train_data, test_data)
    baseline_time = time.time() - t0
    dow_metrics = evaluate(y_test, dow_pred, train_time=0, inference_time=baseline_time)
    print(f"Baseline  — MAE: {dow_metrics['MAE']}, RMSE: {dow_metrics['RMSE']}, "
          f"inference: {dow_metrics['inference_time_sec']}s")

    # KNN
    t0 = time.time()
    knn = KNeighborsRegressor(n_neighbors=10)
    knn.fit(X_train_scaled, y_train)
    knn_train_time = time.time() - t0

    t0 = time.time()
    knn_pred = knn.predict(X_test_scaled)
    knn_inference_time = time.time() - t0

    knn_metrics = evaluate(y_test, knn_pred, knn_train_time, knn_inference_time)
    print(f"KNN       — MAE: {knn_metrics['MAE']}, RMSE: {knn_metrics['RMSE']}, "
          f"train: {knn_metrics['train_time_sec']}s, "
          f"inference: {knn_metrics['inference_time_sec']}s")

    # MLP
    t0 = time.time()
    mlp = MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu',
                       max_iter=1000, random_state=42)
    mlp.fit(X_train_scaled, y_train)
    mlp_train_time = time.time() - t0

    t0 = time.time()
    mlp_pred = mlp.predict(X_test_scaled)
    mlp_inference_time = time.time() - t0

    mlp_metrics = evaluate(y_test, mlp_pred, mlp_train_time, mlp_inference_time)
    print(f"MLP       — MAE: {mlp_metrics['MAE']}, RMSE: {mlp_metrics['RMSE']}, "
          f"train: {mlp_metrics['train_time_sec']}s, "
          f"inference: {mlp_metrics['inference_time_sec']}s")

    # save metrics
    last_year = df.index.max().year
    metrics = {
        'data_range': f"{df.index.min().date()} to {df.index.max().date()}",
        'total_rows': len(data),
        'train_rows': len(train_data),
        'test_rows': len(test_data),
        'smard_data_used': SMARD_AVAILABLE,
        'baseline': dow_metrics,
        'knn': knn_metrics,
        'mlp': mlp_metrics
    }
    with open('metrics/results.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print("\nMetrics saved to metrics/results.json")


if __name__ == '__main__':
    train()