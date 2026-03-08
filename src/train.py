import json
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from preprocess import load_data, add_features, split_data


def dow_average_baseline(train, test, n_weeks=5, decay=0.85):
    predictions = []
    for date, row in test.iterrows():
        dow = date.dayofweek
        month = date.month
        same_dow_month = train[
            (train.index.dayofweek == dow) &
            (train.index.month == month)
        ]
        last_n = same_dow_month.iloc[-n_weeks:]
        weights = np.array([decay ** i for i in range(len(last_n)-1, -1, -1)])
        weights = weights / weights.sum()
        pred = np.average(last_n['Consumption'].values, weights=weights)
        predictions.append(pred)
    return np.array(predictions)


def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {'MAE': round(mae, 2), 'RMSE': round(rmse, 2)}


def train():
    # load and prepare data
    df = load_data()
    data = add_features(df)
    train_data, test_data, features, target = split_data(data)

    X_train = train_data[features]
    y_train = train_data[target]
    X_test = test_data[features]
    y_test = test_data[target]

    # scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # baseline
    dow_pred = dow_average_baseline(train_data, test_data)
    dow_metrics = evaluate(y_test, dow_pred)
    print(f"Baseline  — MAE: {dow_metrics['MAE']}, RMSE: {dow_metrics['RMSE']}")

    # KNN
    knn = KNeighborsRegressor(n_neighbors=10)
    knn.fit(X_train_scaled, y_train)
    knn_pred = knn.predict(X_test_scaled)
    knn_metrics = evaluate(y_test, knn_pred)
    print(f"KNN       — MAE: {knn_metrics['MAE']}, RMSE: {knn_metrics['RMSE']}")

    # MLP
    mlp = MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu',
                       max_iter=500, random_state=42)
    mlp.fit(X_train_scaled, y_train)
    mlp_pred = mlp.predict(X_test_scaled)
    mlp_metrics = evaluate(y_test, mlp_pred)
    print(f"MLP       — MAE: {mlp_metrics['MAE']}, RMSE: {mlp_metrics['RMSE']}")

    # save metrics
    metrics = {
        'baseline': dow_metrics,
        'knn': knn_metrics,
        'mlp': mlp_metrics
    }
    with open('metrics/results.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print("\nMetrics saved to metrics/results.json")


if __name__ == '__main__':
    train()