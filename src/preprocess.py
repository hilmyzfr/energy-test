import pandas as pd
import numpy as np

def load_data(path='data/raw/opsd_germany_daily.csv'):
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df[['Consumption']]

def add_features(df):
    data = df.copy()
    data['dayofweek'] = data.index.dayofweek
    data['month'] = data.index.month
    data['is_weekend'] = (data.index.dayofweek >= 5).astype(int)
    data['lag_1'] = data['Consumption'].shift(1)
    data['lag_7'] = data['Consumption'].shift(7)
    data['rolling_7'] = data['Consumption'].shift(1).rolling(7).mean()
    data = data.dropna()
    return data

def split_data(data):
    train = data[data.index.year < 2017]
    test = data[data.index.year == 2017]
    features = ['dayofweek', 'month', 'is_weekend', 'lag_1', 'lag_7', 'rolling_7']
    target = 'Consumption'
    return train, test, features, target

if __name__ == '__main__':
    df = load_data()
    data = add_features(df)
    train, test, features, target = split_data(data)
    print(f"Train: {train.shape[0]} rows")
    print(f"Test:  {test.shape[0]} rows")