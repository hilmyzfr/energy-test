import pandas as pd
import numpy as np
import requests
import holidays

def load_data(path='data/raw/opsd_germany_daily.csv'):
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df[['Consumption']]

def fetch_temperature(start_date, end_date):
    """
    Fetch daily mean temperature for Berlin from Open-Meteo.
    Uses the historical archive API for past dates and the forecast API
    for future dates. Falls back to a sensible default if either fails.
    """
    # Try archive API first (works for past dates)
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": 52.52,
        "longitude": 13.41,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "temperature_2m_mean",
        "timezone": "Europe/Berlin"
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()

        if 'daily' in data and data['daily'].get('time'):
            temp_df = pd.DataFrame({
                'date': pd.to_datetime(data['daily']['time']),
                'temperature': data['daily']['temperature_2m_mean']
            })
            temp_df = temp_df.set_index('date')
            return temp_df
    except Exception:
        pass

    # Try forecast API for future/recent dates
    try:
        forecast_url = "https://api.open-meteo.com/v1/forecast"
        forecast_params = {
            "latitude": 52.52,
            "longitude": 13.41,
            "daily": "temperature_2m_mean",
            "timezone": "Europe/Berlin",
            "start_date": start_date,
            "end_date": end_date,
        }
        response = requests.get(forecast_url, params=forecast_params, timeout=10)
        data = response.json()

        if 'daily' in data and data['daily'].get('time'):
            temp_df = pd.DataFrame({
                'date': pd.to_datetime(data['daily']['time']),
                'temperature': data['daily']['temperature_2m_mean']
            })
            temp_df = temp_df.set_index('date')
            return temp_df
    except Exception:
        pass

    # Fallback: return a reasonable default for Germany (~8°C annual average)
    print(f"Warning: Could not fetch temperature for {start_date} to {end_date}, using default 8.0°C")
    date_range = pd.date_range(start_date, end_date)
    temp_df = pd.DataFrame({
        'date': date_range,
        'temperature': [8.0] * len(date_range)
    })
    temp_df = temp_df.set_index('date')
    return temp_df

def add_holiday_flag(df):
    """Add German public holiday flag."""
    de_holidays = holidays.Germany()
    df['is_holiday'] = df.index.map(lambda x: 1 if x in de_holidays else 0)
    return df

def add_features(df, temperature):
    data = df.copy()
    
    # merge temperature
    data = data.join(temperature, how='left')
    
    # time features
    data['dayofweek'] = data.index.dayofweek
    data['month'] = data.index.month
    data['is_weekend'] = (data.index.dayofweek >= 5).astype(int)
    
    # holiday flag
    data = add_holiday_flag(data)
    
    # lag features
    data['lag_1'] = data['Consumption'].shift(1)
    data['lag_7'] = data['Consumption'].shift(7)
    data['rolling_7'] = data['Consumption'].shift(1).rolling(7).mean()
    
    # interaction features — cold holidays/weekends still need heating
    data['holiday_temp'] = data['is_holiday'] * data['temperature']
    data['weekend_temp'] = data['is_weekend'] * data['temperature']
    
    data = data.dropna()
    return data

def split_data(data):
    train = data[data.index.year < 2017]
    test = data[data.index.year == 2017]
    features = ['dayofweek', 'month', 'is_weekend', 'is_holiday',
                'temperature', 'lag_1', 'lag_7', 'rolling_7',
                'holiday_temp', 'weekend_temp']
    target = 'Consumption'
    return train, test, features, target

if __name__ == '__main__':
    df = load_data()
    print("Fetching temperature data...")
    temperature = fetch_temperature('2006-01-01', '2017-12-31')
    data = add_features(df, temperature)
    train, test, features, target = split_data(data)
    print(f"Train: {train.shape[0]} rows")
    print(f"Test:  {test.shape[0]} rows")
    print(f"Features: {features}")