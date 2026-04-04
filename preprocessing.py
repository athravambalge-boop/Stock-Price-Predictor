import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def add_features(data):
    enriched = data.copy()

    enriched['MA20'] = enriched['Close'].rolling(window=20).mean()
    enriched['MA50'] = enriched['Close'].rolling(window=50).mean()
    enriched['MA100'] = enriched['Close'].rolling(window=100).mean()
    enriched['EMA20'] = enriched['Close'].ewm(span=20, adjust=False).mean()

    delta = enriched['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    enriched['RSI'] = 100 - (100 / (1 + rs))

    high_low = enriched['High'] - enriched['Low']
    high_close_prev = (enriched['High'] - enriched['Close'].shift(1)).abs()
    low_close_prev = (enriched['Low'] - enriched['Close'].shift(1)).abs()
    true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    enriched['ATR14'] = true_range.rolling(window=14).mean()

    rolling_std20 = enriched['Close'].rolling(window=20).std()
    enriched['BollingerUpper'] = enriched['MA20'] + 2 * rolling_std20
    enriched['BollingerLower'] = enriched['MA20'] - 2 * rolling_std20
    enriched['BollingerWidth'] = (enriched['BollingerUpper'] - enriched['BollingerLower']) / enriched['MA20']

    enriched['VolumeChange'] = enriched['Volume'].pct_change()
    direction = np.sign(enriched['Close'].diff()).fillna(0)
    enriched['OBV'] = (direction * enriched['Volume']).fillna(0).cumsum()

    enriched['Return1'] = enriched['Close'].pct_change()
    enriched['LogReturn'] = np.log(enriched['Close'] / enriched['Close'].shift(1))
    enriched['Volatility20'] = enriched['LogReturn'].rolling(window=20).std()

    enriched.replace([np.inf, -np.inf], np.nan, inplace=True)
    enriched.dropna(inplace=True)
    return enriched


def create_sequences(features, target, sequence_length=60):
    X, y = [], []
    for i in range(sequence_length, len(features)):
        X.append(features[i-sequence_length:i])
        y.append(target[i])
    return np.array(X), np.array(y)


def prepare_lstm_data(data, feature_cols, target_col='LogReturn', sequence_length=60, train_ratio=0.8):
    features = data[feature_cols].values
    target = data[[target_col]].values
    close = data['Close'].values
    dates = data.index

    X, y, y_price, prev_close, y_dates = [], [], [], [], []
    for i in range(sequence_length, len(data)):
        X.append(features[i-sequence_length:i])
        y.append(target[i])
        y_price.append(close[i])
        prev_close.append(close[i-1])
        y_dates.append(dates[i])

    X = np.array(X)
    y = np.array(y)
    y_price = np.array(y_price).reshape(-1, 1)
    prev_close = np.array(prev_close).reshape(-1, 1)
    y_dates = np.array(y_dates)

    split = int(train_ratio * len(X))

    X_train_raw, X_test_raw = X[:split], X[split:]
    y_train_raw, y_test_raw = y[:split], y[split:]

    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_2d = X_train_raw.reshape(-1, X_train_raw.shape[-1])
    feature_scaler.fit(X_train_2d)

    X_train_scaled = feature_scaler.transform(X_train_2d).reshape(X_train_raw.shape)
    X_test_scaled = feature_scaler.transform(X_test_raw.reshape(-1, X_test_raw.shape[-1])).reshape(X_test_raw.shape)

    target_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler.fit(y_train_raw)
    y_train_scaled = target_scaler.transform(y_train_raw)
    y_test_scaled = target_scaler.transform(y_test_raw)

    return {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train_scaled,
        'y_test': y_test_scaled,
        'y_test_price': y_price[split:],
        'prev_close_test': prev_close[split:],
        'dates_test': y_dates[split:],
        'feature_scaler': feature_scaler,
        'target_scaler': target_scaler,
        'split_index': split,
    }