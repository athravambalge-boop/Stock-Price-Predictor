import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def add_features(data):
    enriched = data.copy()

    # Moving Averages
    enriched['MA20'] = enriched['Close'].rolling(window=20).mean()
    enriched['MA50'] = enriched['Close'].rolling(window=50).mean()
    enriched['MA100'] = enriched['Close'].rolling(window=100).mean()
    enriched['EMA20'] = enriched['Close'].ewm(span=20, adjust=False).mean()
    enriched['EMA50'] = enriched['Close'].ewm(span=50, adjust=False).mean()

    # RSI (Relative Strength Index)
    delta = enriched['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    enriched['RSI'] = 100 - (100 / (1 + rs))

    # ATR (Average True Range)
    high_low = enriched['High'] - enriched['Low']
    high_close_prev = (enriched['High'] - enriched['Close'].shift(1)).abs()
    low_close_prev = (enriched['Low'] - enriched['Close'].shift(1)).abs()
    true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    enriched['ATR14'] = true_range.rolling(window=14).mean()

    # Bollinger Bands
    rolling_std20 = enriched['Close'].rolling(window=20).std()
    enriched['BollingerUpper'] = enriched['MA20'] + 2 * rolling_std20
    enriched['BollingerLower'] = enriched['MA20'] - 2 * rolling_std20
    enriched['BollingerWidth'] = (enriched['BollingerUpper'] - enriched['BollingerLower']) / enriched['MA20']

    # MACD (Moving Average Convergence Divergence)
    ema12 = enriched['Close'].ewm(span=12, adjust=False).mean()
    ema26 = enriched['Close'].ewm(span=26, adjust=False).mean()
    enriched['MACD'] = ema12 - ema26
    enriched['MACD_Signal'] = enriched['MACD'].ewm(span=9, adjust=False).mean()
    enriched['MACD_Histogram'] = enriched['MACD'] - enriched['MACD_Signal']

    # Stochastic Oscillator
    low14 = enriched['Low'].rolling(window=14).min()
    high14 = enriched['High'].rolling(window=14).max()
    enriched['Stochastic_K'] = 100 * ((enriched['Close'] - low14) / (high14 - low14))
    enriched['Stochastic_D'] = enriched['Stochastic_K'].rolling(window=3).mean()

    # Volume indicators
    enriched['VolumeChange'] = enriched['Volume'].pct_change()
    direction = np.sign(enriched['Close'].diff()).fillna(0)
    enriched['OBV'] = (direction * enriched['Volume']).fillna(0).cumsum()
    enriched['OBV_EMA'] = enriched['OBV'].ewm(span=20, adjust=False).mean()

    # Price-based features
    enriched['Return1'] = enriched['Close'].pct_change()
    enriched['LogReturn'] = np.log(enriched['Close'] / enriched['Close'].shift(1))
    enriched['Volatility20'] = enriched['LogReturn'].rolling(window=20).std()
    
    # Additional lag features
    enriched['Close_Lag1'] = enriched['Close'].shift(1)
    enriched['Close_Lag5'] = enriched['Close'].shift(5)
    enriched['Return_Lag1'] = enriched['Return1'].shift(1)
    enriched['Return_Lag5'] = enriched['Return1'].shift(5)
    
    # Williams %R
    enriched['Williams_R'] = -100 * ((high14 - enriched['Close']) / (high14 - low14))
    
    # CCI (Commodity Channel Index)
    tp = (enriched['High'] + enriched['Low'] + enriched['Close']) / 3
    sma_tp = tp.rolling(window=20).mean()
    mad = tp.rolling(window=20).apply(lambda x: np.abs(x - x.mean()).mean())
    enriched['CCI'] = (tp - sma_tp) / (0.015 * mad)

    enriched.replace([np.inf, -np.inf], np.nan, inplace=True)
    enriched.dropna(inplace=True)
    return enriched


def create_sequences(features, target, sequence_length=60):
    X, y = [], []
    for i in range(sequence_length, len(features)):
        X.append(features[i-sequence_length:i])
        y.append(target[i])
    return np.array(X), np.array(y)


def build_sequence_dataset(data, feature_cols, target_col='LogReturn', sequence_length=60):
    features = data[feature_cols].values
    target = data[[target_col]].values
    close = data['Close'].values
    dates = data.index

    X, y, y_price, prev_close, y_dates = [], [], [], [], []
    for i in range(sequence_length, len(data)):
        X.append(features[i - sequence_length:i])
        y.append(target[i])
        y_price.append(close[i])
        prev_close.append(close[i - 1])
        y_dates.append(dates[i])

    return {
        'X_raw': np.array(X),
        'y_raw': np.array(y),
        'y_price': np.array(y_price).reshape(-1, 1),
        'prev_close': np.array(prev_close).reshape(-1, 1),
        'dates': np.array(y_dates),
    }


def prepare_lstm_data(data, feature_cols, target_col='LogReturn', sequence_length=60, train_ratio=0.8):
    sequence_bundle = build_sequence_dataset(
        data,
        feature_cols=feature_cols,
        target_col=target_col,
        sequence_length=sequence_length,
    )

    X = sequence_bundle['X_raw']
    y = sequence_bundle['y_raw']
    y_price = sequence_bundle['y_price']
    prev_close = sequence_bundle['prev_close']
    y_dates = sequence_bundle['dates']

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


def prepare_full_training_data(data, feature_cols, target_col='LogReturn', sequence_length=60):
    sequence_bundle = build_sequence_dataset(
        data,
        feature_cols=feature_cols,
        target_col=target_col,
        sequence_length=sequence_length,
    )

    X_all_raw = sequence_bundle['X_raw']
    y_all_raw = sequence_bundle['y_raw']

    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    X_all_2d = X_all_raw.reshape(-1, X_all_raw.shape[-1])
    feature_scaler.fit(X_all_2d)
    X_all_scaled = feature_scaler.transform(X_all_2d).reshape(X_all_raw.shape)

    target_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler.fit(y_all_raw)
    y_all_scaled = target_scaler.transform(y_all_raw)

    latest_feature_window = np.array(data[feature_cols].tail(sequence_length).values).reshape(1, sequence_length, len(feature_cols))
    latest_feature_window_scaled = feature_scaler.transform(
        latest_feature_window.reshape(-1, latest_feature_window.shape[-1])
    ).reshape(latest_feature_window.shape)

    return {
        'X_all': X_all_scaled,
        'y_all': y_all_scaled,
        'feature_scaler': feature_scaler,
        'target_scaler': target_scaler,
        'latest_feature_window': latest_feature_window_scaled,
        'last_close': float(data['Close'].iloc[-1]),
    }