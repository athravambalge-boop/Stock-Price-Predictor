import numpy as np
from sklearn.preprocessing import MinMaxScaler

def add_features(data):
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data['MA100'] = data['Close'].rolling(window=100).mean()

    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))

    data.dropna(inplace=True)
    return data

def scale_data(data):
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled = scaler.fit_transform(data[['Close']])
    return scaled, scaler

def create_sequences(data):
    X, y = [], []
    for i in range(60, len(data)):
        X.append(data[i-60:i])
        y.append(data[i])
    return np.array(X), np.array(y)