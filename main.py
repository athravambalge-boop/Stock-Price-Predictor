from data_loader import load_data
from preprocessing import add_features, scale_data, create_sequences
from models import build_lstm
from train import train_lstm, predict_lstm
import matplotlib.pyplot as plt
import numpy as np

def main():
    data = load_data("TCS.NS")
    data = add_features(data)

    scaled, scaler = scale_data(data)
    X, y = create_sequences(scaled)

    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = build_lstm((X_train.shape[1], 1))
    model = train_lstm(model, X_train, y_train)

    pred = predict_lstm(model, X_test, scaler)
    actual = scaler.inverse_transform(y_test.reshape(-1,1))

    plt.plot(actual, label="Actual")
    plt.plot(pred, label="Predicted")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()