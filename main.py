from data_loader import load_data
from preprocessing import add_features, prepare_lstm_data
from models import build_lstm
from train import train_lstm, predict_lstm, returns_to_price, evaluate_forecast, set_global_seed
import matplotlib.pyplot as plt


FEATURE_COLS = [
    'Close', 'MA20', 'MA50', 'MA100', 'EMA20', 'EMA50', 'RSI',
    'ATR14', 'BollingerUpper', 'BollingerLower', 'BollingerWidth',
    'MACD', 'MACD_Signal', 'MACD_Histogram',
    'Stochastic_K', 'Stochastic_D',
    'VolumeChange', 'OBV', 'OBV_EMA', 'Volatility20',
    'Close_Lag1', 'Close_Lag5', 'Return_Lag1', 'Return_Lag5',
    'Williams_R', 'CCI'
]

def main():
    set_global_seed(42)

    data = load_data("TCS.NS")
    data = add_features(data)
    prepared = prepare_lstm_data(
        data,
        feature_cols=FEATURE_COLS,
        target_col='LogReturn',
        sequence_length=60,
        train_ratio=0.8,
    )

    X_train, X_test = prepared['X_train'], prepared['X_test']
    y_train = prepared['y_train']
    target_scaler = prepared['target_scaler']
    y_test_price = prepared['y_test_price']
    prev_close_test = prepared['prev_close_test']
    dates_test = prepared['dates_test']

    model = build_lstm(X_train.shape[1:], seed=42)
    model, _ = train_lstm(model, X_train, y_train)

    pred_returns = predict_lstm(model, X_test, target_scaler)
    pred_price = returns_to_price(pred_returns, prev_close_test)
    actual_price = y_test_price
    metrics = evaluate_forecast(actual_price, pred_price, prev_close_test)
    print(
        f"RMSE={metrics['rmse']:.3f}, MAE={metrics['mae']:.3f}, "
        f"MAPE={metrics['mape']:.3f}%, Direction={metrics['directional_accuracy']:.2f}%"
    )

    plt.figure(figsize=(12, 5))
    plt.plot(dates_test, actual_price, label="Actual")
    plt.plot(dates_test, pred_price, label="Predicted")
    plt.legend()
    plt.title("TCS.NS Price Forecast")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()