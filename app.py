import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from data_loader import load_data
from preprocessing import add_features, scale_data, create_sequences
from models import build_lstm
from train import train_lstm, predict_lstm

from sklearn.metrics import mean_squared_error

# ==============================
# 🎯 TITLE
# ==============================
st.title("📈 Stock Price Predictor (LSTM)")
st.write("Select a stock and predict trends")

# ==============================
# 📊 STOCK OPTIONS
# ==============================
stock_options = {
    "TCS": "TCS.NS",
    "Infosys": "INFY.NS",
    "Reliance": "RELIANCE.NS",
    "HDFC Bank": "HDFCBANK.NS"
}

selected_stock = st.selectbox("Select Stock", list(stock_options.keys()))
ticker = stock_options[selected_stock]

from datetime import datetime

# Date selectors
from datetime import datetime

today = datetime.today()

start_date = st.date_input(
    "Start Date",
    value=datetime(2015, 1, 1),
    max_value=today
)

end_date = st.date_input(
    "End Date",
    value=today,
    max_value=today
)

if start_date >= end_date:
    st.error("Start date must be before end date")
    st.stop()

# ==============================
# 📥 LOAD DATA
# ==============================
data = load_data(ticker, start=start_date, end=end_date)

# ==============================
# 🧠 PREPROCESS
# ==============================
data = add_features(data)
scaled, scaler = scale_data(data)
X, y = create_sequences(scaled)

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# ==============================
# 🔘 TRAIN BUTTON
# ==============================
if st.button("Train Model"):

    with st.spinner("Training model... ⏳"):
        model = build_lstm((X_train.shape[1], 1))
        model = train_lstm(model, X_train, y_train)

        # Predictor outputs
        pred = predict_lstm(model, X_test, scaler)
        actual = scaler.inverse_transform(y_test.reshape(-1,1))

    st.success("Model trained successfully!")

    # ==============================
    # 📊 METRICS
    # ==============================
    rmse = np.sqrt(mean_squared_error(actual, pred))
    current_price = float(data['Close'].iloc[-1])

    # Next day predictor value
    last_60_days = scaled[-60:]
    last_60_days = np.reshape(last_60_days, (1, 60, 1))

    next_day_pred = model.predict(last_60_days)
    next_day_pred = scaler.inverse_transform(next_day_pred)

    # Display metrics
    col1, col2, col3 = st.columns(3)

    col1.metric("Current Price", f"₹{current_price:.2f}")
    next_price = float(next_day_pred[0][0])
    col2.metric("Predicted Next Day", f"₹{next_price:.2f}")
    col3.metric("RMSE", f"{rmse:.2f}")

    # ==============================
    # 📉 GRAPH WITH REAL DATES
    # ==============================
    st.subheader(f"{selected_stock} Stock Price Predictor")

    dates = data.index[-len(y_test):]

    fig, ax = plt.subplots()

    ax.plot(dates, actual, label="Actual Price")
    ax.plot(dates, pred, label="Predicted Price")

    ax.set_xlabel("Date")
    ax.set_ylabel("Stock Price (INR)")
    ax.set_title(f"{selected_stock} Stock Price Predictor")

    plt.xticks(rotation=45)
    ax.legend()

    st.pyplot(fig)