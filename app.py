import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from data_loader import load_data
from preprocessing import add_features, prepare_lstm_data
from models import build_lstm
from train import (
    train_lstm,
    predict_lstm,
    returns_to_price,
    evaluate_forecast,
    baseline_predictions,
    walk_forward_cv_rmse,
)


def extract_close_series(data):
    close = data['Close']
    if hasattr(close, 'ndim') and close.ndim > 1:
        close = close.iloc[:, 0]
    return close.dropna()


@st.cache_data(ttl=3600)
def get_cached_data(ticker, start, end):
    return load_data(ticker, start=start, end=end)


@st.cache_data(ttl=3600)
def get_cached_features(data):
    return add_features(data.copy())


def find_swing_levels(series, window=3, tail_points=8):
    highs = []
    lows = []
    values = series.values
    idx = series.index

    for i in range(window, len(series) - window):
        segment = values[i - window:i + window + 1]
        center = values[i]
        if center == np.max(segment):
            highs.append((idx[i], center))
        if center == np.min(segment):
            lows.append((idx[i], center))

    recent_highs = [v for _, v in highs[-tail_points:]]
    recent_lows = [v for _, v in lows[-tail_points:]]

    resistance = float(np.mean(recent_highs)) if recent_highs else float(series.tail(20).max())
    support = float(np.mean(recent_lows)) if recent_lows else float(series.tail(20).min())
    return support, resistance


def analyze_chart(data):
    close = extract_close_series(data)

    if len(close) < 20:
        return {
            "trend": "Not enough data to analyze the chart meaningfully.",
            "support": None,
            "resistance": None,
            "breakout": "Need at least 20 data points to estimate breakout levels.",
            "breakout_status": "range",
            "breakout_level": None,
        }

    recent_60 = close.tail(min(60, len(close)))
    recent_20 = close.tail(20)

    weekly_close = close.resample('W').last().dropna()
    weekly_support = float(weekly_close.tail(26).min()) if len(weekly_close) >= 5 else float(recent_20.min())
    weekly_resistance = float(weekly_close.tail(26).max()) if len(weekly_close) >= 5 else float(recent_20.max())

    swing_support, swing_resistance = find_swing_levels(close, window=3, tail_points=8)

    short_ma = close.rolling(window=20).mean().iloc[-1]
    long_ma = close.rolling(window=50).mean().iloc[-1] if len(close) >= 50 else close.rolling(window=max(5, len(close))).mean().iloc[-1]

    start_price = recent_60.iloc[0]
    end_price = recent_60.iloc[-1]
    slope_pct = ((end_price - start_price) / start_price) * 100 if start_price != 0 else 0

    if short_ma > long_ma and slope_pct > 2:
        trend_text = f"The chart is in an uptrend for the selected timeline. The 20-day average is above the longer trend and price has risen about {slope_pct:.1f}% over this period."
    elif short_ma < long_ma and slope_pct < -2:
        trend_text = f"The chart is in a downtrend for the selected timeline. The 20-day average is below the longer trend and price has fallen about {abs(slope_pct):.1f}% over this period."
    else:
        trend_text = f"The chart looks sideways to mildly directional in the selected timeline. Price change is about {slope_pct:.1f}% and the trend has not broken clearly either way."

    support_level = float(np.mean([swing_support, weekly_support]))
    resistance_level = float(np.mean([swing_resistance, weekly_resistance]))
    current_price = float(close.iloc[-1])
    breakout_status = "range"
    breakout_level = resistance_level

    if current_price >= resistance_level * 1.01:
        breakout_text = f"Price is above the recent resistance near ₹{resistance_level:.2f}, which suggests a breakout may already be in progress."
        breakout_status = "breakout"
    elif current_price >= resistance_level * 0.98:
        breakout_text = f"Price is testing resistance near ₹{resistance_level:.2f}. A close above that level could confirm a breakout."
        breakout_status = "testing"
    elif current_price <= support_level * 0.99:
        breakout_text = f"Price is below the recent support near ₹{support_level:.2f}, which points to a possible breakdown."
        breakout_status = "breakdown"
        breakout_level = support_level
    else:
        breakout_text = f"The nearest resistance is around ₹{resistance_level:.2f} and support is around ₹{support_level:.2f}. The chart is still inside this range."

    return {
        "trend": trend_text,
        "support": support_level,
        "resistance": resistance_level,
        "breakout": breakout_text,
        "breakout_status": breakout_status,
        "breakout_level": breakout_level,
    }


def style_time_axis(ax, date_index):
    if len(date_index) < 2:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        return

    start = date_index.min()
    end = date_index.max()
    total_months = ((end.year - start.year) * 12) + (end.month - start.month) + 1

    if total_months > 120:
        locator = mdates.YearLocator(base=1)
        formatter = mdates.DateFormatter("%Y")
    elif total_months > 48:
        locator = mdates.MonthLocator(interval=6)
        formatter = mdates.DateFormatter("%b %Y")
    elif total_months > 18:
        locator = mdates.MonthLocator(interval=3)
        formatter = mdates.DateFormatter("%b %Y")
    else:
        locator = mdates.MonthLocator(interval=1)
        formatter = mdates.DateFormatter("%b %Y")

    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

# ==============================
# 🎯 TITLE
# ==============================
st.title("📈 Nifty 50 Stock Price Predictor (Multivariate LSTM)")
st.write("Select any Nifty 50 company and predict trends using Close, MA50, MA100, and RSI")

# ==============================
# 📊 STOCK OPTIONS
# ==============================
stock_options = {
    "Adani Enterprises": "ADANIENT.NS",
    "Adani Ports & SEZ": "ADANIPORTS.NS",
    "Apollo Hospitals": "APOLLOHOSP.NS",
    "Asian Paints": "ASIANPAINT.NS",
    "Axis Bank": "AXISBANK.NS",
    "Bajaj Auto": "BAJAJ-AUTO.NS",
    "Bajaj Finance": "BAJFINANCE.NS",
    "Bajaj Finserv": "BAJAJFINSV.NS",
    "Bharat Electronics": "BEL.NS",
    "Bharti Airtel": "BHARTIARTL.NS",
    "Cipla": "CIPLA.NS",
    "Coal India": "COALINDIA.NS",
    "Dr. Reddy's Laboratories": "DRREDDY.NS",
    "Eicher Motors": "EICHERMOT.NS",
    "Eternal": "ETERNAL.NS",
    "Grasim Industries": "GRASIM.NS",
    "HCLTech": "HCLTECH.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "HDFC Life": "HDFCLIFE.NS",
    "Hindalco Industries": "HINDALCO.NS",
    "Hindustan Unilever": "HINDUNILVR.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "IndiGo": "INDIGO.NS",
    "Infosys": "INFY.NS",
    "ITC": "ITC.NS",
    "Jio Financial Services": "JIOFIN.NS",
    "JSW Steel": "JSWSTEEL.NS",
    "Kotak Mahindra Bank": "KOTAKBANK.NS",
    "Larsen & Toubro": "LT.NS",
    "Mahindra & Mahindra": "M&M.NS",
    "Maruti Suzuki": "MARUTI.NS",
    "Max Healthcare": "MAXHEALTH.NS",
    "Nestle India": "NESTLEIND.NS",
    "NTPC": "NTPC.NS",
    "Oil and Natural Gas Corporation": "ONGC.NS",
    "Power Grid": "POWERGRID.NS",
    "Reliance Industries": "RELIANCE.NS",
    "SBI Life Insurance Company": "SBILIFE.NS",
    "Shriram Finance": "SHRIRAMFIN.NS",
    "State Bank of India": "SBIN.NS",
    "Sun Pharma": "SUNPHARMA.NS",
    "Tata Consultancy Services": "TCS.NS",
    "Tata Consumer Products": "TATACONSUM.NS",
    "Tata Motors Passenger Vehicles": "TMPV.NS",
    "Tata Steel": "TATASTEEL.NS",
    "Tech Mahindra": "TECHM.NS",
    "Titan Company": "TITAN.NS",
    "Trent": "TRENT.NS",
    "UltraTech Cement": "ULTRACEMCO.NS",
    "Wipro": "WIPRO.NS",
}

selected_stock = st.selectbox(
    "Select Nifty 50 Company",
    list(stock_options.keys()),
    index=None,
    placeholder="Choose a Nifty 50 company"
)

if not selected_stock:
    st.info("Select a company to view price summary, chart breakdown, and predictions.")
    st.stop()

ticker = stock_options[selected_stock]

st.caption("Covered companies: all current Nifty 50 constituents")

from datetime import datetime

# Date selectors
from datetime import datetime

today = datetime.today()

start_date = st.date_input(
    "Start Date",
    value=datetime(2015, 1, 1),
    max_value=today
)

end_date = today.date()
st.caption(f"End Date is fixed to current date: {end_date.strftime('%d %b %Y')}")

if start_date >= end_date:
    st.error("Start date must be before current date")
    st.stop()

# ==============================
# 📥 LOAD DATA
# ==============================
data = get_cached_data(ticker, start=start_date, end=end_date)
raw_data = data.copy()
full_history = get_cached_data(ticker, start="1900-01-01", end=today)

if raw_data.empty or extract_close_series(raw_data).empty:
    st.error("No price data found for the selected timeline. Please choose an earlier start date.")
    st.stop()

chart_insight = analyze_chart(raw_data)

full_close = extract_close_series(full_history)
selected_close = extract_close_series(raw_data)

if full_close.empty:
    st.error("Unable to fetch full historical data for this stock at the moment. Please try again.")
    st.stop()

all_time_high_date = full_close.idxmax()
all_time_high = float(full_close.max())
all_time_low_date = full_close.idxmin()
all_time_low = float(full_close.min())

selected_high_date = selected_close.idxmax()
selected_high = float(selected_close.max())
selected_low_date = selected_close.idxmin()
selected_low = float(selected_close.min())

last_52_weeks = full_close.tail(252)
fifty_two_week_high_date = last_52_weeks.idxmax()
fifty_two_week_high = float(last_52_weeks.max())
fifty_two_week_low_date = last_52_weeks.idxmin()
fifty_two_week_low = float(last_52_weeks.min())

st.subheader("Price Summary")
summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)

summary_col1.metric(
    "All-Time High",
    f"₹{all_time_high:.2f}",
    f"▲ {all_time_high_date.strftime('%d %b %Y')}"
)
summary_col2.metric(
    "All-Time Low",
    f"₹{all_time_low:.2f}",
    f"▼ {all_time_low_date.strftime('%d %b %Y')}",
    delta_color="inverse"
)
summary_col3.metric(
    "Selected Range High",
    f"₹{selected_high:.2f}",
    f"▲ {selected_high_date.strftime('%d %b %Y')}"
)
summary_col4.metric(
    "Selected Range Low",
    f"₹{selected_low:.2f}",
    f"▼ {selected_low_date.strftime('%d %b %Y')}",
    delta_color="inverse"
)

week_col1, week_col2 = st.columns(2)

week_col1.metric(
    "52-Week High",
    f"₹{fifty_two_week_high:.2f}",
    f"▲ {fifty_two_week_high_date.strftime('%d %b %Y')}"
)
week_col2.metric(
    "52-Week Low",
    f"₹{fifty_two_week_low:.2f}",
    f"▼ {fifty_two_week_low_date.strftime('%d %b %Y')}",
    delta_color="inverse"
)

st.subheader("Data & Model Health")
health_col1, health_col2, health_col3, health_col4 = st.columns(4)
health_col1.metric("Rows Downloaded", f"{len(raw_data)}")
health_col2.metric("Date Span", f"{raw_data.index.min().strftime('%d %b %Y')} to {raw_data.index.max().strftime('%d %b %Y')}")
health_col3.metric("Selected Timeline Days", f"{(raw_data.index.max() - raw_data.index.min()).days}")
health_col4.metric("Close Data Points", f"{len(selected_close)}")

# ==============================
# 🧠 PREPROCESS
# ==============================
processed_data = get_cached_features(data)

feature_cols = [
    'Close', 'MA20', 'MA50', 'MA100', 'EMA20', 'RSI',
    'ATR14', 'BollingerUpper', 'BollingerLower', 'BollingerWidth',
    'VolumeChange', 'OBV', 'Volatility20'
]

if len(processed_data) < 61:
    st.warning(
        "Not enough historical data to train the model for this start date. "
        "Please select an earlier start date so at least 61 processed rows are available."
    )
    st.stop()

lstm_data = prepare_lstm_data(
    processed_data,
    feature_cols=feature_cols,
    target_col='LogReturn',
    sequence_length=60,
    train_ratio=0.8,
)

X_train, X_test = lstm_data['X_train'], lstm_data['X_test']
y_train, y_test = lstm_data['y_train'], lstm_data['y_test']
target_scaler = lstm_data['target_scaler']
y_test_price = lstm_data['y_test_price']
prev_close_test = lstm_data['prev_close_test']
dates_test = lstm_data['dates_test']

health_pre_col1, health_pre_col2, health_pre_col3 = st.columns(3)
health_pre_col1.metric("Rows After Features", f"{len(processed_data)}")
health_pre_col2.metric("Train Sequences", f"{len(X_train)}")
health_pre_col3.metric("Test Sequences", f"{len(X_test)}")

training_context = f"{ticker}_{start_date}_{end_date}"
if st.session_state.get("trained_context") != training_context:
    st.session_state.pop("trained_outputs", None)
    st.session_state["trained_context"] = training_context

# ==============================
# 🔘 TRAIN BUTTON
# ==============================
if st.button("Train Model"):

    with st.spinner("Training model... ⏳"):
        model = build_lstm(X_train.shape[1:])
        model, history = train_lstm(model, X_train, y_train)

        pred_returns = predict_lstm(model, X_test, target_scaler)
        pred_price = returns_to_price(pred_returns, prev_close_test)
        actual_price = y_test_price

        lstm_metrics = evaluate_forecast(actual_price, pred_price, prev_close_test)

        cv_result = walk_forward_cv_rmse(build_lstm, X_train, y_train, target_scaler, n_splits=3)

        baseline = baseline_predictions(X_train, y_train, X_test, prev_close_test)
        lr_returns = target_scaler.inverse_transform(baseline['linear_return'])
        rf_returns = target_scaler.inverse_transform(baseline['rf_return'])
        lr_price = returns_to_price(lr_returns, prev_close_test)
        rf_price = returns_to_price(rf_returns, prev_close_test)

        baseline_metrics = {
            'Naive (Last Close)': evaluate_forecast(actual_price, baseline['naive_close'], prev_close_test),
            'Linear Regression': evaluate_forecast(actual_price, lr_price, prev_close_test),
            'Random Forest': evaluate_forecast(actual_price, rf_price, prev_close_test),
            'LSTM': lstm_metrics,
        }

    st.success("Model trained successfully!")

    # Next-day prediction from most recent scaled feature window
    latest_feature_window = np.array(processed_data[feature_cols].tail(60).values).reshape(1, 60, len(feature_cols))
    latest_feature_window_scaled = lstm_data['feature_scaler'].transform(
        latest_feature_window.reshape(-1, latest_feature_window.shape[-1])
    ).reshape(latest_feature_window.shape)
    next_day_return = predict_lstm(model, latest_feature_window_scaled, target_scaler)
    next_day_price = returns_to_price(next_day_return, [[float(processed_data['Close'].iloc[-1])]])

    st.session_state["trained_outputs"] = {
        "rmse": float(lstm_metrics['rmse']),
        "mae": float(lstm_metrics['mae']),
        "mape": float(lstm_metrics['mape']),
        "directional_accuracy": float(lstm_metrics['directional_accuracy']),
        "current_price": float(selected_close.iloc[-1]),
        "next_price": float(next_day_price[0][0]),
        "dates": dates_test,
        "actual_price": actual_price,
        "pred_price": pred_price,
        "residuals": lstm_metrics['residuals'],
        "baseline_metrics": baseline_metrics,
        "cv_result": cv_result,
        "history": history.history,
    }

trained_outputs = st.session_state.get("trained_outputs")

if trained_outputs:

    # Display metrics
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Current Price", f"₹{trained_outputs['current_price']:.2f}")
    col2.metric("Predicted Next Day", f"₹{trained_outputs['next_price']:.2f}")
    col3.metric("RMSE", f"{trained_outputs['rmse']:.2f}")
    col4.metric("Direction Accuracy", f"{trained_outputs['directional_accuracy']:.1f}%")

    metric_col1, metric_col2 = st.columns(2)
    metric_col1.metric("MAE", f"{trained_outputs['mae']:.2f}")
    metric_col2.metric("MAPE", f"{trained_outputs['mape']:.2f}%")
    st.caption("Prediction target: next-day close price inferred from predicted log-return")

    # ==============================
    # 📉 GRAPH WITH REAL DATES
    # ==============================
    st.subheader(f"{selected_stock} Stock Price Predictor")
    st.caption("Model input: 60-step multivariate window with trend, momentum, volatility, and volume features")

    fig, ax = plt.subplots(figsize=(14, 5.8))

    ax.plot(
        trained_outputs["dates"],
        trained_outputs["actual_price"],
        label="Actual Price",
        color="#1f77b4",
        linewidth=2.2,
    )
    ax.plot(
        trained_outputs["dates"],
        trained_outputs["pred_price"],
        label="LSTM Predicted Price",
        color="#ff7f0e",
        linewidth=2.2,
        alpha=0.9,
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("Stock Price (INR)")
    ax.set_title(f"{selected_stock} Stock Price Predictor")
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.35)
    style_time_axis(ax, trained_outputs["dates"])
    fig.autofmt_xdate(rotation=45)
    ax.legend(loc="upper left", frameon=False)
    st.pyplot(fig, clear_figure=True)

    st.subheader("Model Diagnostics")
    residual_fig, residual_ax = plt.subplots(figsize=(14, 3.8))
    residual_ax.plot(trained_outputs["dates"], trained_outputs["residuals"], color="#9333ea", linewidth=1.4)
    residual_ax.axhline(0, color="#111827", linestyle="--", linewidth=1)
    residual_ax.set_title("Residuals (Actual - Predicted)")
    residual_ax.set_xlabel("Date")
    residual_ax.set_ylabel("Residual")
    residual_ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.3)
    style_time_axis(residual_ax, trained_outputs["dates"])
    residual_fig.autofmt_xdate(rotation=45)
    st.pyplot(residual_fig, clear_figure=True)

    baseline_rows = []
    for model_name, metrics in trained_outputs['baseline_metrics'].items():
        baseline_rows.append({
            'Model': model_name,
            'RMSE': round(metrics['rmse'], 3),
            'MAE': round(metrics['mae'], 3),
            'MAPE (%)': round(metrics['mape'], 3),
            'Direction Accuracy (%)': round(metrics['directional_accuracy'], 2),
        })
    baseline_df = pd.DataFrame(baseline_rows).sort_values('RMSE')
    st.subheader("Baseline Comparison")
    st.dataframe(baseline_df, use_container_width=True, hide_index=True)

    cv = trained_outputs['cv_result']
    st.subheader("Walk-Forward Validation")
    cv_cols = st.columns(len(cv['fold_rmse']) + 1)
    for idx, score in enumerate(cv['fold_rmse'], start=1):
        cv_cols[idx - 1].metric(f"Fold {idx} RMSE", f"{score:.2f}")
    cv_cols[-1].metric("Mean RMSE", f"{cv['mean_rmse']:.2f}")

    history = trained_outputs['history']
    if history.get('loss'):
        health_train_col1, health_train_col2 = st.columns(2)
        health_train_col1.metric("Epochs Trained", f"{len(history['loss'])}")
        if history.get('val_loss'):
            health_train_col2.metric("Best Val Loss", f"{min(history['val_loss']):.6f}")

    st.subheader("Chart Breakdown")

    breakdown_fig, breakdown_ax = plt.subplots(figsize=(14, 5.5))
    breakdown_ax.plot(
        raw_data.index,
        selected_close,
        label="Close Price",
        color="#2563eb",
        linewidth=2.0,
    )

    trendline_20 = selected_close.rolling(window=20).mean()
    trendline_50 = selected_close.rolling(window=50).mean()
    breakdown_ax.plot(
        raw_data.index,
        trendline_20,
        label="Trendline (20)",
        color="#7c3aed",
        linewidth=1.6,
        alpha=0.9,
    )
    breakdown_ax.plot(
        raw_data.index,
        trendline_50,
        label="Trendline (50)",
        color="#0f766e",
        linewidth=1.6,
        alpha=0.9,
    )

    if chart_insight["support"] is not None:
        breakdown_ax.axhline(
            y=chart_insight["support"],
            color="green",
            linestyle="--",
            linewidth=1.5,
            label=f"Support ₹{chart_insight['support']:.2f}"
        )
    if chart_insight["resistance"] is not None:
        breakdown_ax.axhline(
            y=chart_insight["resistance"],
            color="red",
            linestyle="--",
            linewidth=1.5,
            label=f"Resistance ₹{chart_insight['resistance']:.2f}"
        )

    latest_date = raw_data.index[-1]
    latest_price = float(selected_close.iloc[-1])
    status = chart_insight.get("breakout_status")
    level = chart_insight.get("breakout_level")

    if level is not None:
        breakdown_ax.axhline(
            y=level,
            color="#f59e0b",
            linestyle="-.",
            linewidth=1.5,
            alpha=0.95,
            label=f"Breakout Level ₹{level:.2f}",
        )

    if status in {"breakout", "breakdown", "testing"} and level is not None:
        if status == "breakout":
            signal_color = "#16a34a"
            signal_label = f"Breakout above ₹{level:.2f}"
        elif status == "breakdown":
            signal_color = "#dc2626"
            signal_label = f"Breakdown below ₹{level:.2f}"
        else:
            signal_color = "#d97706"
            signal_label = f"Testing breakout near ₹{level:.2f}"

        breakdown_ax.scatter(
            [latest_date],
            [latest_price],
            color=signal_color,
            s=70,
            zorder=5,
            label=signal_label,
        )
        breakdown_ax.axhline(
            y=level,
            color=signal_color,
            linestyle=":",
            linewidth=1.4,
            alpha=0.9,
        )

    breakdown_ax.set_xlabel("Date")
    breakdown_ax.set_ylabel("Stock Price (INR)")
    breakdown_ax.set_title(f"{selected_stock} Chart Breakdown")
    breakdown_ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.35)
    style_time_axis(breakdown_ax, raw_data.index)
    breakdown_fig.autofmt_xdate(rotation=45)
    breakdown_ax.legend(loc="upper left", frameon=False)
    st.pyplot(breakdown_fig, clear_figure=True)

    st.caption(f"Analysis timeline: {start_date.strftime('%d %b %Y')} to {end_date.strftime('%d %b %Y')}")
    st.write(chart_insight["trend"])
    st.write(chart_insight["breakout"])