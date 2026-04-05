# Stock Price Predictor

Streamlit app for Nifty 50 stocks that combines multivariate LSTM forecasting, baseline model comparison, chart breakdown analysis, and model diagnostics.

## What The App Does

- Lets you select any current Nifty 50 stock.
- Uses user-selected start date with end date fixed to the current date.
- Downloads market data from Yahoo Finance.
- Builds technical features (trend, momentum, volatility, and volume).
- Trains an LSTM model to predict next-step log-return, then converts predictions back to price.
- Compares LSTM against baseline models.
- Displays chart breakdown with trendlines, support, resistance, and breakout level.
- Shows a unified market-impact news feed for the selected stock.

## Key Features

- Nifty 50 stock picker (no default preselection).
- Price Summary panel:
	- All-Time High/Low with date
	- Selected Range High/Low with date
	- 52-Week High/Low with date
- Data and Model Health panel:
	- Rows downloaded
	- Date span and timeline size
	- Rows after feature engineering
	- Train/test sequence counts
- Leakage-safe preprocessing:
	- Scalers fit only on training data
- Training reliability:
	- EarlyStopping
	- ReduceLROnPlateau
	- ModelCheckpoint
- Validation and diagnostics:
	- Walk-forward validation (TimeSeriesSplit)
	- RMSE, MAE, MAPE, directional accuracy
	- Residual chart
- Baseline comparison table:
	- Naive (last close)
	- Linear Regression
	- Random Forest
	- LSTM
- Market-impact news panel:
	- Today-only headline filter
	- Multi-source aggregation (Yahoo Finance, Moneycontrol, Google News RSS)
	- Impact tags (Geopolitics, Politics, Macro, Commodities, Corporate)
	- Timestamp display in IST

## Market News Feed

The app includes a "Today's Market-Moving News" section that prioritizes headlines likely to affect markets.

- Aggregates from:
	- Yahoo Finance ticker feed
	- Moneycontrol RSS feeds
	- Google News RSS market searches
- Filters to today's items and removes duplicates across sources.
- Prioritizes headlines using impact keywords (for example war, geopolitics, policy, interest rates, oil, inflation, and earnings).
- Displays news times in IST.

## Feature Engineering

The app computes these indicators before training:

- MA20, MA50, MA100, EMA20
- RSI (14)
- ATR (14)
- Bollinger Upper/Lower/Width
- VolumeChange
- OBV
- Volatility20

## Chart Breakdown

The breakdown chart includes:

- Close price
- Trendline (20)
- Trendline (50)
- Support line
- Resistance line
- Breakout level line
- Breakout/Breakdown/Testing marker

## Project Structure

- `app.py`: Streamlit UI and end-to-end workflow
- `data_loader.py`: Yahoo Finance downloader and column normalization
- `preprocessing.py`: Feature engineering and leakage-safe sequence prep
- `models.py`: LSTM architecture
- `train.py`: Training utilities, evaluation, baselines, walk-forward CV
- `main.py`: local script entry point

## Setup

1. Create and activate a Python virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

## Notes

- Data quality and availability depend on Yahoo Finance.
- If timeline is too short, training is blocked with a guided warning.
- `best_lstm.keras` is a generated checkpoint file and should not be committed.
