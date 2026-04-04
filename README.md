# Stock Price Predictor

A Streamlit app that downloads historical stock data, builds technical features, and trains an LSTM model to predict the next price movement for selected Indian stocks.

## Features

- Select from TCS, Infosys, Reliance, and HDFC Bank
- Choose a custom date range
- Train an LSTM model on historical prices
- View actual vs predicted prices and RMSE
- Generate a next-day price prediction

## Project Structure

- `app.py` - Streamlit interface for selecting stocks, dates, and training the model
- `data_loader.py` - Downloads market data with yfinance
- `preprocessing.py` - Adds features, scales data, and creates sequences
- `models.py` - Defines the LSTM model
- `train.py` - Trains the model and produces predictions
- `main.py` - Simple script entry point for local plotting

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the App

Start the Streamlit interface with:

```bash
streamlit run app.py
```

## Notes

- The app relies on live data from Yahoo Finance.
- The model uses only the `Close` price for scaling and sequence generation.
- Training can take a short time depending on the selected date range and machine performance.
