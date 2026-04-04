from datetime import datetime
import pandas as pd
import yfinance as yf

def load_data(ticker, start="2015-01-01", end=None):
    if end is None:
        end = datetime.today().strftime('%Y-%m-%d')
    data = yf.download(ticker, start=start, end=end)

    # yfinance can return MultiIndex columns even for a single ticker.
    # Flatten to standard OHLCV columns so downstream code always gets Series.
    if isinstance(data.columns, pd.MultiIndex):
        unique_tickers = data.columns.get_level_values(-1).unique()
        if len(unique_tickers) == 1:
            data.columns = data.columns.get_level_values(0)

    return data