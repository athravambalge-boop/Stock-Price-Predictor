from datetime import datetime
import time
import pandas as pd
import yfinance as yf

def _normalize_ohlcv_columns(data):
    # yfinance can return MultiIndex columns even for a single ticker.
    if isinstance(data.columns, pd.MultiIndex):
        unique_tickers = data.columns.get_level_values(-1).unique()
        if len(unique_tickers) == 1:
            data.columns = data.columns.get_level_values(0)

    expected = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in expected:
        if col not in data.columns:
            data[col] = pd.NA

    return data


def load_data(ticker, start="2015-01-01", end=None, retries=3, backoff_seconds=1.0):
    if end is None:
        end = datetime.today().strftime('%Y-%m-%d')

    last_error = None
    for attempt in range(retries):
        try:
            data = yf.download(
                ticker,
                start=start,
                end=end,
                auto_adjust=False,
                progress=False,
                threads=False,
            )
            if data is None or data.empty:
                return pd.DataFrame()
            return _normalize_ohlcv_columns(data)
        except Exception as exc:
            last_error = exc
            if attempt < retries - 1:
                time.sleep(backoff_seconds * (2 ** attempt))

    if last_error:
        raise last_error
    return pd.DataFrame()