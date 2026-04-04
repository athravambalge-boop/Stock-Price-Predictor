from datetime import datetime
import yfinance as yf

def load_data(ticker, start="2015-01-01", end=None):
    if end is None:
        end = datetime.today().strftime('%Y-%m-%d')
    return yf.download(ticker, start=start, end=end)