import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import quote_plus
from urllib.request import Request, urlopen
from xml.etree import ElementTree as ET


IST = timezone.utc
try:
    # Fixed offset for India Standard Time (UTC+05:30).
    from datetime import timedelta
    IST = timezone(timedelta(hours=5, minutes=30))
except Exception:
    pass

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


def parse_datetime_safe(value):
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(value, tz=timezone.utc)
        except Exception:
            return None

    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return None

        try:
            parsed = parsedate_to_datetime(cleaned)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        except Exception:
            pass

        if cleaned.endswith("Z"):
            cleaned = cleaned.replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(cleaned)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        except Exception:
            return None

    return None


def format_datetime_utc(dt_obj):
    if not dt_obj:
        return "Time not available"
    return dt_obj.astimezone(IST).strftime("%d %b %Y, %H:%M IST")


def news_impact_score(headline, company_name, ticker_symbol):
    text = (headline or "").lower()

    high_impact_terms = [
        "war", "missile", "attack", "ceasefire", "sanction", "geopolitical",
        "election", "parliament", "congress", "government", "policy", "tariff",
        "federal reserve", "fed", "rbi", "interest rate", "inflation", "cpi",
        "opec", "oil", "crude", "recession", "gdp", "unemployment",
    ]
    medium_impact_terms = [
        "market", "stocks", "equity", "bond", "currency", "rupee", "dollar",
        "earnings", "guidance", "merger", "acquisition", "regulator", "sec",
        "outlook", "forecast", "exports", "imports", "trade",
    ]

    score = 0
    score += sum(3 for term in high_impact_terms if term in text)
    score += sum(1 for term in medium_impact_terms if term in text)

    company_terms = [
        token.lower()
        for token in company_name.replace("&", " ").replace("-", " ").split()
        if len(token) > 2
    ]
    ticker_root = (ticker_symbol or "").replace(".NS", "").strip().lower()

    if ticker_root and ticker_root in text:
        score += 2
    if any(term in text for term in company_terms):
        score += 2

    return score


def market_impact_tags(headline):
    text = (headline or "").lower()
    tags = []

    if any(word in text for word in ["war", "attack", "missile", "conflict", "ceasefire", "sanction", "geopolitical"]):
        tags.append("Geopolitics")
    if any(word in text for word in ["election", "government", "parliament", "policy", "tariff", "political"]):
        tags.append("Politics")
    if any(word in text for word in ["fed", "rbi", "interest rate", "inflation", "cpi", "central bank"]):
        tags.append("Macro")
    if any(word in text for word in ["oil", "crude", "opec", "gas"]):
        tags.append("Commodities")
    if any(word in text for word in ["earnings", "guidance", "results", "profit"]):
        tags.append("Corporate")

    if not tags:
        tags.append("Markets")

    return tags


@st.cache_data(ttl=1800)
def get_cached_news(ticker_symbol, limit=8):
    try:
        news_items = yf.Ticker(ticker_symbol).news or []
    except Exception:
        return []

    parsed_items = []
    for item in news_items:
        content = item.get("content") if isinstance(item.get("content"), dict) else {}

        title = (
            item.get("title")
            or content.get("title")
            or content.get("description")
            or "Untitled headline"
        )

        link = item.get("link") or content.get("clickThroughUrl")
        if isinstance(link, dict):
            link = link.get("url")
        if not link:
            canonical = item.get("canonicalUrl") or content.get("canonicalUrl")
            if isinstance(canonical, dict):
                link = canonical.get("url")

        publisher = (
            item.get("publisher")
            or (content.get("provider", {}) or {}).get("displayName")
            or "Unknown source"
        )

        publish_time_raw = (
            item.get("providerPublishTime")
            or content.get("pubDate")
            or content.get("displayTime")
        )
        parsed_dt = parse_datetime_safe(publish_time_raw)
        published_at = format_datetime_utc(parsed_dt)

        related_tickers = item.get("relatedTickers") or content.get("tickers")
        if not isinstance(related_tickers, list):
            related_tickers = []

        parsed_items.append(
            {
                "title": title,
                "link": link,
                "publisher": publisher,
                "published_at": published_at,
                "published_dt": parsed_dt,
                "related_tickers": related_tickers,
            }
        )

        if len(parsed_items) >= limit:
            break

    return parsed_items


@st.cache_data(ttl=1800)
def get_moneycontrol_news(company_name, ticker_symbol, limit=8):
    # Moneycontrol provides RSS feeds; fetch market/business streams and filter by selected stock.
    feed_urls = [
        "https://www.moneycontrol.com/rss/latestnews.xml",
        "https://www.moneycontrol.com/rss/business.xml",
        "https://www.moneycontrol.com/rss/marketreports.xml",
    ]

    query_terms = [t.lower() for t in company_name.replace("&", " ").replace("-", " ").split() if len(t) > 2]
    ticker_root = (ticker_symbol or "").replace(".NS", "").strip().lower()
    if ticker_root:
        query_terms.append(ticker_root)

    parsed_items = []
    seen_links = set()

    for feed_url in feed_urls:
        try:
            request = Request(feed_url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(request, timeout=8) as response:
                xml_payload = response.read()
            root = ET.fromstring(xml_payload)
        except Exception:
            continue

        for node in root.findall(".//item"):
            title = (node.findtext("title") or "").strip()
            link = (node.findtext("link") or "").strip()
            raw_pubdate = (node.findtext("pubDate") or "").strip()
            parsed_dt = parse_datetime_safe(raw_pubdate)
            published_at = format_datetime_utc(parsed_dt)

            if not title or not link or link in seen_links:
                continue

            text_blob = f"{title} {link}".lower()
            relevance = any(term in text_blob for term in query_terms)

            parsed_items.append(
                {
                    "title": title,
                    "link": link,
                    "publisher": "Moneycontrol",
                    "published_at": published_at,
                    "published_dt": parsed_dt,
                    "related_tickers": [ticker_symbol] if relevance else [],
                    "relevance": relevance,
                }
            )
            seen_links.add(link)

    parsed_items.sort(
        key=lambda x: (
            x["relevance"],
            x.get("published_dt") or datetime(1970, 1, 1, tzinfo=timezone.utc),
        ),
        reverse=True,
    )
    return parsed_items[:limit]


@st.cache_data(ttl=900)
def get_global_market_news(company_name, ticker_symbol, limit=24):
    search_queries = [
        f"{company_name} stock market",
        "global stock market today",
        "war and geopolitics impact on markets",
        "political news impact on markets",
        "oil crude and commodities market moves",
        "central bank inflation interest rates markets",
        "india politics and markets",
    ]

    feed_urls = [
        f"https://news.google.com/rss/search?q={quote_plus(query + ' when:1d')}&hl=en-IN&gl=IN&ceid=IN:en"
        for query in search_queries
    ]

    parsed_items = []
    seen_links = set()

    for feed_url in feed_urls:
        try:
            request = Request(feed_url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(request, timeout=10) as response:
                xml_payload = response.read()
            root = ET.fromstring(xml_payload)
        except Exception:
            continue

        for node in root.findall(".//item"):
            title = (node.findtext("title") or "").strip()
            link = (node.findtext("link") or "").strip()
            publisher = (node.findtext("source") or "Google News").strip() or "Google News"
            parsed_dt = parse_datetime_safe(node.findtext("pubDate") or "")

            if not title or not link or link in seen_links:
                continue

            parsed_items.append(
                {
                    "title": title,
                    "link": link,
                    "publisher": publisher,
                    "published_at": format_datetime_utc(parsed_dt),
                    "published_dt": parsed_dt,
                    "related_tickers": [],
                }
            )
            seen_links.add(link)

    parsed_items.sort(
        key=lambda x: x.get("published_dt") or datetime(1970, 1, 1, tzinfo=timezone.utc),
        reverse=True,
    )
    return parsed_items[:limit]


@st.cache_data(ttl=900)
def get_market_impact_news(company_name, ticker_symbol, limit=20, only_today=True):
    yahoo_news = get_cached_news(ticker_symbol, limit=12)
    moneycontrol_news = get_moneycontrol_news(company_name, ticker_symbol, limit=12)
    global_news = get_global_market_news(company_name, ticker_symbol, limit=36)

    all_news = yahoo_news + moneycontrol_news + global_news

    local_now = datetime.now().astimezone()
    today_local = local_now.date()

    deduped = {}
    for item in all_news:
        title = (item.get("title") or "").strip()
        link = (item.get("link") or "").strip()
        if not title:
            continue

        published_dt = item.get("published_dt")
        if only_today and published_dt is not None:
            if published_dt.astimezone(local_now.tzinfo).date() != today_local:
                continue

        key = (title.lower(), link)
        score = news_impact_score(title, company_name, ticker_symbol)
        tags = market_impact_tags(title)

        enriched = {
            "title": title,
            "link": link,
            "publisher": item.get("publisher", "Unknown source"),
            "published_at": item.get("published_at", "Time not available"),
            "published_dt": published_dt,
            "related_tickers": item.get("related_tickers", []),
            "impact_score": score,
            "impact_tags": tags,
        }

        existing = deduped.get(key)
        if not existing or enriched["impact_score"] > existing["impact_score"]:
            deduped[key] = enriched

    merged = list(deduped.values())
    merged.sort(
        key=lambda x: (
            x["impact_score"],
            x.get("published_dt") or datetime(1970, 1, 1, tzinfo=timezone.utc),
        ),
        reverse=True,
    )

    return merged[:limit]


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
st.warning(
    "Disclaimer: This app is for educational and informational use only. "
    "It does not provide financial advice, investment recommendations, or guarantees. "
    "Market data and news may be delayed or incomplete. Always do your own research before investing."
)

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

st.subheader("Today's Market-Moving News")
market_news = get_market_impact_news(selected_stock, ticker, limit=20, only_today=True)

if market_news:
    st.caption(
        "Sources: Yahoo Finance, Moneycontrol, and Google News market searches. "
        "Feed is filtered to today and prioritized by market impact (war, geopolitics, politics, macro, commodities, and company-specific updates)."
    )
    for news in market_news:
        headline = news["title"]
        link = news["link"]
        publisher = news["publisher"]
        published_at = news["published_at"]
        related_tickers = news["related_tickers"]
        tags = news.get("impact_tags", [])

        if link:
            st.markdown(f"- [{headline}]({link})")
        else:
            st.markdown(f"- {headline}")

        metadata = f"{publisher} • {published_at}"
        if tags:
            metadata += f" • Tags: {', '.join(tags[:3])}"
        if related_tickers:
            metadata += f" • Related: {', '.join(related_tickers[:5])}"
        st.caption(metadata)
else:
    st.info("No market-impact headlines from today are available right now. Please check again later.")

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