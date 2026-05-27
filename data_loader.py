import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta


def _download(ticker, start, end, interval):
    df = yf.download(ticker, start=start, end=end, interval=interval, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[keep].dropna()
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df


def fetch_data(ticker: str = "BTC-USD", interval: str = "1h") -> pd.DataFrame:
    """Fetch hourly OHLCV — tries progressively shorter windows until data arrives."""
    end = datetime.utcnow()
    for days in [700, 600, 500, 365]:
        start = end - timedelta(days=days)
        df = _download(ticker, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), interval)
        if not df.empty:
            return df
    raise RuntimeError(f"yfinance returned no hourly data for {ticker}. Check connection.")


def fetch_daily(ticker: str = "BTC-USD") -> pd.DataFrame:
    """Fetch daily OHLCV for the daily trend filter."""
    end   = datetime.utcnow()
    start = end - timedelta(days=720)
    df = _download(ticker, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), "1d")
    return df  # empty is handled gracefully in backtester


def fetch_fear_greed() -> pd.Series:
    """Fetch Fear & Greed Index from alternative.me — returns empty Series on failure."""
    try:
        import urllib.request
        import json
        url = "https://api.alternative.me/fng/?limit=720&format=json"
        with urllib.request.urlopen(url, timeout=10) as r:
            data = json.loads(r.read())["data"]
        records = {
            pd.Timestamp.fromtimestamp(int(d["timestamp"])).normalize(): int(d["value"])
            for d in data
        }
        return pd.Series(records, name="fear_greed").sort_index()
    except Exception:
        return pd.Series(dtype=float, name="fear_greed")
