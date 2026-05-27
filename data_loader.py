import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta


def fetch_data(ticker: str = "BTC-USD", days: int = 720, interval: str = "1h") -> pd.DataFrame:
    """Fetch hourly OHLCV data for the specified ticker and look-back window.

    Yahoo Finance's 1h endpoint enforces a hard 730-day rolling limit;
    using 720 days keeps us safely inside that boundary.
    """
    end = datetime.now()
    start = end - timedelta(days=days)

    df = yf.download(
        ticker,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        interval=interval,
        progress=False,
    )

    # yfinance returns MultiIndex columns (price_type, ticker) — keep only price_type
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Drop any duplicate columns (e.g. 'Adj Close') and keep standard OHLCV
    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[keep].dropna()

    if df.empty:
        raise RuntimeError(
            f"yfinance returned no data for {ticker} ({interval}, last {days} days). "
            "Check your internet connection or try again later."
        )

    df.index = pd.to_datetime(df.index)

    # Normalise to timezone-naive so downstream code doesn't hit tz errors
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    return df
