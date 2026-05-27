"""
Regime-based backtester.

Pipeline:
  1. Compute 3 HMM features  → train GaussianHMM (7 states)
  2. Auto-label Bull Run / Bear/Crash states by mean return
  3. Compute 8 technical-indicator signals (voting system)
  4. Simulate entries/exits with 48-hour cooldown + 2.5x leverage
"""

import warnings
import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

COOLDOWN_HOURS = 48
LEVERAGE = 2.5
N_STATES = 7


# ─── Feature Engineering ─────────────────────────────────────────────────────

def _squeeze(s) -> pd.Series:
    """Return a plain 1-D Series regardless of DataFrame/Series input."""
    if isinstance(s, pd.DataFrame):
        return s.iloc[:, 0]
    return s


def compute_hmm_features(df: pd.DataFrame) -> pd.DataFrame:
    close = _squeeze(df["Close"])
    high = _squeeze(df["High"])
    low = _squeeze(df["Low"])
    volume = _squeeze(df["Volume"])

    returns = close.pct_change()
    price_range = (high - low) / close.replace(0, np.nan)

    # Log-volume difference avoids NaN from zero-volume bars (pct_change blows up on 0).
    # min_periods=5 prevents sparse NaN cascade across the 24-bar window.
    log_vol = np.log1p(volume.clip(lower=0))
    vol_volatility = log_vol.diff().rolling(24, min_periods=5).std()

    features = pd.DataFrame(
        {"returns": returns, "range": price_range, "vol_volatility": vol_volatility}
    )
    return features


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close = _squeeze(df["Close"])
    high = _squeeze(df["High"])
    low = _squeeze(df["Low"])
    volume = _squeeze(df["Volume"])

    # RSI — Wilder smoothing (alpha = 1/14)
    delta = close.diff()
    avg_gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    avg_loss = (-delta).clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)

    # Momentum — 10-bar percent change
    momentum = close.pct_change(10) * 100

    # Volatility — 24-bar rolling std of returns (as %)
    volatility = close.pct_change().rolling(24).std() * 100

    # Volume vs 20-bar SMA
    volume_above_sma = (volume > volume.rolling(20).mean()).astype(float)

    # EMA 50 / 200
    ema50 = close.ewm(span=50, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()

    # MACD (12, 26, 9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal_line = macd.ewm(span=9, adjust=False).mean()

    # ADX (14-period Wilder)
    up_move = high.diff()
    down_move = -(low.diff())
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean().replace(0, np.nan)
    plus_di = 100 * plus_dm.ewm(alpha=1 / 14, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1 / 14, adjust=False).mean() / atr
    di_sum = (plus_di + minus_di).replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / di_sum
    adx = dx.ewm(alpha=1 / 14, adjust=False).mean()

    return pd.DataFrame(
        {
            "rsi": rsi,
            "momentum": momentum,
            "volatility": volatility,
            "volume_above_sma": volume_above_sma,
            "ema50": ema50,
            "ema200": ema200,
            "macd": macd,
            "signal_line": signal_line,
            "adx": adx,
        },
        index=close.index,
    )


# ─── HMM Training + Regime Labelling ─────────────────────────────────────────

def train_hmm(features_clean: pd.DataFrame):
    scaler = StandardScaler()
    X = scaler.fit_transform(features_clean.values)

    model = hmm.GaussianHMM(
        n_components=N_STATES,
        covariance_type="full",
        n_iter=300,
        random_state=42,
        tol=1e-4,
    )
    model.fit(X)
    return model, scaler


def label_regimes(model, scaler, features_clean: pd.DataFrame):
    """Return a Series mapping each timestamp to its regime label."""
    X = scaler.transform(features_clean.values)
    raw_states = model.predict(X)

    # Mean return per state → identify bull (max) and bear (min) automatically
    state_mean_return = {
        s: features_clean["returns"][raw_states == s].mean()
        for s in range(N_STATES)
        if (raw_states == s).any()
    }
    bull_state = max(state_mean_return, key=state_mean_return.get)
    bear_state = min(state_mean_return, key=state_mean_return.get)

    def _label(s):
        if s == bull_state:
            return "Bull Run"
        if s == bear_state:
            return "Bear/Crash"
        return "Neutral"

    regime_series = pd.Series(
        [_label(s) for s in raw_states],
        index=features_clean.index,
        name="regime",
    )
    return regime_series, bull_state, bear_state, state_mean_return


# ─── Voting System ────────────────────────────────────────────────────────────

def _count_votes(row: pd.Series, close: float) -> int:
    votes = 0
    if not pd.isna(row["rsi"]):
        votes += int(row["rsi"] < 90)
    if not pd.isna(row["momentum"]):
        votes += int(row["momentum"] > 1.0)
    if not pd.isna(row["volatility"]):
        votes += int(row["volatility"] < 6.0)
    votes += int(bool(row["volume_above_sma"]))
    if not pd.isna(row["adx"]):
        votes += int(row["adx"] > 25)
    if not pd.isna(row["ema50"]):
        votes += int(close > row["ema50"])
    if not pd.isna(row["ema200"]):
        votes += int(close > row["ema200"])
    if not pd.isna(row["macd"]) and not pd.isna(row["signal_line"]):
        votes += int(row["macd"] > row["signal_line"])
    return votes


# ─── Backtest Engine ──────────────────────────────────────────────────────────

def run_backtest(
    df: pd.DataFrame,
    initial_capital: float = 10_000.0,
    leverage: float = LEVERAGE,
):
    # 1. HMM feature computation + model training
    raw_features = compute_hmm_features(df)
    features_clean = raw_features.dropna()
    model, scaler = train_hmm(features_clean)
    regime_series, bull_state, bear_state, state_returns = label_regimes(
        model, scaler, features_clean
    )

    # 2. Technical indicators
    indicators = compute_indicators(df)

    # 3. Align to common valid index
    valid_idx = regime_series.index.intersection(indicators.dropna(how="any").index)
    regime_aligned = regime_series[valid_idx]
    ind_aligned = indicators.loc[valid_idx]
    df_aligned = df.loc[valid_idx]

    # 4. Simulation loop
    capital = initial_capital
    position = 0          # 0 = cash, 1 = long
    entry_price = None
    entry_time = None
    entry_capital = None
    last_exit_time = None

    trades = []
    portfolio_records = []

    for idx in valid_idx:
        row = ind_aligned.loc[idx]
        regime = regime_aligned[idx]
        close = float(_squeeze(df_aligned.loc[[idx], "Close"]).iloc[0])

        # Unrealised portfolio value
        if position == 1:
            pnl_factor = (close - entry_price) / entry_price * leverage
            current_value = max(entry_capital * (1 + pnl_factor), 0.0)
        else:
            current_value = capital

        votes = _count_votes(row, close)

        # ── Exit: regime flipped to Bear/Crash ──────────────────────────────
        if position == 1 and regime == "Bear/Crash":
            pnl_factor = (close - entry_price) / entry_price * leverage
            trade_pnl = entry_capital * pnl_factor
            capital = max(entry_capital + trade_pnl, 0.0)

            trades.append(
                {
                    "entry_time": entry_time,
                    "exit_time": idx,
                    "entry_price": entry_price,
                    "exit_price": close,
                    "pnl_pct": pnl_factor * 100,
                    "pnl_dollar": trade_pnl,
                    "exit_reason": "Regime → Bear/Crash",
                }
            )
            position = 0
            entry_price = entry_time = entry_capital = None
            last_exit_time = idx
            current_value = capital

        # ── Entry: Bull Run + votes ≥ 7 + not in cooldown ───────────────────
        if position == 0 and regime == "Bull Run" and votes >= 7:
            in_cooldown = False
            if last_exit_time is not None:
                hours_since = (idx - last_exit_time).total_seconds() / 3600
                in_cooldown = hours_since < COOLDOWN_HOURS

            if not in_cooldown:
                position = 1
                entry_price = close
                entry_time = idx
                entry_capital = capital

        portfolio_records.append(
            {
                "time": idx,
                "value": current_value,
                "regime": regime,
                "signal": "Long" if position == 1 else "Cash",
                "votes": votes,
                "close": close,
            }
        )

    # Force-close open position at end of data
    if position == 1:
        close = float(_squeeze(df_aligned.iloc[[-1]]["Close"]).iloc[0])
        pnl_factor = (close - entry_price) / entry_price * leverage
        trade_pnl = entry_capital * pnl_factor
        capital = max(entry_capital + trade_pnl, 0.0)
        trades.append(
            {
                "entry_time": entry_time,
                "exit_time": valid_idx[-1],
                "entry_price": entry_price,
                "exit_price": close,
                "pnl_pct": pnl_factor * 100,
                "pnl_dollar": trade_pnl,
                "exit_reason": "End of Data",
            }
        )

    # 5. Build output DataFrames
    portfolio_df = pd.DataFrame(portfolio_records).set_index("time")

    _empty_cols = [
        "entry_time", "exit_time", "entry_price", "exit_price",
        "pnl_pct", "pnl_dollar", "exit_reason",
    ]
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame(columns=_empty_cols)

    # 6. Metrics
    total_return_pct = (capital - initial_capital) / initial_capital * 100
    bh_start = float(_squeeze(df_aligned.iloc[[0]]["Close"]).iloc[0])
    bh_end = float(_squeeze(df_aligned.iloc[[-1]]["Close"]).iloc[0])
    bh_return_pct = (bh_end - bh_start) / bh_start * 100
    alpha = total_return_pct - bh_return_pct

    win_rate = 0.0
    if len(trades_df) > 0:
        win_rate = float((trades_df["pnl_pct"] > 0).mean() * 100)

    pv = portfolio_df["value"]
    max_drawdown = float(((pv - pv.cummax()) / pv.cummax() * 100).min())

    metrics = {
        "total_return": total_return_pct,
        "bh_return": bh_return_pct,
        "alpha": alpha,
        "win_rate": win_rate,
        "max_drawdown": max_drawdown,
        "num_trades": len(trades_df),
        "final_capital": capital,
    }

    return portfolio_df, trades_df, metrics, regime_series, df_aligned
