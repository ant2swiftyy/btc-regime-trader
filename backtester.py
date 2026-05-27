"""
Regime-based backtester — v2 (enhanced indicators)

Pipeline:
  1. Compute 3 HMM features → train GaussianHMM (7 states)
  2. Auto-label Bull Run / Bear/Crash states by mean return
  3. Compute 12 technical signals (voting system)
     Original 8 : RSI, Momentum, Volatility, Volume SMA,
                  ADX, EMA50, EMA200, MACD
     New 4      : Supertrend, VWAP, Stochastic RSI, OBV trend
  4. Enter when Bull Run + votes ≥ 10/12
  5. Exit when regime = Bear/Crash OR Supertrend turns bearish
  6. 48-hour cooldown after any exit
"""

import warnings
import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

COOLDOWN_HOURS    = 48
LEVERAGE          = 2.5
N_STATES          = 7
VOTE_THRESHOLD    = 9     # out of 12
STOP_LOSS_PCT     = 0.12  # hard stop: exit if price drops 12% from entry
TRAIL_STOP_PCT    = 0.09  # trailing stop: exit if price drops 9% from peak
REGIME_CONFIRM    = 3     # require 3 consecutive Bull Run bars before entry
TIME_STOP_HOURS   = 24    # exit flat/losing trade after 24 hours
PARTIAL_TAKE_PCT  = 0.05  # take 50% profit at +5% gain
FEAR_GREED_MAX    = 60    # skip entry if Fear & Greed > 60 (greed = bad time to buy)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _squeeze(s) -> pd.Series:
    if isinstance(s, pd.DataFrame):
        return s.iloc[:, 0]
    return s


# ─── HMM Features ────────────────────────────────────────────────────────────

def compute_hmm_features(df: pd.DataFrame) -> pd.DataFrame:
    close  = _squeeze(df["Close"])
    high   = _squeeze(df["High"])
    low    = _squeeze(df["Low"])
    volume = _squeeze(df["Volume"])

    returns     = close.pct_change()
    price_range = (high - low) / close.replace(0, np.nan)

    log_vol      = np.log1p(volume.clip(lower=0))
    vol_volatility = log_vol.diff().rolling(24, min_periods=5).std()

    return pd.DataFrame(
        {"returns": returns, "range": price_range, "vol_volatility": vol_volatility}
    )


# ─── Supertrend (recursive — must loop) ──────────────────────────────────────

def _supertrend(high: pd.Series, low: pd.Series, close: pd.Series,
                period: int = 10, multiplier: float = 3.0):
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()

    hl2         = (high + low) / 2
    basic_upper = (hl2 + multiplier * atr).values
    basic_lower = (hl2 - multiplier * atr).values
    cls         = close.values
    n           = len(cls)

    upper      = np.full(n, np.nan)
    lower      = np.full(n, np.nan)
    supertrend = np.full(n, np.nan)
    direction  = np.zeros(n, dtype=int)   # +1 bullish, -1 bearish

    upper[0]      = basic_upper[0]
    lower[0]      = basic_lower[0]
    supertrend[0] = upper[0]
    direction[0]  = -1

    for i in range(1, n):
        upper[i] = (
            basic_upper[i]
            if basic_upper[i] < upper[i - 1] or cls[i - 1] > upper[i - 1]
            else upper[i - 1]
        )
        lower[i] = (
            basic_lower[i]
            if basic_lower[i] > lower[i - 1] or cls[i - 1] < lower[i - 1]
            else lower[i - 1]
        )

        if supertrend[i - 1] == upper[i - 1]:   # was bearish
            if cls[i] <= upper[i]:
                supertrend[i] = upper[i];  direction[i] = -1
            else:
                supertrend[i] = lower[i];  direction[i] =  1
        else:                                    # was bullish
            if cls[i] >= lower[i]:
                supertrend[i] = lower[i];  direction[i] =  1
            else:
                supertrend[i] = upper[i];  direction[i] = -1

    return (
        pd.Series(supertrend, index=close.index, name="supertrend"),
        pd.Series(direction,  index=close.index, name="supertrend_dir"),
    )


# ─── Technical Indicators (12 signals) ───────────────────────────────────────

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close  = _squeeze(df["Close"])
    high   = _squeeze(df["High"])
    low    = _squeeze(df["Low"])
    volume = _squeeze(df["Volume"])

    # ── Original 8 ──────────────────────────────────────────────────────────

    # 1. RSI (14-period Wilder)
    delta    = close.diff()
    avg_gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    avg_loss = (-delta).clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    rsi      = 100 - 100 / (1 + avg_gain / avg_loss.replace(0, np.nan))

    # 2. Momentum (10-bar %)
    momentum = close.pct_change(10) * 100

    # 3. Volatility (24-bar rolling std %)
    volatility = close.pct_change().rolling(24).std() * 100

    # 4. Volume above 20-bar SMA
    volume_above_sma = (volume > volume.rolling(20).mean()).astype(float)

    # 5. EMA 50 / 6. EMA 200
    ema50  = close.ewm(span=50,  adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()

    # 7. MACD (12, 26, 9)
    macd        = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    signal_line = macd.ewm(span=9, adjust=False).mean()

    # 8. ADX (14-period Wilder)
    up_move   = high.diff()
    down_move = -(low.diff())
    plus_dm   = up_move.where((up_move > down_move)   & (up_move > 0),   0.0)
    minus_dm  = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    prev_cls  = close.shift(1)
    tr        = pd.concat(
        [high - low, (high - prev_cls).abs(), (low - prev_cls).abs()], axis=1
    ).max(axis=1)
    atr       = tr.ewm(alpha=1 / 14, adjust=False).mean().replace(0, np.nan)
    plus_di   = 100 * plus_dm.ewm(alpha=1 / 14, adjust=False).mean() / atr
    minus_di  = 100 * minus_dm.ewm(alpha=1 / 14, adjust=False).mean() / atr
    di_sum    = (plus_di + minus_di).replace(0, np.nan)
    adx       = (100 * (plus_di - minus_di).abs() / di_sum).ewm(alpha=1 / 14, adjust=False).mean()

    # ── New 4 ───────────────────────────────────────────────────────────────

    # 9. Supertrend (period=10, multiplier=3)
    _, st_dir = _supertrend(high, low, close, period=10, multiplier=3.0)
    supertrend_bullish = (st_dir == 1).astype(float)

    # 10. VWAP — rolling 24-bar (works on 24/7 crypto without daily reset)
    typical = (high + low + close) / 3
    vwap    = (typical * volume).rolling(24, min_periods=5).sum() / \
               volume.rolling(24, min_periods=5).sum()

    # 11. Stochastic RSI — RSI(14) → Stoch(14) → smooth K(3), D(3)
    rsi_min  = rsi.rolling(14).min()
    rsi_max  = rsi.rolling(14).max()
    stoch_k  = ((rsi - rsi_min) / (rsi_max - rsi_min).replace(0, np.nan)
                ).rolling(3).mean() * 100
    stoch_d  = stoch_k.rolling(3).mean()

    # 12. OBV trend — OBV above its 20-bar EMA
    obv      = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    obv_ema  = obv.ewm(span=20, adjust=False).mean()

    return pd.DataFrame(
        {
            # original
            "rsi":              rsi,
            "momentum":         momentum,
            "volatility":       volatility,
            "volume_above_sma": volume_above_sma,
            "ema50":            ema50,
            "ema200":           ema200,
            "macd":             macd,
            "signal_line":      signal_line,
            "adx":              adx,
            # new
            "supertrend_bullish": supertrend_bullish,
            "vwap":               vwap,
            "stoch_k":            stoch_k,
            "stoch_d":            stoch_d,
            "obv":                obv,
            "obv_ema":            obv_ema,
        },
        index=close.index,
    )


# ─── HMM Training + Regime Labelling ─────────────────────────────────────────

def train_hmm(features_clean: pd.DataFrame):
    scaler = StandardScaler()
    X      = scaler.fit_transform(features_clean.values)
    model  = hmm.GaussianHMM(
        n_components=N_STATES, covariance_type="full",
        n_iter=300, random_state=42, tol=1e-4,
    )
    model.fit(X)
    return model, scaler


def label_regimes(model, scaler, features_clean: pd.DataFrame):
    X          = scaler.transform(features_clean.values)
    raw_states = model.predict(X)

    state_mean_return = {
        s: features_clean["returns"][raw_states == s].mean()
        for s in range(N_STATES)
        if (raw_states == s).any()
    }
    bull_state = max(state_mean_return, key=state_mean_return.get)
    bear_state = min(state_mean_return, key=state_mean_return.get)

    def _label(s):
        if s == bull_state:  return "Bull Run"
        if s == bear_state:  return "Bear/Crash"
        return "Neutral"

    return (
        pd.Series([_label(s) for s in raw_states], index=features_clean.index, name="regime"),
        bull_state, bear_state, state_mean_return,
    )


# ─── Voting System (12 conditions, need ≥ 10) ────────────────────────────────

def count_votes(row: pd.Series, close: float) -> int:
    v = 0
    # original 8
    if not pd.isna(row["rsi"]):         v += int(row["rsi"] < 90)
    if not pd.isna(row["momentum"]):    v += int(row["momentum"] > 1.0)
    if not pd.isna(row["volatility"]):  v += int(row["volatility"] < 6.0)
    v += int(bool(row["volume_above_sma"]))
    if not pd.isna(row["adx"]):         v += int(row["adx"] > 25)
    if not pd.isna(row["ema50"]):       v += int(close > row["ema50"])
    if not pd.isna(row["ema200"]):      v += int(close > row["ema200"])
    if not pd.isna(row["macd"]) and not pd.isna(row["signal_line"]):
        v += int(row["macd"] > row["signal_line"])
    # new 4
    v += int(bool(row["supertrend_bullish"]))                          # 9
    if not pd.isna(row["vwap"]):        v += int(close > row["vwap"]) # 10
    if not pd.isna(row["stoch_k"]) and not pd.isna(row["stoch_d"]):   # 11
        v += int(row["stoch_k"] > row["stoch_d"] and row["stoch_k"] < 80)
    if not pd.isna(row["obv"]) and not pd.isna(row["obv_ema"]):       # 12
        v += int(row["obv"] > row["obv_ema"])
    return v


# ─── Backtest Engine ──────────────────────────────────────────────────────────

def run_backtest(
    df: pd.DataFrame,
    initial_capital: float = 10_000.0,
    leverage: float = LEVERAGE,
    daily_df: pd.DataFrame = None,
    fear_greed: pd.Series = None,
):
    # 1. HMM
    raw_features   = compute_hmm_features(df)
    features_clean = raw_features.dropna()
    model, scaler  = train_hmm(features_clean)
    regime_series, bull_state, bear_state, state_returns = label_regimes(
        model, scaler, features_clean
    )

    # 2. Technical indicators
    indicators = compute_indicators(df)

    # 3. Daily trend filter: price above daily EMA50
    daily_trend = pd.Series(dtype=float)
    if daily_df is not None and not daily_df.empty:
        dc = _squeeze(daily_df["Close"])
        daily_ema50 = dc.ewm(span=50, adjust=False).mean()
        daily_trend = (dc > daily_ema50).astype(int)

    # 4. Align to common valid index
    valid_idx      = regime_series.index.intersection(indicators.dropna(how="any").index)
    regime_aligned = regime_series[valid_idx]
    ind_aligned    = indicators.loc[valid_idx]
    df_aligned     = df.loc[valid_idx]

    # 5. Simulation
    capital       = initial_capital
    position      = 0
    entry_price   = entry_time = entry_capital = None
    peak_price    = None
    partial_taken = False
    half_capital  = 0.0
    last_exit_time = None
    bull_streak    = 0

    trades            = []
    portfolio_records = []

    for idx in valid_idx:
        row    = ind_aligned.loc[idx]
        regime = regime_aligned[idx]
        close  = float(_squeeze(df_aligned.loc[[idx], "Close"]).iloc[0])
        date   = pd.Timestamp(idx).normalize()

        bull_streak = bull_streak + 1 if regime == "Bull Run" else 0

        if position == 1 and close > peak_price:
            peak_price = close

        # Live portfolio value
        if position == 1:
            active_cap    = entry_capital / 2 if partial_taken else entry_capital
            pnl_factor    = (close - entry_price) / entry_price * leverage
            current_value = max(active_cap * (1 + pnl_factor) + half_capital, 0.0)
        else:
            current_value = capital

        votes = count_votes(row, close)

        # ── Partial profit take at +PARTIAL_TAKE_PCT ────────────────────────
        if position == 1 and not partial_taken:
            if (close - entry_price) / entry_price >= PARTIAL_TAKE_PCT:
                locked = (entry_capital / 2) * (1 + (close - entry_price) / entry_price * leverage)
                half_capital  = locked
                partial_taken = True

        # ── Exit conditions ──────────────────────────────────────────────────
        exit_reason = None
        if position == 1:
            drop_from_entry = (close - entry_price) / entry_price
            drop_from_peak  = (close - peak_price)  / peak_price
            hours_held      = (idx - entry_time).total_seconds() / 3600
            if regime == "Bear/Crash":
                exit_reason = "Regime: Bear/Crash"
            elif drop_from_entry <= -STOP_LOSS_PCT:
                exit_reason = "Stop Loss"
            elif drop_from_peak <= -TRAIL_STOP_PCT:
                exit_reason = "Trailing Stop"
            elif hours_held >= TIME_STOP_HOURS and drop_from_entry <= 0:
                exit_reason = "Time Stop (24h)"

        if exit_reason:
            active_cap = entry_capital / 2 if partial_taken else entry_capital
            pnl_factor = (close - entry_price) / entry_price * leverage
            new_cap    = max(active_cap * (1 + pnl_factor) + half_capital, 0.0)
            trade_pnl  = new_cap - entry_capital
            capital    = entry_capital + trade_pnl
            trades.append({
                "entry_time":    entry_time,
                "exit_time":     idx,
                "entry_price":   entry_price,
                "exit_price":    close,
                "pnl_pct":       trade_pnl / entry_capital * 100,
                "pnl_dollar":    trade_pnl,
                "exit_reason":   exit_reason,
                "partial_taken": partial_taken,
            })
            position = 0
            entry_price = entry_time = entry_capital = peak_price = None
            partial_taken = False
            half_capital  = 0.0
            last_exit_time = idx
            current_value  = capital

        # ── Entry condition ──────────────────────────────────────────────────
        can_enter = (
            position == 0
            and regime == "Bull Run"
            and bull_streak >= REGIME_CONFIRM
            and votes >= VOTE_THRESHOLD
        )
        if can_enter:
            # Daily trend filter
            if not daily_trend.empty:
                dval = daily_trend.reindex([date], method="ffill")
                if dval.empty or dval.iloc[0] == 0:
                    can_enter = False

            # Fear & Greed filter
            if can_enter and fear_greed is not None and not fear_greed.empty:
                fval = fear_greed.reindex([date], method="ffill")
                if not fval.empty and not pd.isna(fval.iloc[0]):
                    if fval.iloc[0] > FEAR_GREED_MAX:
                        can_enter = False

        if can_enter:
            in_cooldown = False
            if last_exit_time is not None:
                in_cooldown = (idx - last_exit_time).total_seconds() / 3600 < COOLDOWN_HOURS
            if not in_cooldown:
                position      = 1
                entry_price   = close
                peak_price    = close
                entry_time    = idx
                entry_capital = capital
                partial_taken = False
                half_capital  = 0.0

        portfolio_records.append({
            "time":   idx,
            "value":  current_value,
            "regime": regime,
            "signal": "Long" if position == 1 else "Cash",
            "votes":  votes,
            "close":  close,
        })

    # Force-close at end of data
    if position == 1:
        close      = float(_squeeze(df_aligned.iloc[[-1]]["Close"]).iloc[0])
        active_cap = entry_capital / 2 if partial_taken else entry_capital
        pnl_factor = (close - entry_price) / entry_price * leverage
        new_cap    = max(active_cap * (1 + pnl_factor) + half_capital, 0.0)
        trade_pnl  = new_cap - entry_capital
        capital    = entry_capital + trade_pnl
        trades.append({
            "entry_time":    entry_time,
            "exit_time":     valid_idx[-1],
            "entry_price":   entry_price,
            "exit_price":    close,
            "pnl_pct":       trade_pnl / entry_capital * 100,
            "pnl_dollar":    trade_pnl,
            "exit_reason":   "End of Data",
            "partial_taken": partial_taken,
        })

    # 5. Build outputs
    portfolio_df = pd.DataFrame(portfolio_records).set_index("time")
    _cols        = ["entry_time","exit_time","entry_price","exit_price",
                    "pnl_pct","pnl_dollar","exit_reason"]
    trades_df    = pd.DataFrame(trades) if trades else pd.DataFrame(columns=_cols)

    # 6. Metrics
    total_return_pct = (capital - initial_capital) / initial_capital * 100
    bh_start         = float(_squeeze(df_aligned.iloc[[0]]["Close"]).iloc[0])
    bh_end           = float(_squeeze(df_aligned.iloc[[-1]]["Close"]).iloc[0])
    bh_return_pct    = (bh_end - bh_start) / bh_start * 100
    alpha            = total_return_pct - bh_return_pct
    win_rate         = float((trades_df["pnl_pct"] > 0).mean() * 100) if len(trades_df) > 0 else 0.0
    pv               = portfolio_df["value"]
    max_drawdown     = float(((pv - pv.cummax()) / pv.cummax() * 100).min())

    metrics = {
        "total_return": total_return_pct,
        "bh_return":    bh_return_pct,
        "alpha":        alpha,
        "win_rate":     win_rate,
        "max_drawdown": max_drawdown,
        "num_trades":   len(trades_df),
        "final_capital": capital,
    }
    return portfolio_df, trades_df, metrics, regime_series, df_aligned
