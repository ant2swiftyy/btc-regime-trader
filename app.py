"""
BTC Regime Trader — Streamlit Dashboard
========================================
Layout
  • Top bar : Signal badge | Regime badge | Voting score | 4 key metrics
  • Main chart : Candlestick + regime-coloured background + trade markers
  • Portfolio chart : Strategy equity curve vs Buy-and-Hold
  • Trade log table
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from backtester import run_backtest
from data_loader import fetch_data

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BTC Regime Trader",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
/* ── General ── */
[data-testid="stAppViewContainer"] { background: #0e1117; }
[data-testid="stHeader"]           { background: transparent; }

/* ── Signal / Regime badges ── */
.badge {
    display: inline-block;
    padding: 10px 28px;
    border-radius: 30px;
    font-size: 1.3rem;
    font-weight: 800;
    letter-spacing: 1.5px;
    margin-top: 6px;
}
.badge-long    { background:#0a1f14; color:#00e676; border:2px solid #00e676; }
.badge-cash    { background:#1f0a0a; color:#ff5252; border:2px solid #ff5252; }
.badge-bull    { background:#0a1f14; color:#00e676; border:1px solid #00e676; }
.badge-bear    { background:#1f0a0a; color:#ff5252; border:1px solid #ff5252; }
.badge-neutral { background:#12121f; color:#7c83fd; border:1px solid #7c83fd; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: #12141f;
    border: 1px solid #252840;
    border-radius: 10px;
    padding: 14px 18px;
}

/* ── Section headers ── */
.section-header {
    font-size: 1rem;
    font-weight: 600;
    color: #8b95b8;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 4px;
}

/* ── Divider ── */
hr { border-color: #252840 !important; }
</style>
""",
    unsafe_allow_html=True,
)

# ─── Data loading (cached 1 h) ────────────────────────────────────────────────
INITIAL_CAPITAL = 10_000.0


@st.cache_data(ttl=3600, show_spinner=False)
def _load():
    df = fetch_data()
    return run_backtest(df, initial_capital=INITIAL_CAPITAL)


# ─── Helpers ─────────────────────────────────────────────────────────────────
REGIME_FILL = {
    "Bull Run":   "rgba(0,230,118,0.07)",
    "Bear/Crash": "rgba(255,82,82,0.09)",
    "Neutral":    "rgba(124,131,253,0.04)",
}


def _regime_vrects(fig, regime_series: pd.Series, row=None, col=None):
    """Add coloured vertical bands for each regime period."""
    prev, t0 = None, None
    for t, r in regime_series.items():
        key = "Neutral" if r.startswith("Neutral") else r
        if key != prev:
            if prev is not None:
                kw = dict(x0=t0, x1=t, fillcolor=REGIME_FILL.get(prev, "rgba(80,80,80,0.03)"),
                          opacity=1, layer="below", line_width=0)
                if row:
                    kw.update(row=row, col=col)
                fig.add_vrect(**kw)
            prev, t0 = key, t
    if prev is not None:
        kw = dict(x0=t0, x1=regime_series.index[-1],
                  fillcolor=REGIME_FILL.get(prev, "rgba(80,80,80,0.03)"),
                  opacity=1, layer="below", line_width=0)
        if row:
            kw.update(row=row, col=col)
        fig.add_vrect(**kw)


def _dark_layout(fig, height=500):
    fig.update_layout(
        template="plotly_dark",
        height=height,
        margin=dict(l=0, r=0, t=10, b=0),
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font=dict(color="#b0b8d8"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_rangeslider_visible=False,
    )


# ─── Load data ───────────────────────────────────────────────────────────────
with st.spinner("Fetching BTC-USD data and training HMM (7 states)…"):
    portfolio_df, trades_df, metrics, regime_series, df_aligned = _load()

latest = portfolio_df.iloc[-1]
cur_signal = latest["signal"]
cur_regime = latest["regime"]
cur_votes = int(latest["votes"])

# ─── Header ──────────────────────────────────────────────────────────────────
st.markdown("## BTC Regime Trader")
st.markdown(
    "_GaussianHMM · 7 states · 8-factor voting · 2.5× leverage · 48-hour cooldown_"
)
st.divider()

# ─── Top row: Signal | Regime | 4 Metrics ────────────────────────────────────
top_l, top_m, top_r = st.columns([1.2, 1.2, 3.6])

with top_l:
    st.markdown('<p class="section-header">Current Signal</p>', unsafe_allow_html=True)
    sc = "badge-long" if cur_signal == "Long" else "badge-cash"
    st.markdown(f'<div class="badge {sc}">{cur_signal}</div>', unsafe_allow_html=True)

with top_m:
    st.markdown('<p class="section-header">Market Regime</p>', unsafe_allow_html=True)
    rc = "badge-bull" if cur_regime == "Bull Run" else ("badge-bear" if cur_regime == "Bear/Crash" else "badge-neutral")
    st.markdown(f'<div class="badge {rc}">{cur_regime}</div>', unsafe_allow_html=True)
    st.caption(f"Voting score: **{cur_votes}/8** signals active")

with top_r:
    m1, m2, m3, m4 = st.columns(4)
    ret_delta = f"{metrics['alpha']:+.1f}% vs B&H"
    m1.metric("Total Return", f"{metrics['total_return']:.1f}%", delta=ret_delta)
    m2.metric("Alpha vs B&H", f"{metrics['alpha']:+.1f}%")
    m3.metric("Win Rate", f"{metrics['win_rate']:.1f}%")
    m4.metric("Max Drawdown", f"{metrics['max_drawdown']:.1f}%")

st.divider()

# ─── Candlestick chart ───────────────────────────────────────────────────────
st.markdown('<p class="section-header">BTC-USD · Hourly · Regime Background</p>', unsafe_allow_html=True)

days_options = [30, 60, 90, 180, 365, 730]
chart_window = st.select_slider(
    "Chart window (days)",
    options=days_options,
    value=90,
    format_func=lambda x: f"{x}d",
)

cutoff = df_aligned.index[-1] - pd.Timedelta(days=chart_window)
cdf = df_aligned[df_aligned.index >= cutoff]
c_regime = regime_series[regime_series.index.isin(cdf.index)]

fig_candle = go.Figure()
_regime_vrects(fig_candle, c_regime)

fig_candle.add_trace(
    go.Candlestick(
        x=cdf.index,
        open=cdf["Open"],
        high=cdf["High"],
        low=cdf["Low"],
        close=cdf["Close"],
        name="BTC-USD",
        increasing_line_color="#00e676",
        decreasing_line_color="#ff5252",
        increasing_fillcolor="#00e676",
        decreasing_fillcolor="#ff5252",
    )
)

# Trade markers
if not trades_df.empty:
    visible = trades_df[trades_df["exit_time"] >= cdf.index[0]]
    entries = visible[visible["entry_time"] >= cdf.index[0]]
    if not entries.empty:
        fig_candle.add_trace(
            go.Scatter(
                x=entries["entry_time"],
                y=entries["entry_price"],
                mode="markers",
                marker=dict(symbol="triangle-up", color="#00e676", size=11, line=dict(color="#fff", width=0.5)),
                name="Entry (Long)",
                hovertemplate="<b>BUY</b><br>%{x}<br>$%{y:,.0f}<extra></extra>",
            )
        )
    if not visible.empty:
        colors = ["#00e676" if p > 0 else "#ff5252" for p in visible["pnl_pct"]]
        fig_candle.add_trace(
            go.Scatter(
                x=visible["exit_time"],
                y=visible["exit_price"],
                mode="markers",
                marker=dict(symbol="triangle-down", color=colors, size=11, line=dict(color="#fff", width=0.5)),
                name="Exit",
                hovertemplate="<b>SELL</b><br>%{x}<br>$%{y:,.0f}<extra></extra>",
                customdata=visible["pnl_pct"].values,
            )
        )

_dark_layout(fig_candle, height=520)
fig_candle.update_xaxis(title_text="")
fig_candle.update_yaxis(title_text="Price (USD)", tickprefix="$")
st.plotly_chart(fig_candle, use_container_width=True)

# ─── Equity curve ────────────────────────────────────────────────────────────
st.markdown('<p class="section-header">Portfolio Equity · Strategy vs Buy-and-Hold</p>', unsafe_allow_html=True)

bh_start = float(df_aligned["Close"].iloc[0])
bh_values = INITIAL_CAPITAL * (df_aligned["Close"] / bh_start)

fig_eq = go.Figure()
_regime_vrects(fig_eq, regime_series)

fig_eq.add_trace(
    go.Scatter(
        x=portfolio_df.index,
        y=portfolio_df["value"],
        name="HMM Strategy (2.5× lev.)",
        line=dict(color="#00e676", width=2),
        fill="tozeroy",
        fillcolor="rgba(0,230,118,0.05)",
    )
)
fig_eq.add_trace(
    go.Scatter(
        x=df_aligned.index,
        y=bh_values,
        name="Buy & Hold",
        line=dict(color="#7c83fd", width=1.5, dash="dot"),
    )
)
_dark_layout(fig_eq, height=280)
fig_eq.update_yaxis(title_text="Portfolio Value ($)", tickprefix="$", tickformat=",.0f")
st.plotly_chart(fig_eq, use_container_width=True)

# ─── Summary stats row ───────────────────────────────────────────────────────
st.divider()
s1, s2, s3, s4, s5, s6 = st.columns(6)
s1.metric("Starting Capital",    f"${INITIAL_CAPITAL:,.0f}")
s2.metric("Final Capital",       f"${metrics['final_capital']:,.0f}")
s3.metric("Total Trades",        metrics["num_trades"])
s4.metric("Buy & Hold Return",   f"{metrics['bh_return']:.1f}%")
s5.metric("Strategy Return",     f"{metrics['total_return']:.1f}%")
s6.metric("Alpha",               f"{metrics['alpha']:+.1f}%")

# ─── Trade log ───────────────────────────────────────────────────────────────
st.divider()
st.markdown(f'<p class="section-header">Trade Log ({metrics["num_trades"]} completed trades)</p>', unsafe_allow_html=True)

if not trades_df.empty:
    disp = trades_df.copy()
    disp["entry_time"] = pd.to_datetime(disp["entry_time"]).dt.strftime("%Y-%m-%d %H:%M")
    disp["exit_time"]  = pd.to_datetime(disp["exit_time"]).dt.strftime("%Y-%m-%d %H:%M")
    disp["entry_price"] = disp["entry_price"].map("${:,.2f}".format)
    disp["exit_price"]  = disp["exit_price"].map("${:,.2f}".format)
    disp["pnl_pct"]     = disp["pnl_pct"].map("{:+.2f}%".format)
    disp["pnl_dollar"]  = disp["pnl_dollar"].map("${:+,.2f}".format)
    disp.columns = ["Entry Time", "Exit Time", "Entry $", "Exit $", "PnL %", "PnL $", "Reason"]
    st.dataframe(disp, use_container_width=True, hide_index=True)
else:
    st.info("No completed trades in the backtest window. The strategy requires Bull Run regime + 7/8 signals aligned simultaneously.")

# ─── Legend ──────────────────────────────────────────────────────────────────
st.divider()
leg1, leg2, leg3 = st.columns(3)
leg1.markdown("🟢 **Green background** — Bull Run regime")
leg2.markdown("🔴 **Red background** — Bear / Crash regime")
leg3.markdown("⚪ **Grey background** — Neutral regime")

st.caption(
    "⚠️ Simulation only. 2.5× leverage is applied to PnL, not margin. "
    "Past performance does not guarantee future results. Not financial advice."
)
