"""
BTC Regime Trader — Robinhood Edition
Beginner-friendly dashboard with clear buy/sell signals.
"""

import datetime as _dt
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from backtester import run_backtest
from data_loader import fetch_data

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BTC Signal — Robinhood Edition",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #0e1117; }
[data-testid="stHeader"]           { background: transparent; }

.signal-box {
    border-radius: 16px;
    padding: 28px 20px;
    text-align: center;
    margin-bottom: 10px;
}
.signal-buy  { background: #0a2e1a; border: 3px solid #00e676; }
.signal-sell { background: #2e0a0a; border: 3px solid #ff5252; }
.signal-wait { background: #1a1a0a; border: 3px solid #ffd600; }

.signal-emoji { font-size: 3rem; }
.signal-title { font-size: 2rem; font-weight: 900; margin: 8px 0 4px; }
.signal-buy  .signal-title { color: #00e676; }
.signal-sell .signal-title { color: #ff5252; }
.signal-wait .signal-title { color: #ffd600; }
.signal-sub   { font-size: 1rem; color: #aaa; }

.step-box {
    background: #12141f;
    border: 1px solid #252840;
    border-radius: 12px;
    padding: 18px 20px;
    margin-bottom: 10px;
}
.step-title { font-weight: 700; font-size: 1.05rem; color: #fff; margin-bottom: 6px; }
.step-body  { color: #b0b8d8; font-size: 0.95rem; line-height: 1.6; }

.why-box {
    background: #12141f;
    border-left: 4px solid #7c83fd;
    border-radius: 0 12px 12px 0;
    padding: 16px 18px;
}

[data-testid="metric-container"] {
    background: #12141f;
    border: 1px solid #252840;
    border-radius: 10px;
    padding: 14px 18px;
}
hr { border-color: #252840 !important; }
</style>
""", unsafe_allow_html=True)

# ─── Data loading ─────────────────────────────────────────────────────────────
INITIAL_CAPITAL = 10_000.0

@st.cache_data(ttl=3600, show_spinner=False)
def _load():
    df = fetch_data()
    # Run backtest at 1x leverage — Robinhood doesn't offer leveraged BTC
    return run_backtest(df, initial_capital=INITIAL_CAPITAL, leverage=1.0)

REGIME_FILL = {
    "Bull Run":   "rgba(0,230,118,0.07)",
    "Bear/Crash": "rgba(255,82,82,0.09)",
    "Neutral":    "rgba(124,131,253,0.04)",
}

def _regime_vrects(fig, regime_series):
    prev, t0 = None, None
    for t, r in regime_series.items():
        key = "Neutral" if r.startswith("Neutral") else r
        if key != prev:
            if prev is not None:
                fig.add_vrect(x0=t0, x1=t,
                    fillcolor=REGIME_FILL.get(prev, "rgba(80,80,80,0.03)"),
                    opacity=1, layer="below", line_width=0)
            prev, t0 = key, t
    if prev is not None:
        fig.add_vrect(x0=t0, x1=regime_series.index[-1],
            fillcolor=REGIME_FILL.get(prev, "rgba(80,80,80,0.03)"),
            opacity=1, layer="below", line_width=0)

# ─── Load ─────────────────────────────────────────────────────────────────────
with st.spinner("Scanning BTC market conditions…"):
    portfolio_df, trades_df, metrics, regime_series, df_aligned = _load()

load_time = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
last_candle = pd.to_datetime(df_aligned.index[-1]).strftime("%Y-%m-%d %H:%M UTC")

latest   = portfolio_df.iloc[-1]
cur_signal = latest["signal"]   # "Long" or "Cash"
cur_regime = latest["regime"]
cur_votes  = int(latest["votes"])
cur_price  = latest["close"]

# ─── Determine action ────────────────────────────────────────────────────────
if cur_signal == "Long":
    action     = "BUY"
    action_emoji = "🟢"
    box_class  = "signal-buy"
    headline   = "NOW IS A GOOD TIME TO BUY BTC"
    subline    = f"The AI model sees a Bull Run + {cur_votes}/8 indicators agree"
elif cur_regime == "Bear/Crash":
    action     = "SELL / STAY OUT"
    action_emoji = "🔴"
    box_class  = "signal-sell"
    headline   = "NOT A GOOD TIME — STAY OUT"
    subline    = "Market is in Bear/Crash mode. If you hold BTC, consider selling."
else:
    action     = "WAIT"
    action_emoji = "🟡"
    box_class  = "signal-wait"
    headline   = "WAIT — NO CLEAR SIGNAL YET"
    subline    = f"Conditions aren't strong enough yet ({cur_votes}/8 indicators). Hold off."

# ─── Header ──────────────────────────────────────────────────────────────────
hcol1, hcol2 = st.columns([5, 1])
with hcol1:
    st.markdown("## 📊 BTC Signal Dashboard — Robinhood Edition")
with hcol2:
    if st.button("🔄 Refresh", help="Force a fresh market scan"):
        st.cache_data.clear()
        st.rerun()

st.caption(f"📡 Last BTC candle: **{last_candle}** · Scanned at: {load_time} · Auto-updates every 60 min")
st.divider()

# ─── MAIN SIGNAL BOX ─────────────────────────────────────────────────────────
st.markdown(f"""
<div class="signal-box {box_class}">
    <div class="signal-emoji">{action_emoji}</div>
    <div class="signal-title">{headline}</div>
    <div class="signal-sub">{subline}</div>
</div>
""", unsafe_allow_html=True)

# ─── Current BTC price + vote bar ────────────────────────────────────────────
p1, p2, p3 = st.columns(3)
p1.metric("BTC Price Right Now", f"${cur_price:,.2f}")
p2.metric("Signal Strength", f"{cur_votes}/8 indicators aligned",
          delta="Strong" if cur_votes >= 7 else ("Moderate" if cur_votes >= 5 else "Weak"))
p3.metric("Market Mood", cur_regime)

st.divider()

# ─── WHAT TO DO ON ROBINHOOD ─────────────────────────────────────────────────
st.markdown("### 📱 What To Do Right Now on Robinhood")

if action == "BUY":
    st.markdown("""
<div class="step-box">
    <div class="step-title">Step 1 — Open Robinhood</div>
    <div class="step-body">Tap the Robinhood app on your phone.</div>
</div>
<div class="step-box">
    <div class="step-title">Step 2 — Search for Bitcoin</div>
    <div class="step-body">Tap the search icon and type <b>BTC</b> or <b>Bitcoin</b>. Tap on it.</div>
</div>
<div class="step-box">
    <div class="step-title">Step 3 — Tap "Buy"</div>
    <div class="step-body">Enter the dollar amount you want to invest. <b>Only use money you're okay losing.</b> Start small — $50–$100 to test.</div>
</div>
<div class="step-box">
    <div class="step-title">Step 4 — Watch this dashboard daily</div>
    <div class="step-body">Come back here every day. The moment the signal turns <b style="color:#ff5252">🔴 SELL / STAY OUT</b>, open Robinhood and sell your BTC.</div>
</div>
""", unsafe_allow_html=True)

elif action == "SELL / STAY OUT":
    st.markdown("""
<div class="step-box">
    <div class="step-title">⚠️ If you currently hold BTC on Robinhood</div>
    <div class="step-body">Open Robinhood → tap your BTC position → tap <b>Sell</b> → sell all or part of it. The market is in a bearish phase.</div>
</div>
<div class="step-box">
    <div class="step-title">If you don't hold BTC right now</div>
    <div class="step-body">Do nothing. Stay in cash. Wait for a 🟢 BUY signal before entering.</div>
</div>
""", unsafe_allow_html=True)

else:
    st.markdown("""
<div class="step-box">
    <div class="step-title">Do nothing for now</div>
    <div class="step-body">The market conditions aren't strong enough to act. Don't force a trade. Check back in a few hours — signals can change quickly.</div>
</div>
<div class="step-box">
    <div class="step-title">If you already hold BTC</div>
    <div class="step-body">Hold and don't panic sell. The market isn't in crash mode, just unclear. Wait for a clearer signal.</div>
</div>
""", unsafe_allow_html=True)

# ─── WHY section ─────────────────────────────────────────────────────────────
with st.expander("🔍 Why is it saying this? (tap to see)"):
    indicators_explained = [
        ("RSI < 90",             cur_votes >= 1, "Bitcoin isn't overbought"),
        ("Momentum > 1%",        cur_votes >= 2, "Price has upward momentum"),
        ("Volatility < 6%",      cur_votes >= 3, "Market isn't too wild/risky"),
        ("Volume above average", cur_votes >= 4, "More people are buying than usual"),
        ("ADX > 25",             cur_votes >= 5, "There's a real trend, not just chop"),
        ("Above 50hr average",   cur_votes >= 6, "Short-term trend is up"),
        ("Above 200hr average",  cur_votes >= 7, "Long-term trend is up"),
        ("MACD bullish",         cur_votes >= 8, "Momentum indicator is positive"),
    ]
    st.markdown(f"**Market Mood (AI model):** {cur_regime}")
    st.markdown(f"**Signals active: {cur_votes}/8** — need 7 or more to trigger a BUY")
    st.markdown("---")
    for name, active, meaning in indicators_explained:
        icon = "✅" if active else "❌"
        st.markdown(f"{icon} **{name}** — {meaning}")

st.divider()

# ─── CHART ───────────────────────────────────────────────────────────────────
st.markdown("### 📈 BTC Price Chart (last 90 days)")
st.caption("🟢 Green background = Bull Run &nbsp;&nbsp; 🔴 Red background = Bear/Crash &nbsp;&nbsp; ▲ = Buy signal &nbsp;&nbsp; ▼ = Sell signal")

cutoff = df_aligned.index[-1] - pd.Timedelta(days=90)
cdf = df_aligned[df_aligned.index >= cutoff]
c_regime = regime_series[regime_series.index.isin(cdf.index)]

fig = go.Figure()
_regime_vrects(fig, c_regime)

fig.add_trace(go.Candlestick(
    x=cdf.index,
    open=cdf["Open"], high=cdf["High"],
    low=cdf["Low"],   close=cdf["Close"],
    name="BTC-USD",
    increasing_line_color="#00e676",
    decreasing_line_color="#ff5252",
    increasing_fillcolor="#00e676",
    decreasing_fillcolor="#ff5252",
))

# Trade markers
if not trades_df.empty:
    visible = trades_df[trades_df["exit_time"] >= cdf.index[0]]
    entries = visible[visible["entry_time"] >= cdf.index[0]]
    if not entries.empty:
        fig.add_trace(go.Scatter(
            x=entries["entry_time"], y=entries["entry_price"],
            mode="markers",
            marker=dict(symbol="triangle-up", color="#00e676", size=13,
                        line=dict(color="#fff", width=0.5)),
            name="BUY signal",
            hovertemplate="<b>BUY</b><br>%{x}<br>$%{y:,.0f}<extra></extra>",
        ))
    if not visible.empty:
        colors = ["#00e676" if p > 0 else "#ff5252" for p in visible["pnl_pct"]]
        fig.add_trace(go.Scatter(
            x=visible["exit_time"], y=visible["exit_price"],
            mode="markers",
            marker=dict(symbol="triangle-down", color=colors, size=13,
                        line=dict(color="#fff", width=0.5)),
            name="SELL signal",
            hovertemplate="<b>SELL</b><br>%{x}<br>$%{y:,.0f}<extra></extra>",
        ))

fig.update_layout(
    template="plotly_dark", height=460,
    margin=dict(l=0, r=0, t=10, b=0),
    plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
    font=dict(color="#b0b8d8"),
    xaxis_rangeslider_visible=False,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
fig.update_yaxis(tickprefix="$")
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ─── PERFORMANCE (plain English) ─────────────────────────────────────────────
st.markdown("### 📊 How Has This Strategy Performed? (Last 2 Years, No Leverage)")
st.caption("This shows what *would have happened* if you followed every signal over the past 2 years starting with $10,000.")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Strategy Result",
          f"${metrics['final_capital']:,.0f}",
          delta=f"{metrics['total_return']:+.1f}% on $10k")
m2.metric("If You Just Held BTC",
          f"{metrics['bh_return']:+.1f}%",
          delta="Buy & Hold return")
m3.metric("Trades That Were Profitable",
          f"{metrics['win_rate']:.0f}%",
          delta=f"out of {metrics['num_trades']} trades")
m4.metric("Worst Loss Period",
          f"{metrics['max_drawdown']:.1f}%",
          delta="max drawdown", delta_color="inverse")

with st.expander("📘 What do these numbers mean?"):
    st.markdown(f"""
- **Strategy Result** — If you started with $10,000 and followed every BUY/SELL signal, you'd have **${metrics['final_capital']:,.0f}** today.
- **If You Just Held BTC** — Simply buying BTC and never selling returned **{metrics['bh_return']:+.1f}%** over the same period. This is the benchmark to beat.
- **Trades That Were Profitable** — Out of every {metrics['num_trades']} trades the strategy made, about **{metrics['win_rate']:.0f}%** of them made money.
- **Worst Loss Period** — At its worst point, the strategy was down **{metrics['max_drawdown']:.1f}%** from its peak before recovering.
""")

st.divider()

# ─── TRADE LOG ───────────────────────────────────────────────────────────────
with st.expander(f"📋 Full Trade History ({metrics['num_trades']} trades — tap to view)"):
    if not trades_df.empty:
        disp = trades_df.copy()
        disp["entry_time"]  = pd.to_datetime(disp["entry_time"]).dt.strftime("%b %d %Y %H:%M")
        disp["exit_time"]   = pd.to_datetime(disp["exit_time"]).dt.strftime("%b %d %Y %H:%M")
        disp["entry_price"] = disp["entry_price"].map("${:,.2f}".format)
        disp["exit_price"]  = disp["exit_price"].map("${:,.2f}".format)
        disp["pnl_pct"]     = disp["pnl_pct"].map("{:+.2f}%".format)
        disp["pnl_dollar"]  = disp["pnl_dollar"].map("${:+,.2f}".format)
        disp.columns = ["Bought", "Sold", "Buy Price", "Sell Price", "Profit/Loss %", "Profit/Loss $", "Why Sold"]
        st.dataframe(disp, use_container_width=True, hide_index=True)
    else:
        st.info("No completed trades yet.")

# ─── DISCLAIMER ──────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
> ⚠️ **Important:** This tool does not connect to Robinhood or execute any trades.
> It only tells you what the AI model sees in the market. All trades are your own decision.
> Never invest more than you can afford to lose. Crypto is highly volatile.
""")
