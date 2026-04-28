# =============================================================
# PHASE 4: Backtest Engine
# =============================================================
# pip install streamlit yfinance plotly pandas numpy scikit-learn
#
# Run: streamlit run phase4_backtest.py
# =============================================================

import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="PredictAI · Phase 4", page_icon="⚡", layout="wide")
st.title("⚡ Phase 4 · Backtest Engine")
st.caption("จำลองการเทรดในอดีตด้วย Signal จาก Model และวิเคราะห์ผลตอบแทน")

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ ตั้งค่า Backtest")
    symbol     = st.selectbox("สินทรัพย์", ["BTC-USD","ETH-USD","SOL-USD","AAPL","TSLA","NVDA"])
    period     = st.selectbox("ช่วงเวลา", ["1y","2y","3y"], index=1)
    strategy   = st.selectbox(
        "กลยุทธ์",
        ["RSI Mean Reversion", "MA Crossover", "LR Trend Following", "RSI + MA Combo"],
        index=3
    )
    init_capital = st.number_input("ทุนเริ่มต้น ($)", value=10_000, step=1_000)
    commission   = st.number_input("ค่าคอมมิชชั่น (%)", value=0.1, step=0.05)

    st.divider()
    st.subheader("ปรับ Parameters")
    rsi_buy  = st.slider("RSI Buy Signal (<)", 20, 45, 35)
    rsi_sell = st.slider("RSI Sell Signal (>)", 55, 80, 65)
    ma_fast  = st.slider("MA Fast Period", 5, 30, 10)
    ma_slow  = st.slider("MA Slow Period", 20, 100, 50)

# ── Fetch & Features ─────────────────────────────────────────
@st.cache_data(ttl=300)
def fetch(sym, period):
    df = yf.Ticker(sym).history(period=period, interval="1d")
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df

with st.spinner("กำลังเตรียมข้อมูล..."):
    df_raw = fetch(symbol, period)

    close  = df_raw["Close"].copy()
    # RSI
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    df_raw["RSI"] = 100 - 100 / (1 + rs)
    # MAs
    df_raw["MA_fast"] = close.rolling(ma_fast).mean()
    df_raw["MA_slow"] = close.rolling(ma_slow).mean()
    # LR slope (30-day window)
    def lr_slope(prices, window=30):
        slopes = [np.nan] * window
        for i in range(window, len(prices)):
            y = prices.iloc[i-window:i].values
            x = np.arange(window).reshape(-1,1)
            m = LinearRegression().fit(x, y).coef_[0]
            slopes.append(m / prices.iloc[i] * 100)   # normalise as %
        return pd.Series(slopes, index=prices.index)

    df_raw["LR_Slope"] = lr_slope(close)
    df_raw = df_raw.dropna()

# ── Signal generation ─────────────────────────────────────────
def generate_signals(df, strategy, rsi_buy, rsi_sell, ma_fast, ma_slow):
    signals = pd.Series(0, index=df.index)   # 0=hold, 1=buy, -1=sell
    pos = 0  # 0=no position, 1=in position

    for i in range(1, len(df)):
        rsi  = df["RSI"].iloc[i]
        maf  = df["MA_fast"].iloc[i]
        mas  = df["MA_slow"].iloc[i]
        maf_prev = df["MA_fast"].iloc[i-1]
        mas_prev = df["MA_slow"].iloc[i-1]
        slope = df["LR_Slope"].iloc[i]

        if strategy == "RSI Mean Reversion":
            if rsi < rsi_buy and pos == 0:
                signals.iloc[i] = 1; pos = 1
            elif rsi > rsi_sell and pos == 1:
                signals.iloc[i] = -1; pos = 0

        elif strategy == "MA Crossover":
            if maf > mas and maf_prev <= mas_prev and pos == 0:
                signals.iloc[i] = 1; pos = 1
            elif maf < mas and maf_prev >= mas_prev and pos == 1:
                signals.iloc[i] = -1; pos = 0

        elif strategy == "LR Trend Following":
            if slope > 0.05 and pos == 0:
                signals.iloc[i] = 1; pos = 1
            elif slope < -0.02 and pos == 1:
                signals.iloc[i] = -1; pos = 0

        elif strategy == "RSI + MA Combo":
            uptrend = maf > mas
            if rsi < rsi_buy and uptrend and pos == 0:
                signals.iloc[i] = 1; pos = 1
            elif (rsi > rsi_sell or maf < mas) and pos == 1:
                signals.iloc[i] = -1; pos = 0

    # Force close at end
    if pos == 1:
        signals.iloc[-1] = -1

    return signals

signals = generate_signals(df_raw, strategy, rsi_buy, rsi_sell, ma_fast, ma_slow)

# ── Backtest execution ────────────────────────────────────────
def backtest(df, signals, init_capital, commission_pct):
    capital   = float(init_capital)
    shares    = 0.0
    equity    = [capital]
    trades    = []
    entry_price = 0.0
    comm_rate = commission_pct / 100

    for i in range(1, len(df)):
        price = df["Close"].iloc[i]
        sig   = signals.iloc[i]
        date  = df.index[i]

        if sig == 1 and capital > 0:
            cost    = capital * (1 - comm_rate)
            shares  = cost / price
            capital = 0.0
            entry_price = price
            trades.append({
                "date": date, "type": "BUY", "price": price,
                "shares": shares, "value": shares * price, "pnl": None
            })

        elif sig == -1 and shares > 0:
            proceeds = shares * price * (1 - comm_rate)
            pnl      = proceeds - (shares * entry_price)
            pnl_pct  = pnl / (shares * entry_price) * 100
            capital  = proceeds
            trades.append({
                "date": date, "type": "SELL", "price": price,
                "shares": shares, "value": proceeds, "pnl": pnl_pct
            })
            shares = 0.0

        current_equity = capital + (shares * price if shares > 0 else 0)
        equity.append(current_equity)

    return equity, trades

equity, trades = backtest(df_raw, signals, init_capital, commission)

# Buy & Hold benchmark
bh_equity = (df_raw["Close"] / df_raw["Close"].iloc[0] * init_capital).tolist()

# ── Performance metrics ───────────────────────────────────────
def calc_metrics(equity, init_capital, bh_equity):
    eq = np.array(equity)
    bh = np.array(bh_equity)

    total_return = (eq[-1] - init_capital) / init_capital * 100
    bh_return    = (bh[-1] - init_capital) / init_capital * 100
    alpha        = total_return - bh_return

    # Drawdown
    peak     = np.maximum.accumulate(eq)
    dd       = (eq - peak) / peak * 100
    max_dd   = dd.min()

    # Sharpe (annualised, assuming 252 trading days)
    daily_ret = pd.Series(eq).pct_change().dropna()
    sharpe    = (daily_ret.mean() / daily_ret.std() * np.sqrt(252)
                 if daily_ret.std() > 0 else 0)

    # Win rate
    sell_trades = [t for t in trades if t["type"] == "SELL" and t["pnl"] is not None]
    wins        = [t for t in sell_trades if t["pnl"] > 0]
    win_rate    = len(wins) / len(sell_trades) * 100 if sell_trades else 0

    return {
        "total_return": total_return, "bh_return": bh_return, "alpha": alpha,
        "max_dd": max_dd, "sharpe": sharpe, "win_rate": win_rate,
        "n_trades": len(sell_trades), "final_equity": eq[-1]
    }

m = calc_metrics(equity, init_capital, bh_equity)

# ── Display metrics ───────────────────────────────────────────
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Total Return",   f"{m['total_return']:+.2f}%",
          f"BH: {m['bh_return']:+.1f}%")
c2.metric("Alpha vs B&H",   f"{m['alpha']:+.2f}%")
c3.metric("Final Equity",   f"${m['final_equity']:,.0f}")
c4.metric("Sharpe Ratio",   f"{m['sharpe']:.2f}")
c5.metric("Max Drawdown",   f"{m['max_dd']:.2f}%")
c6.metric("Win Rate",       f"{m['win_rate']:.1f}%",
          f"{m['n_trades']} trades")

st.divider()

# ── Charts ────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📈 Equity Curve", "📊 Trade Analysis", "📋 Trade Log"])

with tab1:
    dates = df_raw.index
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.55, 0.25, 0.20],
                        vertical_spacing=0.04)

    # Equity curves
    fig.add_trace(go.Scatter(
        x=dates, y=equity, name="Strategy",
        line=dict(color="#00d084", width=2),
        fill="tozeroy", fillcolor="rgba(0,208,132,0.05)"
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=dates, y=bh_equity, name="Buy & Hold",
        line=dict(color="#555d72", width=1.5, dash="dot")
    ), row=1, col=1)

    # Buy/Sell markers
    buy_dates  = [t["date"] for t in trades if t["type"] == "BUY"]
    buy_prices = [equity[df_raw.index.get_loc(d)] for d in buy_dates]
    sell_dates  = [t["date"] for t in trades if t["type"] == "SELL"]
    sell_prices = [equity[df_raw.index.get_loc(d)] for d in sell_dates]

    fig.add_trace(go.Scatter(
        x=buy_dates, y=buy_prices, mode="markers",
        marker=dict(color="#00d084", size=8, symbol="triangle-up"),
        name="Buy Signal"
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=sell_dates, y=sell_prices, mode="markers",
        marker=dict(color="#ff4d6a", size=8, symbol="triangle-down"),
        name="Sell Signal"
    ), row=1, col=1)

    # RSI
    fig.add_trace(go.Scatter(
        x=dates, y=df_raw["RSI"],
        name="RSI", line=dict(color="#a78bfa", width=1.2)
    ), row=2, col=1)
    fig.add_hline(y=rsi_buy, line_color="rgba(0,208,132,0.4)",
                  line_dash="dot", row=2, col=1)
    fig.add_hline(y=rsi_sell, line_color="rgba(255,77,106,0.4)",
                  line_dash="dot", row=2, col=1)
    fig.add_hline(y=50, line_color="rgba(255,255,255,0.1)", row=2, col=1)

    # Drawdown
    eq_arr = np.array(equity)
    peak   = np.maximum.accumulate(eq_arr)
    dd     = (eq_arr - peak) / peak * 100
    fig.add_trace(go.Scatter(
        x=dates, y=dd, name="Drawdown",
        fill="tozeroy", fillcolor="rgba(255,77,106,0.15)",
        line=dict(color="#ff4d6a", width=1)
    ), row=3, col=1)

    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#0d0f14", plot_bgcolor="#13161e",
        height=680, margin=dict(l=60, r=20, t=30, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        font=dict(family="monospace")
    )
    fig.update_yaxes(gridcolor="#1a1e28")
    fig.update_xaxes(gridcolor="#1a1e28")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    sell_trades = [t for t in trades if t["type"] == "SELL" and t["pnl"] is not None]
    if sell_trades:
        pnl_vals = [t["pnl"] for t in sell_trades]

        col_a, col_b = st.columns(2)

        # PnL distribution histogram
        with col_a:
            st.markdown("**P&L Distribution per Trade**")
            fig3 = go.Figure(go.Histogram(
                x=pnl_vals,
                nbinsx=20,
                marker_color=["#00d084" if v >= 0 else "#ff4d6a" for v in pnl_vals],
                opacity=0.8,
            ))
            fig3.update_layout(
                template="plotly_dark", paper_bgcolor="#0d0f14", plot_bgcolor="#13161e",
                height=300, margin=dict(l=40,r=20,t=20,b=40),
                xaxis_title="P&L (%)", font=dict(family="monospace")
            )
            st.plotly_chart(fig3, use_container_width=True)

        # Cumulative P&L
        with col_b:
            st.markdown("**Cumulative P&L**")
            cum_pnl = np.cumsum(pnl_vals)
            fig4 = go.Figure(go.Scatter(
                y=cum_pnl, mode="lines+markers",
                line=dict(color="#4d9fff", width=2),
                marker=dict(color=["#00d084" if v >= 0 else "#ff4d6a" for v in cum_pnl], size=6)
            ))
            fig4.update_layout(
                template="plotly_dark", paper_bgcolor="#0d0f14", plot_bgcolor="#13161e",
                height=300, margin=dict(l=40,r=20,t=20,b=40),
                yaxis_title="Cumulative P&L (%)", xaxis_title="Trade #",
                font=dict(family="monospace")
            )
            st.plotly_chart(fig4, use_container_width=True)
    else:
        st.info("ยังไม่มี Trade ที่ปิดแล้ว ลองปรับ Parameters ใน Sidebar")

with tab3:
    if trades:
        trade_df = pd.DataFrame([{
            "Date":   t["date"].strftime("%Y-%m-%d"),
            "Type":   t["type"],
            "Price":  f"${t['price']:,.2f}",
            "Value":  f"${t['value']:,.2f}",
            "P&L %":  f"{t['pnl']:+.2f}%" if t["pnl"] is not None else "—",
        } for t in trades])
        st.dataframe(
            trade_df.style.apply(
                lambda col: ["color: #00d084" if "BUY" in str(v) else
                             "color: #ff4d6a" if "SELL" in str(v) else ""
                             for v in col],
                subset=["Type"]
            ),
            use_container_width=True, hide_index=True
        )
    else:
        st.info("ไม่มี Trade ที่ generate ได้ กรุณาปรับ RSI / MA Parameters")

# ── Strategy explanation ──────────────────────────────────────
with st.expander("ℹ️ อ่านเพิ่มเติม: กลยุทธ์ที่เลือก"):
    explanations = {
        "RSI Mean Reversion": f"**ซื้อ** เมื่อ RSI < {rsi_buy} (oversold) · **ขาย** เมื่อ RSI > {rsi_sell} (overbought) — เหมาะกับ Sideways market",
        "MA Crossover":       f"**ซื้อ** เมื่อ MA{ma_fast} ตัดขึ้น MA{ma_slow} (Golden Cross) · **ขาย** เมื่อ MA{ma_fast} ตัดลง (Death Cross) — เหมาะกับ Trending market",
        "LR Trend Following": "**ซื้อ** เมื่อ Slope ของ Linear Regression (30d) เป็นบวก · **ขาย** เมื่อ Slope เป็นลบ",
        "RSI + MA Combo":     f"**ซื้อ** เมื่อ RSI < {rsi_buy} และ MA{ma_fast} > MA{ma_slow} (uptrend) · **ขาย** เมื่อ RSI > {rsi_sell} หรือ MA cross down",
    }
    st.markdown(explanations[strategy])
    st.caption("⚠️ Disclaimer: Backtest ไม่รับประกันผลตอบแทนในอนาคต Past performance ≠ Future results")