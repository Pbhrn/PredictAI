# =============================================================
# PHASE 1: Data Acquisition + Candlestick Chart
# =============================================================
# pip install streamlit yfinance plotly pandas
#
# Run: streamlit run phase1_chart.py
# =============================================================

import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="PredictAI · Phase 1",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Phase 1 · Data Acquisition & Chart")
st.caption("ดึงข้อมูลจาก Yahoo Finance แสดงเป็น Candlestick Chart พร้อม Volume")

# ── Sidebar controls ─────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ ตั้งค่า")
    symbol = st.selectbox(
        "เลือกสินทรัพย์",
        ["BTC-USD", "ETH-USD", "SOL-USD", "AAPL", "TSLA", "NVDA", "MSFT"],
        index=0
    )
    period = st.selectbox(
        "ช่วงเวลา",
        ["3mo", "6mo", "1y", "2y"],
        index=2,
        format_func=lambda x: {"3mo": "3 เดือน", "6mo": "6 เดือน", "1y": "1 ปี", "2y": "2 ปี"}[x]
    )
    interval = st.selectbox(
        "Interval",
        ["1d", "1wk"],
        format_func=lambda x: {"1d": "รายวัน", "1wk": "รายสัปดาห์"}[x]
    )
    show_ma = st.checkbox("แสดง Moving Average", value=True)
    show_volume = st.checkbox("แสดง Volume", value=True)

# ── Fetch data ────────────────────────────────────────────────
@st.cache_data(ttl=300)  # Cache 5 นาที
def fetch_data(sym: str, period: str, interval: str) -> pd.DataFrame:
    ticker = yf.Ticker(sym)
    df = ticker.history(period=period, interval=interval)
    df.index = pd.to_datetime(df.index)
    df.index = df.index.tz_localize(None)  # ตัด timezone ออกเพื่อ Plotly
    return df

with st.spinner(f"กำลังดึงข้อมูล {symbol}..."):
    df = fetch_data(symbol, period, interval)

if df.empty:
    st.error("ไม่พบข้อมูล กรุณาลองใหม่อีกครั้ง")
    st.stop()

# ── Compute MA ────────────────────────────────────────────────
df["MA20"] = df["Close"].rolling(window=20).mean()
df["MA50"] = df["Close"].rolling(window=50).mean()

# ── Stats row ─────────────────────────────────────────────────
latest  = df["Close"].iloc[-1]
prev    = df["Close"].iloc[-2]
change  = (latest - prev) / prev * 100
high52  = df["High"].max()
low52   = df["Low"].min()
avg_vol = df["Volume"].mean()

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("ราคาล่าสุด",  f"${latest:,.2f}",  f"{change:+.2f}%")
col2.metric("52w High",     f"${high52:,.2f}")
col3.metric("52w Low",      f"${low52:,.2f}")
col4.metric("MA20",         f"${df['MA20'].iloc[-1]:,.2f}")
col5.metric("Volume เฉลี่ย", f"{avg_vol:,.0f}")

st.divider()

# ── Build chart ───────────────────────────────────────────────
rows = 2 if show_volume else 1
row_heights = [0.75, 0.25] if show_volume else [1.0]

fig = make_subplots(
    rows=rows, cols=1,
    shared_xaxes=True,
    row_heights=row_heights,
    vertical_spacing=0.03
)

# Candlestick
fig.add_trace(
    go.Candlestick(
        x=df.index,
        open=df["Open"], high=df["High"],
        low=df["Low"],   close=df["Close"],
        name=symbol,
        increasing_line_color="#00d084",
        decreasing_line_color="#ff4d6a",
        increasing_fillcolor="#00d084",
        decreasing_fillcolor="#ff4d6a",
    ),
    row=1, col=1
)

# Moving Averages
if show_ma:
    fig.add_trace(
        go.Scatter(x=df.index, y=df["MA20"], name="MA20",
                   line=dict(color="#f5a623", width=1.5)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df["MA50"], name="MA50",
                   line=dict(color="#4d9fff", width=1.5, dash="dot")),
        row=1, col=1
    )

# Volume
if show_volume:
    colors = ["#00d084" if c >= o else "#ff4d6a"
              for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(
        go.Bar(x=df.index, y=df["Volume"], name="Volume",
               marker_color=colors, opacity=0.7),
        row=2, col=1
    )

fig.update_layout(
    template="plotly_dark",
    paper_bgcolor="#0d0f14",
    plot_bgcolor="#13161e",
    xaxis_rangeslider_visible=False,
    height=600,
    margin=dict(l=60, r=20, t=30, b=30),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    font=dict(family="monospace", size=11),
)
fig.update_xaxes(gridcolor="#1a1e28", showgrid=True)
fig.update_yaxes(gridcolor="#1a1e28", showgrid=True)

st.plotly_chart(fig, use_container_width=True)

# ── Raw data table ────────────────────────────────────────────
with st.expander("📋 ดู Raw Data"):
    display = df[["Open","High","Low","Close","Volume","MA20","MA50"]].tail(30)
    st.dataframe(display.style.format({
        "Open": "{:.2f}", "High": "{:.2f}", "Low": "{:.2f}",
        "Close": "{:.2f}", "Volume": "{:,.0f}",
        "MA20": "{:.2f}", "MA50": "{:.2f}"
    }), use_container_width=True)
