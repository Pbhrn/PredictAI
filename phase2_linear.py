# =============================================================
# PHASE 2: Linear Regression — Next-Day Prediction
# =============================================================
# pip install streamlit yfinance plotly pandas scikit-learn pandas-ta
#
# Run: streamlit run phase2_linear.py
# =============================================================

import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# ── pandas-ta (optional, graceful fallback) ───────────────────
try:
    import pandas_ta as ta
    HAS_TA = True
except ImportError:
    HAS_TA = False

st.set_page_config(page_title="PredictAI · Phase 2", page_icon="📊", layout="wide")
st.title("📊 Phase 2 · Linear Regression — Next-Day Prediction")
st.caption("ใช้ Technical Indicators เป็น Features ทำนายราคาวันถัดไปด้วย Linear Regression")

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ ตั้งค่า")
    symbol   = st.selectbox("สินทรัพย์", ["BTC-USD","ETH-USD","SOL-USD","AAPL","TSLA","NVDA"])
    period   = st.selectbox("ข้อมูลย้อนหลัง", ["1y","2y","3y"], index=1)
    test_pct = st.slider("สัดส่วน Test Set (%)", 10, 30, 20)
    future_days = st.slider("พยากรณ์ล่วงหน้า (วัน)", 7, 60, 30)

    st.divider()
    st.subheader("Features ที่ใช้")
    use_rsi   = st.checkbox("RSI (14)", value=True)
    use_macd  = st.checkbox("MACD Signal", value=True)
    use_ma    = st.checkbox("MA20 / MA50 Ratio", value=True)
    use_vol   = st.checkbox("Volume Change %", value=True)
    use_range = st.checkbox("High-Low Range %", value=True)
    use_lag   = st.checkbox("Lag Returns (1,3,5d)", value=True)

# ── Data fetch ────────────────────────────────────────────────
@st.cache_data(ttl=300)
def fetch(sym, period):
    df = yf.Ticker(sym).history(period=period, interval="1d")
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df

# ── Feature Engineering ───────────────────────────────────────
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    feat = pd.DataFrame(index=df.index)

    close = df["Close"]
    feat["close"] = close

    # Target: next-day close
    feat["target"] = close.shift(-1)

    if use_rsi:
        # RSI manual (no dependency)
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        feat["rsi"] = 100 - 100 / (1 + rs)

    if use_macd:
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd  = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        feat["macd_hist"] = macd - signal

    if use_ma:
        ma20 = close.rolling(20).mean()
        ma50 = close.rolling(50).mean()
        feat["ma20_ratio"] = close / ma20
        feat["ma50_ratio"] = close / ma50

    if use_vol:
        feat["vol_chg"] = df["Volume"].pct_change()

    if use_range:
        feat["hl_range"] = (df["High"] - df["Low"]) / close

    if use_lag:
        for d in [1, 3, 5]:
            feat[f"ret_{d}d"] = close.pct_change(d)

    feat = feat.dropna()
    return feat

with st.spinner("กำลังดึงและเตรียมข้อมูล..."):
    df_raw = fetch(symbol, period)
    feat_df = build_features(df_raw)

if len(feat_df) < 60:
    st.error("ข้อมูลไม่เพียงพอ กรุณาเลือกช่วงเวลาที่ยาวขึ้น")
    st.stop()

# ── Prepare X, y ─────────────────────────────────────────────
feature_cols = [c for c in feat_df.columns if c not in ["close", "target"]]
X = feat_df[feature_cols].values
y = feat_df["target"].values
dates = feat_df.index

# Normalize
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

split_idx = int(len(X_scaled) * (1 - test_pct / 100))
X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
y_train, y_test = y_scaled[:split_idx], y_scaled[split_idx:]
dates_test = dates[split_idx:]

# ── Train ─────────────────────────────────────────────────────
model = LinearRegression()
model.fit(X_train, y_train)

y_pred_scaled = model.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
y_actual = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()

# ── Metrics ───────────────────────────────────────────────────
rmse  = np.sqrt(mean_squared_error(y_actual, y_pred))
mae   = mean_absolute_error(y_actual, y_pred)
mape  = np.mean(np.abs((y_actual - y_pred) / y_actual)) * 100
r2    = r2_score(y_actual, y_pred)

# ── Next-day forecast ─────────────────────────────────────────
last_features = feat_df[feature_cols].iloc[-1].values.reshape(1, -1)
last_scaled   = scaler_X.transform(last_features)

future_preds = []
current_feat = last_features.copy()
for _ in range(future_days):
    pred_scaled = model.predict(scaler_X.transform(current_feat))
    pred_price  = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))[0][0]
    future_preds.append(pred_price)
    # Slide window: nudge last close toward prediction
    if use_lag:
        ret_cols = [i for i, c in enumerate(feature_cols) if c.startswith("ret_")]
        for rc in ret_cols:
            current_feat[0][rc] = (pred_price - feat_df["close"].iloc[-1]) / feat_df["close"].iloc[-1]

future_dates = pd.date_range(start=dates[-1] + pd.Timedelta(days=1), periods=future_days, freq="B")

# ── Metrics display ───────────────────────────────────────────
next_price = future_preds[0]
current_price = df_raw["Close"].iloc[-1]
pred_change = (next_price - current_price) / current_price * 100

st.subheader("📐 ผลลัพธ์ Model")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("ราคาปัจจุบัน", f"${current_price:,.2f}")
c2.metric("พยากรณ์วันพรุ่งนี้", f"${next_price:,.2f}", f"{pred_change:+.2f}%")
c3.metric("RMSE", f"{rmse:.2f}")
c4.metric("MAPE", f"{mape:.2f}%")
c5.metric("R² Score", f"{r2:.4f}")

st.divider()

# ── Chart ─────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📈 Actual vs Predicted", "🔍 Feature Coefficients"])

with tab1:
    fig = go.Figure()

    # Training set
    fig.add_trace(go.Scatter(
        x=dates[:split_idx], y=feat_df["target"].values[:split_idx],
        name="Train (Actual)", line=dict(color="#555d72", width=1)
    ))
    # Test actual
    fig.add_trace(go.Scatter(
        x=dates_test, y=y_actual,
        name="Test (Actual)", line=dict(color="#4d9fff", width=2)
    ))
    # Test predicted
    fig.add_trace(go.Scatter(
        x=dates_test, y=y_pred,
        name="Test (Predicted)", line=dict(color="#f5a623", width=2)
    ))
    # Future forecast
    fig.add_trace(go.Scatter(
        x=future_dates, y=future_preds,
        name=f"{future_days}-day Forecast",
        line=dict(color="#f5a623", width=1.5, dash="dash"),
        fill="tonexty" if False else None,
    ))
    # Vertical split line — ใช้ add_shape แทน add_vline เพื่อหลีกเลี่ยง Plotly Timestamp bug
    split_date = str(dates[split_idx])[:10]
    fig.add_shape(type="line", x0=split_date, x1=split_date, y0=0, y1=1,
                  xref="x", yref="paper",
                  line=dict(color="rgba(255,255,255,0.2)", dash="dot", width=1))
    fig.add_annotation(x=split_date, y=1, xref="x", yref="paper",
                       text="Train / Test Split", showarrow=False,
                       font=dict(color="rgba(255,255,255,0.4)", size=10),
                       xanchor="left", yanchor="top")

    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#0d0f14", plot_bgcolor="#13161e",
        height=500, margin=dict(l=60, r=20, t=30, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        font=dict(family="monospace")
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    coef_df = pd.DataFrame({
        "Feature": feature_cols,
        "Coefficient": model.coef_
    }).sort_values("Coefficient", key=abs, ascending=True)

    colors = ["#00d084" if v >= 0 else "#ff4d6a" for v in coef_df["Coefficient"]]
    fig2 = go.Figure(go.Bar(
        x=coef_df["Coefficient"], y=coef_df["Feature"],
        orientation="h", marker_color=colors
    ))
    fig2.update_layout(
        template="plotly_dark", paper_bgcolor="#0d0f14", plot_bgcolor="#13161e",
        height=400, margin=dict(l=120, r=20, t=30, b=30),
        xaxis_title="Coefficient Value",
        font=dict(family="monospace")
    )
    st.plotly_chart(fig2, use_container_width=True)

# ── Forecast table ────────────────────────────────────────────
with st.expander(f"📋 ตาราง {future_days}-day Forecast"):
    fdf = pd.DataFrame({
        "Date": future_dates.strftime("%Y-%m-%d"),
        "Predicted Price": [f"${p:,.2f}" for p in future_preds],
        "Change vs Today": [f"{(p-current_price)/current_price*100:+.2f}%" for p in future_preds],
        "Signal": ["🟢 BUY" if p > current_price else "🔴 SELL" for p in future_preds]
    })
    st.dataframe(fdf, use_container_width=True, hide_index=True)
