# =============================================================
# PHASE 3: LSTM — 60-Day Lookback → Next-Day Prediction
# =============================================================
# pip install streamlit yfinance plotly pandas numpy
#         tensorflow scikit-learn
#
# Run: streamlit run phase3_lstm.py
# =============================================================

import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

# ── TensorFlow (graceful fallback to placeholder) ─────────────
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    HAS_TF = True
except ImportError:
    HAS_TF = False

st.set_page_config(page_title="PredictAI · Phase 3", page_icon="🧠", layout="wide")
st.title("🧠 Phase 3 · LSTM — Deep Learning Time Series")
st.caption("LSTM จำรูปแบบในอดีต 60 วัน เพื่อทำนายราคาวันถัดไป")

if not HAS_TF:
    st.warning("⚠️ TensorFlow ยังไม่ได้ติดตั้ง: `pip install tensorflow`")
    st.info("Dashboard จะแสดงสถาปัตยกรรมและ simulation ในระหว่างนี้")

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ ตั้งค่า LSTM")
    symbol     = st.selectbox("สินทรัพย์", ["BTC-USD","ETH-USD","SOL-USD","AAPL","TSLA","NVDA"])
    lookback   = st.slider("Lookback Window (วัน)", 30, 120, 60)
    epochs     = st.slider("Epochs", 10, 100, 50)
    batch_size = st.selectbox("Batch Size", [16, 32, 64], index=1)
    test_size  = st.slider("Test Set (วัน)", 30, 90, 60)
    future_days = st.slider("พยากรณ์ล่วงหน้า (วัน)", 7, 30, 14)

    st.divider()
    st.subheader("สถาปัตยกรรม Network")
    units1 = st.slider("LSTM Layer 1 Units", 32, 256, 128)
    units2 = st.slider("LSTM Layer 2 Units", 16, 128, 64)
    dropout_rate = st.slider("Dropout Rate", 0.1, 0.5, 0.2)
    dense_units  = st.slider("Dense Units", 8, 64, 32)

# ── Fetch data ────────────────────────────────────────────────
@st.cache_data(ttl=300)
def fetch(sym):
    df = yf.Ticker(sym).history(period="3y", interval="1d")
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df

# ── Feature engineering ───────────────────────────────────────
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"]
    # RSI
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - 100 / (1 + rs)
    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD_Hist"] = (ema12 - ema26) - (ema12 - ema26).ewm(span=9, adjust=False).mean()
    # MA ratios
    df["MA20_Ratio"] = close / close.rolling(20).mean()
    df["MA50_Ratio"] = close / close.rolling(50).mean()
    # Returns
    df["Ret1d"] = close.pct_change(1)
    df["Ret5d"] = close.pct_change(5)
    df["HL_Range"] = (df["High"] - df["Low"]) / close
    return df.dropna()

# ── Build LSTM sequences ──────────────────────────────────────
def create_sequences(data: np.ndarray, lookback: int):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i])
        y.append(data[i, 0])   # column 0 = Close
    return np.array(X), np.array(y)

# ── Build Keras model ─────────────────────────────────────────
def build_model(input_shape, units1, units2, dense_units, dropout_rate):
    model = Sequential([
        LSTM(units1, return_sequences=True, input_shape=input_shape),
        Dropout(dropout_rate),
        LSTM(units2, return_sequences=False),
        Dropout(dropout_rate),
        Dense(dense_units, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="huber", metrics=["mae"])
    return model

# ── Main ──────────────────────────────────────────────────────
with st.spinner("กำลังดึงข้อมูล..."):
    df_raw = fetch(symbol)
    df     = add_features(df_raw.copy())

feat_cols = ["Close", "RSI", "MACD_Hist", "MA20_Ratio", "MA50_Ratio",
             "Ret1d", "Ret5d", "HL_Range"]
data_arr = df[feat_cols].values

# Scale
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data_arr)

# Create sequences
X_all, y_all = create_sequences(data_scaled, lookback)
dates_all    = df.index[lookback:]

# Train / test split
split = len(X_all) - test_size
X_train, X_test = X_all[:split], X_all[split:]
y_train, y_test = y_all[:split], y_all[split:]
dates_test = dates_all[split:]

st.info(f"ข้อมูลทั้งหมด: **{len(df)}** วัน  |  Train: **{split}** sequences  |  Test: **{len(X_test)}** sequences  |  Features: **{len(feat_cols)}**")

# ── Train / Load ──────────────────────────────────────────────
train_btn = st.button("🚀 เทรน LSTM Model", type="primary", use_container_width=True)

if "lstm_trained" not in st.session_state:
    st.session_state.lstm_trained = False
    st.session_state.lstm_history = None
    st.session_state.lstm_pred    = None
    st.session_state.lstm_future  = None

if train_btn and HAS_TF:
    model = build_model((lookback, len(feat_cols)), units1, units2, dense_units, dropout_rate)
    total_params = model.count_params()

    progress_bar = st.progress(0, text="กำลัง Compile model...")
    status_text  = st.empty()

    history_loss = []
    history_val  = []

    callbacks = [
        EarlyStopping(patience=8, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(factor=0.5, patience=4, verbose=0)
    ]

    for epoch in range(1, epochs + 1):
        h = model.fit(
            X_train, y_train,
            validation_split=0.1,
            batch_size=batch_size,
            epochs=1, verbose=0
        )
        history_loss.append(h.history["loss"][0])
        history_val.append(h.history["val_loss"][0])
        pct = epoch / epochs
        progress_bar.progress(pct, text=f"Epoch {epoch}/{epochs}  —  Loss: {history_loss[-1]:.5f}")
        status_text.text(f"Val Loss: {history_val[-1]:.5f}  |  LR: {float(tf.keras.backend.get_value(model.optimizer.learning_rate)):.6f}")

    progress_bar.empty()
    status_text.empty()
    st.success(f"✅ เทรนเสร็จ! Total parameters: {total_params:,}")

    # Predict on test
    pred_scaled = model.predict(X_test, verbose=0)

    # Inverse transform (only Close column)
    def inv_close(arr):
        dummy = np.zeros((len(arr), data_scaled.shape[1]))
        dummy[:, 0] = arr.ravel()
        return scaler.inverse_transform(dummy)[:, 0]

    y_pred   = inv_close(pred_scaled)
    y_actual = inv_close(y_test.reshape(-1, 1))

    # Multi-step future forecast
    last_seq = data_scaled[-lookback:].copy()
    future_preds = []
    for _ in range(future_days):
        inp = last_seq[-lookback:].reshape(1, lookback, len(feat_cols))
        p   = model.predict(inp, verbose=0)[0][0]
        future_preds.append(p)
        new_row = last_seq[-1].copy()
        new_row[0] = p
        last_seq = np.vstack([last_seq, new_row])

    future_prices = inv_close(np.array(future_preds).reshape(-1, 1))

    st.session_state.lstm_trained = True
    st.session_state.lstm_history = {"loss": history_loss, "val": history_val}
    st.session_state.lstm_pred    = (y_actual, y_pred, dates_test)
    st.session_state.lstm_future  = future_prices

elif train_btn and not HAS_TF:
    st.error("TensorFlow ไม่ได้ติดตั้ง: `pip install tensorflow`")

# ── Visualize results ─────────────────────────────────────────
if st.session_state.lstm_trained:
    y_actual, y_pred, dates_test = st.session_state.lstm_pred
    future_prices = st.session_state.lstm_future
    history       = st.session_state.lstm_history

    # Metrics
    rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
    mae  = mean_absolute_error(y_actual, y_pred)
    mape = np.mean(np.abs((y_actual - y_pred) / y_actual)) * 100
    current_price = df_raw["Close"].iloc[-1]
    next_pred     = future_prices[0]
    pred_chg      = (next_pred - current_price) / current_price * 100

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("ราคาปัจจุบัน", f"${current_price:,.2f}")
    c2.metric("พยากรณ์วันพรุ่งนี้", f"${next_pred:,.2f}", f"{pred_chg:+.2f}%")
    c3.metric("RMSE", f"{rmse:.2f}")
    c4.metric("MAE", f"{mae:.2f}")
    c5.metric("MAPE", f"{mape:.2f}%")

    st.divider()

    tab1, tab2, tab3 = st.tabs(["📈 Actual vs Predicted", "📉 Training Loss", f"🔮 {future_days}-day Forecast"])

    with tab1:
        future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1),
                                     periods=future_days, freq="B")
        fig = go.Figure()
        # Full history (context)
        fig.add_trace(go.Scatter(
            x=df.index[-200:], y=df_raw["Close"].reindex(df.index[-200:]),
            name="Historical", line=dict(color="#555d72", width=1)
        ))
        # Test actual
        fig.add_trace(go.Scatter(
            x=dates_test, y=y_actual,
            name="Actual (Test)", line=dict(color="#4d9fff", width=2)
        ))
        # LSTM predictions
        fig.add_trace(go.Scatter(
            x=dates_test, y=y_pred,
            name="LSTM Predicted", line=dict(color="#a78bfa", width=2)
        ))
        # Forecast
        fig.add_trace(go.Scatter(
            x=future_dates, y=future_prices,
            name=f"{future_days}d Forecast",
            line=dict(color="#a78bfa", width=1.5, dash="dash")
        ))
        # Confidence band (±RMSE)
        fig.add_trace(go.Scatter(
            x=list(future_dates) + list(future_dates[::-1]),
            y=list(future_prices + rmse) + list((future_prices - rmse)[::-1]),
            fill="toself", fillcolor="rgba(167,139,250,0.1)",
            line=dict(color="rgba(167,139,250,0)"),
            name="±1 RMSE Band"
        ))
        split_date = str(dates_test[0])[:10]
        fig.add_shape(type="line", x0=split_date, x1=split_date, y0=0, y1=1,
                      xref="x", yref="paper",
                      line=dict(color="rgba(255,255,255,0.2)", dash="dot", width=1))
        fig.add_annotation(x=split_date, y=1, xref="x", yref="paper",
                           text="Test Start", showarrow=False,
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
        fig2 = go.Figure()
        epochs_range = list(range(1, len(history["loss"]) + 1))
        fig2.add_trace(go.Scatter(
            x=epochs_range, y=history["loss"],
            name="Train Loss", line=dict(color="#a78bfa", width=2)
        ))
        fig2.add_trace(go.Scatter(
            x=epochs_range, y=history["val"],
            name="Val Loss", line=dict(color="#f5a623", width=2, dash="dot")
        ))
        fig2.update_layout(
            template="plotly_dark", paper_bgcolor="#0d0f14", plot_bgcolor="#13161e",
            height=400, xaxis_title="Epoch", yaxis_title="Huber Loss",
            margin=dict(l=60, r=20, t=30, b=40),
            font=dict(family="monospace")
        )
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        future_df = pd.DataFrame({
            "Date": pd.date_range(start=df.index[-1] + pd.Timedelta(days=1),
                                  periods=future_days, freq="B").strftime("%Y-%m-%d"),
            "Predicted Price": [f"${p:,.2f}" for p in future_prices],
            "Change vs Today": [f"{(p - current_price)/current_price*100:+.2f}%"
                                for p in future_prices],
            "Signal": ["🟢 BUY" if p > current_price else "🔴 SELL"
                       for p in future_prices]
        })
        st.dataframe(future_df, use_container_width=True, hide_index=True)

else:
    # Architecture preview (before training)
    st.subheader("🏗️ สถาปัตยกรรม LSTM")
    layers = [
        ("Input", f"{lookback} timesteps × {len(feat_cols)} features", "#4d9fff"),
        ("LSTM Layer 1", f"{units1} units, return_sequences=True", "#a78bfa"),
        (f"Dropout ({dropout_rate})", "Regularization", "#555d72"),
        ("LSTM Layer 2", f"{units2} units, return_sequences=False", "#a78bfa"),
        (f"Dropout ({dropout_rate})", "Regularization", "#555d72"),
        (f"Dense ({dense_units})", "ReLU activation", "#f5a623"),
        ("Output (1)", "Predicted next-day price", "#00d084"),
    ]
    for name, desc, color in layers:
        st.markdown(
            f"<div style='background:#13161e;border-left:3px solid {color};"
            f"padding:8px 16px;margin:4px 0;border-radius:4px;font-family:monospace'>"
            f"<span style='color:{color};font-weight:600'>{name}</span> "
            f"<span style='color:#8890a8;font-size:13px'>— {desc}</span></div>",
            unsafe_allow_html=True
        )
    st.caption("กด 'เทรน LSTM Model' เพื่อเริ่มการฝึก")
