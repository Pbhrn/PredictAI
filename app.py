# =============================================================
# MAIN APP — รวมทุก Phase ไว้ใน Multi-page App
# =============================================================
# Run: streamlit run app.py
# =============================================================

import streamlit as st

st.set_page_config(
    page_title="PredictAI · Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background-color: #0d0f14; }
  [data-testid="stSidebar"]          { background-color: #13161e; border-right: 1px solid #2a2f3d; }
  h1, h2, h3                         { font-family: monospace; }
  .stSelectbox label, .stSlider label { color: #8890a8 !important; }
  .metric-label { font-size: 11px !important; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar navigation ────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤖 PredictAI")
    st.caption("Crypto & Stock Prediction Dashboard")
    st.divider()

    page = st.radio(
        "เลือก Phase",
        options=["🏠 Home", "📈 Phase 1 · Chart", "📊 Phase 2 · Linear Regression",
                 "🧠 Phase 3 · LSTM", "⚡ Phase 4 · Backtest"],
        label_visibility="collapsed"
    )
    st.divider()
    st.caption("Built with Streamlit + yfinance + TensorFlow")
    st.caption("⚠️ ไม่ใช่คำแนะนำการลงทุน")

# ── Pages ─────────────────────────────────────────────────────
if page == "🏠 Home":
    st.title("🤖 PredictAI · Crypto & Stock Prediction")
    st.markdown("---")
    st.markdown("""
    ### ยินดีต้อนรับสู่ PredictAI Dashboard

    เลือก Phase จาก Sidebar เพื่อเริ่มใช้งาน:
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.info("**📈 Phase 1 · Chart**\n\nดึงข้อมูลจาก Yahoo Finance แสดง Candlestick, Volume, Moving Average")
        st.success("**🧠 Phase 3 · LSTM**\n\nDeep Learning ที่จำรูปแบบ 60 วันเพื่อทำนายวันถัดไป พร้อม Training visualization")
    with col2:
        st.warning("**📊 Phase 2 · Linear Regression**\n\nทำนาย Next-Day Price ด้วย Technical Indicators เป็น Features พร้อม Feature Importance")
        st.error("**⚡ Phase 4 · Backtest**\n\nจำลองการเทรดใน 4 กลยุทธ์ วัด Return, Sharpe Ratio, Drawdown")

    st.divider()
    st.subheader("📦 Dependencies")
    st.code("""pip install streamlit yfinance plotly pandas numpy \\
         scikit-learn tensorflow pandas-ta""", language="bash")

elif page == "📈 Phase 1 · Chart":
    exec(open("phase1_chart.py", encoding="utf-8").read())

elif page == "📊 Phase 2 · Linear Regression":
    exec(open("phase2_linear.py", encoding="utf-8").read())

elif page == "🧠 Phase 3 · LSTM":
    exec(open("phase3_lstm.py", encoding="utf-8").read())

elif page == "⚡ Phase 4 · Backtest":
    exec(open("phase4_backtest.py", encoding="utf-8").read())
