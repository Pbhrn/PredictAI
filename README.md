# PredictAI

PredictAI is a Streamlit dashboard for crypto and stock forecasting. It features four phases: interactive Plotly charts, Linear Regression, LSTM deep learning with TensorFlow, and a backtesting engine. It uses feature engineering and MinMaxScaler to predict prices and simulate strategies.

# 📈 PredictAI: Crypto & Stock Prediction Dashboard

**PredictAI** is a comprehensive **Streamlit** web application designed to navigate developers through the complete lifecycle of AI-driven trading. It bridges the gap between raw financial data and actionable deep learning insights across four developmental phases.

---

### 🚀 **Project Roadmap**

* **Phase 1: Data & Visualization** – Interactive Candlestick charts with MA20/MA50 overlays and volume tracking via `yfinance` and `Plotly`.
* **Phase 2: Linear Regression** – Baseline ML forecasting using feature engineering (RSI, MACD) and `MinMaxScaler` normalization.
* **Phase 3: LSTM Deep Learning** – Advanced 2-layer **LSTM Neural Network** using a 60-day sliding window to predict next-day price action.
* **Phase 4: Backtest Engine** – Quantitative simulation of 4 trading strategies to measure Alpha, Sharpe Ratio, and Max Drawdown.

---

### 🛠️ **Tech Stack**

| Category | Tools |
| :--- | :--- |
| **Frontend** | Streamlit |
| **Data & Finance** | Pandas, NumPy, yfinance |
| **Machine Learning** | scikit-learn, TensorFlow, Keras |
| **Visualization** | Plotly |

---

### 🧠 **Key Features**

* **Smart Feature Engineering**: Includes RSI(14), MACD, Volume Change %, and Lag Returns to capture market momentum.
* **Robust Deep Learning**: LSTM architecture features **Dropout layers** for anti-overfitting and **Huber Loss** for outlier resistance.
* **Performance Metrics**: Compare AI strategies against a standard **Buy & Hold** benchmark using real-world trade logs.

---

### 💻 **Quick Start**

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Run Application**:
    ```bash
    streamlit run app.py
    ```
3.  **Access**: Open `http://localhost:8501` in your browser.

---

> [!WARNING]
> **Disclaimer**: This project is for educational purposes only. Past performance does not guarantee future returns.
