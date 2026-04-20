# PredictAI
PredictAI is a Streamlit dashboard for crypto and stock forecasting. It features four phases: interactive Plotly charts , Linear Regression , LSTM deep learning with TensorFlow , and a backtesting engine. It uses feature engineering and MinMaxScaler to predict prices and simulate strategies.
# 📈 PredictAI: Crypto & Stock Prediction Dashboard

[cite_start]**PredictAI** is a comprehensive **Streamlit** web application designed to navigate developers through the complete lifecycle of AI-driven trading[cite: 1, 7]. [cite_start]It bridges the gap between raw financial data and actionable deep learning insights across four developmental phases[cite: 3, 4].

---

### 🚀 **Project Roadmap**

* [cite_start]**Phase 1: Data & Visualization** – Interactive Candlestick charts with MA20/MA50 overlays and volume tracking via `yfinance` and `Plotly`[cite: 17].
* [cite_start]**Phase 2: Linear Regression** – Baseline ML forecasting using feature engineering (RSI, MACD) and `MinMaxScaler` normalization[cite: 18].
* [cite_start]**Phase 3: LSTM Deep Learning** – Advanced 2-layer **LSTM Neural Network** using a 60-day sliding window to predict next-day price action[cite: 19, 34].
* [cite_start]**Phase 4: Backtest Engine** – Quantitative simulation of 4 trading strategies to measure Alpha, Sharpe Ratio, and Max Drawdown[cite: 20, 48].

---

### 🛠️ **Tech Stack**

| Category | Tools |
| :--- | :--- |
| **Frontend** | [cite_start]Streamlit [cite: 5] |
| **Data & Finance** | [cite_start]Pandas, NumPy, yfinance [cite: 5, 20] |
| [cite_start]**Machine Learning** | scikit-learn, TensorFlow, Keras [cite: 5, 19] |
| **Visualization** | [cite_start]Plotly [cite: 5] |

---

### 🧠 **Key Features**

* [cite_start]**Smart Feature Engineering**: Includes RSI(14), MACD, Volume Change %, and Lag Returns to capture market momentum[cite: 23].
* [cite_start]**Robust Deep Learning**: LSTM architecture features **Dropout layers** for anti-overfitting and **Huber Loss** for outlier resistance[cite: 34].
* [cite_start]**Performance Metrics**: Compare AI strategies against a standard **Buy & Hold** benchmark using real-world trade logs[cite: 20, 48].

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
3.  [cite_start]**Access**: Open `http://localhost:8501` in your browser[cite: 15].

---

> [!WARNING]
> **Disclaimer**: This project is for educational purposes only. [cite_start]Past performance does not guarantee future returns[cite: 67].
