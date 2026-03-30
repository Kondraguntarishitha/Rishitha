# 📊 StockSense — AI-Based Stock Risk Analysis & Investment Advisor

A full-stack AI-powered stock risk analysis platform built with Python, Streamlit, and Yahoo Finance.

---

## 🏗️ Project Architecture

```
User (Browser)
     ↓
app.py          →  Layer 2: Frontend  (Streamlit UI)
     ↓
backend.py      →  Layer 3: ML Risk Engine  (metrics + classification + advice)
     ↓
stock_data.py   →  Layer 4: Stock Data API  (Yahoo Finance via yfinance)
```

## 📁 File Structure

```
stock_analyzer/
├── app.py            # Streamlit frontend — all charts, cards, UI
├── backend.py        # Risk engine — metrics, scoring, advice generation
├── stock_data.py     # Data layer — Yahoo Finance fetching + indicators
├── requirements.txt  # Python dependencies
└── README.md
```

---

## ⚙️ Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the app
```bash
streamlit run app.py
```

### 3. Open in browser
```
http://localhost:8501
```

---

## 🔍 How to Use

1. Enter a **stock ticker** in the sidebar (e.g. `AAPL`, `TSLA`, `TCS.NS`)
2. Select the **analysis period** (3M, 6M, 1Y, 2Y, 5Y)
3. Click **Analyze Stock**
4. View full risk dashboard, charts, and investment advice

### Supported Exchanges
| Exchange | Example Tickers |
|----------|----------------|
| NASDAQ   | AAPL, TSLA, MSFT, NVDA |
| NYSE     | JPM, BAC, KO |
| NSE (India) | TCS.NS, INFY.NS, RELIANCE.NS |
| BSE (India) | TCS.BO, INFY.BO |

---

## 📊 What the App Analyzes

### Risk Metrics
| Metric | Description |
|--------|-------------|
| **Volatility** | Annualized standard deviation of daily returns |
| **Sharpe Ratio** | Return earned per unit of total risk (>1 is good) |
| **Sortino Ratio** | Like Sharpe but only penalizes downside risk |
| **Beta** | How much the stock moves relative to S&P 500 |
| **Max Drawdown** | Largest peak-to-trough price decline |
| **VaR 95%** | Worst expected daily loss 95% of the time |
| **Calmar Ratio** | Annual return divided by max drawdown |

### Technical Indicators
- **MA20, MA50, MA200** — Simple Moving Averages
- **EMA12, EMA26** — Exponential Moving Averages
- **MACD + Signal** — Momentum indicator
- **Bollinger Bands** — Volatility bands (±2σ)
- **RSI (14-day)** — Overbought/oversold detector

### Risk Classification Model
```
Score 0–5  → LOW RISK
Score 6–11 → MEDIUM RISK
Score 12+  → HIGH RISK
```
Each of 5 metrics (Volatility, Drawdown, Sharpe, Beta, VaR) contributes 0–4 points.

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend UI | Streamlit |
| Charts | Plotly |
| Data Fetching | yfinance |
| Data Processing | Pandas, NumPy |
| Risk Engine | Custom Python rule-based model |

---

> ⚠️ **Disclaimer**: This tool is for educational purposes only. Not financial advice.
