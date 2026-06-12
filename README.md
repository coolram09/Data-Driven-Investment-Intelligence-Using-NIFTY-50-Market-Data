
NIFTY-50 Investment Intelligence Platform

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Built with Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Live Demo](https://img.shields.io/badge/Live-Demo-success?style=for-the-badge)](https://data-driven-investment-intelligence-using-nifty-50-market-data.streamlit.app/)

An AI-powered **investment intelligence dashboard** built on the NIFTY-50
historical market dataset (Jan 2000 – Apr 2021). The platform transforms
raw OHLCV data into actionable investment intelligence: price forecasts,
risk diagnostics, optimized portfolios, and market anomaly alerts — all
through an interactive Streamlit interface.

>  **Disclaimer**: This project is for educational purposes only. Predictions
> and portfolio recommendations are statistical projections based on historical
> data and **do not constitute financial advice**.

---

##  Features

| Module | Description |
|---|---|
| **Market Overview** | Risk-return landscape, sector-wise performance, top/bottom performers |
|  **Stock Analysis** | Interactive candlestick charts with MA, EMA, Bollinger Bands, RSI, MACD, volatility & momentum |
|  **Stock Predictor Engine** | Random Forest regression (future return) + classification (direction), evaluated via MAE, RMSE, R², and Directional Accuracy on a chronological train/test split |
|  **Risk Assessment** | Volatility, Sharpe Ratio, Sortino Ratio, Max Drawdown, Calmar Ratio, VaR (95%/99%), Beta |
|  **Portfolio Construction** | Mean-variance optimized portfolios for Conservative, Balanced, and Aggressive investor profiles, with full explainability |
|  **Anomaly Detection** | Z-score based detection of volatility/volume spikes and major drawdown episodes |

---

 ### Live Demo

🔗 Streamlit App: https://data-driven-investment-intelligence-using-nifty-50-market-data.streamlit.app/
#### Application Preview

![Market Overview](docs/market_overview.png)

<details>
<summary>View More</summary>

![Stock Analysis](docs/stock_analysis.png)

![Stock Predictor](docs/stock_predictor_engine.png)

![Risk Assessment](docs/risk_assessment.png)

![Portfolio Construction](docs/portfolio_construction.png)

![Market Anomaly](docs/market_anomaly.png)

</details>




##  Quick Start

### 1. Environment Setup

Requires **Python 3.9+**.

```bash
# Clone the repository
git clone https://github.com/coolram09/Data-Driven-Investment-Intelligence-Using-NIFTY-50-Market-Data.git
cd Data-Driven-Investment-Intelligence-Using-NIFTY-50-Market-Data

# (Recommended) create a virtual environment
python -m venv venv
source venv/bin/activate          # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Application

```bash
streamlit run app.py
```

The app will open automatically at **http://localhost:8501**.

---

##  Reproducing Results

All results in the accompanying Technical Report (`Technical_Report.pdf`) are
generated directly from the modules in `src/` against `https://www.kaggle.com/datasets/rohanrao/nifty50-stock-market-data/data`.
To reproduce them:

1. **EDA & Market Overview stats** — open the dashboard's ** Market Overview**
   page; set the lookback slider to **5 years** to match the report.
2. **Predictor metrics** — open ** Stock Predictor**, select a symbol (e.g.
   `RELIANCE`), set horizon to **5 days** and model type to **Random Forest**.
   The displayed MAE / RMSE / R² / Directional Accuracy match the report's
   methodology (chronological 80/20 split, `random_state=42`).
3. **Risk metrics** — open ** Risk Assessment → Single Stock**, select a
   symbol, and keep the risk-free rate at the default **6%**.
4. **Portfolios** — open **Portfolio Construction**, select a profile
   (Conservative / Balanced / Aggressive) with a **5-year lookback**.
5. **Anomalies** — open **Anomaly Detection**, select a symbol, and keep the
   default Z-score thresholds (**3.0**).

All randomness is seeded (`random_state=42` for Random Forest models), so
results are deterministic given the same data and parameters.

To regenerate the report figures/PDF directly from the command line instead
of the dashboard:

```bash
cd report
python generate_analysis.py   # recomputes all stats & saves figures to figs/
python build_report.py        # rebuilds Technical_Report.pdf
```

---

## 📁 Project Structure

```
project/
├── app.py                     # Streamlit dashboard (entry point)
├── requirements.txt           # Python dependencies
├── README.md
├── data/
│   ├── NIFTY50_all.csv         # Consolidated historical OHLCV data (50 stocks, 2000-2021)
│   └── NSE_Symbols.csv         # NSE symbol -> company name reference
├── src/
│   ├── data_utils.py            # Data loading, symbol consolidation, sector mapping
│   ├── indicators.py            # Technical indicators (MA, EMA, RSI, MACD, Bollinger, etc.)
│   ├── predictor.py              # Stock predictor engine (RF regression + classification)
│   ├── risk.py                   # Risk metrics (Sharpe, Sortino, drawdown, VaR, beta)
│   ├── portfolio.py               # Mean-variance portfolio construction
│   └── anomaly.py                  # Market anomaly & drawdown detection
└── report/
    ├── generate_analysis.py        # Regenerates all report figures/stats
    ├── build_report.py             # Builds Technical_Report.pdf
    └── figs/                        # Generated chart images
```

---

##  Methodology Summary

- **Data**: NIFTY-50 historical OHLCV (Jan 2000 – Apr 2021), with renamed
  symbols consolidated into continuous series (e.g. `INFOSYSTCH → INFY`,
  `HEROHONDA → HEROMOTOCO`, `MUNDRAPORT → ADANIPORTS`).
- **Features**: Moving averages, EMAs, RSI, MACD, Bollinger Bands, rolling
  volatility, momentum, and volume ratios.
- **Predictor**: Random Forest Regressor + Classifier, chronological
  (walk-forward) 80/20 train/test split to avoid look-ahead bias.
- **Risk**: Annualized return/volatility, Sharpe, Sortino, Max Drawdown,
  Calmar, VaR, Beta vs. equal-weighted NIFTY-50.
- **Portfolios**: SciPy SLSQP mean-variance optimization over risk-filtered
  candidate universes (Conservative → min variance, Balanced → max Sharpe,
  Aggressive → max return), with per-stock weight caps.
- **Anomaly Detection**: Rolling Z-score thresholds on returns and volume,
  plus peak-to-trough drawdown episode detection (≥30%).

For full details, see [`Technical_Report.pdf`](Technical_Report.pdf).

---

##  Notes & Assumptions

- Symbols renamed over the dataset's history (e.g. `INFOSYSTCH → INFY`,
  `HEROHONDA → HEROMOTOCO`, `MUNDRAPORT → ADANIPORTS`, `TELCO → TATAMOTORS`,
  `UTIBANK → AXISBANK`) have been consolidated into a single continuous
  series per company — see `SYMBOL_CONSOLIDATION` in `src/data_utils.py`.
- Sector/industry classification is manually curated for all 50 constituents
  (`SECTOR_MAP` in `src/data_utils.py`), since the provided dataset does not
  include sector metadata.
- Risk-free rate defaults to **6%** (typical Indian short-term government
  rate) and is adjustable in the dashboard.
- **Only the provided datasets are used** — no live market data, financial
  APIs, news, or alternative data sources, per the challenge constraints.

---


 
##  Acknowledgements

- Dataset: [NIFTY-50 Stock Market Data](https://www.kaggle.com/datasets/rohanrao/nifty50-stock-market-data) (Kaggle)
