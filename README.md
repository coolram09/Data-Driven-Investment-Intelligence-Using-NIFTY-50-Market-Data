# NIFTY-50 Investment Intelligence Platform

An AI-powered investment intelligence dashboard built on the NIFTY-50
historical market dataset (Jan 2000 - Apr 2021).

## Features
- **Market Overview**: risk-return landscape, sector performance, top/bottom performers
- **Stock Analysis**: candlestick charts with MA, EMA, Bollinger Bands, RSI, MACD, volatility, momentum
- **Stock Predictor Engine**: Random Forest regression (future return) + classification (direction),
  evaluated with MAE, RMSE, R², and Directional Accuracy on a chronological train/test split
- **Risk Assessment**: Volatility, Sharpe, Sortino, Max Drawdown, Calmar, VaR, Beta
- **Portfolio Construction**: Mean-variance optimized portfolios for Conservative, Balanced,
  and Aggressive investor profiles, with explanations
- **Anomaly Detection**: Z-score based detection of volatility/volume spikes and major drawdown episodes

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project Structure
```
project/
├── app.py                  # Streamlit dashboard (entry point)
├── data/
│   ├── NIFTY50_all.csv      # Consolidated historical price data
│   └── NSE_Symbols.csv      # NSE symbol -> company name reference
├── src/
│   ├── data_utils.py         # Data loading, symbol consolidation, sector mapping
│   ├── indicators.py         # Technical indicators (MA, EMA, RSI, MACD, Bollinger, etc.)
│   ├── predictor.py           # Stock predictor engine (RF regression + classification)
│   ├── risk.py                # Risk metrics (Sharpe, Sortino, drawdown, VaR, beta)
│   ├── portfolio.py            # Mean-variance portfolio construction
│   └── anomaly.py              # Market anomaly & drawdown detection
└── requirements.txt
```

## Notes & Assumptions
- Symbols that were renamed over the dataset's history (e.g. INFOSYSTCH -> INFY,
  HEROHONDA -> HEROMOTOCO, MUNDRAPORT -> ADANIPORTS, etc.) have been consolidated
  into a single continuous series per company (see `SYMBOL_CONSOLIDATION` in
  `src/data_utils.py`).
- Sector/industry classification is manually curated for the 50 constituents
  (`SECTOR_MAP` in `src/data_utils.py`), since the provided dataset does not
  include sector metadata.
- Risk-free rate defaults to 6% (typical Indian short-term government rate) and
  is adjustable in the dashboard.
- Only the provided datasets are used — no live market data, APIs, or
  alternative data sources.
- Predictions are statistical projections based on historical patterns and do
  not constitute financial advice.
