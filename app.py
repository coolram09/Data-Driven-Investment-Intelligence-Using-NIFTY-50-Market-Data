"""
NIFTY-50 Investment Intelligence Platform
==========================================
A Streamlit dashboard providing:
  - Market Overview
  - Stock Analysis (technical indicators)
  - Stock Predictor Engine (return / direction forecasting)
  - Risk Assessment
  - Portfolio Construction (Conservative / Balanced / Aggressive)
  - Market Anomaly Detection
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from data_utils import load_processed_data, get_company_name, get_sector, get_symbol_list
from indicators import add_all_indicators
from risk import risk_summary, classify_risk_level, sharpe_ratio, annualized_return, annualized_volatility
from predictor import train_predictor, predict_latest, FEATURE_COLUMNS
from portfolio import build_portfolio, stock_stats_table, PROFILE_CONFIG
from anomaly import detect_anomalies, detect_extreme_drawdown_periods


st.set_page_config(
    page_title="NIFTY-50 Investment Intelligence Platform",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown("""
<style>
.block-container {
    padding-top: 1rem;
}

h1 {
    color: #00C896;
}

[data-testid="stMetricValue"] {
    font-size: 2rem;
}

[data-testid="stSidebar"] {
    background-color: #1e1e2f;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading and processing NIFTY-50 dataset...")
def get_data():
    return load_processed_data()


@st.cache_data(show_spinner="Computing technical indicators...")
def get_symbol_data(_df, symbol):
    sub = _df[_df["Symbol"] == symbol]
    return add_all_indicators(sub)


@st.cache_resource(show_spinner="Training predictive models...")
def get_trained_model(_df, symbol, horizon, model_type):
    sub = _df[_df["Symbol"] == symbol]
    return train_predictor(sub, horizon=horizon, model_type=model_type)


@st.cache_data(show_spinner="Computing market-wide statistics...")
def get_stats_table(_df, lookback_years):
    return stock_stats_table(_df, lookback_years=lookback_years)


@st.cache_data(show_spinner="Optimizing portfolio...")
def get_portfolio(_df, profile, lookback_years):
    return build_portfolio(_df, profile, lookback_years=lookback_years)


data = get_data()
all_symbols = get_symbol_list(data)
data_max_date = data["Date"].max()
data_min_date = data["Date"].min()


def fmt_pct(x):
    return f"{x*100:.2f}%" if pd.notna(x) else "N/A"


def fmt_num(x, decimals=2):
    return f"{x:.{decimals}f}" if pd.notna(x) else "N/A"


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.title("NIFTY-50")
st.sidebar.markdown("### Investment Intelligence Platform")
page = st.sidebar.radio(
    "Navigate",
    [
        " Market Overview",
        " Stock Analysis",
        " Stock Predictor",
        " Risk Assessment",
        "Portfolio Construction",
        " Anomaly Detection",
    ],
)

st.sidebar.markdown("---")
st.sidebar.caption(
    f"Dataset: NIFTY-50 historical data\n\n"
    f"Range: {data_min_date.date()} to {data_max_date.date()}\n\n"
    f"50 companies, {len(data):,} daily records"
)


# ===========================================================================
# PAGE 1: MARKET OVERVIEW
# ===========================================================================
if page == " Market Overview":
    st.title(" NIFTY-50 Market Overview")
    st.markdown("""
### Analyze • Predict • Manage Risk • Build Portfolios

A complete investment analytics platform built on
NIFTY-50 historical market data.
""")

    lookback = st.slider("Lookback period for stats (years)", 1, 10, 5)
    stats = get_stats_table(data, lookback)

    # Add sector / company info
    stats = stats.copy()
    stats["Company"] = [get_company_name(s) for s in stats.index]
    stats["Sector"] = [get_sector(s)[0] for s in stats.index]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Stocks", len(stats))
    col2.metric("Avg. Annual Return", fmt_pct(stats["Annualized Return"].mean()))
    col3.metric("Avg. Annual Volatility", fmt_pct(stats["Annualized Volatility"].mean()))
    col4.metric("Avg. Sharpe Ratio", fmt_num(stats["Sharpe Ratio"].mean()))

    st.subheader("Risk-Return Landscape")
    fig = px.scatter(
        stats.reset_index(), x="Annualized Volatility", y="Annualized Return",
        color="Sector", hover_name="Symbol",
        hover_data={"Company": True, "Sharpe Ratio": ":.2f"},
        size=stats["Sharpe Ratio"].clip(lower=0.01).values,
        title=f"Risk vs. Return ({lookback}-year lookback)",
        labels={"Annualized Volatility": "Annualized Volatility",
                "Annualized Return": "Annualized Return"},
    )
    fig.update_layout(xaxis_tickformat=".0%", yaxis_tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader(f"Top 10 Performers ({lookback}y)")
        top10 = stats.sort_values("Annualized Return", ascending=False).head(10)
        st.dataframe(
            top10[["Company", "Sector", "Annualized Return", "Annualized Volatility", "Sharpe Ratio"]]
            .style.format({"Annualized Return": "{:.2%}", "Annualized Volatility": "{:.2%}", "Sharpe Ratio": "{:.2f}"}),
            use_container_width=True,
        )
    with col_b:
        st.subheader(f"Bottom 10 Performers ({lookback}y)")
        bottom10 = stats.sort_values("Annualized Return", ascending=True).head(10)
        st.dataframe(
            bottom10[["Company", "Sector", "Annualized Return", "Annualized Volatility", "Sharpe Ratio"]]
            .style.format({"Annualized Return": "{:.2%}", "Annualized Volatility": "{:.2%}", "Sharpe Ratio": "{:.2f}"}),
            use_container_width=True,
        )

    st.subheader("Sector-wise Average Performance")
    sector_stats = stats.groupby("Sector")[["Annualized Return", "Annualized Volatility", "Sharpe Ratio"]].mean()
    sector_stats = sector_stats.sort_values("Annualized Return", ascending=False)
    fig2 = px.bar(
        sector_stats.reset_index(), x="Sector", y="Annualized Return",
        color="Sharpe Ratio", title="Average Annualized Return by Sector",
        color_continuous_scale="RdYlGn",
    )
    fig2.update_layout(yaxis_tickformat=".0%")
    st.plotly_chart(fig2, use_container_width=True)


# ===========================================================================
# PAGE 2: STOCK ANALYSIS
# ===========================================================================
elif page == " Stock Analysis":
    st.title(" Stock Analysis")

    symbol = st.selectbox("Select a stock", all_symbols,
                           format_func=lambda s: f"{s} — {get_company_name(s)}")
    sector, industry = get_sector(symbol)
    st.caption(f"**{get_company_name(symbol)}** | Sector: {sector} | Industry: {industry}")

    df = get_symbol_data(data, symbol)

    min_d, max_d = df["Date"].min().date(), df["Date"].max().date()
    default_start = max(min_d, max_d - pd.Timedelta(days=365 * 3))
    date_range = st.slider("Date range", min_value=min_d, max_value=max_d,
                            value=(default_start, max_d))
    plot_df = df[(df["Date"].dt.date >= date_range[0]) & (df["Date"].dt.date <= date_range[1])]

    # Price + Bollinger Bands + Moving averages
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=plot_df["Date"], open=plot_df["Open"], high=plot_df["High"],
        low=plot_df["Low"], close=plot_df["Close"], name="Price"
    ))
    fig.add_trace(go.Scatter(x=plot_df["Date"], y=plot_df["MA_20"], name="MA 20", line=dict(width=1)))
    fig.add_trace(go.Scatter(x=plot_df["Date"], y=plot_df["MA_50"], name="MA 50", line=dict(width=1)))
    fig.add_trace(go.Scatter(x=plot_df["Date"], y=plot_df["BB_Upper"], name="BB Upper",
                              line=dict(width=1, dash="dot", color="gray")))
    fig.add_trace(go.Scatter(x=plot_df["Date"], y=plot_df["BB_Lower"], name="BB Lower",
                              line=dict(width=1, dash="dot", color="gray"), fill="tonexty",
                              fillcolor="rgba(150,150,150,0.1)"))
    fig.update_layout(title=f"{symbol} Price, Moving Averages & Bollinger Bands",
                       xaxis_rangeslider_visible=False, height=500)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=plot_df["Date"], y=plot_df["RSI_14"], name="RSI (14)"))
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
        fig_rsi.update_layout(title="Relative Strength Index (RSI 14)", height=300, yaxis_range=[0, 100])
        st.plotly_chart(fig_rsi, use_container_width=True)

    with col2:
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=plot_df["Date"], y=plot_df["MACD"], name="MACD"))
        fig_macd.add_trace(go.Scatter(x=plot_df["Date"], y=plot_df["MACD_Signal"], name="Signal"))
        fig_macd.add_trace(go.Bar(x=plot_df["Date"], y=plot_df["MACD_Hist"], name="Histogram",
                                   marker_color="lightgray"))
        fig_macd.update_layout(title="MACD", height=300)
        st.plotly_chart(fig_macd, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Scatter(x=plot_df["Date"], y=plot_df["Volatility_20"], name="20-day Volatility"))
        fig_vol.update_layout(title="Annualized 20-day Rolling Volatility", height=300, yaxis_tickformat=".0%")
        st.plotly_chart(fig_vol, use_container_width=True)

    with col4:
        fig_mom = go.Figure()
        fig_mom.add_trace(go.Scatter(x=plot_df["Date"], y=plot_df["Momentum_20"], name="20-day Momentum"))
        fig_mom.update_layout(title="20-day Momentum (Rate of Change)", height=300, yaxis_tickformat=".0%")
        st.plotly_chart(fig_mom, use_container_width=True)

    # Latest snapshot
    st.subheader("Latest Snapshot")
    latest = df.iloc[-1]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Close", f"₹{latest['Close']:.2f}", f"{latest['Return']*100:.2f}%")
    c2.metric("RSI (14)", f"{latest['RSI_14']:.1f}")
    c3.metric("20d Volatility (ann.)", fmt_pct(latest["Volatility_20"]))
    c4.metric("20d Momentum", fmt_pct(latest["Momentum_20"]))
    bb_signal = "Overbought" if latest["BB_PctB"] > 1 else ("Oversold" if latest["BB_PctB"] < 0 else "Neutral")
    c5.metric("Bollinger %B", f"{latest['BB_PctB']:.2f}", bb_signal)


# ===========================================================================
# PAGE 3: STOCK PREDICTOR ENGINE
# ===========================================================================
elif page == " Stock Predictor":
    st.title(" Stock Predictor Engine")
    st.markdown(
        "Forecasts future returns and price direction using a "
        "**Random Forest** model trained on technical indicators "
        "(moving averages, RSI, MACD, Bollinger Bands, volatility, "
        "momentum, volume). Evaluation uses a **chronological train/test "
        "split** (no shuffling) — the model is tested on the most recent "
        "unseen data only."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        symbol = st.selectbox("Select a stock", all_symbols,
                               format_func=lambda s: f"{s} — {get_company_name(s)}", key="pred_symbol")
    with col2:
        horizon = st.select_slider("Forecast horizon (trading days)", options=[1, 3, 5, 10, 20], value=5)
    with col3:
        model_type = st.selectbox("Model type", ["random_forest", "linear"],
                                   format_func=lambda x: "Random Forest" if x == "random_forest" else "Linear Regression + RF Classifier")

    with st.spinner("Training model..."):
        result = get_trained_model(data, symbol, horizon, model_type)

    metrics = result["metrics"]

    st.subheader("Model Evaluation (held-out test period)")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("MAE (return)", f"{metrics['MAE']:.4f}")
    m2.metric("RMSE (return)", f"{metrics['RMSE']:.4f}")
    m3.metric("R² Score", f"{metrics['R2']:.3f}")
    m4.metric("Directional Accuracy", fmt_pct(metrics["Directional Accuracy (Classifier)"]))
    st.caption(
        f"Trained on {metrics['n_train']:,} days, tested on the most recent "
        f"{metrics['n_test']:,} days. Regression-based directional accuracy: "
        f"{fmt_pct(metrics['Directional Accuracy (Regression)'])}."
    )

    # Forecast vs actual chart
    test_dates = pd.to_datetime(result["test_dates"])
    actual_price = result["test_close"]
    pred_return = result["pred_reg"]
    implied_pred_price = actual_price / (1 + result["y_test_reg"]) * (1 + pred_return)
    # Note: actual_price here is Close at time t; FutureReturn predicts price at t+horizon.
    # For visualization we compare actual future return vs predicted future return.

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test_dates, y=result["y_test_reg"], name=f"Actual {horizon}-day return", line=dict(width=1)))
    fig.add_trace(go.Scatter(x=test_dates, y=pred_return, name=f"Predicted {horizon}-day return", line=dict(width=1)))
    fig.update_layout(title=f"Actual vs Predicted {horizon}-Day Forward Return ({symbol})",
                       yaxis_tickformat=".1%", height=400)
    st.plotly_chart(fig, use_container_width=True)

    # Latest forecast
    st.subheader("Latest Forecast")
    latest_pred = predict_latest(data[data["Symbol"] == symbol], result)
    f1, f2, f3, f4 = st.columns(4)
    f1.metric("As of", pd.Timestamp(latest_pred["as_of_date"]).strftime("%Y-%m-%d"))
    f2.metric("Last Close", f"₹{latest_pred['last_close']:.2f}")
    f3.metric(f"Predicted {horizon}-day return", fmt_pct(latest_pred["predicted_return"]))
    f4.metric(f"Predicted Direction", latest_pred["predicted_direction"],
              f"P(up) = {latest_pred['prob_up']:.0%}")
    st.info(
        f"Based on data up to **{pd.Timestamp(latest_pred['as_of_date']).strftime('%Y-%m-%d')}**, "
        f"the model projects a price of **₹{latest_pred['predicted_price']:.2f}** "
        f"({fmt_pct(latest_pred['predicted_return'])}) over the next **{horizon} trading days**, "
        f"with **{latest_pred['prob_up']:.0%} probability** of an upward move. "
        f"⚠️ This is a statistical projection based on historical patterns, not financial advice."
    )

    # Feature importance
    if result["feature_importance"] is not None:
        st.subheader("Feature Importance (Regression Model)")
        fi = result["feature_importance"].head(10).sort_values()
        fig_fi = px.bar(x=fi.values, y=fi.index, orientation="h",
                         labels={"x": "Importance", "y": "Feature"},
                         title="Top 10 Most Important Features")
        st.plotly_chart(fig_fi, use_container_width=True)


# ===========================================================================
# PAGE 4: RISK ASSESSMENT
# ===========================================================================
elif page == " Risk Assessment":
    st.title(" Risk Assessment")

    tab1, tab2 = st.tabs(["Single Stock", "Compare Multiple Stocks"])

    with tab1:
        symbol = st.selectbox("Select a stock", all_symbols,
                               format_func=lambda s: f"{s} — {get_company_name(s)}", key="risk_symbol")
        rf_rate = st.slider("Risk-free rate (annual, for Sharpe/Sortino)", 0.0, 0.10, 0.06, 0.005,
                             format="%.3f")
        df = data[data["Symbol"] == symbol]

        # Use NIFTY index proxy: equal-weighted average of all stocks as market return
        market_returns = data.groupby("Date")["Return"].mean().rename("Market")

        summary = risk_summary(df, risk_free_rate=rf_rate, market_returns=market_returns)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Annualized Return", fmt_pct(summary["Annualized Return"]))
        c2.metric("Annualized Volatility", fmt_pct(summary["Annualized Volatility"]))
        c3.metric("Sharpe Ratio", fmt_num(summary["Sharpe Ratio"]))
        c4.metric("Sortino Ratio", fmt_num(summary["Sortino Ratio"]))

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Max Drawdown", fmt_pct(summary["Max Drawdown"]))
        c6.metric("Calmar Ratio", fmt_num(summary["Calmar Ratio"]))
        c7.metric("VaR 95% (daily)", fmt_pct(summary["VaR 95% (daily)"]))
        c8.metric("Beta (vs NIFTY-50 avg)", fmt_num(summary.get("Beta")))

        risk_level = classify_risk_level(summary["Annualized Volatility"])
        st.markdown(f"### Overall Risk Classification: **{risk_level}**")
        if risk_level == "Low":
            st.success("This stock has historically exhibited relatively low volatility, "
                        "suitable for conservative investors seeking capital preservation.")
        elif risk_level == "Medium":
            st.warning("This stock has historically exhibited moderate volatility, "
                        "suitable for balanced investors seeking growth with managed risk.")
        else:
            st.error("This stock has historically exhibited high volatility, "
                     "suitable only for aggressive investors with high risk tolerance.")

        # Drawdown chart
        df2 = df.sort_values("Date").copy()
        cum_max = df2["Close"].cummax()
        drawdown = (df2["Close"] - cum_max) / cum_max
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(x=df2["Date"], y=drawdown, fill="tozeroy", name="Drawdown",
                                     line=dict(color="firebrick")))
        fig_dd.update_layout(title=f"{symbol} — Historical Drawdown", yaxis_tickformat=".0%", height=350)
        st.plotly_chart(fig_dd, use_container_width=True)

    with tab2:
        st.subheader("Risk-Adjusted Comparison")
        lookback = st.slider("Lookback (years)", 1, 10, 5, key="risk_lookback")
        selected = st.multiselect("Select stocks to compare", all_symbols,
                                   default=["RELIANCE", "TCS", "HDFCBANK", "ITC"],
                                   format_func=lambda s: f"{s} — {get_company_name(s)}")
        if selected:
            stats = get_stats_table(data, lookback)
            comp = stats.loc[selected].copy()
            comp["Company"] = [get_company_name(s) for s in comp.index]
            st.dataframe(
                comp[["Company", "Annualized Return", "Annualized Volatility", "Sharpe Ratio", "Risk Level"]]
                .style.format({"Annualized Return": "{:.2%}", "Annualized Volatility": "{:.2%}", "Sharpe Ratio": "{:.2f}"}),
                use_container_width=True,
            )
            fig = px.bar(comp.reset_index(), x="Symbol", y=["Annualized Return", "Annualized Volatility"],
                          barmode="group", title="Return vs Volatility Comparison")
            fig.update_layout(yaxis_tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)


# ===========================================================================
# PAGE 5: PORTFOLIO CONSTRUCTION
# ===========================================================================
elif page == "Portfolio Construction":
    st.title("Portfolio Construction")
    st.markdown(
        "Generates a recommended portfolio allocation for three investor "
        "profiles using **Modern Portfolio Theory** (mean-variance "
        "optimization) over a candidate universe of NIFTY-50 stocks "
        "filtered by historical risk level."
    )

    col1, col2 = st.columns(2)
    with col1:
        profile = st.selectbox("Select Investor Profile", list(PROFILE_CONFIG.keys()))
    with col2:
        lookback = st.slider("Lookback period (years)", 2, 10, 5)
#added exception to handle any potential error streamliy
    try:
        result = get_portfolio(data, profile, lookback)
    except Exception as e:
        st.error(f"Portfolio Error: {e}")
        st.stop()

    st.info(f"**{profile} Investor**: {result['description']}")

    perf = result["performance"]
    p1, p2, p3 = st.columns(3)
    p1.metric("Expected Annual Return", fmt_pct(perf["Expected Annual Return"]))
    p2.metric("Expected Annual Volatility", fmt_pct(perf["Expected Annual Volatility"]))
    p3.metric("Sharpe Ratio", fmt_num(perf["Sharpe Ratio"]))

    weights = result["weights"]
    weights_df = pd.DataFrame({
        "Symbol": weights.index,
        "Company": [get_company_name(s) for s in weights.index],
        "Sector": [get_sector(s)[0] for s in weights.index],
        "Weight": weights.values,
    })

    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.subheader("Recommended Allocation")
        fig = px.pie(weights_df, values="Weight", names="Symbol",
                      title=f"{profile} Portfolio Allocation", hole=0.4,
                      hover_data=["Company"])
        st.plotly_chart(fig, use_container_width=True)
    with col_b:
        st.subheader("Allocation Details")
        st.dataframe(
            weights_df.style.format({"Weight": "{:.1%}"}),
            use_container_width=True, hide_index=True,
        )

    st.subheader("Stock-Level Statistics (selected holdings)")
    cand_stats = result["candidate_stats"].copy()
    cand_stats["Company"] = [get_company_name(s) for s in cand_stats.index]
    st.dataframe(
        cand_stats[["Company", "Annualized Return", "Annualized Volatility", "Sharpe Ratio", "Risk Level"]]
        .style.format({"Annualized Return": "{:.2%}", "Annualized Volatility": "{:.2%}", "Sharpe Ratio": "{:.2f}"}),
        use_container_width=True,
    )

    with st.expander("Why these stocks? (Explainability)"):
        st.markdown(f"""
        - **Investment objective**: {PROFILE_CONFIG[profile]['objective'].replace('_', ' ').title()}
        - **Eligible risk levels**: {', '.join(PROFILE_CONFIG[profile]['risk_levels'])}
        - **Maximum single-stock weight cap**: {PROFILE_CONFIG[profile]['max_weight']:.0%}
        - The candidate universe is first filtered to stocks whose **{lookback}-year
          annualized volatility** falls in the eligible risk level(s).
        - Among those, the top candidates by {'historical return' if profile == 'Aggressive' else 'Sharpe ratio'}
          are passed to a **mean-variance optimizer**, which solves for the
          weight combination that {'minimizes portfolio variance' if profile == 'Conservative' else ('maximizes the Sharpe ratio' if profile == 'Balanced' else 'maximizes expected return')},
          subject to the weight cap above.
        """)


# ===========================================================================
# PAGE 6: ANOMALY DETECTION
# ===========================================================================
elif page == " Anomaly Detection":
    st.title(" Market Anomaly Detection")
    st.markdown(
        "Flags days with statistically unusual price moves or trading "
        "volume (Z-score based on a 60-day rolling window), and "
        "identifies extended drawdown episodes (≥30% decline from a "
        "recent peak)."
    )

    symbol = st.selectbox("Select a stock", all_symbols,
                           format_func=lambda s: f"{s} — {get_company_name(s)}", key="anomaly_symbol")
    df = data[data["Symbol"] == symbol]

    col1, col2 = st.columns(2)
    with col1:
        return_z = st.slider("Return Z-score threshold", 2.0, 5.0, 3.0, 0.5)
    with col2:
        volume_z = st.slider("Volume Z-score threshold", 2.0, 5.0, 3.0, 0.5)

    anomalies = detect_anomalies(df, return_z_thresh=return_z, volume_z_thresh=volume_z)
    drawdowns = detect_extreme_drawdown_periods(df, threshold=-0.30)

    st.subheader(f"Detected Anomaly Events ({len(anomalies)})")
    if len(anomalies):
        df2 = df.sort_values("Date")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df2["Date"], y=df2["Close"], name="Close", line=dict(color="lightblue")))
        fig.add_trace(go.Scatter(x=anomalies["Date"], y=anomalies["Close"], mode="markers",
                                  name="Anomaly", marker=dict(color="red", size=8, symbol="x")))
        fig.update_layout(title=f"{symbol} — Price with Detected Anomalies", height=400)
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            anomalies[["Date", "Close", "Return", "Volume", "Events"]]
            .sort_values("Date", ascending=False)
            .style.format({"Close": "₹{:.2f}", "Return": "{:.2%}", "Volume": "{:,.0f}"}),
            use_container_width=True,
        )
    else:
        st.write("No anomalies detected with the current thresholds.")

    st.subheader(f"Major Drawdown Episodes (≥ 30% decline) — {len(drawdowns)} found")
    if len(drawdowns):
        st.dataframe(
            drawdowns.style.format({
                "Peak Price": "₹{:.2f}", "Trough Price": "₹{:.2f}", "Drawdown": "{:.1%}",
            }),
            use_container_width=True,
        )
    else:
        st.write("No major drawdown episodes (≥30%) detected for this stock.")


st.sidebar.markdown("---")
st.sidebar.caption(
    "⚠️ **Disclaimer**: This platform is for educational purposes only, "
    "built on historical NIFTY-50 data (Jan 2000 – Apr 2021). Predictions "
    "and recommendations do not constitute financial advice."
)
