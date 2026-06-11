"""
Risk Assessment Module.

Computes standard risk / risk-adjusted return metrics for a single stock's
(or portfolio's) daily return series.
"""

import pandas as pd
import numpy as np

TRADING_DAYS = 252


def annualized_return(returns: pd.Series) -> float:
    returns = returns.dropna()
    if len(returns) == 0:
        return np.nan
    mean_daily = returns.mean()
    return (1 + mean_daily) ** TRADING_DAYS - 1


def annualized_volatility(returns: pd.Series) -> float:
    returns = returns.dropna()
    if len(returns) == 0:
        return np.nan
    return returns.std() * np.sqrt(TRADING_DAYS)


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.06) -> float:
    """Annualized Sharpe ratio. Default risk-free rate ~ 6% (typical Indian
    short-term govt rate) — can be overridden."""
    returns = returns.dropna()
    if len(returns) == 0:
        return np.nan
    excess_daily = returns - (risk_free_rate / TRADING_DAYS)
    std = excess_daily.std()
    if std == 0 or np.isnan(std):
        return np.nan
    return (excess_daily.mean() / std) * np.sqrt(TRADING_DAYS)


def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.06) -> float:
    """Annualized Sortino ratio (uses downside deviation only)."""
    returns = returns.dropna()
    if len(returns) == 0:
        return np.nan
    excess_daily = returns - (risk_free_rate / TRADING_DAYS)
    downside = excess_daily[excess_daily < 0]
    if len(downside) == 0:
        return np.nan
    downside_std = downside.std()
    if downside_std == 0 or np.isnan(downside_std):
        return np.nan
    return (excess_daily.mean() / downside_std) * np.sqrt(TRADING_DAYS)


def max_drawdown(prices: pd.Series) -> float:
    """Maximum drawdown (most negative peak-to-trough decline), as a
    negative fraction (e.g. -0.45 = -45%)."""
    prices = prices.dropna()
    if len(prices) == 0:
        return np.nan
    cumulative_max = prices.cummax()
    drawdown = (prices - cumulative_max) / cumulative_max
    return drawdown.min()


def calmar_ratio(returns: pd.Series, prices: pd.Series) -> float:
    """Annualized return / |max drawdown|."""
    mdd = max_drawdown(prices)
    if mdd == 0 or np.isnan(mdd):
        return np.nan
    return annualized_return(returns) / abs(mdd)


def value_at_risk(returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical Value at Risk (daily), as a negative fraction."""
    returns = returns.dropna()
    if len(returns) == 0:
        return np.nan
    return np.percentile(returns, (1 - confidence) * 100)


def beta(returns: pd.Series, market_returns: pd.Series) -> float:
    """Beta of the asset relative to a market benchmark series."""
    df = pd.concat([returns, market_returns], axis=1).dropna()
    if len(df) < 2:
        return np.nan
    cov = np.cov(df.iloc[:, 0], df.iloc[:, 1])[0, 1]
    var = np.var(df.iloc[:, 1])
    if var == 0:
        return np.nan
    return cov / var


def risk_summary(df: pd.DataFrame, risk_free_rate: float = 0.06,
                  market_returns: pd.Series = None) -> dict:
    """Compute a full risk summary dict for a single-symbol DataFrame
    containing 'Close' and 'Return' columns."""
    returns = df["Return"]
    prices = df["Close"]

    summary = {
        "Annualized Return": annualized_return(returns),
        "Annualized Volatility": annualized_volatility(returns),
        "Sharpe Ratio": sharpe_ratio(returns, risk_free_rate),
        "Sortino Ratio": sortino_ratio(returns, risk_free_rate),
        "Max Drawdown": max_drawdown(prices),
        "Calmar Ratio": calmar_ratio(returns, prices),
        "VaR 95% (daily)": value_at_risk(returns, 0.95),
        "VaR 99% (daily)": value_at_risk(returns, 0.99),
    }
    if market_returns is not None:
        summary["Beta"] = beta(returns, market_returns)
    return summary


def classify_risk_level(volatility: float) -> str:
    """Bucket annualized volatility into a qualitative risk level."""
    if np.isnan(volatility):
        return "Unknown"
    if volatility < 0.25:
        return "Low"
    elif volatility < 0.40:
        return "Medium"
    else:
        return "High"


def portfolio_returns(returns_df: pd.DataFrame, weights: dict) -> pd.Series:
    """Given a DataFrame of per-symbol daily returns (columns = symbols,
    aligned on Date index) and a dict of weights, compute the weighted
    portfolio daily return series."""
    symbols = [s for s in weights if s in returns_df.columns]
    w = np.array([weights[s] for s in symbols])
    w = w / w.sum()
    sub = returns_df[symbols].dropna()
    return sub.dot(w)


if __name__ == "__main__":
    from data_utils import load_processed_data

    data = load_processed_data()
    rel = data[data["Symbol"] == "RELIANCE"]
    print(risk_summary(rel))
