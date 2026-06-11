"""
Portfolio Construction Module.

Builds recommended portfolios for three investor profiles (Conservative,
Balanced, Aggressive) using Modern Portfolio Theory (mean-variance
optimization) over a candidate universe of NIFTY-50 stocks, filtered by
each stock's historical risk profile.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from risk import annualized_return, annualized_volatility, sharpe_ratio, classify_risk_level

TRADING_DAYS = 252

PROFILE_CONFIG = {
    "Conservative": {
        "risk_levels": ["Low"],
        "max_weight": 0.25,
        "objective": "min_volatility",
        "description": (
            "Prioritizes capital preservation. Invests only in stocks with "
            "historically low volatility, with a portfolio constructed to "
            "minimize overall variance and capped single-stock exposure."
        ),
    },
    "Balanced": {
        "risk_levels": ["Low", "Medium"],
        "max_weight": 0.20,
        "objective": "max_sharpe",
        "description": (
            "Seeks a balance between growth and stability by combining "
            "low- and medium-volatility stocks, optimizing for the best "
            "risk-adjusted return (Sharpe ratio)."
        ),
    },
    "Aggressive": {
        "risk_levels": ["Low", "Medium", "High"],
        "max_weight": 0.30,
        "objective": "max_return",
        "description": (
            "Targets maximum growth, willing to accept higher volatility. "
            "Draws from the full universe of stocks (including high "
            "volatility names) and tilts toward the highest historical "
            "returns, subject to a single-stock exposure cap."
        ),
    },
}


def build_returns_matrix(df: pd.DataFrame, symbols=None, start_date=None, end_date=None) -> pd.DataFrame:
    """Pivot the long-format price/return DataFrame into a wide matrix of
    daily returns (Date index x Symbol columns)."""
    sub = df.copy()
    if symbols is not None:
        sub = sub[sub["Symbol"].isin(symbols)]
    if start_date is not None:
        sub = sub[sub["Date"] >= start_date]
    if end_date is not None:
        sub = sub[sub["Date"] <= end_date]

    matrix = sub.pivot(index="Date", columns="Symbol", values="Return")
    return matrix


def stock_stats_table(df: pd.DataFrame, lookback_years: int = 5) -> pd.DataFrame:
    """Compute per-symbol annualized return, volatility, Sharpe and risk
    classification, restricted to the last `lookback_years` years of data
    (so newer/older listings are compared on a consistent basis where
    possible)."""
    end_date = df["Date"].max()
    start_date = end_date - pd.DateOffset(years=lookback_years)

    rows = []
    for sym, g in df.groupby("Symbol"):
        g = g[g["Date"] >= start_date]
        if len(g) < 100:
            g = df[df["Symbol"] == sym]  # fall back to full history
        ret = g["Return"]
        ann_ret = annualized_return(ret)
        ann_vol = annualized_volatility(ret)
        rows.append({
            "Symbol": sym,
            "Annualized Return": ann_ret,
            "Annualized Volatility": ann_vol,
            "Sharpe Ratio": sharpe_ratio(ret),
            "Risk Level": classify_risk_level(ann_vol),
            "Data Points": len(g),
        })
    return pd.DataFrame(rows).set_index("Symbol")


def _portfolio_perf(weights, mean_returns, cov_matrix):
    port_return = np.dot(weights, mean_returns) * TRADING_DAYS
    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(TRADING_DAYS)
    return port_return, port_vol


def _negative_sharpe(weights, mean_returns, cov_matrix, risk_free_rate):
    p_ret, p_vol = _portfolio_perf(weights, mean_returns, cov_matrix)
    if p_vol == 0:
        return 0
    return -(p_ret - risk_free_rate) / p_vol


def _portfolio_volatility(weights, mean_returns, cov_matrix):
    return _portfolio_perf(weights, mean_returns, cov_matrix)[1]


def _negative_return(weights, mean_returns, cov_matrix):
    return -_portfolio_perf(weights, mean_returns, cov_matrix)[0]


def optimize_portfolio(returns_matrix: pd.DataFrame, objective: str = "max_sharpe",
                        max_weight: float = 0.25, risk_free_rate: float = 0.06) -> pd.Series:
    """Mean-variance optimization over the columns (symbols) of
    `returns_matrix`. Returns a Series of weights indexed by symbol."""
    returns_matrix = returns_matrix.dropna(axis=0, how="any")
    symbols = returns_matrix.columns.tolist()
    n = len(symbols)

    mean_returns = returns_matrix.mean().values
    cov_matrix = returns_matrix.cov().values

    effective_max = max(max_weight, 1.0 / n)
    bounds = tuple((0.0, effective_max) for _ in range(n))
    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1},)
    init_guess = np.repeat(1 / n, n)

    if objective == "max_sharpe":
        result = minimize(_negative_sharpe, init_guess,
                           args=(mean_returns, cov_matrix, risk_free_rate),
                           method="SLSQP", bounds=bounds, constraints=constraints)
    elif objective == "min_volatility":
        result = minimize(_portfolio_volatility, init_guess,
                           args=(mean_returns, cov_matrix),
                           method="SLSQP", bounds=bounds, constraints=constraints)
    elif objective == "max_return":
        result = minimize(_negative_return, init_guess,
                           args=(mean_returns, cov_matrix),
                           method="SLSQP", bounds=bounds, constraints=constraints)
    else:
        raise ValueError(f"Unknown objective: {objective}")

    weights = result.x
    weights = np.clip(weights, 0, None)
    weights = weights / weights.sum()
    return pd.Series(weights, index=symbols).sort_values(ascending=False)


def build_portfolio(df: pd.DataFrame, profile: str, lookback_years: int = 5,
                     min_weight_threshold: float = 0.01,
                     candidate_limit: int = 15) -> dict:
    """Construct a recommended portfolio for the given investor profile.

    Returns a dict with weights, expected performance, stock-level stats
    and a textual rationale.
    """
    if profile not in PROFILE_CONFIG:
        raise ValueError(f"Unknown profile: {profile}")
    config = PROFILE_CONFIG[profile]

    stats = stock_stats_table(df, lookback_years=lookback_years)
    candidates = stats[stats["Risk Level"].isin(config["risk_levels"])].copy()

    # Rank candidates by Sharpe (or return for aggressive) and limit universe
    if config["objective"] == "max_return":
        candidates = candidates.sort_values("Annualized Return", ascending=False)
    else:
        candidates = candidates.sort_values("Sharpe Ratio", ascending=False)
    candidates = candidates.head(candidate_limit)

    end_date = df["Date"].max()
    start_date = end_date - pd.DateOffset(years=lookback_years)
    returns_matrix = build_returns_matrix(df, symbols=candidates.index.tolist(),
                                           start_date=start_date)
    # Require columns with reasonably full history over the window
    valid_cols = returns_matrix.columns[returns_matrix.notna().mean() > 0.9]
    returns_matrix = returns_matrix[valid_cols]

    weights = optimize_portfolio(returns_matrix, objective=config["objective"],
                                  max_weight=config["max_weight"])
    weights = weights[weights >= min_weight_threshold]
    weights = weights / weights.sum()

    port_returns = returns_matrix[weights.index].dropna().dot(weights)
    perf = {
        "Expected Annual Return": annualized_return(port_returns),
        "Expected Annual Volatility": annualized_volatility(port_returns),
        "Sharpe Ratio": sharpe_ratio(port_returns),
    }

    return {
        "profile": profile,
        "description": config["description"],
        "weights": weights,
        "performance": perf,
        "candidate_stats": stats.loc[weights.index],
        "all_candidates": candidates,
    }


if __name__ == "__main__":
    from data_utils import load_processed_data

    data = load_processed_data()
    for profile in ["Conservative", "Balanced", "Aggressive"]:
        result = build_portfolio(data, profile)
        print("=" * 50)
        print(profile)
        print(result["weights"])
        print(result["performance"])
