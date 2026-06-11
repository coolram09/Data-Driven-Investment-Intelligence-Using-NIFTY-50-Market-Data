"""
Technical indicator computation for the NIFTY-50 Investment Intelligence
Platform.

All functions operate on a single symbol's price DataFrame (sorted by Date,
ascending) and return the DataFrame with new columns appended.
"""

import pandas as pd
import numpy as np


def add_moving_averages(df: pd.DataFrame, windows=(5, 10, 20, 50, 100, 200)) -> pd.DataFrame:
    df = df.copy()
    for w in windows:
        df[f"MA_{w}"] = df["Close"].rolling(window=w, min_periods=w).mean()
    return df


def add_ema(df: pd.DataFrame, spans=(12, 26, 50)) -> pd.DataFrame:
    df = df.copy()
    for s in spans:
        df[f"EMA_{s}"] = df["Close"].ewm(span=s, adjust=False).mean()
    return df


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    df = df.copy()
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50)  # neutral when avg_loss == 0
    df[f"RSI_{period}"] = rsi
    return df


def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    df = df.copy()
    ema_fast = df["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    df["MACD"] = macd_line
    df["MACD_Signal"] = signal_line
    df["MACD_Hist"] = macd_line - signal_line
    return df


def add_bollinger_bands(df: pd.DataFrame, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    df = df.copy()
    rolling_mean = df["Close"].rolling(window=window, min_periods=window).mean()
    rolling_std = df["Close"].rolling(window=window, min_periods=window).std()
    df["BB_Mid"] = rolling_mean
    df["BB_Upper"] = rolling_mean + num_std * rolling_std
    df["BB_Lower"] = rolling_mean - num_std * rolling_std
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / df["BB_Mid"]
    df["BB_PctB"] = (df["Close"] - df["BB_Lower"]) / (df["BB_Upper"] - df["BB_Lower"])
    return df


def add_volatility(df: pd.DataFrame, windows=(10, 20, 30)) -> pd.DataFrame:
    """Rolling annualized volatility based on daily log returns."""
    df = df.copy()
    if "LogReturn" not in df.columns:
        df["LogReturn"] = np.log(df["Close"] / df["Close"].shift(1))
    for w in windows:
        df[f"Volatility_{w}"] = df["LogReturn"].rolling(window=w, min_periods=w).std() * np.sqrt(252)
    return df


def add_momentum(df: pd.DataFrame, windows=(5, 10, 20, 60)) -> pd.DataFrame:
    """Rate-of-change momentum: percentage change over N days."""
    df = df.copy()
    for w in windows:
        df[f"Momentum_{w}"] = df["Close"].pct_change(periods=w)
    return df


def add_avg_volume(df: pd.DataFrame, windows=(10, 20)) -> pd.DataFrame:
    df = df.copy()
    for w in windows:
        df[f"AvgVolume_{w}"] = df["Volume"].rolling(window=w, min_periods=w).mean()
        df[f"VolumeRatio_{w}"] = df["Volume"] / df[f"AvgVolume_{w}"]
    return df


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Convenience function: apply the full indicator suite to a single-symbol
    DataFrame, sorted ascending by date."""
    df = df.sort_values("Date").reset_index(drop=True)
    df = add_moving_averages(df)
    df = add_ema(df)
    df = add_rsi(df)
    df = add_macd(df)
    df = add_bollinger_bands(df)
    df = add_volatility(df)
    df = add_momentum(df)
    df = add_avg_volume(df)
    return df


def add_indicators_all_symbols(df: pd.DataFrame) -> pd.DataFrame:
    """Apply add_all_indicators per symbol and concatenate."""
    out = []
    for sym, g in df.groupby("Symbol"):
        out.append(add_all_indicators(g))
    return pd.concat(out, ignore_index=True)


if __name__ == "__main__":
    from data_utils import load_processed_data

    data = load_processed_data()
    sample = data[data["Symbol"] == "RELIANCE"]
    sample = add_all_indicators(sample)
    print(sample.tail(3)[
        ["Date", "Close", "MA_20", "EMA_12", "RSI_14", "MACD", "BB_Upper",
         "Volatility_20", "Momentum_10"]
    ])
