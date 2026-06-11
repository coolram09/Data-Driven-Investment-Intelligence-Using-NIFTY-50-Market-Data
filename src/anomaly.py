"""
Market Anomaly Detection Module.

Flags days where a stock exhibited unusual behavior relative to its own
recent history:
  - Sudden volatility spikes (return Z-score beyond threshold)
  - Extreme single-day drawdowns / rallies
  - Unusual trading volume (volume Z-score beyond threshold)
"""

import pandas as pd
import numpy as np


def detect_anomalies(df: pd.DataFrame, return_window: int = 60, volume_window: int = 60,
                      return_z_thresh: float = 3.0, volume_z_thresh: float = 3.0) -> pd.DataFrame:
    """Given a single-symbol DataFrame (sorted by Date), return a DataFrame
    of flagged anomaly events with a description."""
    df = df.sort_values("Date").reset_index(drop=True).copy()

    df["Return_RollMean"] = df["Return"].rolling(return_window, min_periods=20).mean()
    df["Return_RollStd"] = df["Return"].rolling(return_window, min_periods=20).std()
    df["Return_Z"] = (df["Return"] - df["Return_RollMean"]) / df["Return_RollStd"]

    df["Volume_RollMean"] = df["Volume"].rolling(volume_window, min_periods=20).mean()
    df["Volume_RollStd"] = df["Volume"].rolling(volume_window, min_periods=20).std()
    df["Volume_Z"] = (df["Volume"] - df["Volume_RollMean"]) / df["Volume_RollStd"]

    anomalies = []
    for _, row in df.iterrows():
        events = []
        if pd.notna(row["Return_Z"]) and abs(row["Return_Z"]) >= return_z_thresh:
            direction = "spike up" if row["Return_Z"] > 0 else "sharp drop"
            events.append(f"Volatility {direction} (return {row['Return']*100:.1f}%, "
                           f"z-score {row['Return_Z']:.1f})")
        if pd.notna(row["Volume_Z"]) and row["Volume_Z"] >= volume_z_thresh:
            events.append(f"Unusual trading volume (z-score {row['Volume_Z']:.1f})")

        if events:
            anomalies.append({
                "Date": row["Date"],
                "Close": row["Close"],
                "Return": row["Return"],
                "Volume": row["Volume"],
                "Return_Z": row["Return_Z"],
                "Volume_Z": row["Volume_Z"],
                "Events": "; ".join(events),
            })

    return pd.DataFrame(anomalies)


def detect_extreme_drawdown_periods(df: pd.DataFrame, threshold: float = -0.30) -> pd.DataFrame:
    """Identify periods where the stock fell `threshold` (e.g. -30%) or
    more from a recent peak, returning the peak date, trough date and
    drawdown magnitude for each distinct episode."""
    df = df.sort_values("Date").reset_index(drop=True).copy()
    prices = df["Close"].values
    dates = df["Date"].values

    cum_max = -np.inf
    cum_max_date = None
    episodes = []
    in_drawdown = False
    episode_start = None
    trough = np.inf
    trough_date = None

    for i, (price, date) in enumerate(zip(prices, dates)):
        if price > cum_max:
            if in_drawdown:
                dd = (trough - cum_max) / cum_max
                if dd <= threshold:
                    episodes.append({
                        "Peak Date": cum_max_date,
                        "Peak Price": cum_max,
                        "Trough Date": trough_date,
                        "Trough Price": trough,
                        "Drawdown": dd,
                        "Recovery Date": date,
                    })
                in_drawdown = False
            cum_max = price
            cum_max_date = date
            trough = np.inf
        else:
            in_drawdown = True
            if price < trough:
                trough = price
                trough_date = date

    # Handle ongoing drawdown at end of data
    if in_drawdown:
        dd = (trough - cum_max) / cum_max
        if dd <= threshold:
            episodes.append({
                "Peak Date": cum_max_date,
                "Peak Price": cum_max,
                "Trough Date": trough_date,
                "Trough Price": trough,
                "Drawdown": dd,
                "Recovery Date": None,
            })

    return pd.DataFrame(episodes)


if __name__ == "__main__":
    from data_utils import load_processed_data

    data = load_processed_data()
    rel = data[data["Symbol"] == "RELIANCE"]
    anomalies = detect_anomalies(rel)
    print(f"Found {len(anomalies)} anomaly days")
    print(anomalies.tail(5))
    print(detect_extreme_drawdown_periods(rel))
