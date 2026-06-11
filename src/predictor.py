"""
Stock Predictor Engine.

Builds a feature set from technical indicators and trains:
  1. A regression model to forecast the future N-day return.
  2. A classification model to forecast the direction of price movement
     (up / down) over the same horizon.

Models are evaluated using a chronological (no-shuffle) train/test split,
which is appropriate for time-series data.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score

from indicators import add_all_indicators

FEATURE_COLUMNS = [
    "MA_5", "MA_10", "MA_20", "MA_50",
    "EMA_12", "EMA_26", "EMA_50",
    "RSI_14",
    "MACD", "MACD_Signal", "MACD_Hist",
    "BB_Width", "BB_PctB",
    "Volatility_10", "Volatility_20", "Volatility_30",
    "Momentum_5", "Momentum_10", "Momentum_20", "Momentum_60",
    "VolumeRatio_10", "VolumeRatio_20",
    "Return", "LogReturn",
]


def build_feature_table(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """Given a single-symbol price DataFrame (with returns), compute
    indicators and the target variables:
      - FutureReturn:  % change in Close over the next `horizon` days
      - Direction:     1 if FutureReturn > 0 else 0
    Returns a DataFrame ready for model training (NaNs dropped).
    """
    df = add_all_indicators(df)
    df = df.copy()
    df["FutureClose"] = df["Close"].shift(-horizon)
    df["FutureReturn"] = (df["FutureClose"] - df["Close"]) / df["Close"]
    df["Direction"] = (df["FutureReturn"] > 0).astype(int)

    cols_needed = FEATURE_COLUMNS + ["FutureReturn", "Direction", "Date", "Close"]
    df = df.dropna(subset=cols_needed)
    return df


def chronological_split(df: pd.DataFrame, test_size: float = 0.2):
    n = len(df)
    split_idx = int(n * (1 - test_size))
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]
    return train, test


def train_predictor(df: pd.DataFrame, horizon: int = 5, test_size: float = 0.2,
                     model_type: str = "random_forest"):
    """Train regression + classification models for a single symbol.

    Returns a dict with trained models, test predictions, and evaluation
    metrics.
    """
    feat_df = build_feature_table(df, horizon=horizon)
    if len(feat_df) < 100:
        raise ValueError("Not enough data to train a reliable model "
                          f"(only {len(feat_df)} usable rows).")

    train, test = chronological_split(feat_df, test_size=test_size)

    X_train, y_train_reg = train[FEATURE_COLUMNS], train["FutureReturn"]
    X_test, y_test_reg = test[FEATURE_COLUMNS], test["FutureReturn"]
    y_train_clf, y_test_clf = train["Direction"], test["Direction"]

    if model_type == "linear":
        reg_model = LinearRegression()
        clf_model = RandomForestClassifier(
            n_estimators=200, max_depth=6, min_samples_leaf=10,
            random_state=42, n_jobs=-1
        )
    else:
        reg_model = RandomForestRegressor(
            n_estimators=300, max_depth=8, min_samples_leaf=5,
            random_state=42, n_jobs=-1
        )
        clf_model = RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=5,
            random_state=42, n_jobs=-1
        )

    reg_model.fit(X_train, y_train_reg)
    clf_model.fit(X_train, y_train_clf)

    pred_reg = reg_model.predict(X_test)
    pred_clf = clf_model.predict(X_test)
    pred_clf_proba = clf_model.predict_proba(X_test)[:, 1]

    metrics = {
        "MAE": mean_absolute_error(y_test_reg, pred_reg),
        "RMSE": np.sqrt(mean_squared_error(y_test_reg, pred_reg)),
        "R2": r2_score(y_test_reg, pred_reg),
        "Directional Accuracy (Regression)": float(
            np.mean((pred_reg > 0) == (y_test_reg.values > 0))
        ),
        "Directional Accuracy (Classifier)": accuracy_score(y_test_clf, pred_clf),
        "n_train": len(train),
        "n_test": len(test),
    }

    feature_importance = None
    if hasattr(reg_model, "feature_importances_"):
        feature_importance = pd.Series(
            reg_model.feature_importances_, index=FEATURE_COLUMNS
        ).sort_values(ascending=False)

    result = {
        "reg_model": reg_model,
        "clf_model": clf_model,
        "metrics": metrics,
        "test_dates": test["Date"].values,
        "y_test_reg": y_test_reg.values,
        "pred_reg": pred_reg,
        "y_test_clf": y_test_clf.values,
        "pred_clf": pred_clf,
        "pred_clf_proba": pred_clf_proba,
        "test_close": test["Close"].values,
        "feature_importance": feature_importance,
        "horizon": horizon,
        "feat_df": feat_df,
    }
    return result


def predict_latest(df: pd.DataFrame, result: dict):
    """Use trained models on the most recent available data row to produce
    a forward-looking forecast."""
    feat_df = result["feat_df"]
    latest = feat_df.iloc[[-1]]
    X_latest = latest[FEATURE_COLUMNS]

    pred_return = result["reg_model"].predict(X_latest)[0]
    pred_direction = result["clf_model"].predict(X_latest)[0]
    pred_proba_up = result["clf_model"].predict_proba(X_latest)[0, 1]

    last_close = latest["Close"].values[0]
    last_date = latest["Date"].values[0]
    target_price = last_close * (1 + pred_return)

    return {
        "as_of_date": last_date,
        "last_close": last_close,
        "predicted_return": pred_return,
        "predicted_price": target_price,
        "predicted_direction": "Up" if pred_direction == 1 else "Down",
        "prob_up": pred_proba_up,
        "horizon_days": result["horizon"],
    }


if __name__ == "__main__":
    from data_utils import load_processed_data

    data = load_processed_data()
    rel = data[data["Symbol"] == "RELIANCE"]
    res = train_predictor(rel, horizon=5)
    print(res["metrics"])
    print(predict_latest(rel, res))
    print(res["feature_importance"].head(8))
