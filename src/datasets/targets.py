from __future__ import annotations

import pandas as pd


def make_return_targets(daily_ohlcv: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """
    Input daily OHLCV schema:
      timestamp, symbol, open, high, low, close, volume

    Output:
      timestamp, symbol, next_{h}d_return
    """
    df = daily_ohlcv.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    # ensure numeric
    df["close"] = pd.to_numeric(df["close"], errors="coerce")

    out = df[["timestamp", "symbol"]].copy()
    for h in horizons:
        # future return: close[t+h]/close[t]-1
        fut = df.groupby("symbol", sort=False)["close"].shift(-h)
        out[f"next_{h}d_return"] = (fut / df["close"]) - 1.0

    return out
