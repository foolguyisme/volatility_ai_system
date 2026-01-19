from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb


def _detect_date_col(df: pd.DataFrame) -> str:
    for c in ["date", "datetime", "time"]:
        if c in df.columns:
            return c
    raise KeyError("No date-like column found. Expected one of: date/datetime/time")


def _detect_symbol_col(df: pd.DataFrame) -> Optional[str]:
    for c in ["symbol", "ticker", "asset", "stock", "sid"]:
        if c in df.columns:
            return c
    return None


def _ensure_datetime(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    return out


def predict_for_date(
    features_df: pd.DataFrame,
    booster: lgb.Booster,
    date_str: str,
) -> pd.DataFrame:
    """
    Return predictions for a given date.
    If multiple symbols exist, returns one row per symbol for that date.
    """
    date_col = _detect_date_col(features_df)
    sym_col = _detect_symbol_col(features_df)
    df = _ensure_datetime(features_df, date_col)

    target_date = pd.to_datetime(date_str)

    mask = df[date_col].dt.date == target_date.date()
    day_df = df.loc[mask].copy()
    if day_df.empty:
        df = df.sort_values(date_col)
        df2 = df[df[date_col] <= target_date]
        if df2.empty:
            raise ValueError(f"No rows <= {date_str} in features parquet.")
        latest_dt = df2[date_col].max()
        day_df = df2[df2[date_col] == latest_dt].copy()

    feature_names = booster.feature_name()
    missing = [c for c in feature_names if c not in day_df.columns]
    if missing:
        raise KeyError(f"Missing feature columns in features_df: {missing[:10]} ... total={len(missing)}")

    X = day_df[feature_names].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    preds = booster.predict(X, num_iteration=booster.best_iteration)

    out_cols = [date_col]
    if sym_col:
        out_cols.append(sym_col)
    out = day_df[out_cols].copy()
    out["pred"] = preds
    return out
