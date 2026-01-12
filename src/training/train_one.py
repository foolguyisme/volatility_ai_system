from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from src.datasets.targets import make_return_targets
from src.models.lgbm import LGBMRegressorWrapper


def _ensure_dirs(*dirs: str) -> None:
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def _time_split(df: pd.DataFrame, valid_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Time-based split across the whole dataset (already sorted by timestamp).
    """
    n = len(df)
    n_valid = int(round(n * valid_size))
    n_valid = max(1, min(n_valid, n - 1))
    train = df.iloc[: n - n_valid].copy()
    valid = df.iloc[n - n_valid :].copy()
    return train, valid


def train_from_parquets(
    raw_daily_path: str,
    features_path: str,
    horizons: list[int],
    target_col: str,
    valid_size: float,
    model_params: Dict,
    out_dirs: Dict[str, str],
) -> Dict:
    df_daily = pd.read_parquet(raw_daily_path)
    df_feat = pd.read_parquet(features_path)

    # targets
    df_y = make_return_targets(df_daily, horizons=horizons)

    # merge X + y
    df = df_feat.merge(df_y, on=["timestamp", "symbol"], how="left")

    # keep only rows with target
    df = df.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    df = df.dropna(subset=[target_col])

    # Build X matrix
    feature_cols = [c for c in df.columns if c not in ("timestamp", "symbol") and not c.startswith("next_")]
    X = df[feature_cols]
    y = df[target_col].astype("float64")

    # Handle inf
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0.0)

    # Time split
    train_df, valid_df = _time_split(df, valid_size=valid_size)

    X_train = train_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y_train = train_df[target_col].astype("float64")

    X_valid = valid_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y_valid = valid_df[target_col].astype("float64")

    # Train
    model = LGBMRegressorWrapper(model_params).fit(X_train, y_train)
    pred_valid = model.predict(X_valid)

    rmse = float(np.sqrt(mean_squared_error(y_valid, pred_valid)))

    # Save artifacts
    _ensure_dirs(out_dirs["models_dir"], out_dirs["oof_dir"], out_dirs["reports_dir"])

    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(out_dirs["models_dir"], f"lgbm_{target_col}_{run_id}.txt")
    oof_path = os.path.join(out_dirs["oof_dir"], f"oof_{target_col}_{run_id}.parquet")
    report_path = os.path.join(out_dirs["reports_dir"], f"report_{target_col}_{run_id}.json")

    # lightgbm native save
    model.model.booster_.save_model(model_path)

    oof = valid_df[["timestamp", "symbol"]].copy()
    oof["y_true"] = y_valid.values
    oof["y_pred"] = pred_valid
    oof.to_parquet(oof_path, index=False)

    report = {
        "target": target_col,
        "rmse": rmse,
        "n_rows_total": int(len(df)),
        "n_train": int(len(train_df)),
        "n_valid": int(len(valid_df)),
        "feature_cols": int(len(feature_cols)),
        "model_path": model_path,
        "oof_path": oof_path,
        "run_id": run_id,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return report
