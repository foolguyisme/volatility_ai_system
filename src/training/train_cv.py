from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from src.datasets.targets import make_return_targets
from src.models.lgbm import LGBMRegressorWrapper
from src.training.cv import WalkForwardConfig, walk_forward_splits


def _prepare_train_table(raw_daily_path: str, features_path: str, horizons: List[int]) -> pd.DataFrame:
    df_daily = pd.read_parquet(raw_daily_path)
    df_feat = pd.read_parquet(features_path)

    df_y = make_return_targets(df_daily, horizons=horizons)

    df = df_feat.merge(df_y, on=["timestamp", "symbol"], how="left")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    return df


def train_cv_for_target(
    df: pd.DataFrame,
    target_col: str,
    cv_cfg: WalkForwardConfig,
    model_params: Dict,
) -> Tuple[float, float, List[float], pd.DataFrame]:
    df = df.dropna(subset=[target_col]).reset_index(drop=True)

    feature_cols = [c for c in df.columns if c not in ("timestamp", "symbol") and not c.startswith("next_")]
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = df[target_col].astype("float64").values

    splits = list(walk_forward_splits(len(df), cv_cfg))
    rmses: List[float] = []

    oof = np.full(len(df), np.nan, dtype="float64")

    for i, (tr_idx, va_idx) in enumerate(splits, start=1):
        X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
        X_va, y_va = X.iloc[va_idx], y[va_idx]

        model = LGBMRegressorWrapper(model_params).fit(X_tr, y_tr)
        pred = model.predict(X_va)

        oof[va_idx] = pred
        rmse = float(np.sqrt(mean_squared_error(y_va, pred)))
        rmses.append(rmse)

        vmin = df.loc[va_idx, "timestamp"].min().date()
        vmax = df.loc[va_idx, "timestamp"].max().date()
        print(f"[cv] target={target_col} fold={i}/{cv_cfg.n_folds} rmse={rmse:.6f} "
              f"train={len(tr_idx)} valid={len(va_idx)} valid_range={vmin}..{vmax}")

    rmse_mean = float(np.mean(rmses))
    rmse_std = float(np.std(rmses))

    oof_df = df[["timestamp", "symbol"]].copy()
    oof_df["y_true"] = y
    oof_df["y_pred"] = oof
    oof_df["target"] = target_col

    return rmse_mean, rmse_std, rmses, oof_df


def train_cv_from_parquets(cfg: dict) -> dict:
    raw_daily = cfg["inputs"]["raw_daily_parquet"]
    features = cfg["inputs"]["features_parquet"]
    horizons = list(cfg["targets"]["horizons"])

    cv = cfg["cv"]
    cv_cfg = WalkForwardConfig(
        n_folds=int(cv["n_folds"]),
        valid_size=float(cv["valid_size"]),
        expanding=bool(cv.get("expanding", True)),
    )

    model_params = cfg["model"]["params"]

    targets = cfg["training"]["targets"]
    out = cfg["output"]
    models_dir = out["models_dir"]
    oof_dir = out["oof_dir"]

    df = _prepare_train_table(raw_daily, features, horizons=horizons)

    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    results = {}
    all_oof = []

    for t in targets:
        print(f"\n[train_cv] target={t}")
        rmse_mean, rmse_std, fold_rmses, oof_df = train_cv_for_target(
            df=df, target_col=t, cv_cfg=cv_cfg, model_params=model_params
        )

        # 用全資料重訓一個最終模型
        df_t = df.dropna(subset=[t]).reset_index(drop=True)
        feat_cols = [c for c in df_t.columns if c not in ("timestamp", "symbol") and not c.startswith("next_")]
        X_all = df_t[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        y_all = df_t[t].astype("float64").values

        model = LGBMRegressorWrapper(model_params).fit(X_all, y_all)

        model_path = os.path.join(models_dir, f"lgbm_cv_{t}_{run_id}.txt")
        model.model.booster_.save_model(model_path)

        oof_path = os.path.join(oof_dir, f"oof_cv_{t}_{run_id}.parquet")
        oof_df.to_parquet(oof_path, index=False)

        print(f"[train_cv] done target={t} rmse_mean={rmse_mean:.6f} rmse_std={rmse_std:.6f}")
        print(f"[train_cv] model={model_path}")
        print(f"[train_cv] oof={oof_path}")

        results[t] = {
            "rmse_mean": rmse_mean,
            "rmse_std": rmse_std,
            "fold_rmses": fold_rmses,
            "model_path": model_path,
            "oof_path": oof_path,
        }

        all_oof.append(oof_df)

    return {
        "run_id": run_id,
        "results": results,
    }
