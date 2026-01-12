from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_squared_error

from src.datasets.targets import make_return_targets
from src.models.lgbm import LGBMRegressorWrapper
from src.training.cv import WalkForwardConfig, walk_forward_splits


@dataclass
class PreparedData:
    X: pd.DataFrame
    y: np.ndarray
    meta: pd.DataFrame
    feature_cols: List[str]


def _ensure_dirs(*dirs: str) -> None:
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def prepare_xy(
    raw_daily_path: str,
    features_path: str,
    horizons: List[int],
    target_col: str,
) -> PreparedData:
    df_daily = pd.read_parquet(raw_daily_path)
    df_feat = pd.read_parquet(features_path)

    df_y = make_return_targets(df_daily, horizons=horizons)
    df = df_feat.merge(df_y, on=["timestamp", "symbol"], how="left")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    df = df.dropna(subset=[target_col]).reset_index(drop=True)

    feature_cols = [c for c in df.columns if c not in ("timestamp", "symbol") and not c.startswith("next_")]

    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = df[target_col].astype("float64").values
    meta = df[["timestamp", "symbol"]].copy()

    return PreparedData(X=X, y=y, meta=meta, feature_cols=feature_cols)


def cv_rmse(
    X: pd.DataFrame,
    y: np.ndarray,
    cv_cfg: WalkForwardConfig,
    model_params: Dict,
) -> Tuple[float, List[float]]:
    splits = list(walk_forward_splits(len(X), cv_cfg))
    if len(splits) == 0:
        raise RuntimeError("No valid splits generated. Try smaller n_folds or valid_size.")

    rmses: List[float] = []

    for tr_idx, va_idx in splits:
        X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
        X_va, y_va = X.iloc[va_idx], y[va_idx]

        model = LGBMRegressorWrapper(model_params).fit(X_tr, y_tr)
        pred = model.predict(X_va)

        rmse = float(np.sqrt(mean_squared_error(y_va, pred)))
        rmses.append(rmse)

    return float(np.mean(rmses)), rmses


def run_optuna_lgbm_cv(
    raw_daily_path: str,
    features_path: str,
    horizons: List[int],
    target_col: str,
    cv_cfg: WalkForwardConfig,
    fixed_params: Dict,
    n_trials: int,
    seed: int,
    reports_dir: str,
    models_dir: str,
    timeout_sec: int | None = None,
) -> Dict:
    _ensure_dirs(reports_dir, models_dir)

    data = prepare_xy(
        raw_daily_path=raw_daily_path,
        features_path=features_path,
        horizons=horizons,
        target_col=target_col,
    )

    # 讓 Optuna 可重現
    sampler = optuna.samplers.TPESampler(seed=seed)

    def objective(trial: optuna.Trial) -> float:
        # 只調關鍵參數（先精簡，50 trials 才有效）
        params = dict(fixed_params)

        params.update(
            {
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 16, 256, log=True),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 80, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 1.0, log=True),
                "n_estimators": trial.suggest_int("n_estimators", 800, 5000, log=True),
            }
        )

        mean_rmse, fold_rmses = cv_rmse(data.X, data.y, cv_cfg=cv_cfg, model_params=params)

        # 也存一下每折，方便你之後分析
        trial.set_user_attr("fold_rmses", fold_rmses)
        return mean_rmse

    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, timeout=timeout_sec, show_progress_bar=True)

    best_params = dict(fixed_params)
    best_params.update(study.best_params)

    # 用 best params 在全資料重訓一個 final model
    final_model = LGBMRegressorWrapper(best_params).fit(data.X, data.y)
    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(models_dir, f"lgbm_optuna_cv_{target_col}_{run_id}.txt")
    final_model.model.booster_.save_model(model_path)

    report = {
        "target": target_col,
        "n_trials": n_trials,
        "best_value_rmse_mean": float(study.best_value),
        "best_params": study.best_params,
        "fixed_params": fixed_params,
        "model_path": model_path,
        "n_rows_used": int(len(data.X)),
        "n_features": int(len(data.feature_cols)),
        "cv": {
            "n_folds": cv_cfg.n_folds,
            "valid_size": cv_cfg.valid_size,
            "expanding": cv_cfg.expanding,
        },
        "run_id": run_id,
    }

    report_path = os.path.join(reports_dir, f"report_optuna_cv_{target_col}_{run_id}.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n[optuna] best rmse_mean =", report["best_value_rmse_mean"])
    print("[optuna] best params =", study.best_params)
    print("[optuna] saved model =", model_path)
    print("[optuna] saved report =", report_path)

    return report
