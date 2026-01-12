import argparse
import os
import yaml

from src.training.cv import WalkForwardConfig
from src.training.optuna_cv import run_optuna_lgbm_cv


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train_optuna.yaml")
    args = ap.parse_args()

    cfg = load_yaml(args.config)

    raw_daily = cfg["inputs"]["raw_daily_parquet"]
    features = cfg["inputs"]["features_parquet"]
    horizons = list(cfg["targets"]["horizons"])

    target = cfg["training"]["target"]

    cv_cfg = WalkForwardConfig(
        n_folds=int(cfg["cv"]["n_folds"]),
        valid_size=float(cfg["cv"]["valid_size"]),
        expanding=bool(cfg["cv"].get("expanding", True)),
    )

    n_trials = int(cfg["optuna"]["n_trials"])
    seed = int(cfg["optuna"].get("seed", 42))
    timeout_sec = cfg["optuna"].get("timeout_sec", None)

    fixed_params = cfg["model"]["fixed_params"]

    out = cfg["output"]
    reports_dir = out["reports_dir"]
    models_dir = out["models_dir"]
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    run_optuna_lgbm_cv(
        raw_daily_path=raw_daily,
        features_path=features,
        horizons=horizons,
        target_col=target,
        cv_cfg=cv_cfg,
        fixed_params=fixed_params,
        n_trials=n_trials,
        seed=seed,
        reports_dir=reports_dir,
        models_dir=models_dir,
        timeout_sec=timeout_sec,
    )


if __name__ == "__main__":
    main()
