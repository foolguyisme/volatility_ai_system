import argparse
import os
import yaml
import json
from datetime import datetime

from src.training.cv import WalkForwardConfig
from src.training.optuna_cv import run_optuna_lgbm_cv


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _now_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _safe_mkdir(p: str) -> None:
    if p:
        os.makedirs(p, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train_optuna.yaml")
    ap.add_argument("--tag", default="optuna", help="default=optuna")
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
    reports_dir = out.get("reports_dir", "artifacts/reports")
    models_dir = out.get("models_dir", "artifacts/models")
    _safe_mkdir(reports_dir)
    _safe_mkdir(models_dir)


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

    #  Fallback：找出最新的 optuna report 檔，補上 tag 與 rmse_mean 
    import glob

    patt = os.path.join(reports_dir, f"report_optuna_cv_{target}_*.json")
    files = sorted(glob.glob(patt))
    if not files:
        print("[08_optuna] warning: no optuna report found to normalize:", patt)
        return

    latest = files[-1]
    with open(latest, "r", encoding="utf-8") as f:
        j = json.load(f)

    # normalize
    j["tag"] = args.tag or "optuna"
    j["config_path"] = args.config
    j.setdefault("run_type", "optuna_cv")
    j.setdefault("run_id", _now_run_id())

    # 把 best_value_rmse_mean 放到 rmse_mean（讓 compare 統一讀 rmse_mean）
    if "rmse_mean" not in j or j.get("rmse_mean") is None:
        if "best_value_rmse_mean" in j:
            j["rmse_mean"] = j["best_value_rmse_mean"]
        elif "best_value" in j:
            j["rmse_mean"] = j["best_value"]

    with open(latest, "w", encoding="utf-8") as f:
        json.dump(j, f, ensure_ascii=False, indent=2)

    print("[08_optuna] normalized report =", latest)


if __name__ == "__main__":
    main()
