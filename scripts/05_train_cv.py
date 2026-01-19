import argparse
import os
import yaml
import json
from datetime import datetime

from src.training.train_cv import train_cv_from_parquets


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
    ap.add_argument("--config", default="configs/train_cv.yaml")
    ap.add_argument("--tag", default=None, help="e.g. baseline / optuna / v1 / v2 ...")
    args = ap.parse_args()

    cfg = load_yaml(args.config)

    tag = args.tag
    model_params = cfg["model"]["params"]

    print("[05_train_cv] using config =", args.config)
    print("[05_train_cv] tag =", tag)
    print("[05_train_cv] model.params =", model_params)

    out = cfg["output"]
    models_dir = out.get("models_dir", "artifacts/models")
    oof_dir = out.get("oof_dir", "artifacts/oof")
    reports_dir = out.get("reports_dir", "artifacts/reports")

    _safe_mkdir(models_dir)
    _safe_mkdir(oof_dir)
    _safe_mkdir(reports_dir)

    # 依照src會回傳 dict: {target_name: report_dict, ...} 或 list[report_dict]
    results = train_cv_from_parquets(cfg)

    run_id = _now_run_id()

    # 統一整理成 list[dict]
    per_target_reports = []
    if isinstance(results, dict):
        for _, rep in results.items():
            if isinstance(rep, dict):
                per_target_reports.append(rep)
    elif isinstance(results, list):
        per_target_reports = [r for r in results if isinstance(r, dict)]
    elif isinstance(results, dict) is False and isinstance(results, str):
        # 你之前碰到 'str' object has no attribute get，這裡防禦一下
        per_target_reports = []
    else:
        per_target_reports = []

    # 對每個target存一份report，讓09_compare 可以直接吃
    saved = 0
    for rep in per_target_reports:
        t = rep.get("target") or rep.get("target_col")
        if not t:
            continue

        rep["config_path"] = args.config
        rep["model_params_used"] = model_params
        if tag:
            rep["tag"] = tag
        rep.setdefault("run_id", run_id)

        fname = f"report_cv_{tag+'_' if tag else ''}{t}_{run_id}.json"
        rp = os.path.join(reports_dir, fname)
        with open(rp, "w", encoding="utf-8") as f:
            json.dump(rep, f, ensure_ascii=False, indent=2)
        saved += 1

    # 額外存summary
    summary = {
        "run_id": run_id,
        "tag": tag,
        "config_path": args.config,
        "n_targets": saved,
        "targets": [r.get("target") for r in per_target_reports if isinstance(r, dict)],
    }
    summary_path = os.path.join(reports_dir, f"report_cv_{tag or 'run'}_ALL_{run_id}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[05_train_cv] saved summary report =", summary_path)
    print("[05_train_cv] saved per-target reports =", saved)


if __name__ == "__main__":
    main()
