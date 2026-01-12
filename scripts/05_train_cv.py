import argparse
import os
import yaml
import json

from src.training.train_cv import train_cv_from_parquets


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train_cv.yaml")
    args = ap.parse_args()

    cfg = load_yaml(args.config)

    # ✅ 關鍵：印出你實際吃到的參數（用來對照）
    model_params = cfg["model"]["params"]
    print("[05_train_cv] using config =", args.config)
    print("[05_train_cv] model.params =", model_params)

    # 確保輸出資料夾存在
    out = cfg["output"]
    os.makedirs(out["models_dir"], exist_ok=True)
    os.makedirs(out["oof_dir"], exist_ok=True)
    os.makedirs(out.get("reports_dir", "artifacts/reports"), exist_ok=True)

    report = train_cv_from_parquets(cfg)

    # ✅ 額外把「本次實際用的 params」寫進 report，避免之後搞混
    report["config_path"] = args.config
    report["model_params_used"] = model_params

    if "reports_dir" in out:
        rp = os.path.join(out["reports_dir"], f"report_train_cv_{report['run_id']}.json")
        with open(rp, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print("[05_train_cv] saved report =", rp)


if __name__ == "__main__":
    main()
