import argparse
import os
import yaml

from src.training.train_one import train_from_parquets


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train.yaml")
    args = ap.parse_args()

    cfg = load_yaml(args.config)

    raw_daily = cfg["inputs"]["raw_daily_parquet"]
    features = cfg["inputs"]["features_parquet"]
    horizons = list(cfg["targets"]["horizons"])
    valid_size = float(cfg["split"]["valid_size"])

    model_params = cfg["model"]["params"]
    out = cfg["output"]

    out_dirs = {
        "models_dir": out["models_dir"],
        "oof_dir": out["oof_dir"],
        "reports_dir": out["reports_dir"],
    }
    for d in out_dirs.values():
        os.makedirs(d, exist_ok=True)

    targets = cfg["training"]["targets"]

    for t in targets:
        print(f"\n[train] target={t}")

        report = train_from_parquets(
            raw_daily_path=raw_daily,
            features_path=features,
            horizons=horizons,
            target_col=t,
            valid_size=valid_size,
            model_params=model_params,
            out_dirs=out_dirs,
        )

        report_path = os.path.join(
            out_dirs["reports_dir"],
            f"report_{t}_{report['run_id']}.json",
        )

        print(f"[train] done target={t}")
        print(f"[train] rmse={report['rmse']}")
        print(f"[train] model={report['model_path']}")
        print(f"[train] oof={report['oof_path']}")
        print(f"[train] report={report_path}")


if __name__ == "__main__":
    main()
