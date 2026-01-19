import argparse
import glob
import json
import os
import shutil
from typing import Dict, List, Optional, Tuple


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _pick_best_report(
    report_paths: List[str],
    target: str,
    prefer_tag: Optional[str] = "optuna",
) -> Tuple[str, dict]:
    """
    Choose best report for a target.
    Priority:
      1) If prefer_tag provided: pick reports with tag==prefer_tag first.
      2) Within candidates: choose smallest score among:
            - best_value_rmse_mean (optuna report)
            - rmse_mean           (cv report)
            - rmse               (single split report)
      3) Tie-break: latest filename (lexicographic)
    """
    candidates = []
    for p in report_paths:
        j = _load_json(p)
        if j.get("target") != target:
            continue
        candidates.append((p, j))

    if not candidates:
        raise FileNotFoundError(f"No reports found for target={target}")

    def score(j: dict) -> float:
        # optuna report stores best_value_rmse_mean
        if j.get("best_value_rmse_mean") is not None:
            return float(j["best_value_rmse_mean"])
        if j.get("rmse_mean") is not None:
            return float(j["rmse_mean"])
        if j.get("rmse") is not None:
            return float(j["rmse"])
        # worst fallback
        return 1e9

    if prefer_tag:
        tagged = [(p, j) for (p, j) in candidates if j.get("tag") == prefer_tag]
        if tagged:
            candidates = tagged

    # sort by score then by filename (latest)
    candidates.sort(key=lambda x: (score(x[1]), x[0]))
    best_p, best_j = candidates[0]
    return best_p, best_j


def _resolve_model_path(report: dict) -> str:
    # different report styles
    for k in ["model_path", "model"]:
        if k in report and report[k]:
            return str(report[k])
    raise KeyError("Report has no model_path/model field.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports_dir", default="artifacts/reports")
    ap.add_argument("--production_dir", default="artifacts/models/production")
    ap.add_argument("--targets", nargs="+", default=["next_1d_return", "next_7d_return"])
    ap.add_argument("--prefer_tag", default="optuna", help="Prefer this tag first, e.g. optuna/baseline")
    ap.add_argument("--pattern", default="artifacts/reports/report_*.json")
    args = ap.parse_args()

    os.makedirs(args.production_dir, exist_ok=True)

    report_paths = sorted(glob.glob(args.pattern))
    if not report_paths:
        raise FileNotFoundError(f"No report files found with pattern: {args.pattern}")

    exported: Dict[str, Dict] = {}

    for t in args.targets:
        best_report_path, best_report = _pick_best_report(
            report_paths=report_paths,
            target=t,
            prefer_tag=args.prefer_tag,
        )
        model_path = _resolve_model_path(best_report)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"model_path not found on disk: {model_path}")

        dst_model = os.path.join(args.production_dir, f"{t}.txt")
        shutil.copy2(model_path, dst_model)

        dst_meta = os.path.join(args.production_dir, f"{t}.meta.json")
        meta = {
            "target": t,
            "source_report": best_report_path,
            "source_model_path": model_path,
            "tag": best_report.get("tag"),
            "score": best_report.get("best_value_rmse_mean")
                     or best_report.get("rmse_mean")
                     or best_report.get("rmse"),
            "run_id": best_report.get("run_id"),
        }
        with open(dst_meta, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        exported[t] = {"model": dst_model, "meta": dst_meta}
        print(f"[export] target={t}")
        print(f"  report = {best_report_path}")
        print(f"  model  = {model_path}")
        print(f"  -> production model = {dst_model}")

    print("[export] done. production_dir =", args.production_dir)


if __name__ == "__main__":
    main()
