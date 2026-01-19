# src/reporting/compare_runs.py
from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class RunInfo:
    path: str
    run_id: str
    target: str
    rmse_mean: Optional[float]
    rmse_std: Optional[float]
    fold_rmses: Optional[List[float]]
    model_path: Optional[str]
    oof_path: Optional[str]
    params: Optional[Dict[str, Any]]
    tag: str           # baseline / optuna / other
    created_at: str


def _safe_float(x) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def _load_json(p: str) -> Dict[str, Any]:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _mtime_str(path: str) -> str:
    ts = os.path.getmtime(path)
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def _guess_tag(report: Dict[str, Any], path: str) -> str:
    """
    Prefer explicit report['tag'] / report['source'] if present.
    Else fall back to filename heuristic.
    """
    for k in ("tag", "source", "kind"):
        v = report.get(k)
        if isinstance(v, str) and v.strip():
            vv = v.strip().lower()
            if "optuna" in vv:
                return "optuna"
            if "baseline" in vv or "train_cv" in vv or "cv" in vv:
                return "baseline"
            return vv

    bn = os.path.basename(path).lower()
    if "optuna" in bn:
        return "optuna"
    if "train_cv" in bn or "baseline" in bn:
        return "baseline"
    return "other"


def _normalize_report(p: str) -> RunInfo:
    j = _load_json(p)

    run_id = str(j.get("run_id") or os.path.splitext(os.path.basename(p))[0])
    target = str(j.get("target") or j.get("training", {}).get("target") or "unknown_target")

    rmse_mean = _safe_float(j.get("rmse_mean", j.get("rmse")))
    rmse_std = _safe_float(j.get("rmse_std"))

    fold_rmses = j.get("fold_rmses") or j.get("fold_rmse")
    if not isinstance(fold_rmses, list):
        fold_rmses = None

    model_path = j.get("model_path") or j.get("model")
    oof_path = j.get("oof_path") or j.get("oof")
    params = j.get("params") or j.get("model_params") or j.get("model", {}).get("params")
    if not isinstance(params, dict):
        params = None

    tag = _guess_tag(j, p)
    created_at = j.get("created_at") or _mtime_str(p)

    return RunInfo(
        path=p,
        run_id=run_id,
        target=target,
        rmse_mean=rmse_mean,
        rmse_std=rmse_std,
        fold_rmses=fold_rmses,
        model_path=model_path,
        oof_path=oof_path,
        params=params,
        tag=tag,
        created_at=created_at,
    )


def _best_run(runs: List[RunInfo]) -> Optional[RunInfo]:
    runs = [r for r in runs if r.rmse_mean is not None]
    if not runs:
        return None
    runs.sort(key=lambda r: r.rmse_mean)  # lower is better
    return runs[0]


def _latest_run(runs: List[RunInfo]) -> Optional[RunInfo]:
    if not runs:
        return None
    runs.sort(key=lambda r: r.created_at)
    return runs[-1]


def build_summary(reports_glob: str = "artifacts/reports/*.json") -> Dict[str, Any]:
    paths = sorted(glob.glob(reports_glob))
    runs = [_normalize_report(p) for p in paths]

    by_target: Dict[str, List[RunInfo]] = {}
    for r in runs:
        if r.target == "unknown_target":
            continue
        by_target.setdefault(r.target, []).append(r)

    baseline_best = {}
    optuna_best = {}
    any_best = {}

    for t, rs in by_target.items():
        any_best_run = _best_run(rs)
        if any_best_run:
            any_best[t] = any_best_run.__dict__

        base_rs = [r for r in rs if r.tag == "baseline"]
        opt_rs = [r for r in rs if r.tag == "optuna"]

        b = _best_run(base_rs) or _latest_run(base_rs)
        o = _best_run(opt_rs) or _latest_run(opt_rs)

        if b:
            baseline_best[t] = b.__dict__
        if o:
            optuna_best[t] = o.__dict__

    deltas = {}
    for t in sorted(by_target.keys()):
        b = baseline_best.get(t)
        o = optuna_best.get(t)
        if b and o and b.get("rmse_mean") is not None and o.get("rmse_mean") is not None:
            b_rmse = float(b["rmse_mean"])
            o_rmse = float(o["rmse_mean"])
            deltas[t] = {
                "baseline_rmse_mean": b_rmse,
                "optuna_rmse_mean": o_rmse,
                "abs_improve": b_rmse - o_rmse,
                "rel_improve_pct": (b_rmse - o_rmse) / max(1e-12, b_rmse) * 100.0,
            }

    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "reports_glob": reports_glob,
        "targets": sorted(list(by_target.keys())),
        "best_per_target": any_best,
        "baseline_best_per_target": baseline_best,
        "optuna_best_per_target": optuna_best,
        "deltas": deltas,
        "n_reports": len(runs),
    }


def save_summary(out_path: str = "artifacts/reports/summary_cv_compare.json",
                 reports_glob: str = "artifacts/reports/*.json") -> str:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    summary = build_summary(reports_glob)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return out_path
