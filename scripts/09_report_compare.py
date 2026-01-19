import os
import glob
import json


REPORTS_DIR = "artifacts/reports"
OUT_PATH = os.path.join(REPORTS_DIR, "summary_cv_compare.json")


def load_json(p: str) -> dict:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def get_score(j: dict):
    for k in ("rmse_mean", "rmse", "best_value_rmse_mean", "best_value"):
        v = j.get(k)
        if v is not None:
            return float(v)
    return None


def main():
    ps = sorted(glob.glob(os.path.join(REPORTS_DIR, "report_*.json")))
    reports = []
    for p in ps:
        try:
            j = load_json(p)
            j["_path"] = p
            reports.append(j)
        except Exception:
            continue

    print(f"[report] n_reports = {len(reports)}")

    latest = {}
    for j in reports:
        target = j.get("target") or j.get("target_col")
        tag = j.get("tag")
        if not target or not tag:
            continue
        key = (tag, target)
        latest[key] = j

    for j in reports:
        p = os.path.basename(j.get("_path", ""))
        target = j.get("target") or j.get("target_col")
        if not target:
            continue
        if "optuna" in p and not j.get("tag"):
            latest[("optuna", target)] = j
        if ("baseline" in p or "cv_baseline" in p) and not j.get("tag"):
            latest[("baseline", target)] = j

    targets = sorted({t for (_, t) in latest.keys()})
    rows = []
    for t in targets:
        b = latest.get(("baseline", t))
        o = latest.get(("optuna", t))
        b_score = get_score(b) if b else None
        o_score = get_score(o) if o else None
        delta = (o_score - b_score) if (o_score is not None and b_score is not None) else None

        rows.append(
            {
                "target": t,
                "baseline_score": b_score,
                "optuna_score": o_score,
                "delta_optuna_minus_baseline": delta,
                "baseline_path": b.get("_path") if b else None,
                "optuna_path": o.get("_path") if o else None,
            }
        )

    out = {"rows": rows}
    os.makedirs(REPORTS_DIR, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("[report] saved summary ->", OUT_PATH)
    for r in rows:
        if r["delta_optuna_minus_baseline"] is not None:
            print(f"[report] {r['target']}: delta(optuna-baseline) = {r['delta_optuna_minus_baseline']:.6f}")
        else:
            print(f"[report] {r['target']}: delta(optuna-baseline) = None")


if __name__ == "__main__":
    main()
