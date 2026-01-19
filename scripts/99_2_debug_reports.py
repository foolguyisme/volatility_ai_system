import argparse
import glob
import json
import os


def safe_load_json(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        return {"__load_error__": str(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pattern",
        default="artifacts/reports/report_*next_1d_return*.json",
        help="glob pattern for reports",
    )
    ap.add_argument("--tail", type=int, default=20, help="print last N files")
    args = ap.parse_args()

    ps = sorted(glob.glob(args.pattern))
    print("pattern =", args.pattern)
    print("files =", len(ps))
    if not ps:
        return

    show = ps[-args.tail :]
    for p in show:
        j = safe_load_json(p)
        base = os.path.basename(p)

        print("\n----", base)
        if "__load_error__" in j:
            print(" load_error =", j["__load_error__"])
            continue

        print(" tag =", j.get("tag"))
        print(" target =", j.get("target"))
        print(" rmse =", j.get("rmse"))
        print(" rmse_mean =", j.get("rmse_mean"))
        print(" best_value =", j.get("best_value"))

        keys = sorted(list(j.keys()))
        print(" keys contains:", keys)


if __name__ == "__main__":
    main()
