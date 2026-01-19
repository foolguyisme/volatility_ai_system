import argparse
import json
import os
import pandas as pd

from src.inference.load_model import load_lgbm_model
from src.inference.predict import predict_for_date


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data/features/features.parquet")
    ap.add_argument("--production_dir", default="artifacts/models/production")
    ap.add_argument("--date", default=None, help="YYYY-MM-DD. If not set, use latest available date in features.")
    ap.add_argument("--targets", nargs="+", default=["next_1d_return", "next_7d_return"])
    ap.add_argument("--out_json", default=None, help="Optional output JSON path.")
    args = ap.parse_args()

    df_feat = pd.read_parquet(args.features)

    # if date not given, pick latest date row(s)
    if args.date is None:
        # try common date column names
        date_col = None
        for c in ["date", "datetime", "time"]:
            if c in df_feat.columns:
                date_col = c
                break
        if date_col is None:
            raise KeyError("features parquet has no date column (date/datetime/time).")
        df_feat[date_col] = pd.to_datetime(df_feat[date_col])
        latest = df_feat[date_col].max()
        args.date = str(latest.date())

    result = {
        "date": args.date,
        "predictions": {}
    }

    for t in args.targets:
        model_path = os.path.join(args.production_dir, f"{t}.txt")
        booster = load_lgbm_model(model_path)
        pred_df = predict_for_date(df_feat, booster, args.date)

        # pack output: if multiple symbols -> list, else scalar
        if "symbol" in pred_df.columns:
            rows = []
            for _, r in pred_df.iterrows():
                rows.append({"symbol": r["symbol"], "pred": float(r["pred"])})
            result["predictions"][t] = rows
        else:
            # single series
            result["predictions"][t] = float(pred_df["pred"].iloc[0])

    s = json.dumps(result, ensure_ascii=False, indent=2)
    print(s)

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            f.write(s)
        print("[predict] saved ->", args.out_json)


if __name__ == "__main__":
    main()
