# scripts/11_make_signal_prod.py
from __future__ import annotations

import argparse
import os
import pandas as pd

from src.inference.load_model import load_lgbm_model
from src.inference.predict import predict_for_date
from src.signals.make_signals import make_signal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data/features/features.parquet")
    ap.add_argument("--production_dir", default="artifacts/models/production")
    ap.add_argument("--date", default=None, help="YYYY-MM-DD (default: latest)")
    ap.add_argument("--out", default="artifacts/signals/signals.parquet")
    ap.add_argument("--topk", type=int, default=20, help="top K signals by score")
    args = ap.parse_args()

    if not os.path.exists(args.features):
        raise FileNotFoundError(f"features parquet not found: {args.features}")

    model_1d_path = os.path.join(args.production_dir, "next_1d_return.txt")
    model_7d_path = os.path.join(args.production_dir, "next_7d_return.txt")
    if not os.path.exists(model_1d_path):
        raise FileNotFoundError(f"production model not found: {model_1d_path}")
    if not os.path.exists(model_7d_path):
        raise FileNotFoundError(f"production model not found: {model_7d_path}")

    df_feat = pd.read_parquet(args.features)

    required_cols = {"ts", "symbol"}
    missing = required_cols - set(df_feat.columns)
    if missing:
        raise ValueError(
            f"features missing required columns: {sorted(list(missing))}. "
            f"Available cols (head): {list(df_feat.columns)[:20]}"
        )

    model_1d = load_lgbm_model(model_1d_path)
    model_7d = load_lgbm_model(model_7d_path)

    pred_1d = predict_for_date(df_feat, model_1d, args.date)
    pred_7d = predict_for_date(df_feat, model_7d, args.date)

    pred_1d = pred_1d.rename(columns={"pred": "pred_next_1d_return"})
    pred_7d = pred_7d.rename(columns={"pred": "pred_next_7d_return"})

    df = pred_1d.merge(pred_7d, on=["ts", "symbol"], how="inner")

    rows = []
    for _, r in df.iterrows():
        sig = make_signal(float(r["pred_next_1d_return"]), float(r["pred_next_7d_return"]))
        rows.append(
            {
                "ts": r["ts"],
                "symbol": r["symbol"],
                "pred_next_1d_return": float(r["pred_next_1d_return"]),
                "pred_next_7d_return": float(r["pred_next_7d_return"]),
                "action": sig.action,
                "score": float(sig.score),
                "reason": sig.reason,
            }
        )

    out_df = pd.DataFrame(rows)
    out_df = out_df.sort_values("score", ascending=False).head(args.topk)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if os.path.exists(args.out):
        old = pd.read_parquet(args.out)
        out_df = pd.concat([old, out_df], ignore_index=True)
        out_df = out_df.drop_duplicates(subset=["symbol", "ts"], keep="last")

    out_df.to_parquet(args.out, index=False)

    print("[signal] saved ->", args.out)
    print(out_df[["symbol", "action", "score"]].head(10))


if __name__ == "__main__":
    main()
