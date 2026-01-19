# scripts/10_make_signal.py
from __future__ import annotations

import argparse
import os
import pandas as pd

from src.inference.predict import predict_one
from src.signals.make_signals import make_signal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True, help="e.g. AAPL, NVDA, TSLA")
    ap.add_argument("--features", default="data/features/features.parquet")
    ap.add_argument("--model_1d", required=True, help="path to 1d model .txt")
    ap.add_argument("--model_7d", required=True, help="path to 7d model .txt")
    ap.add_argument("--out", default="artifacts/signals/signals.parquet")
    args = ap.parse_args()

    # basic checks
    if not os.path.exists(args.features):
        raise FileNotFoundError(f"features parquet not found: {args.features}")
    if not os.path.exists(args.model_1d):
        raise FileNotFoundError(f"1d model not found: {args.model_1d}")
    if not os.path.exists(args.model_7d):
        raise FileNotFoundError(f"7d model not found: {args.model_7d}")

    ts, p1, p7 = predict_one(
        features_parquet=args.features,
        symbol=args.symbol,
        model_1d_path=args.model_1d,
        model_7d_path=args.model_7d,
    )

    # normalize ts
    try:
        ts = pd.to_datetime(ts)
    except Exception:
        pass

    sig = make_signal(p1, p7)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    row = {
        "ts": ts,
        "symbol": args.symbol,
        "pred_next_1d_return": float(p1),
        "pred_next_7d_return": float(p7),
        "action": sig.action,
        "score": float(sig.score),
        "reason": sig.reason,
    }

    df = pd.DataFrame([row])

    if os.path.exists(args.out):
        old = pd.read_parquet(args.out)
        df = pd.concat([old, df], ignore_index=True)

        if "ts" in df.columns and "symbol" in df.columns:
            df = df.drop_duplicates(subset=["symbol", "ts"], keep="last")

    cols = ["ts", "symbol", "pred_next_1d_return", "pred_next_7d_return", "action", "score", "reason"]
    df = df[[c for c in cols if c in df.columns] + [c for c in df.columns if c not in cols]]

    df.to_parquet(args.out, index=False)

    print(f"[signal] ts={ts} symbol={args.symbol}")
    print(f"[signal] {sig.action} | score={sig.score:.2f}")
    print(f"[signal] {sig.reason}")
    print(f"[signal] saved -> {args.out}")


if __name__ == "__main__":
    main()
