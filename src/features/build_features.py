from __future__ import annotations

import os
import yaml
import pandas as pd

from src.features.handcraft.ohlcv_features import build_ohlcv_features, merge_timeframes


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _must_exist(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")


def run(config_path: str = "configs/features.yaml") -> str:
    cfg = load_yaml(config_path)

    daily_path = cfg["inputs"]["daily_parquet"]
    hourly_path = cfg["inputs"]["hourly_parquet"]
    out_path = cfg["outputs"]["features_parquet"]

    print(f"[build_features] daily_parquet={daily_path}")
    print(f"[build_features] hourly_parquet={hourly_path}")
    print(f"[build_features] out={out_path}")

    _must_exist(daily_path)
    _must_exist(hourly_path)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    df_daily = pd.read_parquet(daily_path)
    df_hourly = pd.read_parquet(hourly_path)

    print(f"[build_features] loaded daily rows={len(df_daily)} cols={len(df_daily.columns)}")
    print(f"[build_features] loaded hourly rows={len(df_hourly)} cols={len(df_hourly.columns)}")

    dcfg = cfg["daily"]
    hcfg = cfg["hourly"]
    mcfg = cfg.get("merge", {})

    daily_feat = build_ohlcv_features(
        df=df_daily,
        timeframe=dcfg.get("timeframe", "1d"),
        windows=list(dcfg.get("windows", [1, 7, 15])),
        rsi_windows=list(dcfg.get("rsi_windows", [14])),
        add_calendar=bool(dcfg.get("add_calendar", True)),
    )

    hourly_feat = build_ohlcv_features(
        df=df_hourly,
        timeframe=hcfg.get("timeframe", "1h"),
        windows=list(hcfg.get("windows", [1, 24, 168, 360])),
        rsi_windows=list(hcfg.get("rsi_windows", [14])),
        add_calendar=bool(hcfg.get("add_calendar", True)),
    )

    merged = merge_timeframes(
        daily_feat=daily_feat,
        hourly_feat=hourly_feat,
        agg_hours=list(mcfg.get("agg_hours", [24, 168, 360])),
    )

    merged.to_parquet(out_path, index=False)
    print(f"[build_features] saved: {out_path} | rows={len(merged)} cols={len(merged.columns)}")
    return out_path
