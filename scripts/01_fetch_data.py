import argparse
import os
import yaml

from src.datasources.tradingview_ohlcv import fetch_daily_and_hourly


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/datasource_tradingview.yaml")
    ap.add_argument("--force", action="store_true", help="Force refresh (ignore cache)")
    args = ap.parse_args()

    cfg = load_yaml(args.config)

    symbols = cfg["symbols"]
    lookback = cfg.get("lookback", {"1d": "5y", "1h": "730d"})
    auto_adjust = bool(cfg.get("auto_adjust", True))

    out_cfg = cfg.get("output", {})
    raw_dir = out_cfg.get("raw_dir", "data/raw")
    cache_dir = out_cfg.get("cache_dir", "data/cache")
    daily_path = out_cfg.get("daily_path", os.path.join(raw_dir, "ohlcv_1d.parquet"))
    hourly_path = out_cfg.get("hourly_path", os.path.join(raw_dir, "ohlcv_1h.parquet"))

    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    dfs = fetch_daily_and_hourly(
        symbols=symbols,
        lookback=lookback,
        cache_dir=cache_dir,
        auto_adjust=auto_adjust,
        force_refresh=args.force,
    )

    dfs["1d"].to_parquet(daily_path, index=False)
    dfs["1h"].to_parquet(hourly_path, index=False)

    print(f"[fetch] saved daily:  {daily_path} | rows={len(dfs['1d'])}")
    print(f"[fetch] saved hourly: {hourly_path} | rows={len(dfs['1h'])}")


if __name__ == "__main__":
    main()
