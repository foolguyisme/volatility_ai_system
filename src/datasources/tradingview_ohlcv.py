from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass
from typing import Dict, List

import pandas as pd
import yfinance as yf

SUPPORTED_INTERVALS = {"1d", "1h"}


@dataclass
class FetchSpec:
    symbols: List[str]
    interval: str      # "1d" or "1h"
    period: str        # e.g. "5y", "730d"
    auto_adjust: bool = True


def _ensure_dirs(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _cache_key(spec: FetchSpec) -> str:
    payload = f"{','.join(spec.symbols)}|{spec.interval}|{spec.period}|{spec.auto_adjust}"
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def _normalize_ohlcv(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    統一 schema:
      timestamp, symbol, open, high, low, close, volume
    並處理 yfinance 可能回傳 MultiIndex 欄位的情況。
    """
    if df is None or getattr(df, "empty", True):
        return pd.DataFrame(columns=["timestamp", "symbol", "open", "high", "low", "close", "volume"])

    dfx = df.copy()

    # ✅ MultiIndex 欄位（例如 ('Open','AAPL')）處理
    if isinstance(dfx.columns, pd.MultiIndex):
        # 常見形式：level0 = OHLCV, level1 = Ticker
        if symbol in dfx.columns.get_level_values(-1):
            dfx = dfx.xs(symbol, axis=1, level=-1, drop_level=True)
        else:
            # 兜底：取第一個 ticker
            first_ticker = dfx.columns.get_level_values(-1)[0]
            dfx = dfx.xs(first_ticker, axis=1, level=-1, drop_level=True)

    out = dfx.reset_index()

    # index 欄位可能叫 Date 或 Datetime
    if "Datetime" in out.columns:
        out = out.rename(columns={"Datetime": "timestamp"})
    elif "Date" in out.columns:
        out = out.rename(columns={"Date": "timestamp"})
    else:
        out = out.rename(columns={out.columns[0]: "timestamp"})

    out = out.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )

    for c in ["open", "high", "low", "close", "volume"]:
        if c not in out.columns:
            out[c] = pd.NA

    out["symbol"] = symbol
    out = out[["timestamp", "symbol", "open", "high", "low", "close", "volume"]]

    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")

    # ✅ 確保每個欄位都是 Series；若因重複欄名變 DataFrame，就取第一欄
    for c in ["open", "high", "low", "close", "volume"]:
        col = out[c]
        if isinstance(col, pd.DataFrame):
            col = col.iloc[:, 0]
        out[c] = pd.to_numeric(col, errors="coerce")

    out = out.dropna(subset=["timestamp"]).sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    return out


def fetch_ohlcv(spec: FetchSpec, cache_dir: str, force_refresh: bool = False, sleep_sec: float = 0.2) -> pd.DataFrame:
    if spec.interval not in SUPPORTED_INTERVALS:
        raise ValueError(f"Unsupported interval: {spec.interval}. Supported: {sorted(SUPPORTED_INTERVALS)}")

    _ensure_dirs(cache_dir)
    key = _cache_key(spec)
    cache_path = os.path.join(cache_dir, f"ohlcv_{spec.interval}_{key}.parquet")

    if (not force_refresh) and os.path.exists(cache_path):
        return pd.read_parquet(cache_path)

    frames: List[pd.DataFrame] = []
    for sym in spec.symbols:
        try:
            df = yf.download(
                tickers=sym,
                period=spec.period,
                interval=spec.interval,
                auto_adjust=spec.auto_adjust,
                progress=False,
                threads=False,
            )
            frames.append(_normalize_ohlcv(df, sym))
        except Exception as e:
            print(f"[fetch_ohlcv] WARN symbol={sym} interval={spec.interval} failed: {e}")
        time.sleep(sleep_sec)

    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out.to_parquet(cache_path, index=False)
    return out


def fetch_daily_and_hourly(
    symbols: List[str],
    lookback: Dict[str, str],
    cache_dir: str,
    auto_adjust: bool = True,
    force_refresh: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    回傳 {"1d": df_daily, "1h": df_hourly}
    """
    daily_spec = FetchSpec(symbols=symbols, interval="1d", period=lookback.get("1d", "5y"), auto_adjust=auto_adjust)
    hourly_spec = FetchSpec(symbols=symbols, interval="1h", period=lookback.get("1h", "730d"), auto_adjust=auto_adjust)

    df_1d = fetch_ohlcv(daily_spec, cache_dir=cache_dir, force_refresh=force_refresh)
    df_1h = fetch_ohlcv(hourly_spec, cache_dir=cache_dir, force_refresh=force_refresh)

    return {"1d": df_1d, "1h": df_1h}
