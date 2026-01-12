from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_log_return(close: pd.Series) -> pd.Series:
    c = close.astype("float64").clip(lower=1e-12)
    return np.log(c).diff()


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)

    # min_periods 不能大於 window
    minp = 1 if window <= 2 else max(2, window // 3)

    roll_up = up.rolling(window, min_periods=minp).mean()
    roll_down = down.rolling(window, min_periods=minp).mean()

    rs = roll_up / (roll_down + 1e-12)
    return 100.0 - (100.0 / (1.0 + rs))


def build_ohlcv_features(
    df: pd.DataFrame,
    timeframe: str,
    windows: list[int],
    rsi_windows: list[int] | None = None,
    add_calendar: bool = True,
) -> pd.DataFrame:
    required = {"timestamp", "symbol", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in OHLCV df: {missing}")

    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out = out.dropna(subset=["timestamp"])
    out = out.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    for c in ["open", "high", "low", "close", "volume"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    feats = []
    for sym, g in out.groupby("symbol", sort=False):
        g = g.copy()

        g["lr"] = _safe_log_return(g["close"])
        g["ret"] = g["close"].pct_change()
        g["hl_range"] = (g["high"] - g["low"]) / (g["close"].abs() + 1e-12)
        g["oc_change"] = (g["close"] - g["open"]) / (g["open"].abs() + 1e-12)

        tr = _true_range(g["high"], g["low"], g["close"])
        g["atr_1"] = tr

        for w in windows:
            # ✅ 修正：min_periods 不能大於 window
            minp = 1 if w <= 2 else max(2, w // 3)

            g[f"{timeframe}_lr_mean_{w}"] = g["lr"].rolling(w, min_periods=minp).mean()
            g[f"{timeframe}_lr_std_{w}"] = g["lr"].rolling(w, min_periods=minp).std()
            g[f"{timeframe}_rv_{w}"] = np.sqrt((g["lr"] ** 2).rolling(w, min_periods=minp).sum())

            g[f"{timeframe}_mom_{w}"] = g["close"].pct_change(w)
            g[f"{timeframe}_high_low_{w}"] = (
                (g["high"].rolling(w, min_periods=minp).max() - g["low"].rolling(w, min_periods=minp).min())
                / (g["close"].abs() + 1e-12)
            )

            g[f"{timeframe}_vol_mean_{w}"] = g["volume"].rolling(w, min_periods=minp).mean()
            g[f"{timeframe}_vol_std_{w}"] = g["volume"].rolling(w, min_periods=minp).std()

            g[f"{timeframe}_atr_{w}"] = tr.rolling(w, min_periods=minp).mean()

        if rsi_windows:
            for rw in rsi_windows:
                g[f"{timeframe}_rsi_{rw}"] = _rsi(g["close"], rw)

        if add_calendar:
            ts = g["timestamp"]
            g[f"{timeframe}_dow"] = ts.dt.dayofweek.astype("int16")
            g[f"{timeframe}_hour"] = ts.dt.hour.astype("int16")
            g[f"{timeframe}_month"] = ts.dt.month.astype("int16")

        keep_cols = ["timestamp", "symbol"] + [c for c in g.columns if c.startswith(f"{timeframe}_")]
        feats.append(g[keep_cols])

    feat_df = pd.concat(feats, ignore_index=True)
    feat_df = feat_df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    return feat_df


def merge_timeframes(
    daily_feat: pd.DataFrame,
    hourly_feat: pd.DataFrame,
    agg_hours: list[int] | None = None,
) -> pd.DataFrame:
    if agg_hours is None:
        agg_hours = [24, 168, 360]

    dfh = hourly_feat.copy()
    dfh["timestamp"] = pd.to_datetime(dfh["timestamp"], utc=True, errors="coerce")
    dfh = dfh.dropna(subset=["timestamp"]).sort_values(["symbol", "timestamp"])

    dfh["day"] = dfh["timestamp"].dt.floor("D")

    cand_cols = [
        c for c in dfh.columns
        if c.startswith("1h_") and (("rv_" in c) or ("lr_std_" in c) or ("atr_" in c) or ("vol_mean_" in c))
    ]

    base = dfh[["symbol", "timestamp", "day"] + cand_cols].copy()
    base = base.set_index("timestamp")

    rows = []
    for sym, g in base.groupby("symbol", sort=False):
        g = g.sort_index()
        last_per_day = g.groupby("day").tail(1).copy()

        for h in agg_hours:
            win = f"{h}h"
            minp = max(1, h // 6)

            rolled = g[cand_cols].rolling(win, min_periods=minp).mean()
            rolled = rolled.loc[last_per_day.index].add_prefix(f"1h2d_mean_{h}h_")
            last_per_day = pd.concat([last_per_day, rolled], axis=1)

            rolled_std = g[cand_cols].rolling(win, min_periods=minp).std()
            rolled_std = rolled_std.loc[last_per_day.index].add_prefix(f"1h2d_std_{h}h_")
            last_per_day = pd.concat([last_per_day, rolled_std], axis=1)

        out = last_per_day.reset_index(drop=False)[["symbol", "day"] + [c for c in last_per_day.columns if c.startswith("1h2d_")]]
        rows.append(out)

    hourly_to_daily = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["symbol", "day"])
    hourly_to_daily = hourly_to_daily.rename(columns={"day": "timestamp"})

    daily = daily_feat.copy()
    daily["timestamp"] = pd.to_datetime(daily["timestamp"], utc=True, errors="coerce").dt.floor("D")

    merged = daily.merge(hourly_to_daily, on=["symbol", "timestamp"], how="left")
    merged = merged.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    return merged
