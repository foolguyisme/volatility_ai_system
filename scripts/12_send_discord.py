from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone

import pandas as pd
import yaml

from src.bot.discord_webhook import send_discord_webhook
from src.bot.dedupe import check_and_log_dedupe, mark_error, mark_sent
from src.bot.formatters import format_signals_text


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_env_vars(val: str) -> str:
    if isinstance(val, str) and val.startswith("${") and val.endswith("}"):
        key = val[2:-1]
        got = os.environ.get(key)
        if not got:
            raise RuntimeError(f"Missing env var: {key}")
        return got
    return val


def infer_date_str(df: pd.DataFrame) -> str:
    for col in ["date", "datetime", "ts"]:
        if col in df.columns:
            s = pd.to_datetime(df[col]).dropna()
            if len(s) > 0:
                return str(s.max().date())
    return datetime.now(timezone.utc).date().isoformat()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/bot.yaml")
    ap.add_argument("--signals", default=None, help="override signals path")
    ap.add_argument("--topk", type=int, default=None)
    ap.add_argument("--min_abs_score", type=float, default=None)
    args = ap.parse_args()

    cfg = load_yaml(args.config)

    webhook_url = resolve_env_vars(cfg["discord"]["webhook_url"])
    username = cfg["discord"].get("username")
    avatar_url = cfg["discord"].get("avatar_url") or None

    signals_path = args.signals or cfg["signals"]["path"]
    topk = args.topk if args.topk is not None else int(cfg["signals"].get("topk", 20))
    min_abs_score = args.min_abs_score if args.min_abs_score is not None else float(cfg["signals"].get("min_abs_score", 0.0))

    df = pd.read_parquet(signals_path)
    date_str = infer_date_str(df)

    message = format_signals_text(
        df=df,
        date_str=date_str,
        topk=topk,
        min_abs_score=min_abs_score,
        stock_col="stock_id",
        score_col="score",
    )

    log_dir = cfg.get("dedupe", {}).get("log_dir", "artifacts/bot_logs")
    dedupe_enable = bool(cfg.get("dedupe", {}).get("enable", True))

    dedupe = check_and_log_dedupe(log_dir=log_dir, date_str=date_str, message_text=message, enable=dedupe_enable)
    if not dedupe.should_send:
        print(f"[send_discord] skip: {dedupe.reason}")
        return

    try:
        send_discord_webhook(
            webhook_url=webhook_url,
            content=message,
            username=username,
            avatar_url=avatar_url,
            embeds=None,
        )
        mark_sent(log_dir=log_dir, date_str=date_str, message_hash=dedupe.message_hash)
        print("[send_discord] sent")
    except Exception as e:
        mark_error(log_dir=log_dir, date_str=date_str, message_hash=dedupe.message_hash, error=str(e))
        raise


if __name__ == "__main__":
    main()
