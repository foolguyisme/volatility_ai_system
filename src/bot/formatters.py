from __future__ import annotations

from datetime import date
from typing import Iterable, Tuple

import pandas as pd


DISCORD_CONTENT_LIMIT = 1900  # 留一些 buffer


def format_signals_text(
    df: pd.DataFrame,
    date_str: str,
    topk: int = 20,
    min_abs_score: float = 0.0,
    stock_col: str = "stock_id",
    score_col: str = "score",
) -> str:
    d = df.copy()
    if stock_col not in d.columns or score_col not in d.columns:
        raise ValueError(f"signals missing columns: need '{stock_col}' and '{score_col}', got {list(d.columns)}")

    d = d[[stock_col, score_col]].dropna()
    d = d[d[score_col].abs() >= float(min_abs_score)]
    d = d.sort_values(score_col, ascending=False).head(int(topk))

    lines = []
    lines.append(f"✅ **{date_str} Signals (Top {len(d)})**")
    if len(d) == 0:
        lines.append("今日無有效訊號（門檻過濾後為空）。")
        return "\n".join(lines)

    for i, row in enumerate(d.itertuples(index=False), start=1):
        stock_id = getattr(row, stock_col)
        score = getattr(row, score_col)
        lines.append(f"{i:>2}. `{stock_id}`  score={score:.4f}")

    text = "\n".join(lines)
    if len(text) > DISCORD_CONTENT_LIMIT:
        text = text[:DISCORD_CONTENT_LIMIT] + "\n…(truncated)"
    return text


def make_simple_embed(title: str, description: str) -> list[dict]:
    if len(description) > 3800:
        description = description[:3800] + "\n…(truncated)"
    return [{
        "title": title,
        "description": description,
    }]
