from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Tuple

import numpy as np


@dataclass
class WalkForwardConfig:
    n_folds: int = 5
    valid_size: float = 0.15
    expanding: bool = True


def walk_forward_splits(n: int, cfg: WalkForwardConfig) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    產生多折 walk-forward splits。
    資料需先依時間排序（我們在 train_cv 內會 sort by timestamp）。

    - 每折 valid 長度固定 = round(n * valid_size)
    - folds 從最早開始往後推
    - expanding=True: train 是從 0 到 valid_start（擴張視窗）
      expanding=False: train 視窗固定長度（這版先不做固定長度，保留接口）
    """
    if n <= 10:
        raise ValueError("Dataset too small for walk-forward CV.")

    vlen = int(round(n * cfg.valid_size))
    vlen = max(1, min(vlen, n - 2))

    # 最後一折的 valid_end = n
    # 往前推 cfg.n_folds 個 windows
    last_valid_end = n
    fold_ends = [last_valid_end - i * vlen for i in range(cfg.n_folds)][::-1]

    for valid_end in fold_ends:
        valid_start = valid_end - vlen
        if valid_start <= 0:
            continue

        if cfg.expanding:
            train_start = 0
        else:
            train_start = 0  # 先保留，之後可改固定長度

        train_idx = np.arange(train_start, valid_start)
        valid_idx = np.arange(valid_start, valid_end)

        if len(train_idx) < 10 or len(valid_idx) < 1:
            continue

        yield train_idx, valid_idx
