# src/signals/make_signals.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Tuple

import numpy as np


@dataclass
class SignalResult:
    action: str          # BUY / HOLD / SELL
    score: float         # a combined score
    reason: str          # short explanation


def _z(x: float, scale: float) -> float:
    if np.isnan(x):
        return 0.0
    return float(x / max(1e-12, scale))


def make_signal(
    pred_1d: float,
    pred_7d: float,
    # simple default scales (you can later calibrate with OOF std)
    scale_1d: float = 0.02,
    scale_7d: float = 0.06,
    buy_th: float = 0.75,
    sell_th: float = -0.75,
) -> SignalResult:
    """
    Combine 1d and 7d predictions into a score.
    Score = 0.6*z1d + 0.4*z7d by default.
    """
    z1 = _z(pred_1d, scale_1d)
    z7 = _z(pred_7d, scale_7d)
    score = 0.6 * z1 + 0.4 * z7

    if score >= buy_th:
        action = "BUY"
    elif score <= sell_th:
        action = "SELL"
    else:
        action = "HOLD"

    reason = f"pred_1d={pred_1d:.4f}, pred_7d={pred_7d:.4f}, score={score:.2f}"
    return SignalResult(action=action, score=float(score), reason=reason)
