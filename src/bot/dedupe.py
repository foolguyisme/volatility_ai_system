from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class DedupeResult:
    should_send: bool
    message_hash: str
    reason: str


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def check_and_log_dedupe(
    log_dir: str,
    date_str: str,
    message_text: str,
    enable: bool = True,
) -> DedupeResult:
    """
    If enabled, prevents sending the same message twice per day (same hash).
    Writes a JSON log for traceability.
    """
    os.makedirs(log_dir, exist_ok=True)
    msg_hash = _sha256(message_text)
    log_path = os.path.join(log_dir, f"{date_str}.json")

    if not enable:
        _write_log(log_path, date_str, msg_hash, "disabled", sent=True, error=None)
        return DedupeResult(True, msg_hash, "dedupe disabled")

    if os.path.exists(log_path):
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                prev = json.load(f)
            prev_hash = prev.get("message_hash")
            prev_sent = bool(prev.get("sent", False))
            if prev_sent and prev_hash == msg_hash:
                return DedupeResult(False, msg_hash, "already sent (same hash)")
        except Exception:
            pass

    _write_log(log_path, date_str, msg_hash, "ready_to_send", sent=False, error=None)
    return DedupeResult(True, msg_hash, "not sent yet")


def mark_sent(log_dir: str, date_str: str, message_hash: str) -> None:
    log_path = os.path.join(log_dir, f"{date_str}.json")
    _write_log(log_path, date_str, message_hash, "sent", sent=True, error=None)


def mark_error(log_dir: str, date_str: str, message_hash: str, error: str) -> None:
    log_path = os.path.join(log_dir, f"{date_str}.json")
    _write_log(log_path, date_str, message_hash, "error", sent=False, error=error)


def _write_log(log_path: str, date_str: str, message_hash: str, status: str, sent: bool, error: Optional[str]) -> None:
    payload = {
        "date": date_str,
        "message_hash": message_hash,
        "status": status,
        "sent": sent,
        "error": error,
        "ts": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
