from __future__ import annotations

import json
import urllib.request
from typing import Any, Dict, Optional


def send_discord_webhook(
    webhook_url: str,
    content: Optional[str] = None,
    username: Optional[str] = None,
    avatar_url: Optional[str] = None,
    embeds: Optional[list[dict[str, Any]]] = None,
) -> None:
    """
    Send message to Discord via webhook.
    Uses only stdlib (urllib), no external dependencies.
    """
    payload: Dict[str, Any] = {}
    if content:
        payload["content"] = content
    if username:
        payload["username"] = username
    if avatar_url:
        payload["avatar_url"] = avatar_url
    if embeds:
        payload["embeds"] = embeds

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        webhook_url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        _ = resp.status
