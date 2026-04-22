"""
Bestand: app/redis_bridge.py
Relatief pad: ./app/redis_bridge.py
Functie: Redis pub/sub voor worker → portal (trading_updates + system_stats).
"""

from __future__ import annotations

import json
import os
from typing import Any

TRADING_UPDATES_CHANNEL = "trading_updates"
SYSTEM_STATS_CHANNEL = "system_stats"


def _should_publish() -> bool:
    role = str(os.getenv("AI_TRADING_PROCESS", "all") or "all").strip().lower()
    return role in ("worker", "all")


def publish_trading_update(payload: dict[str, Any]) -> None:
    """Sync publish (worker-thread / request-context); faalt stil als REDIS_URL ontbreekt of Redis down."""
    if not _should_publish():
        return
    url = str(os.getenv("REDIS_URL", "") or "").strip()
    if not url:
        return
    try:
        import redis

        r = redis.Redis.from_url(
            url,
            socket_connect_timeout=1.5,
            socket_timeout=1.5,
            decode_responses=True,
        )
        try:
            r.publish(TRADING_UPDATES_CHANNEL, json.dumps(payload, default=str))
        finally:
            r.close()
    except Exception as exc:
        print(f"[REDIS] publish trading_updates mislukt: {exc}")


def publish_system_stats_update(payload: dict[str, Any]) -> None:
    """Worker → portal: compact system_stats JSON (zelfde shape als `/ws/system-stats`)."""
    if not _should_publish():
        return
    url = str(os.getenv("REDIS_URL", "") or "").strip()
    if not url:
        return
    try:
        import redis

        r = redis.Redis.from_url(
            url,
            socket_connect_timeout=1.5,
            socket_timeout=1.5,
            decode_responses=True,
        )
        try:
            r.publish(SYSTEM_STATS_CHANNEL, json.dumps(payload, default=str))
        finally:
            r.close()
    except Exception as exc:
        print(f"[REDIS] publish system_stats mislukt: {exc}")
