"""
Bestand: app/services/fear_greed.py
Relatief pad: ./app/services/fear_greed.py
Functie: Haalt Fear & Greed index op via alternative.me met dagelijkse cache.
"""

from __future__ import annotations

import time
from typing import Any

import requests


class FearGreedService:
    def __init__(self, ttl_seconds: int = 60 * 60 * 6, timeout_seconds: int = 12) -> None:
        self.ttl_seconds = ttl_seconds
        self.timeout_seconds = timeout_seconds
        self._cache_at = 0.0
        self._cache_payload: dict[str, Any] = {
            "fear_greed_value": 50.0,
            "fear_greed_score": 0.5,
            "classification": "Neutral",
            "timestamp": None,
        }

    def fetch_index(self, force: bool = False) -> dict[str, Any]:
        now = time.time()
        if (not force) and (now - self._cache_at) < float(self.ttl_seconds):
            return self._cache_payload
        try:
            resp = requests.get(
                "https://api.alternative.me/fng/",
                params={"limit": 1, "format": "json"},
                timeout=self.timeout_seconds,
            )
            if resp.status_code != 200:
                return self._cache_payload
            payload = resp.json()
            rows = payload.get("data") if isinstance(payload, dict) else []
            if not isinstance(rows, list) or not rows:
                return self._cache_payload
            row = rows[0] if isinstance(rows[0], dict) else {}
            raw = float(row.get("value", 50.0) or 50.0)
            out = {
                "fear_greed_value": raw,
                "fear_greed_score": max(0.0, min(1.0, raw / 100.0)),
                "classification": str(row.get("value_classification") or "Neutral"),
                "timestamp": row.get("timestamp"),
            }
            self._cache_payload = out
            self._cache_at = now
            return out
        except Exception:
            return self._cache_payload
