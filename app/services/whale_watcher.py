"""
Bestand: app/services/whale_watcher.py
Relatief pad: ./app/services/whale_watcher.py
Functie: Whale pressure via CryptoCompare nieuws (headline-scan); geen Whale Alert API.
"""

from __future__ import annotations

import math
import re
import time
from datetime import datetime, timedelta

from app.datetime_util import UTC
from typing import Any

import requests

# Sterkste patronen eerst (substring / regex overlap).
_WHALE_PATTERNS = [
    (re.compile(r"million\s+btc\b", re.I), 0.55),
    (re.compile(r"million\s+bitcoin\b", re.I), 0.55),
    (re.compile(r"million\s+eth\b", re.I), 0.52),
    (re.compile(r"million\s+ethereum\b", re.I), 0.52),
    (re.compile(r"\bbillion\b", re.I), 0.5),
    (re.compile(r"large\s+transfer", re.I), 0.48),
    (re.compile(r"\bwhale\b", re.I), 0.42),
    (re.compile(r"million\s*\$", re.I), 0.35),
    (re.compile(r"million\s+usd\b", re.I), 0.35),
    (re.compile(r"\baccumulation\b", re.I), 0.25),
]


class WhaleWatcherService:
    def __init__(self, ttl_seconds: int = 300, timeout_seconds: int = 12) -> None:
        self.ttl_seconds = ttl_seconds
        self.timeout_seconds = timeout_seconds
        self._cache_at = 0.0
        self._cache_payload: dict[str, Any] = {
            "whale_pressure": 0.0,
            "whale_headline_hits": 0,
            "headlines_scanned": 0,
            "window_minutes": 60,
            "source": "cryptocompare_news",
        }

    def _fetch_cryptocompare_news(self, api_key: str | None, limit: int = 100) -> list[dict[str, Any]]:
        key = str(api_key or "").strip()
        headers: dict[str, str] = {}
        if key:
            headers["authorization"] = f"Apikey {key}"
        try:
            resp = requests.get(
                "https://min-api.cryptocompare.com/data/v2/news/",
                params={"lang": "EN"},
                headers=headers,
                timeout=self.timeout_seconds,
            )
            if resp.status_code != 200:
                return []
            payload = resp.json()
            rows = payload.get("Data") if isinstance(payload, dict) else []
            if not isinstance(rows, list):
                return []
            out: list[dict[str, Any]] = []
            for row in rows[: max(1, min(limit, 200))]:
                if not isinstance(row, dict):
                    continue
                title = str(row.get("title") or "").strip()
                body = str(row.get("body") or "").strip()
                ts = row.get("published_on")
                out.append({"title": title, "body": body, "published_on": ts})
            return out
        except Exception:
            return []

    @staticmethod
    def _headline_whale_weight(text: str) -> float:
        if not text or not str(text).strip():
            return 0.0
        best = 0.0
        for pattern, w in _WHALE_PATTERNS:
            if pattern.search(text):
                best = max(best, w)
        return best

    def _filter_recent(
        self, articles: list[dict[str, Any]], lookback_minutes: int
    ) -> list[dict[str, Any]]:
        if not articles:
            return []
        cutoff = datetime.now(UTC) - timedelta(minutes=max(15, lookback_minutes))
        out: list[dict[str, Any]] = []
        for a in articles:
            ts = a.get("published_on")
            try:
                pub = datetime.fromtimestamp(int(ts), tz=UTC) if ts is not None else None
            except Exception:
                pub = None
            if pub is None or pub >= cutoff:
                out.append(a)
        return out if out else articles

    def fetch_exchange_pressure(self, api_key: str | None, lookback_minutes: int = 60) -> dict[str, Any]:
        """Leest CryptoCompare headlines en zet een score 0..1 (whale / grote verplaatsing)."""
        now = time.time()
        if (now - self._cache_at) < float(self.ttl_seconds):
            return self._cache_payload

        articles = self._fetch_cryptocompare_news(api_key, limit=100)
        recent = self._filter_recent(articles, lookback_minutes=lookback_minutes)

        raw_score = 0.0
        hits = 0
        for a in recent:
            title = str(a.get("title") or "")
            body = str(a.get("body") or "")[: 1200]
            blob = f"{title}\n{body}"
            w = self._headline_whale_weight(blob)
            if w > 0:
                hits += 1
                raw_score += w

        # Meerdere treffers -> hogere druk; tanh voorkomt saturatie.
        pressure = float(math.tanh(raw_score / 2.2))
        out = {
            "whale_pressure": max(0.0, min(1.0, pressure)),
            "whale_headline_hits": int(hits),
            "headlines_scanned": int(len(recent)),
            "window_minutes": int(lookback_minutes),
            "source": "cryptocompare_news",
        }
        self._cache_payload = out
        self._cache_at = now
        return out
