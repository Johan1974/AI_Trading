"""
Bestand: app/services/news_service.py
Relatief pad: ./app/services/news_service.py
Functie: CryptoCompare nieuwsfeed service voor FinBERT input en portal news feed.
"""

from __future__ import annotations

from datetime import datetime

from app.datetime_util import UTC
import time
from typing import Any

import requests


class CryptoCompareNewsService:
    def __init__(self, ttl_seconds: int = 60, timeout_seconds: int = 12) -> None:
        self.ttl_seconds = ttl_seconds
        self.timeout_seconds = timeout_seconds
        self._cache_at = 0.0
        self._cache_payload: list[dict[str, Any]] = []
        self.last_fetch_ok = True

    def fetch_latest_news(self, api_key: str | None, limit: int = 60, force: bool = False) -> list[dict[str, Any]]:
        now = time.time()
        if (not force) and (now - self._cache_at) < float(self.ttl_seconds) and self._cache_payload:
            self.last_fetch_ok = True
            return self._cache_payload

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
                self.last_fetch_ok = False
                return self._cache_payload
            payload = resp.json()
            rows = payload.get("Data") if isinstance(payload, dict) else []
            if not isinstance(rows, list):
                self.last_fetch_ok = False
                return self._cache_payload

            out: list[dict[str, Any]] = []
            for row in rows[: max(1, min(limit, 200))]:
                if not isinstance(row, dict):
                    continue
                title = str(row.get("title") or "").strip()
                if not title:
                    continue
                body = str(row.get("body") or "").strip()
                ts = row.get("published_on")
                published_at = None
                try:
                    if ts is not None:
                        published_at = datetime.fromtimestamp(int(ts), tz=UTC).isoformat().replace("+00:00", "Z")
                except Exception:
                    published_at = None
                out.append(
                    {
                        "id": str(row.get("id") or row.get("guid") or row.get("url") or title),
                        "title": title,
                        "description": body,
                        "url": row.get("url"),
                        "publishedAt": published_at,
                        "source": {"name": str(row.get("source_info", {}).get("name") or "CryptoCompare")},
                        "news_channel": "API",
                    }
                )
            if out:
                self._cache_payload = out
                self._cache_at = now
            self.last_fetch_ok = bool(out)
            return out or self._cache_payload
        except Exception:
            self.last_fetch_ok = False
            return self._cache_payload
