"""
Bestand: app/services/coinmarketcap.py
Relatief pad: ./app/services/coinmarketcap.py
Functie: Haalt globale crypto-marktmetrics op via CoinMarketCap API.
"""

from __future__ import annotations

from datetime import datetime

from app.datetime_util import UTC
from typing import Any

import requests


class CoinMarketCapService:
    def __init__(self, ttl_seconds: int = 1800) -> None:
        self.ttl_seconds = max(60, int(ttl_seconds))
        self._cache_at: datetime | None = None
        self._cache_payload: dict[str, Any] | None = None

    def fetch_global_metrics(self, api_key: str | None, force: bool = False) -> dict[str, Any]:
        key = str(api_key or "").strip()
        if not key:
            return {
                "btc_dominance_pct": 0.0,
                "total_market_cap_usd": 0.0,
                "total_volume_24h_usd": 0.0,
                "updated_at": None,
                "source": "CMC",
                "ok": False,
            }
        now = datetime.now(UTC)
        if not force and self._cache_payload is not None and self._cache_at is not None:
            age = (now - self._cache_at).total_seconds()
            if age < self.ttl_seconds:
                return self._cache_payload

        try:
            resp = requests.get(
                "https://pro-api.coinmarketcap.com/v1/global-metrics/quotes/latest",
                headers={"X-CMC_PRO_API_KEY": key, "Accept": "application/json"},
                timeout=12,
            )
            if resp.status_code != 200:
                raise RuntimeError(f"http_{resp.status_code}")
            payload = resp.json() if resp.content else {}
            data = payload.get("data", {}) if isinstance(payload, dict) else {}
            quote = data.get("quote", {}).get("USD", {}) if isinstance(data, dict) else {}
            result = {
                "btc_dominance_pct": float(data.get("btc_dominance") or 0.0),
                "total_market_cap_usd": float(quote.get("total_market_cap") or 0.0),
                "total_volume_24h_usd": float(quote.get("total_volume_24h") or 0.0),
                "updated_at": now.isoformat(),
                "source": "CMC",
                "ok": True,
            }
            self._cache_payload = result
            self._cache_at = now
            return result
        except Exception:
            return {
                "btc_dominance_pct": 0.0,
                "total_market_cap_usd": 0.0,
                "total_volume_24h_usd": 0.0,
                "updated_at": now.isoformat(),
                "source": "CMC",
                "ok": False,
            }
