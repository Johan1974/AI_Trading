"""
Bestand: app/services/news_engine.py
Relatief pad: ./app/services/news_engine.py
Functie: Combineert API+RSS nieuws met freshness- en lag-berekening.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from app.datetime_util import UTC
from typing import Any

from app.services.news_service import CryptoCompareNewsService
from app.services.rss_engine import RssEngineService


class NewsEngineService:
    def __init__(
        self,
        api_service: CryptoCompareNewsService,
        rss_service: RssEngineService,
        freshness_minutes: int = 15,
    ) -> None:
        self.api_service = api_service
        self.rss_service = rss_service
        self.freshness_minutes = max(1, int(freshness_minutes))

    @staticmethod
    def _parse_ts(ts: str | None) -> datetime | None:
        if not ts:
            return None
        try:
            dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            return dt.astimezone(UTC)
        except Exception:
            return None

    def fetch_fresh_news(self, cryptocompare_key: str | None, limit: int = 80) -> list[dict[str, Any]]:
        api_rows = self.api_service.fetch_latest_news(api_key=cryptocompare_key, limit=limit)
        rss_rows = self.rss_service.fetch_unprocessed_articles(limit=limit)
        now = datetime.now(UTC)
        cutoff = now - timedelta(minutes=self.freshness_minutes)

        out: list[dict[str, Any]] = []
        seen: set[str] = set()
        for row in api_rows:
            self.rss_service.mark_processed(str(row.get("url") or ""))

        for row in (api_rows + rss_rows):
            title = str(row.get("title") or "").strip()
            url = str(row.get("url") or "").strip()
            key = f"{url}|{title}".strip("|")
            if not key or key in seen:
                continue
            published = self._parse_ts(row.get("publishedAt"))
            if published is None or published < cutoff:
                continue
            lag_sec = max(0, int((now - published).total_seconds()))
            patched = dict(row)
            patched["news_lag_sec"] = lag_sec
            patched["processed_at"] = now.isoformat().replace("+00:00", "Z")
            out.append(patched)
            seen.add(key)
            if len(out) >= max(1, min(limit, 200)):
                break
        return out

    @staticmethod
    def prioritize_for_elite_tickers(
        rows: list[dict[str, Any]],
        elite_tickers: list[str],
        coin_aliases: dict[str, list[str]] | None = None,
    ) -> list[dict[str, Any]]:
        alias_map = coin_aliases or {}
        elite = [str(t or "").upper() for t in elite_tickers if str(t or "").strip()]
        if not elite:
            return rows

        def _score(row: dict[str, Any]) -> tuple[int, int]:
            title = str(row.get("title") or "")
            desc = str(row.get("description") or "")
            text = f"{title} {desc}".upper()
            strong = int(abs(float(row.get("sentiment", 0.0) or 0.0)) >= 0.8)
            match = 0
            for tk in elite:
                base = tk.split("-", 1)[0]
                if base and base in text:
                    match = 1
                    break
                for alias in alias_map.get(base, []):
                    if str(alias).upper() in text:
                        match = 1
                        break
                if match:
                    break
            return (match, strong)

        scored = sorted(rows, key=lambda r: _score(r), reverse=True)
        return scored

