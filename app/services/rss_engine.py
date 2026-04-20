"""
Bestand: app/services/rss_engine.py
Relatief pad: ./app/services/rss_engine.py
Functie: Zero-key RSS engine met lokale deduplicatiecache.
"""

from __future__ import annotations

import json
from datetime import datetime

from app.datetime_util import UTC
from pathlib import Path
from typing import Any

import feedparser


class RssEngineService:
    FEEDS = [
        "https://cointelegraph.com/rss",
        "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "https://decrypt.co/feed",
    ]

    def __init__(
        self,
        cache_file: Path | None = None,
        max_cache_items: int = 5000,
    ) -> None:
        self.cache_file = cache_file or (Path.home() / "AI_Trading" / "storage" / "cache" / "processed_news.json")
        self.max_cache_items = max_cache_items
        self._processed_links = self._load_cache()

    def _load_cache(self) -> list[str]:
        try:
            if not self.cache_file.exists():
                return []
            payload = json.loads(self.cache_file.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                return [str(x) for x in payload if str(x).strip()]
        except Exception:
            pass
        return []

    def _save_cache(self) -> None:
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        trimmed = self._processed_links[-self.max_cache_items :]
        self.cache_file.write_text(json.dumps(trimmed, ensure_ascii=True, indent=2), encoding="utf-8")

    def is_processed(self, link: str) -> bool:
        value = str(link or "").strip()
        if not value:
            return False
        return value in self._processed_links

    def mark_processed(self, link: str) -> None:
        value = str(link or "").strip()
        if not value:
            return
        if value not in self._processed_links:
            self._processed_links.append(value)
            self._save_cache()

    def fetch_unprocessed_articles(self, limit: int = 80) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for feed_url in self.FEEDS:
            try:
                parsed = feedparser.parse(feed_url)
            except Exception:
                continue
            source_name = feed_url.split("/")[2]
            for entry in (parsed.entries or []):
                title = str(getattr(entry, "title", "") or "").strip()
                link = str(getattr(entry, "link", "") or "").strip()
                summary = str(getattr(entry, "summary", "") or "").strip()
                published_at = None
                try:
                    parsed_struct = getattr(entry, "published_parsed", None) or getattr(entry, "updated_parsed", None)
                    if parsed_struct:
                        published_at = datetime(*parsed_struct[:6], tzinfo=UTC).isoformat().replace("+00:00", "Z")
                except Exception:
                    published_at = None
                if not title or not link:
                    continue
                if self.is_processed(link):
                    continue
                out.append(
                    {
                        "title": title,
                        "description": summary,
                        "url": link,
                        "publishedAt": published_at,
                        "source": {"name": source_name},
                        "news_channel": "RSS",
                    }
                )
                self.mark_processed(link)
                if len(out) >= max(1, min(limit, 200)):
                    return out
        return out

    def feeds_healthy(self) -> bool:
        for feed_url in self.FEEDS:
            try:
                parsed = feedparser.parse(feed_url)
                if getattr(parsed, "entries", None):
                    return True
            except Exception:
                continue
        return False
