"""
Bestand: app/services/macro_calendar.py
Relatief pad: ./app/services/macro_calendar.py
Functie: Lightweight macro event feed met focus op CPI/FED high-impact windows.
"""

from __future__ import annotations

from datetime import date, datetime

from app.datetime_util import UTC
import time
from typing import Any

import requests


class MacroCalendarService:
    def __init__(self, ttl_seconds: int = 60 * 30, timeout_seconds: int = 12) -> None:
        self.ttl_seconds = ttl_seconds
        self.timeout_seconds = timeout_seconds
        self._cache_at = 0.0
        self._cache_payload: dict[str, Any] = {
            "macro_volatility_window": False,
            "high_impact_today": [],
            "event_count_today": 0,
        }

    def fetch_today_macro_context(self, force: bool = False) -> dict[str, Any]:
        now = time.time()
        if (not force) and (now - self._cache_at) < float(self.ttl_seconds):
            return self._cache_payload
        try:
            resp = requests.get(
                "https://cdn-nfs.faireconomy.media/ff_calendar_thisweek.json",
                timeout=self.timeout_seconds,
            )
            if resp.status_code != 200:
                return self._cache_payload
            rows = resp.json()
            if not isinstance(rows, list):
                return self._cache_payload
            utc_today = datetime.now(UTC).date()
            matched: list[str] = []
            for item in rows:
                if not isinstance(item, dict):
                    continue
                impact = str(item.get("impact") or "").lower()
                title = str(item.get("title") or "")
                country = str(item.get("country") or "")
                dt = self._parse_row_date(item)
                if dt != utc_today:
                    continue
                title_upper = title.upper()
                keyword_hit = (
                    "CPI" in title_upper
                    or "FED" in title_upper
                    or "FOMC" in title_upper
                    or "RATE" in title_upper
                )
                if impact in {"high", "holiday"} and keyword_hit and country in {"USD", "US"}:
                    matched.append(title)
            out = {
                "macro_volatility_window": bool(matched),
                "high_impact_today": matched[:8],
                "event_count_today": len(matched),
            }
            self._cache_payload = out
            self._cache_at = now
            return out
        except Exception:
            return self._cache_payload

    def _parse_row_date(self, item: dict[str, Any]) -> date | None:
        dt_raw = item.get("date")
        if not dt_raw:
            return None
        text = str(dt_raw)
        for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
            try:
                return datetime.strptime(text, fmt).date()
            except Exception:
                continue
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(UTC).date()
        except Exception:
            return None
