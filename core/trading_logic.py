"""
Trade-discipline helpers for minimum hold time and confidence gating.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from app.datetime_util import UTC


MIN_HOLD_MINUTES = 15
MIN_BUY_CONFIDENCE = 0.75


def _parse_ts(value: str) -> datetime | None:
    txt = str(value or "").strip()
    if not txt:
        return None
    try:
        dt = datetime.fromisoformat(txt.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt.astimezone(UTC)
    except Exception:
        return None


def should_block_sell_for_min_hold(wallet: dict[str, Any], market: str, min_hold_minutes: int = MIN_HOLD_MINUTES) -> bool:
    target = str(market or "").upper()
    lots: list[Any] | None = None
    if isinstance(wallet, dict):
        bym = wallet.get("open_lots_by_market")
        if isinstance(bym, dict):
            raw = bym.get(target)
            if isinstance(raw, list) and raw:
                lots = [x for x in raw if isinstance(x, dict)]
        if not lots:
            flat = wallet.get("open_lots")
            if isinstance(flat, list):
                lots = [x for x in flat if isinstance(x, dict) and str(x.get("market") or "").upper() == target]
    if not isinstance(lots, list) or not lots:
        return False
    now = datetime.now(UTC)
    oldest_entry: datetime | None = None
    for lot in lots:
        if not isinstance(lot, dict):
            continue
        dt = _parse_ts(str(lot.get("entry_ts_utc") or ""))
        if dt is None:
            continue
        if oldest_entry is None or dt < oldest_entry:
            oldest_entry = dt
    if oldest_entry is None:
        return False
    held_minutes = (now - oldest_entry).total_seconds() / 60.0
    return held_minutes < max(1, int(min_hold_minutes))

