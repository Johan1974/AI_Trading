"""GET /api/v1/news/sentiment — laatste nieuws-items + TTL-status vanuit de news-container."""

from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path

from fastapi import APIRouter

router = APIRouter(prefix="/api/v1/news", tags=["news"])

_DB_PATH = Path(os.getenv("NEWS_SENTIMENT_DB_PATH", "data/news_sentiment.db"))
_TTL = int(os.getenv("NEWS_SENTIMENT_TTL_SEC", "900"))
_REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")


def _redis_client():
    try:
        import redis as redis_lib
        return redis_lib.from_url(_REDIS_URL, decode_responses=True, socket_connect_timeout=1, socket_timeout=1)
    except Exception:
        return None


@router.get("/sentiment")
def get_news_sentiment(limit: int = 10):
    items: list[dict] = []
    if _DB_PATH.exists():
        try:
            with sqlite3.connect(str(_DB_PATH)) as conn:
                rows = conn.execute(
                    """SELECT ts_utc, market, headline, description,
                              finbert_score, finbert_confidence, finbert_label
                       FROM news_items
                       ORDER BY id DESC
                       LIMIT ?""",
                    (min(limit, 50),),
                ).fetchall()
                for row in rows:
                    items.append({
                        "ts_utc": row[0],
                        "market": row[1],
                        "headline": row[2],
                        "description": row[3],
                        "score": row[4],
                        "confidence": row[5],
                        "label": row[6],
                    })
        except Exception:
            pass

    ttl_status: dict[str, dict] = {}
    try:
        r = _redis_client()
        if r:
            markets = {item["market"] for item in items}
            now = time.time()
            for mkt in markets:
                ttl_sec = r.ttl(f"news:sentiment:{mkt}")
                raw = r.get(f"news:sentiment:{mkt}")
                score_ts: float | None = None
                if raw:
                    try:
                        score_ts = json.loads(raw).get("ts")
                    except Exception:
                        pass
                ttl_status[mkt] = {
                    "ttl_sec": int(ttl_sec) if ttl_sec and ttl_sec > 0 else 0,
                    "valid": ttl_sec > 0 if ttl_sec else False,
                    "scored_at": score_ts,
                    "age_sec": round(now - score_ts, 0) if score_ts else None,
                }
    except Exception:
        pass

    return {
        "items": items,
        "ttl_status": ttl_status,
        "ttl_max_sec": _TTL,
        "db_available": _DB_PATH.exists(),
    }
