"""
News-fetching + FinBERT scoring loop voor de aparte news-container.

Elke NEWS_FETCH_INTERVAL_SEC seconden:
  1. Haalt nieuws op via CryptoCompare API + RSS feeds
  2. Filtert per markt op coin-naam
  3. Scoort via FinBERT (CPU)
  4. Schrijft resultaat naar:
     - Redis: SET news:sentiment:{MARKET} {json} EX NEWS_SENTIMENT_TTL_SEC
     - SQLite: news_sentiment.db / news_items tabel (voor dashboard)
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

import redis as redis_lib

logging.basicConfig(level=logging.INFO, format="%(asctime)s [NEWS-SCORER] %(message)s")
_log = logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
TTL = int(os.getenv("NEWS_SENTIMENT_TTL_SEC", "900"))
INTERVAL = int(os.getenv("NEWS_FETCH_INTERVAL_SEC", "300"))
DB_PATH = Path(os.getenv("NEWS_SENTIMENT_DB_PATH", "data/news_sentiment.db"))
TICKERS = [t.strip().upper() for t in os.getenv("TICKERS", "BTC-EUR,ETH-EUR,XRP-EUR,SOL-EUR,ADA-EUR").split(",") if t.strip()]
CRYPTOCOMPARE_KEY = os.getenv("CRYPTOCOMPARE_KEY", "")
MAX_ITEMS_PER_MARKET = 100


def _init_db(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS news_items (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_utc              TEXT    NOT NULL,
            market              TEXT    NOT NULL,
            headline            TEXT    NOT NULL,
            description         TEXT    NOT NULL DEFAULT '',
            finbert_score       REAL    NOT NULL,
            finbert_confidence  REAL    NOT NULL,
            finbert_label       TEXT    NOT NULL DEFAULT 'neutral'
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ni_market_ts ON news_items(market, ts_utc)")


def _coin_from_market(market: str) -> str:
    return market.split("-")[0] if "-" in market else market


_COIN_ALIASES: dict[str, list[str]] = {
    "BTC": ["bitcoin", "btc"],
    "ETH": ["ethereum", "eth", "ether"],
    "XRP": ["ripple", "xrp"],
    "SOL": ["solana", "sol"],
    "ADA": ["cardano", "ada"],
    "DOGE": ["dogecoin", "doge"],
    "HYPE": ["hype", "hyperliquid"],
    "TON": ["ton", "toncoin"],
    "LINK": ["chainlink", "link"],
    "SUI": ["sui"],
    "USDC": ["usdc", "usd coin"],
}


def _filter_for_coin(articles: list[dict], coin: str) -> list[dict]:
    aliases = _COIN_ALIASES.get(coin.upper(), [coin.lower()])
    aliases = [coin.lower()] + aliases
    result = []
    for a in articles:
        text = (a.get("title", "") + " " + a.get("description", "")).lower()
        if any(alias in text for alias in aliases):
            result.append(a)
    return result


def _fetch_articles() -> list[dict]:
    articles: list[dict] = []
    if CRYPTOCOMPARE_KEY:
        try:
            import requests
            resp = requests.get(
                "https://min-api.cryptocompare.com/data/v2/news/",
                params={"lang": "EN", "api_key": CRYPTOCOMPARE_KEY},
                timeout=10,
            )
            if resp.ok:
                for item in resp.json().get("Data", []):
                    articles.append({
                        "title": item.get("title", ""),
                        "description": item.get("body", "")[:500],
                        "source": item.get("source", ""),
                    })
        except Exception as exc:
            _log.warning("CryptoCompare fetch failed: %s", exc)
    try:
        import feedparser
        rss_feeds = [
            "https://cointelegraph.com/rss",
            "https://www.coindesk.com/arc/outboundfeeds/rss/",
        ]
        for url in rss_feeds:
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries[:20]:
                    articles.append({
                        "title": getattr(entry, "title", ""),
                        "description": getattr(entry, "summary", "")[:500],
                        "source": url,
                    })
            except Exception:
                pass
    except ImportError:
        pass
    return articles


def _get_active_markets(r: redis_lib.Redis) -> list[str]:
    try:
        raw = r.get("trading:active_markets")
        if raw:
            parsed = json.loads(raw)
            if isinstance(parsed, list) and parsed:
                return [str(m).upper() for m in parsed]
    except Exception:
        pass
    return TICKERS


def run_loop() -> None:
    _log.info("Initialiseer FinBERT (CPU)...")
    from app.ai.sentiment.finbert_sentiment import FinBertSentimentAnalyzer
    finbert = FinBertSentimentAnalyzer()
    _log.info("FinBERT geladen. Start news-loop (interval=%ds, TTL=%ds).", INTERVAL, TTL)

    r = redis_lib.from_url(REDIS_URL, decode_responses=True, socket_connect_timeout=3)
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    while True:
        cycle_start = time.monotonic()
        try:
            markets = _get_active_markets(r)
            articles = _fetch_articles()
            if not articles:
                _log.info("Scan complete: no new headlines, keeping current data.")
            else:
                _log.info("Opgehaald: %d artikelen voor %d markten.", len(articles), len(markets))

            with sqlite3.connect(str(DB_PATH)) as conn:
                _init_db(conn)
                ts = datetime.now(timezone.utc).isoformat()

                for market in markets:
                    coin = _coin_from_market(market)
                    filtered = _filter_for_coin(articles, coin) or articles[:5]
                    texts = [
                        f"{a.get('title', '')}. {a.get('description', '')}".strip()
                        for a in filtered
                    ][:20]

                    result = finbert.score_with_breakdown(texts)
                    agg = result["aggregate"]
                    items = result.get("items", [])

                    payload = {
                        "score": round(float(agg.score), 4),
                        "confidence": round(float(agg.confidence), 4),
                        "ts": time.time(),
                        "items": items[:10],
                    }
                    r.setex(f"news:sentiment:{market}", TTL, json.dumps(payload))

                    for i, art in enumerate(filtered[:10]):
                        item = items[i] if i < len(items) else {}
                        conn.execute(
                            """INSERT INTO news_items
                               (ts_utc, market, headline, description, finbert_score, finbert_confidence, finbert_label)
                               VALUES (?,?,?,?,?,?,?)""",
                            (
                                ts,
                                market,
                                str(art.get("title", ""))[:500],
                                str(art.get("description", ""))[:1000],
                                float(item.get("signed_score", agg.score)),
                                float(item.get("confidence", agg.confidence)),
                                str(item.get("label", "neutral")),
                            ),
                        )
                    conn.execute(
                        "DELETE FROM news_items WHERE market=? AND id NOT IN "
                        "(SELECT id FROM news_items WHERE market=? ORDER BY id DESC LIMIT ?)",
                        (market, market, MAX_ITEMS_PER_MARKET),
                    )
                    _log.info("%s → score=%.4f conf=%.4f (%d artikelen)", market, agg.score, agg.confidence, len(filtered))

        except Exception as exc:
            _log.error("Cycle-fout: %s", exc)

        elapsed = time.monotonic() - cycle_start
        sleep_for = max(0.0, INTERVAL - elapsed)
        _log.info("Cyclus klaar in %.1fs. Volgende run over %.0fs.", elapsed, sleep_for)
        time.sleep(sleep_for)


if __name__ == "__main__":
    run_loop()
