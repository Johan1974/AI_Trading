"""
Bestand: app/services/ingestion.py
Relatief pad: ./app/services/ingestion.py
Functie: Haalt OHLCV-marktdata en nieuwsdata op met minimale validatie.
"""

from typing import Any
from datetime import datetime, timedelta
import contextlib
import io

from app.datetime_util import UTC
import email.utils
import xml.etree.ElementTree as ET

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


RSS_FEEDS = [
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://cointelegraph.com/rss",
]


def _get_resilient_session(retries: int = 3, backoff_factor: float = 0.5, timeout: float = 12.0) -> requests.Session:
    """Creëert een geharde request session die NOOIT oneindig blokkeert."""
    session = requests.Session()
    retry_strategy = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=[408, 429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    
    request_orig = session.request
    session.request = lambda method, url, **kwargs: request_orig(
        method, url, timeout=kwargs.pop('timeout', timeout), **kwargs
    )
    return session


def _safe_parse_datetime(text: str | None) -> datetime | None:
    if not text:
        return None
    raw = str(text).strip()
    try:
        if raw.endswith("Z"):
            return datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(UTC)
        return datetime.fromisoformat(raw).astimezone(UTC)
    except Exception:
        pass
    try:
        dt = email.utils.parsedate_to_datetime(raw)
        if dt is None:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt.astimezone(UTC)
    except Exception:
        return None


def _fetch_rss_news_articles(max_items: int = 60) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    session = _get_resilient_session(retries=2, timeout=10.0)
    
    for feed_url in RSS_FEEDS:
        try:
            resp = session.get(feed_url)
            if resp.status_code != 200 or not resp.text:
                continue
            root = ET.fromstring(resp.text)
            for item in root.findall(".//item"):
                title = (item.findtext("title") or "").strip()
                link = (item.findtext("link") or "").strip()
                description = (item.findtext("description") or "").strip()
                pub = (item.findtext("pubDate") or "").strip()
                key = f"{title}|{link}"
                if not title or key in seen:
                    continue
                seen.add(key)
                dt = _safe_parse_datetime(pub)
                items.append(
                    {
                        "title": title,
                        "description": description,
                        "url": link,
                        "publishedAt": (dt.isoformat().replace("+00:00", "Z") if dt else None),
                        "source": {"name": feed_url.split("/")[2]},
                    }
                )
        except Exception:
            continue
    items.sort(
        key=lambda x: _safe_parse_datetime(x.get("publishedAt")) or datetime(1970, 1, 1, tzinfo=UTC),
        reverse=True,
    )
    return items[:max_items]


def fetch_market_data(ticker: str, lookback_days: int) -> Any:
    import pandas as pd
    import yfinance as yf

    symbol = str(ticker or "").upper().strip()
    session = _get_resilient_session(retries=3, timeout=12.0)

    # Bitvavo fallback for EUR crypto pairs when yfinance coverage is weak/missing.
    # This prevents Elite-8 cycles from failing on newer/smaller assets.
    def _bitvavo_daily_fallback() -> Any:
        if "-" not in symbol or not symbol.endswith("-EUR"):
            return pd.DataFrame()
        try:
            url = f"https://api.bitvavo.com/v2/{symbol}/candles"
            resp = session.get(url, params={"interval": "1d", "limit": max(60, int(lookback_days) + 10)})
            if resp.status_code != 200:
                return pd.DataFrame()
            candles = resp.json()
            if not isinstance(candles, list) or not candles:
                return pd.DataFrame()
            rows: list[dict[str, Any]] = []
            for c in candles:
                if not isinstance(c, list) or len(c) < 6:
                    continue
                try:
                    ts = int(c[0])
                    rows.append(
                        {
                            "Date": datetime.fromtimestamp(ts / 1000, tz=UTC),
                            "Open": float(c[1]),
                            "High": float(c[2]),
                            "Low": float(c[3]),
                            "Close": float(c[4]),
                            "Volume": float(c[5]),
                        }
                    )
                except Exception:
                    continue
            if not rows:
                return pd.DataFrame()
            dfb = pd.DataFrame(rows).sort_values("Date").reset_index(drop=True)
            return dfb
        except Exception:
            return pd.DataFrame()

    # yfinance can print noisy errors for unknown symbols (e.g. EDU-EUR).
    # Capture/suppress those and fail gracefully so startup logs remain clean.
    try:
        with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            df = yf.download(
                symbol,
                period=f"{lookback_days}d",
                interval="1d",
                auto_adjust=True,
                progress=False,
                session=session,
                ignore_tz=True
            )
    except Exception as exc:
        df = pd.DataFrame()
    if df.empty or len(df) < 40:
        dfb = _bitvavo_daily_fallback()
        if not dfb.empty and len(dfb) >= 40:
            return dfb
        raise ValueError(f"Onvoldoende marktdata voor ticker {symbol}.")
    if df.empty or len(df) < 40:
        raise ValueError(f"Onvoldoende marktdata voor ticker {symbol}.")
    return df.reset_index()


def fetch_news_articles(news_query: str, news_api_key: str | None) -> list[dict[str, Any]]:
    cryptocompare_key = str(news_api_key or "").strip()
    session = _get_resilient_session(retries=2, timeout=10.0)
    
    if cryptocompare_key:
        try:
            resp = session.get(
                "https://min-api.cryptocompare.com/data/v2/news/",
                params={"lang": "EN"},
                headers={"authorization": f"Apikey {cryptocompare_key}"}
            )
            if resp.status_code == 200:
                payload = resp.json()
                rows = payload.get("Data", []) if isinstance(payload, dict) else []
                if isinstance(rows, list) and rows:
                    converted: list[dict[str, Any]] = []
                    for row in rows[:60]:
                        if not isinstance(row, dict):
                            continue
                        title = str(row.get("title") or "").strip()
                        if not title:
                            continue
                        ts = row.get("published_on")
                        published = None
                        try:
                            if ts is not None:
                                published = datetime.fromtimestamp(int(ts), tz=UTC).isoformat().replace("+00:00", "Z")
                        except Exception:
                            published = None
                        converted.append(
                            {
                                "title": title,
                                "description": str(row.get("body") or ""),
                                "url": row.get("url"),
                                "publishedAt": published,
                                "source": {
                                    "name": str((row.get("source_info") or {}).get("name") or "CryptoCompare")
                                },
                            }
                        )
                    if converted:
                        return converted
        except Exception:
            pass

    rss_items = _fetch_rss_news_articles(max_items=40)
    now = datetime.now(UTC)
    recent_rss = [
        x
        for x in rss_items
        if (_safe_parse_datetime(x.get("publishedAt")) or datetime(1970, 1, 1, tzinfo=UTC))
        >= (now - timedelta(hours=36))
    ]

    if not news_api_key:
        return recent_rss

    everything_url = "https://newsapi.org/v2/everything"
    windows = [timedelta(hours=6), timedelta(hours=24), timedelta(days=3)]
    api_items: list[dict[str, Any]] = []
    for window in windows:
        params = {
            "q": news_query,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": 50,
            "from": (now - window).isoformat(),
            "apiKey": news_api_key,
        }
        try:
            resp = session.get(everything_url, params=params)
            if resp.status_code == 200:
                articles = resp.json().get("articles", [])
                if articles:
                    api_items = articles
                    break
        except Exception:
            continue

    # Fallback to top-headlines for fresher stream when everything-query is stale.
    try:
        top_headlines_url = "https://newsapi.org/v2/top-headlines"
        top_params = {
            "q": "crypto",
            "language": "en",
            "pageSize": 50,
            "apiKey": news_api_key,
        }
        resp = session.get(top_headlines_url, params=top_params)
        if resp.status_code == 200:
            articles = resp.json().get("articles", [])
            if articles:
                api_items = articles
    except Exception:
        pass

    # Final fallback: best-effort recent feed without explicit from-window.
    if not api_items:
        try:
            params = {
                "q": news_query,
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": 50,
                "apiKey": news_api_key,
            }
            resp = session.get(everything_url, params=params)
            if resp.status_code == 200:
                api_items = resp.json().get("articles", [])
        except Exception:
            pass

    merged: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for source_items in (api_items, recent_rss):
        for article in source_items:
            key = f"{article.get('title','')}|{article.get('url','')}"
            if key in seen_keys:
                continue
            seen_keys.add(key)
            merged.append(article)

    merged.sort(
        key=lambda x: _safe_parse_datetime(x.get("publishedAt")) or datetime(1970, 1, 1, tzinfo=UTC),
        reverse=True,
    )
    return merged[:60]
