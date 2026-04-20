"""
Bestand: app/rl/data.py
Relatief pad: ./app/rl/data.py
Functie: Haalt historische Bitvavo data op en bouwt RL-features met event-annotaties.
"""

from __future__ import annotations

import os
from datetime import datetime

from app.datetime_util import UTC
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from app.rl.events import MAJOR_CRYPTO_EVENTS_UTC
from app.services.fear_greed import FearGreedService
from app.services.macro_calendar import MacroCalendarService
from app.services.whale_watcher import WhaleWatcherService

STORAGE_ROOT = Path.home() / "AI_Trading" / "storage"
PRICE_STORAGE_DIR = STORAGE_ROOT / "historical_prices"
NEWS_STORAGE_DIR = STORAGE_ROOT / "historical_news"
FEATURE_STORAGE_DIR = STORAGE_ROOT / "rl_features"
FEAR_GREED_SERVICE = FearGreedService()
WHALE_WATCHER_SERVICE = WhaleWatcherService()
MACRO_CALENDAR_SERVICE = MacroCalendarService()


def _ensure_storage_dirs() -> None:
    for path in (PRICE_STORAGE_DIR, NEWS_STORAGE_DIR, FEATURE_STORAGE_DIR):
        path.mkdir(parents=True, exist_ok=True)


def _pair_safe_name(market: str) -> str:
    return (market or "UNKNOWN").upper().replace("/", "-").replace(":", "-")


def _price_file_path(market: str, interval: str, start_dt: datetime, end_dt: datetime) -> Path:
    safe_pair = _pair_safe_name(market)
    return PRICE_STORAGE_DIR / (
        f"{safe_pair}_{interval}_{start_dt.strftime('%Y%m%d%H%M%S')}_{end_dt.strftime('%Y%m%d%H%M%S')}.parquet"
    )


def _news_file_path(market: str, start_dt: datetime, end_dt: datetime) -> Path:
    safe_pair = _pair_safe_name(market)
    return NEWS_STORAGE_DIR / (
        f"{safe_pair}_{start_dt.strftime('%Y%m%d%H%M%S')}_{end_dt.strftime('%Y%m%d%H%M%S')}.parquet"
    )


def _feature_file_path(market: str, start_dt: datetime, end_dt: datetime) -> Path:
    safe_pair = _pair_safe_name(market)
    return FEATURE_STORAGE_DIR / (
        f"{safe_pair}_{start_dt.strftime('%Y%m%d%H%M%S')}_{end_dt.strftime('%Y%m%d%H%M%S')}.parquet"
    )


def _save_parquet(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        return
    _ensure_storage_dirs()
    df.to_parquet(path, index=False)


def _save_news_parquet(
    market: str,
    start_dt: datetime,
    end_dt: datetime,
    articles: list[dict[str, Any]],
) -> None:
    if not articles:
        return
    rows = []
    for article in articles:
        rows.append(
            {
                "market": market.upper(),
                "published_at": pd.to_datetime(article.get("publishedAt"), utc=True, errors="coerce"),
                "title": str(article.get("title") or ""),
                "summary": str(article.get("description") or ""),
                "source": str((article.get("source") or {}).get("name") or ""),
                "url": str(article.get("url") or ""),
            }
        )
    news_df = pd.DataFrame(rows).dropna(subset=["published_at"])
    _save_parquet(news_df, _news_file_path(market=market, start_dt=start_dt, end_dt=end_dt))


def fetch_bitvavo_historical_candles(
    market: str,
    interval: str,
    start_dt: datetime,
    end_dt: datetime,
    page_limit: int = 1000,
) -> pd.DataFrame:
    if start_dt.tzinfo is None or end_dt.tzinfo is None:
        raise ValueError("start_dt en end_dt moeten timezone-aware zijn (UTC).")

    base_url = f"https://api.bitvavo.com/v2/{market}/candles"
    all_rows: list[dict[str, Any]] = []
    cursor_end_ms = int(end_dt.timestamp() * 1000)
    start_ms = int(start_dt.timestamp() * 1000)

    while True:
        params = {
            "interval": interval,
            "limit": page_limit,
            "end": cursor_end_ms,
        }
        resp = requests.get(base_url, params=params, timeout=25)
        resp.raise_for_status()
        payload = resp.json()
        if not isinstance(payload, list) or not payload:
            break

        page = []
        for candle in payload:
            ts_ms, open_, high, low, close, volume = candle
            if int(ts_ms) < start_ms:
                continue
            page.append(
                {
                    "timestamp": pd.to_datetime(int(ts_ms), unit="ms", utc=True),
                    "open": float(open_),
                    "high": float(high),
                    "low": float(low),
                    "close": float(close),
                    "volume": float(volume),
                }
            )

        if page:
            all_rows.extend(page)

        oldest_ms = int(payload[-1][0])
        if oldest_ms <= start_ms or len(payload) < page_limit:
            break
        cursor_end_ms = oldest_ms - 1

    if not all_rows:
        raise ValueError("Geen historische Bitvavo candles gevonden voor periode.")

    df = pd.DataFrame(all_rows).drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    df = df.reset_index(drop=True)
    _save_parquet(
        df=df,
        path=_price_file_path(
            market=market,
            interval=interval,
            start_dt=start_dt.astimezone(UTC),
            end_dt=end_dt.astimezone(UTC),
        ),
    )
    return df


def _fetch_historical_news_articles(
    news_query: str,
    news_api_key: str | None,
    start_dt: datetime,
    end_dt: datetime,
    page_size: int = 100,
) -> list[dict[str, Any]]:
    if not news_api_key:
        return []
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": news_query,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": max(10, min(page_size, 100)),
        "from": start_dt.astimezone(UTC).isoformat(),
        "to": end_dt.astimezone(UTC).isoformat(),
        "apiKey": news_api_key,
    }
    try:
        resp = requests.get(url, params=params, timeout=20)
        if resp.status_code != 200:
            return []
        payload = resp.json()
        if not isinstance(payload, dict):
            return []
        return payload.get("articles", []) or []
    except Exception:
        return []


def _headline_sentiment_and_confidence(headline: str, summary: str) -> tuple[float, float]:
    text = f"{headline} {summary}".lower()
    positive_terms = [
        "surge",
        "record",
        "bull",
        "adoption",
        "approval",
        "breakout",
        "rally",
        "growth",
    ]
    negative_terms = [
        "crash",
        "ban",
        "hack",
        "fraud",
        "bear",
        "drop",
        "collapse",
        "selloff",
    ]
    pos_hits = sum(1 for term in positive_terms if term in text)
    neg_hits = sum(1 for term in negative_terms if term in text)
    raw = float(pos_hits - neg_hits)
    sentiment = float(np.tanh(raw / 2.0))
    confidence = float(min(1.0, (pos_hits + neg_hits) / 4.0))
    return sentiment, confidence


def _attach_news_features(
    df: pd.DataFrame,
    market: str,
    news_query: str,
    news_api_key: str | None,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    if df.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float)
    start_dt = df["timestamp"].min().to_pydatetime()
    end_dt = df["timestamp"].max().to_pydatetime()
    coin = (market or "").upper().split("-", 1)[0]
    query = news_query or f"{coin} OR crypto OR bitcoin OR ethereum"
    articles = _fetch_historical_news_articles(
        news_query=query,
        news_api_key=news_api_key,
        start_dt=start_dt,
        end_dt=end_dt,
    )
    if not articles:
        news_df = pd.DataFrame(columns=["timestamp", "sentiment_score", "news_confidence", "social_volume"])
    else:
        news_df = pd.DataFrame(
            [
                {
                    "timestamp": pd.to_datetime(a.get("publishedAt"), utc=True, errors="coerce"),
                    "title": str(a.get("title") or ""),
                    "summary": str(a.get("description") or ""),
                }
                for a in articles
            ]
        ).dropna(subset=["timestamp"])
        if not news_df.empty:
            signals = news_df.apply(
                lambda r: _headline_sentiment_and_confidence(r["title"], r["summary"]),
                axis=1,
            )
            news_df["sentiment_score"] = [float(x[0]) for x in signals]
            news_df["news_confidence"] = [float(x[1]) for x in signals]
            news_df["social_volume"] = 0.0

    if news_df.empty:
        zeros = pd.Series(np.zeros(len(df), dtype=float), index=df.index)
        return zeros.copy(), zeros.copy(), zeros.copy()
    news_df["bucket"] = news_df["timestamp"].dt.floor("h")
    grouped = news_df.groupby("bucket", as_index=False).agg(
        sentiment_score=("sentiment_score", "mean"),
        news_confidence=("news_confidence", "mean"),
        social_volume=("social_volume", "sum"),
    )
    df = df.copy()
    df["bucket"] = df["timestamp"].dt.floor("h")
    merged = df.merge(grouped, how="left", on="bucket")
    sentiment = merged["sentiment_score"].fillna(0.0).astype(float)
    confidence = merged["news_confidence"].fillna(0.0).astype(float)
    social_volume = merged["social_volume"].fillna(0.0).astype(float)
    return sentiment, confidence, social_volume


def build_rl_training_frame(
    candles_df: pd.DataFrame,
    event_window_hours: int = 24,
    market: str = "BTC-EUR",
    news_query: str = "crypto OR bitcoin OR ethereum OR macro economy",
    news_api_key: str | None = None,
    cryptocompare_key: str | None = None,
    cmc_metrics: dict[str, Any] | None = None,
) -> pd.DataFrame:
    df = candles_df.copy()
    df["returns"] = df["close"].pct_change().fillna(0.0)
    df["log_returns"] = np.log1p(df["returns"]).fillna(0.0)
    df["volatility_24"] = df["returns"].rolling(24).std().fillna(0.0)
    df["volatility_72"] = df["returns"].rolling(72).std().fillna(0.0)
    df["range_pct"] = ((df["high"] - df["low"]) / df["close"]).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    df["volume_change"] = df["volume"].pct_change().replace([np.inf, -np.inf], 0.0).fillna(0.0)

    df["major_news_event"] = 0.0
    df["event_impact"] = 0.0

    for event in MAJOR_CRYPTO_EVENTS_UTC:
        event_ts = pd.Timestamp(event["timestamp"]).tz_convert("UTC")
        window = pd.Timedelta(hours=event_window_hours)
        mask = (df["timestamp"] >= (event_ts - window)) & (df["timestamp"] <= (event_ts + window))
        df.loc[mask, "major_news_event"] = 1.0
        df.loc[mask, "event_impact"] = np.maximum(df.loc[mask, "event_impact"], float(event["impact"]))

    df["feature_price_momentum"] = df["close"].pct_change(6).fillna(0.0)
    df["feature_news_volatility_interaction"] = df["major_news_event"] * df["volatility_24"]
    df["feature_event_impact_momentum"] = df["event_impact"] * df["feature_price_momentum"]

    # Technicals via lightweight pandas formulas (no external TA dependency).
    delta = df["close"].diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta.clip(upper=0.0))
    avg_gain = gain.rolling(14, min_periods=14).mean()
    avg_loss = loss.rolling(14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    df["rsi_14"] = pd.Series(rsi).fillna(50.0).astype(float)
    df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema_gap_pct"] = ((df["close"] - df["ema_20"]) / df["ema_20"]).replace(
        [np.inf, -np.inf], 0.0
    ).fillna(0.0)
    mid = df["close"].rolling(20, min_periods=20).mean()
    std = df["close"].rolling(20, min_periods=20).std()
    upper = mid + (2.0 * std)
    lower = mid - (2.0 * std)
    width = ((upper - lower) / mid.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
    pos = ((df["close"] - lower) / (upper - lower).replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
    df["bollinger_width"] = width.fillna(0.0).astype(float)
    df["bollinger_position"] = pos.fillna(0.5).astype(float)

    # Historical news alignment for pre-training period:
    # aggregate article-level signals into the same hourly buckets as price candles.
    hist_sentiment, hist_conf, hist_social_volume = _attach_news_features(
        df=df,
        market=market,
        news_query=news_query,
        news_api_key=news_api_key,
    )
    sentiment_proxy = np.tanh((df["event_impact"] * np.sign(df["feature_price_momentum"])) * 1.8).astype(float)
    df["sentiment_score"] = np.where(np.abs(hist_sentiment.values) > 1e-8, hist_sentiment.values, sentiment_proxy)
    df["news_confidence"] = np.where(hist_conf.values > 1e-8, hist_conf.values, np.clip(df["event_impact"], 0.0, 1.0))
    df["social_volume"] = np.log1p(np.clip(hist_social_volume.values, 0.0, None))
    fear_greed = FEAR_GREED_SERVICE.fetch_index()
    cc_key = (cryptocompare_key or news_api_key or "").strip() or None
    whale = WHALE_WATCHER_SERVICE.fetch_exchange_pressure(api_key=cc_key, lookback_minutes=60)
    macro = MACRO_CALENDAR_SERVICE.fetch_today_macro_context()
    df["fear_greed_score"] = float(fear_greed.get("fear_greed_score", 0.5) or 0.5)
    df["whale_pressure"] = float(whale.get("whale_pressure", 0.0) or 0.0)
    cmc = cmc_metrics or {}
    dom = float(cmc.get("btc_dominance_pct", 0.0) or 0.0)
    if dom <= 0.0:
        dom = float(os.getenv("GENESIS_BTC_DOM_FALLBACK", "52.0") or 52.0)
    df["btc_dominance_pct"] = max(1.0, min(95.0, dom))
    df["macro_volatility_window"] = 1.0 if bool(macro.get("macro_volatility_window")) else 0.0
    df["price_action"] = df["returns"].astype(float)
    # Orderbook imbalance proxy for historical candles when L2 snapshots are unavailable.
    candle_range = (df["high"] - df["low"]).replace(0.0, np.nan)
    directional_move = (df["close"] - df["open"]).fillna(0.0)
    raw_imb = (directional_move / candle_range).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    df["orderbook_imbalance"] = np.tanh(raw_imb * np.log1p(df["volume"].clip(lower=0.0)))

    ema_fast = df["close"].ewm(span=12, adjust=False).mean()
    ema_slow = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = (ema_fast - ema_slow).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    if "bucket" in df.columns:
        df = df.drop(columns=["bucket"])
    out = df.reset_index(drop=True)

    if news_api_key:
        try:
            _save_news_parquet(
                market=market,
                start_dt=out["timestamp"].min().to_pydatetime(),
                end_dt=out["timestamp"].max().to_pydatetime(),
                articles=_fetch_historical_news_articles(
                    news_query=news_query,
                    news_api_key=news_api_key,
                    start_dt=out["timestamp"].min().to_pydatetime(),
                    end_dt=out["timestamp"].max().to_pydatetime(),
                ),
            )
        except Exception:
            pass
    try:
        _save_parquet(
            df=out,
            path=_feature_file_path(
                market=market,
                start_dt=out["timestamp"].min().to_pydatetime(),
                end_dt=out["timestamp"].max().to_pydatetime(),
            ),
        )
    except Exception:
        pass
    return out


def default_training_period() -> tuple[datetime, datetime]:
    return (
        datetime(2024, 1, 1, tzinfo=UTC),
        datetime(2025, 12, 31, 23, 59, tzinfo=UTC),
    )
