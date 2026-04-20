"""
Bestand: app/services/data_aggregator.py
Relatief pad: ./app/services/data_aggregator.py
Functie: Combineert Bitvavo candles, Fear & Greed en NewsAPI in een genormaliseerd feature-DataFrame.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from app.datetime_util import UTC
from typing import Any

import numpy as np
import pandas as pd
import requests


@dataclass
class AggregatorConfig:
    market: str = "BTC-EUR"
    interval: str = "1h"
    candle_limit: int = 200
    news_query: str = "bitcoin OR crypto OR regulation OR macro"
    news_api_key: str | None = None
    volatility_window: int = 12


class DataAggregator:
    def __init__(self, config: AggregatorConfig) -> None:
        self.config = config

    def fetch_bitvavo_candles(self) -> pd.DataFrame:
        url = f"https://api.bitvavo.com/v2/{self.config.market}/candles"
        params = {"interval": self.config.interval, "limit": self.config.candle_limit}
        resp = requests.get(url, params=params, timeout=20)
        resp.raise_for_status()

        candles = resp.json()
        if not isinstance(candles, list) or not candles:
            raise ValueError("Geen Bitvavo candle data ontvangen.")

        rows: list[dict[str, Any]] = []
        for candle in candles:
            ts_ms, open_, high, low, close, volume = candle
            rows.append(
                {
                    "timestamp": pd.to_datetime(int(ts_ms), unit="ms", utc=True),
                    "open": float(open_),
                    "high": float(high),
                    "low": float(low),
                    "close": float(close),
                    "volume": float(volume),
                }
            )

        df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
        return df

    def fetch_fear_greed_index(self) -> pd.DataFrame:
        url = "https://api.alternative.me/fng/"
        resp = requests.get(url, params={"limit": 0, "format": "json"}, timeout=20)
        resp.raise_for_status()
        payload = resp.json()
        data = payload.get("data", [])
        if not data:
            return pd.DataFrame(columns=["timestamp", "fng_value", "fng_classification"])

        rows = []
        for row in data:
            unix_ts = int(row.get("timestamp", "0"))
            rows.append(
                {
                    "timestamp": pd.to_datetime(unix_ts, unit="s", utc=True),
                    "fng_value": float(row.get("value", 0)),
                    "fng_classification": str(row.get("value_classification", "unknown")).lower(),
                }
            )
        return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)

    def fetch_news_stream(self) -> pd.DataFrame:
        if not self.config.news_api_key:
            return pd.DataFrame(columns=["timestamp", "news_count", "news_sentiment_raw"])

        now = datetime.now(UTC)
        lookback = max(2, int(np.ceil(self.config.candle_limit / 24)))
        from_dt = now - timedelta(days=lookback)
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": self.config.news_query,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": 100,
            "from": from_dt.isoformat(),
            "to": now.isoformat(),
            "apiKey": self.config.news_api_key,
        }
        resp = requests.get(url, params=params, timeout=25)
        resp.raise_for_status()
        articles = resp.json().get("articles", [])
        if not articles:
            return pd.DataFrame(columns=["timestamp", "news_count", "news_sentiment_raw"])

        rows = []
        for article in articles:
            published_at = article.get("publishedAt")
            if not published_at:
                continue
            text = f"{article.get('title', '')} {article.get('description', '')}".lower()
            rows.append(
                {
                    "timestamp": pd.to_datetime(published_at, utc=True, errors="coerce"),
                    "sentiment": self._heuristic_sentiment(text),
                }
            )

        raw_news = pd.DataFrame(rows).dropna(subset=["timestamp"])
        if raw_news.empty:
            return pd.DataFrame(columns=["timestamp", "news_count", "news_sentiment_raw"])

        grouped = (
            raw_news.set_index("timestamp")
            .resample(self.config.interval)
            .agg(news_count=("sentiment", "size"), news_sentiment_raw=("sentiment", "mean"))
            .reset_index()
        )
        grouped["news_sentiment_raw"] = grouped["news_sentiment_raw"].fillna(0.0)
        return grouped

    def build_normalized_frame(self) -> pd.DataFrame:
        market_df = self.fetch_bitvavo_candles()
        fng_df = self.fetch_fear_greed_index()
        news_df = self.fetch_news_stream()

        market_df = market_df.set_index("timestamp")
        market_df["returns"] = market_df["close"].pct_change().fillna(0.0)
        market_df["volatility"] = (
            market_df["returns"].rolling(window=self.config.volatility_window).std().fillna(0.0)
        )

        if not fng_df.empty:
            fng_aligned = fng_df.set_index("timestamp")[["fng_value", "fng_classification"]]
            market_df = market_df.join(fng_aligned, how="left")
            market_df["fng_value"] = market_df["fng_value"].ffill().fillna(50.0)
            market_df["fng_classification"] = market_df["fng_classification"].ffill().fillna("neutral")
        else:
            market_df["fng_value"] = 50.0
            market_df["fng_classification"] = "neutral"

        if not news_df.empty:
            news_aligned = news_df.set_index("timestamp")
            market_df = market_df.join(news_aligned, how="left")
        else:
            market_df["news_count"] = 0
            market_df["news_sentiment_raw"] = 0.0

        market_df["news_count"] = market_df["news_count"].fillna(0).astype(int)
        market_df["news_sentiment_raw"] = market_df["news_sentiment_raw"].fillna(0.0)
        news_mean = market_df["news_count"].mean()
        news_std = market_df["news_count"].std()
        market_df["news_peak_zscore"] = (
            (market_df["news_count"] - news_mean) / news_std if news_std and news_std > 0 else 0.0
        )
        market_df["price_volatility_interaction"] = market_df["news_peak_zscore"] * market_df["volatility"]

        normalized = market_df.reset_index()
        normalized["timestamp"] = normalized["timestamp"].dt.tz_convert("UTC")
        return normalized

    @staticmethod
    def _heuristic_sentiment(text: str) -> float:
        positive = ("surge", "rally", "growth", "approval", "breakout", "bull")
        negative = ("hack", "crash", "ban", "lawsuit", "fear", "bear")
        pos_hits = sum(1 for token in positive if token in text)
        neg_hits = sum(1 for token in negative if token in text)
        raw = pos_hits - neg_hits
        return float(max(-1.0, min(1.0, raw / 3.0)))
