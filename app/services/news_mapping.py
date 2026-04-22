"""
Bestand: app/services/news_mapping.py
Relatief pad: ./app/services/news_mapping.py
Functie: Koppelt nieuws aan actieve coins met sentimentfiltering voor terminal ticker tape.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from app.datetime_util import UTC
import time
from typing import Any

from app.services.ingestion import fetch_news_articles


@dataclass
class _CacheState:
    updated_at: float = 0.0
    payload: list[dict[str, Any]] | None = None


class NewsMappingService:
    def __init__(self, sentiment: Any = None) -> None:
        if sentiment is not None:
            self.sentiment = sentiment
        else:
            from app.ai.sentiment.finbert_sentiment import FinBertSentimentAnalyzer

            self.sentiment = FinBertSentimentAnalyzer()
        self.cache = _CacheState()
        self.cache_ttl_seconds = 90.0
        self.alias_to_coin: dict[str, str] = {
            "bitcoin": "BTC",
            "btc": "BTC",
            "halving": "BTC",
            "etf": "BTC",
            "ethereum": "ETH",
            "eth": "ETH",
            "ether": "ETH",
            "solana": "SOL",
            "sol": "SOL",
            "solana labs": "SOL",
            "ripple": "XRP",
            "xrp": "XRP",
            "cardano": "ADA",
            "ada": "ADA",
            "dogecoin": "DOGE",
            "doge": "DOGE",
            "chainlink": "LINK",
            "link": "LINK",
            "litecoin": "LTC",
            "ltc": "LTC",
            "polkadot": "DOT",
            "dot": "DOT",
            "avalanche": "AVAX",
            "avax": "AVAX",
            "tron": "TRX",
            "trx": "TRX",
            "uniswap": "UNI",
            "uni": "UNI",
            "binance coin": "BNB",
            "bnb": "BNB",
            "polygon": "MATIC",
            "matic": "MATIC",
            "sui": "SUI",
            "near": "NEAR",
            "aptos": "APT",
            "apt": "APT",
            "arbitrum": "ARB",
            "arb": "ARB",
            "optimism": "OP",
            "op": "OP",
            "pepe": "PEPE",
            "open campus": "EDU",
            "edu": "EDU",
            "layerzero": "ZRO",
            "zro": "ZRO",
            "merlin": "MERL",
            "merl": "MERL",
            "gwei": "GWEI",
            "portal": "PORTAL",
            "aave": "AAVE",
            "prom": "PROM",
            "highstreet": "HIGH",
            "high": "HIGH",
        }
        self.generic_crypto_terms = {
            "CRYPTO",
            "BITCOIN",
            "ETHEREUM",
            "BLOCKCHAIN",
            "DEFI",
            "TOKEN",
            "ALTCOIN",
            "STABLECOIN",
            "ETF",
            "BINANCE",
            "COINBASE",
            "WEB3",
        }

    def _active_top_coins(self, active_markets: list[dict[str, Any]], limit: int = 20) -> list[str]:
        coins: list[str] = []
        for row in active_markets[:limit]:
            coin = str(row.get("base", "")).upper()
            if coin and coin not in coins:
                coins.append(coin)
        return coins

    def _resolve_coin(self, text_upper: str, top_coins: list[str]) -> tuple[str, list[str]]:
        matched_terms: list[str] = []
        text_tokens = {
            token.strip(".,:;!?()[]{}<>\"'`")
            for token in text_upper.split()
            if token.strip()
        }
        for alias, coin in self.alias_to_coin.items():
            alias_upper = alias.upper()
            if (" " in alias and alias_upper in text_upper) or (alias_upper in text_tokens):
                matched_terms.append(alias.upper())
                return coin, matched_terms
        for coin in top_coins:
            if coin in text_tokens:
                matched_terms.append(coin)
                return coin, matched_terms
        # Alleen MKT als er echt geen coin-match beschikbaar is.
        matched_terms.append("NO_DIRECT_MATCH")
        return "MKT", matched_terms

    def _is_recent(self, published_at: str | None, max_age_hours: int = 36) -> bool:
        if not published_at:
            return False
        try:
            dt = datetime.fromisoformat(str(published_at).replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            return dt.astimezone(UTC) >= (datetime.now(UTC) - timedelta(hours=max_age_hours))
        except Exception:
            return False

    def _is_crypto_relevant(self, text_upper: str, top_coins: list[str]) -> bool:
        tokens = {
            token.strip(".,:;!?()[]{}<>\"'`")
            for token in text_upper.split()
            if token.strip()
        }
        if any(str(term).upper() in text_upper for term in self.alias_to_coin.keys()):
            return True
        if any(coin in tokens for coin in top_coins):
            return True
        if any(term in tokens for term in self.generic_crypto_terms):
            return True
        return False

    def get_mapped_news(
        self,
        news_query: str,
        news_api_key: str | None,
        active_markets: list[dict[str, Any]],
        prefetched_articles: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        now = time.time()
        # Alleen TTL-cache als we zelf artikelen ophalen. Bij caller-supplied `prefetched_articles`
        # (ticker / signal-pipeline) moet altijd opnieuw gemapt worden — anders blijft oude of lege
        # cache hangen terwijl verse CryptoCompare-rows genegeerd worden.
        if prefetched_articles is None:
            if self.cache.payload is not None and (now - self.cache.updated_at) < self.cache_ttl_seconds:
                return self.cache.payload
        prev_payload = self.cache.payload or []

        articles = prefetched_articles if isinstance(prefetched_articles, list) else fetch_news_articles(news_query, news_api_key)
        texts = [f"{a.get('title', '')}. {a.get('description', '')}".strip() for a in articles[:40]]
        scored = self.sentiment.score_with_breakdown(texts)
        scored_rows = scored.get("items", [])

        top_coins = self._active_top_coins(active_markets=active_markets, limit=20)
        mapped: list[dict[str, Any]] = []
        for idx, article in enumerate(articles[:40]):
            title = str(article.get("title") or "").strip()
            description = str(article.get("description") or "").strip()
            if not title:
                continue
            row = scored_rows[idx] if idx < len(scored_rows) else {}
            finbert_sentiment = float(row.get("signed_score", 0.0) or 0.0)
            sentiment = finbert_sentiment
            combined = f"{title} {description}".upper()
            coin, matched_terms = self._resolve_coin(text_upper=combined, top_coins=top_coins)
            if not self._is_recent(article.get("publishedAt"), max_age_hours=48):
                continue
            if not self._is_crypto_relevant(combined, top_coins=top_coins):
                continue

            tokenized = {
                token.strip(".,:;!?()[]{}<>\"'`")
                for token in combined.split()
                if token.strip()
            }
            mentions_top_coin = any(c in tokenized for c in top_coins)
            strong_sentiment = sentiment > 0.4 or sentiment < -0.4
            extreme_mkt_sentiment = sentiment > 0.8 or sentiment < -0.8
            if not strong_sentiment and not mentions_top_coin and coin == "MKT":
                continue
            if coin == "MKT" and not extreme_mkt_sentiment:
                continue

            mapped.append(
                {
                    "text": title,
                    "title": title,
                    "summary": description,
                    "url": article.get("url"),
                    "coin": coin,
                    "sentiment": round(sentiment, 4),
                    "finbert_sentiment": round(finbert_sentiment, 4),
                    "confidence": float(row.get("confidence", 0.0) or 0.0),
                    "social_volume": 0.0,
                    "source": (article.get("source") or {}).get("name"),
                    "source_icon": "NEWS",
                    "published_at": article.get("publishedAt"),
                    "news_lag_sec": int(article.get("news_lag_sec") or 0),
                    "keywords": matched_terms,
                    "affected_tickers": [coin],
                    "is_urgent": bool(article.get("urgent", False)),
                    "explanation": (
                        f"Mapped to {coin} via terms: {', '.join(matched_terms)}"
                        if matched_terms
                        else "Mapped by market mention and sentiment filter"
                    ),
                    "ai_reasoning": (
                        f"FinBERT label={row.get('label', 'neutral')} confidence="
                        f"{float(row.get('confidence', 0.0) or 0.0):.3f}; "
                        f"sentiment={sentiment:.3f}; "
                        f"decisive_keywords={', '.join(matched_terms) or 'none'}"
                    ),
                }
            )

        mapped.sort(
            key=lambda x: (
                1 if bool(x.get("is_urgent")) else 0,
                str(x.get("published_at") or ""),
                abs(float(x.get("sentiment", 0.0))),
            ),
            reverse=True,
        )
        output = mapped[:30]
        if not output and prev_payload:
            # Keep a non-empty feed when upstream news source is temporarily stale/empty.
            output = prev_payload[:30]
        self.cache = _CacheState(updated_at=now, payload=output)
        return output
