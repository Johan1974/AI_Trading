"""
Bestand: app/services/features.py
Relatief pad: ./app/services/features.py
Functie: Berekent basale sentiment- en prijsfeatures uit datafeeds.
"""

import numpy as np


def compute_simple_news_sentiment(articles: list[dict]) -> float:
    if not articles:
        return 0.0

    positive_words = {
        "surge",
        "growth",
        "profit",
        "strong",
        "bull",
        "rally",
        "beat",
        "optimism",
        "recovery",
        "upside",
    }
    negative_words = {
        "crash",
        "loss",
        "weak",
        "bear",
        "fear",
        "recession",
        "inflation",
        "war",
        "downturn",
        "risk",
    }

    score = 0
    for article in articles:
        text = f"{article.get('title', '')} {article.get('description', '')}".lower()
        pos_hits = sum(1 for w in positive_words if w in text)
        neg_hits = sum(1 for w in negative_words if w in text)
        score += pos_hits - neg_hits

    return float(max(-1.0, min(1.0, score / max(1, len(articles) * 2))))


def build_trend_window(close_prices: np.ndarray, window: int = 30) -> tuple[np.ndarray, np.ndarray]:
    recent = close_prices[-window:]
    x = np.arange(len(recent)).reshape(-1, 1)
    y = recent.reshape(-1, 1)
    return x, y
