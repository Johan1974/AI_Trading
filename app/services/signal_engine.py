"""
Bestand: app/services/signal_engine.py
Relatief pad: ./app/services/signal_engine.py
Functie: Orkestreert technical + FinBERT sentiment + judge tot één handelssignaal.
"""

from typing import Any

import numpy as np

from app.ai.judge.weighted_judge import WeightedJudge
from app.ai.sentiment.finbert_sentiment import FinBertSentimentAnalyzer
from app.ai.technical.sklearn_technical import SklearnTechnicalAnalyzer


class SignalEngine:
    def __init__(self, sentiment: FinBertSentimentAnalyzer | None = None) -> None:
        self.technical = SklearnTechnicalAnalyzer(window=30)
        self.sentiment = sentiment if sentiment is not None else FinBertSentimentAnalyzer()
        self.judge = WeightedJudge(technical_weight=0.65, sentiment_weight=0.35)

    def evaluate(self, close_prices: np.ndarray, news_articles: list[dict[str, Any]]) -> dict[str, Any]:
        texts = [
            f"{article.get('title', '')}. {article.get('description', '')}".strip()
            for article in news_articles
        ]
        technical_result = self.technical.score(close_prices=close_prices)
        sentiment_payload = self.sentiment.score_with_breakdown(texts=texts)
        sentiment_result = sentiment_payload["aggregate"]
        judge_result = self.judge.decide(technical=technical_result, sentiment=sentiment_result)
        return {
            "technical": technical_result,
            "sentiment": sentiment_result,
            "sentiment_items": sentiment_payload.get("items", []),
            "judge": judge_result,
        }
