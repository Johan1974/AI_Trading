"""
Bestand: app/services/signal_engine.py
Relatief pad: ./app/services/signal_engine.py
Functie: Orkestreert technical + FinBERT sentiment + judge tot één handelssignaal.
"""

import os
from typing import Any

from app.ai.judge.weighted_judge import WeightedJudge


class SignalEngine:
    def __init__(self, sentiment: Any = None) -> None:
        if str(os.getenv("AI_TRADING_PROCESS", "") or "").strip().lower() == "portal":
            from app.portal_stubs import PortalTechnicalAnalyzer

            self.technical = PortalTechnicalAnalyzer(window=30)
        else:
            from app.ai.technical.sklearn_technical import SklearnTechnicalAnalyzer

            self.technical = SklearnTechnicalAnalyzer(window=30)
        if sentiment is not None:
            self.sentiment = sentiment
        else:
            from app.ai.sentiment.finbert_sentiment import FinBertSentimentAnalyzer

            self.sentiment = FinBertSentimentAnalyzer()
        self.judge = WeightedJudge(technical_weight=0.65, sentiment_weight=0.35)

    def evaluate(self, close_prices: Any, news_articles: list[dict[str, Any]]) -> dict[str, Any]:
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
