"""
Bestand: app/ai/judge/weighted_judge.py
Relatief pad: ./app/ai/judge/weighted_judge.py
Functie: Weegt technical- en sentiment-scores en bepaalt eindsignaal.
"""

from app.ai.base import Judge
from app.ai.types import JudgeResult, SentimentResult, TechnicalResult


class WeightedJudge(Judge):
    def __init__(self, technical_weight: float = 0.65, sentiment_weight: float = 0.35) -> None:
        total = technical_weight + sentiment_weight
        if total <= 0:
            raise ValueError("Gewichten moeten samen groter dan 0 zijn.")
        self.technical_weight = technical_weight / total
        self.sentiment_weight = sentiment_weight / total

    def decide(self, technical: TechnicalResult, sentiment: SentimentResult) -> JudgeResult:
        composite = (technical.score * self.technical_weight) + (
            sentiment.score * self.sentiment_weight
        )
        signal = "HOLD"
        if composite > 0.2:
            signal = "BUY"
        elif composite < -0.2:
            signal = "SELL"
        return JudgeResult(
            composite_score=round(composite, 4),
            signal=signal,
            technical_weight=round(self.technical_weight, 4),
            sentiment_weight=round(self.sentiment_weight, 4),
        )


def adjust_weights(
    coin: str,
    avg_sentiment_top_losses: float,
    avg_sentiment_top_wins: float,
    current_technical_weight: float = 0.65,
    current_sentiment_weight: float = 0.35,
    step: float = 0.05,
) -> dict[str, float | str]:
    """
    Geeft een voorstel voor gewichtsaanpassing op basis van sentiment-outcome correlatie.
    Als sentiment historisch vaker verlies drijft, verlaag sentiment-gewicht.
    """
    tech = float(current_technical_weight)
    sent = float(current_sentiment_weight)
    delta = float(max(0.01, min(0.20, step)))
    reason = "keep_weights"

    if avg_sentiment_top_losses > avg_sentiment_top_wins:
        sent = max(0.10, sent - delta)
        tech = min(0.90, tech + delta)
        reason = "sentiment_underperforming_reduce_sentiment_weight"
    elif avg_sentiment_top_wins > avg_sentiment_top_losses:
        sent = min(0.90, sent + delta)
        tech = max(0.10, tech - delta)
        reason = "sentiment_helping_increase_sentiment_weight"

    total = tech + sent
    tech /= total
    sent /= total
    return {
        "coin": coin.upper(),
        "suggested_technical_weight": round(tech, 4),
        "suggested_sentiment_weight": round(sent, 4),
        "reason": reason,
    }
