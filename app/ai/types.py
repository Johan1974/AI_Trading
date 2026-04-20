"""
Bestand: app/ai/types.py
Relatief pad: ./app/ai/types.py
Functie: Centrale dataclasses voor technische score, sentimentscore en judge-uitkomst.
"""

from dataclasses import dataclass
from typing import Literal


@dataclass
class TechnicalResult:
    score: float
    predicted_return_pct: float
    model_name: str


@dataclass
class SentimentResult:
    score: float
    confidence: float
    model_name: str


@dataclass
class JudgeResult:
    composite_score: float
    signal: Literal["BUY", "SELL", "HOLD"]
    technical_weight: float
    sentiment_weight: float
