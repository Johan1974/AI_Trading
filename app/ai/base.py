"""
Bestand: app/ai/base.py
Relatief pad: ./app/ai/base.py
Functie: Abstracte base classes voor technical model, sentiment model en judge module.
"""

from abc import ABC, abstractmethod
from typing import Any, Sequence

from app.ai.types import JudgeResult, SentimentResult, TechnicalResult


class TechnicalAnalyzer(ABC):
    @abstractmethod
    def score(self, close_prices: Any) -> TechnicalResult:
        raise NotImplementedError


class SentimentAnalyzer(ABC):
    @abstractmethod
    def score(self, texts: Sequence[str]) -> SentimentResult:
        raise NotImplementedError


class Judge(ABC):
    @abstractmethod
    def decide(self, technical: TechnicalResult, sentiment: SentimentResult) -> JudgeResult:
        raise NotImplementedError
