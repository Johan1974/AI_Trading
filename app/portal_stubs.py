"""
Bestand: app/portal_stubs.py
Relatief pad: ./app/portal_stubs.py
Functie: Lichte placeholders voor RL/trader/sentiment wanneer AI_TRADING_PROCESS=portal (slim image zonder torch).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from app.ai.base import SentimentAnalyzer, TechnicalAnalyzer
from app.ai.types import SentimentResult, TechnicalResult


class PortalTechnicalAnalyzer(TechnicalAnalyzer):
    """Sklearn-technical vervanger zonder numpy/scikit-learn op portal-image."""

    def __init__(self, window: int = 30) -> None:
        self.window = int(window)
        self.model_name = "portal-technical-neutral"

    def score(self, close_prices: Any) -> TechnicalResult:
        return TechnicalResult(score=0.0, predicted_return_pct=0.0, model_name=str(self.model_name))


class PortalNeutralSentiment(SentimentAnalyzer):
    """FinBERT-vervanger: neutraal sentiment zonder transformers/torch."""

    def __init__(self, model_id: str = "ProsusAI/finbert") -> None:
        self.model_id = model_id
        self.model_name = f"portal-neutral:{model_id}"

    def score(self, texts: Sequence[str]) -> SentimentResult:
        out = self.score_with_breakdown(texts)
        agg = out.get("aggregate")
        if isinstance(agg, SentimentResult):
            return agg
        return SentimentResult(score=0.0, confidence=0.0, model_name=str(self.model_name))

    def score_with_breakdown(self, texts: Sequence[str]) -> dict[str, Any]:
        clean = [t for t in texts if t and str(t).strip()]
        items: list[dict[str, Any]] = [
            {"index": i, "label": "neutral", "confidence": 0.0, "signed_score": 0.0} for i in range(len(clean))
        ]
        agg = SentimentResult(score=0.0, confidence=0.0, model_name=str(self.model_name))
        return {"aggregate": agg, "items": items}


@dataclass
class PortalTraderConfig:
    initial_capital_eur: float = 10000.0
    lookback_days: int = 400
    model_dir: str = "artifacts/rl"


class PortalPaperTrader:
    """Minimale Trader-interface zonder PPO/torch (portal image)."""

    def __init__(self, config: PortalTraderConfig, agent_service: Any = None) -> None:
        self.config = config
        self.agent = agent_service
        self.mode = "paper"
        self.initial_capital_eur = float(config.initial_capital_eur)
        self.current_pair: str | None = None

    def initialize(self, pair: str = "") -> None:
        self.current_pair = str(pair or "BTC-EUR").upper()

    def decide(
        self,
        latest_row: dict[str, float],
        account: dict[str, float] | None = None,
        trade_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "action": "HOLD",
            "confidence": 0.0,
            "expected_reward_pct": 0.0,
            "feature_weights": {},
            "reasoning": "portal_stub",
            "state_features": [
                "price_action",
                "rsi_14",
                "macd",
                "sentiment_score",
                "btc_dominance_pct",
                "whale_pressure",
            ],
            "mode": self.mode,
            "initial_capital_eur": self.initial_capital_eur,
        }


class PortalRLAgentStub:
    """Minimale RLAgentService-surface voor imports en read-only endpoints."""

    feature_names: tuple[str, ...] = (
        "price_action",
        "rsi_14",
        "macd",
        "sentiment_score",
        "btc_dominance_pct",
        "whale_pressure",
    )

    def __init__(self, model_dir: str = "artifacts/rl") -> None:
        self.model_dir = model_dir
        self.last_decision: Any = None
        self.last_training_progress: dict[str, Any] = {}
        self.last_network_logs: dict[str, Any] = {"approx_kl": []}
        self.last_training_stats: dict[str, Any] = {"global_step_count": 0}

    def training_monitor(self) -> dict[str, Any]:
        return {"stats": dict(self.last_training_stats)}

    def online_update(self, **kwargs: Any) -> None:
        return

    def save_hourly_checkpoint(self, pair: str) -> None:
        return

    def ingest_paper_stop_loss(self) -> None:
        return

    def ensure_pretrained(self, **kwargs: Any) -> None:
        return

    def decide(self, **kwargs: Any) -> Any:
        return SimpleNamespace(
            action_name="HOLD",
            confidence=0.0,
            expected_reward_pct=0.0,
            feature_weights={},
            reasoning="portal_stub",
        )


def portal_get_rl_ppo_device() -> str:
    return "cpu"
