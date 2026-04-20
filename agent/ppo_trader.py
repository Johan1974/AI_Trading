"""
Bestand: agent/ppo_trader.py
Relatief pad: ./agent/ppo_trader.py
Functie: Wrapper rondom RLAgentService voor PPO paper trading setup.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.rl.agent_rl import RLAgentService


@dataclass
class PPOTraderConfig:
    initial_capital_eur: float = 10000.0
    lookback_days: int = 400
    model_dir: str = "artifacts/rl"


class PPOTrader:
    def __init__(self, config: PPOTraderConfig, agent_service: RLAgentService | None = None) -> None:
        self.config = config
        self.agent = agent_service or RLAgentService(model_dir=config.model_dir)
        self.mode = "paper"
        self.initial_capital_eur = float(config.initial_capital_eur)
        self.current_pair: str | None = None

    def initialize(self, pair: str) -> None:
        target = str(pair or "BTC-EUR").upper()
        self.agent.ensure_pretrained(pair=target, lookback_days=self.config.lookback_days)
        self.current_pair = target

    def decide(self, latest_row: dict[str, float], account: dict[str, float] | None = None) -> dict[str, Any]:
        decision = self.agent.decide(latest_row=latest_row, account=account)
        return {
            "action": decision.action_name,
            "confidence": float(decision.confidence),
            "expected_reward_pct": float(decision.expected_reward_pct),
            "feature_weights": dict(decision.feature_weights),
            "reasoning": str(decision.reasoning),
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

