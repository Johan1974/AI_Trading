"""
Bestand: agent/trader.py
Relatief pad: ./agent/trader.py
Functie: Genesis trader interface voor PPO paper trading.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent.ppo_trader import PPOTrader, PPOTraderConfig
from app.rl.agent_rl import RLAgentService


@dataclass
class TraderConfig(PPOTraderConfig):
    pass


class Trader(PPOTrader):
    def __init__(self, config: TraderConfig, agent_service: RLAgentService | None = None) -> None:
        super().__init__(config=config, agent_service=agent_service)

    def describe_state(self) -> list[str]:
        return [
            "bitvavo_price",
            "rsi_14",
            "macd",
            "sentiment_score",
            "btc_dominance_pct",
            "volatility_24",
        ]

    def decide(self, latest_row: dict[str, float], account: dict[str, float] | None = None) -> dict[str, Any]:
        payload = super().decide(latest_row=latest_row, account=account)
        payload["state_features"] = self.describe_state()
        return payload

