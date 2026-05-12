"""
BESTANDSNAAM: app/services/risk.py
FUNCTIE: Past eenvoudige risk-regels toe op modeluitvoer (signaal, SL/TP en sizing).
"""

import os
from dataclasses import dataclass
from typing import Literal

from core.risk_manager import risk_controls_for_close


def signal_from_expected_return(expected_return_pct: float) -> Literal["BUY", "SELL", "HOLD"]:
    thr = float(os.getenv("EXPECTED_RETURN_SIGNAL_THRESHOLD_PCT", "1.0") or 1.0)
    if not (thr == thr) or thr <= 0:
        thr = 1.0
    if expected_return_pct > thr:
        return "BUY"
    if expected_return_pct < -thr:
        return "SELL"
    return "HOLD"


def compute_risk_controls(latest_close: float) -> dict[str, float]:
    return risk_controls_for_close(latest_close)


@dataclass
class RiskDecision:
    approved: bool
    adjusted_signal: Literal["BUY", "SELL", "HOLD"]
    adjusted_size_fraction: float
    reason: str


class RiskManager:
    def __init__(
        self,
        max_budget_fraction_per_trade: float | None = None,
        max_spread_bps_for_trading: float | None = None,
        emergency_negative_sentiment_threshold: float = -0.8,
    ) -> None:
        mb = max_budget_fraction_per_trade
        if mb is None:
            try:
                mb = float(os.getenv("RISK_MAX_BUDGET_FRACTION_PER_TRADE", "0.03") or 0.03)
            except (TypeError, ValueError):
                mb = 0.03
        sp = max_spread_bps_for_trading
        if sp is None:
            try:
                sp = float(os.getenv("RISK_MAX_SPREAD_BPS_FOR_TRADING", "45") or 45)
            except (TypeError, ValueError):
                sp = 45.0
        self.max_budget_fraction_per_trade = mb
        self.max_spread_bps_for_trading = sp
        self.emergency_negative_sentiment_threshold = emergency_negative_sentiment_threshold

    def evaluate(
        self,
        proposed_signal: Literal["BUY", "SELL", "HOLD"],
        proposed_size_fraction: float,
        spread_bps: float,
        sentiment_score: float,
    ) -> RiskDecision:
        size = min(max(0.0, proposed_size_fraction), self.max_budget_fraction_per_trade)

        if sentiment_score <= self.emergency_negative_sentiment_threshold:
            return RiskDecision(
                approved=False,
                adjusted_signal="SELL",
                adjusted_size_fraction=size,
                reason="emergency_exit_negative_sentiment_shock",
            )

        if spread_bps > self.max_spread_bps_for_trading:
            return RiskDecision(
                approved=False,
                adjusted_signal="HOLD",
                adjusted_size_fraction=size,
                reason="volatility_filter_spread_too_high",
            )

        if proposed_signal == "HOLD":
            return RiskDecision(
                approved=False,
                adjusted_signal="HOLD",
                adjusted_size_fraction=size,
                reason="no_trade_signal",
            )

        return RiskDecision(
            approved=True,
            adjusted_signal=proposed_signal,
            adjusted_size_fraction=size,
            reason="approved",
        )
