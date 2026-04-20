"""
Bestand: app/services/risk.py
Relatief pad: ./app/services/risk.py
Functie: Past eenvoudige risk-regels toe op modeluitvoer (signaal, SL/TP en sizing).
"""

from typing import Literal


def signal_from_expected_return(expected_return_pct: float) -> Literal["BUY", "SELL", "HOLD"]:
    if expected_return_pct > 1.0:
        return "BUY"
    if expected_return_pct < -1.0:
        return "SELL"
    return "HOLD"


def compute_risk_controls(latest_close: float) -> dict[str, float]:
    stop_loss_pct = 1.25
    take_profit_pct = 2.5
    position_size_fraction = 0.02
    return {
        "stop_loss_price": round(latest_close * (1.0 - stop_loss_pct / 100.0), 2),
        "take_profit_price": round(latest_close * (1.0 + take_profit_pct / 100.0), 2),
        "position_size_fraction": position_size_fraction,
    }
