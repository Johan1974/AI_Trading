"""
Bestand: app/services/paper.py
Relatief pad: ./app/services/paper.py
Functie: Beheert eenvoudige paper-trading portfolio-updates en PnL/equity tijdreeks.
"""

from datetime import datetime
from typing import Any


def initialize_portfolio() -> dict[str, Any]:
    return {
        "cash": 10000.0,
        "position_qty": 0.0,
        "position_symbol": None,
        "last_price": None,
        "equity": 10000.0,
        "realized_pnl": 0.0,
        "history": [],
    }


def apply_paper_order(
    portfolio: dict[str, Any],
    ticker: str,
    price: float,
    signal: str,
    size_fraction: float,
    fee_bps: float = 5.0,
) -> dict[str, Any]:
    if signal not in {"BUY", "SELL"}:
        snapshot = _snapshot(portfolio, ticker, price, "HOLD")
        portfolio["history"].append(snapshot)
        portfolio["history"] = portfolio["history"][-300:]
        return snapshot

    notional = max(0.0, portfolio["equity"] * size_fraction)
    fee = notional * (fee_bps / 10000.0)
    qty = (notional / price) if price > 0 else 0.0

    if signal == "BUY" and portfolio["cash"] >= (notional + fee):
        portfolio["cash"] -= (notional + fee)
        portfolio["position_qty"] += qty
        portfolio["position_symbol"] = ticker
    elif signal == "SELL" and portfolio["position_qty"] > 0 and portfolio["position_symbol"] == ticker:
        sell_qty = min(portfolio["position_qty"], qty if qty > 0 else portfolio["position_qty"])
        proceeds = sell_qty * price
        portfolio["cash"] += max(0.0, proceeds - fee)
        portfolio["position_qty"] -= sell_qty
        if portfolio["position_qty"] <= 1e-9:
            portfolio["position_qty"] = 0.0
            portfolio["position_symbol"] = None

    portfolio["last_price"] = price
    position_value = portfolio["position_qty"] * price
    portfolio["equity"] = portfolio["cash"] + position_value
    portfolio["realized_pnl"] = portfolio["equity"] - 10000.0

    snapshot = _snapshot(portfolio, ticker, price, signal)
    portfolio["history"].append(snapshot)
    portfolio["history"] = portfolio["history"][-300:]
    return snapshot


def _snapshot(portfolio: dict[str, Any], ticker: str, price: float, action: str) -> dict[str, Any]:
    return {
        "ts": datetime.utcnow().isoformat(),
        "ticker": ticker,
        "action": action,
        "price": round(price, 2),
        "cash": round(portfolio["cash"], 2),
        "position_qty": round(portfolio["position_qty"], 8),
        "equity": round(portfolio["equity"], 2),
        "realized_pnl": round(portfolio["realized_pnl"], 2),
    }
