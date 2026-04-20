"""
Bestand: app/services/execution.py
Relatief pad: ./app/services/execution.py
Functie: Simuleert orderuitvoering voor paper-trading met eenvoudige kosteninschatting.
"""

from typing import Any


def build_paper_order(signal: str, ticker: str, price: float, size_fraction: float) -> dict[str, Any]:
    if signal == "HOLD":
        return {"status": "skipped", "reason": "no_trade_signal"}

    side = "BUY" if signal == "BUY" else "SELL"
    estimated_fee_bps = 5.0
    return {
        "status": "simulated",
        "side": side,
        "ticker": ticker,
        "reference_price": round(price, 2),
        "size_fraction": size_fraction,
        "estimated_fee_bps": estimated_fee_bps,
    }
