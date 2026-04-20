"""
Bestand: app/services/execution.py
Relatief pad: ./app/services/execution.py
Functie: Simuleert orderuitvoering voor paper-trading met eenvoudige kosteninschatting.
"""

from typing import Any

from app.services.dry_run import DRY_RUN_FEE_RATE, dry_run_trade_logger


@dry_run_trade_logger(source="paper_execution")
def build_paper_order(
    signal: str,
    ticker: str,
    price: float,
    size_fraction: float,
    budget_eur: float = 10000.0,
) -> dict[str, Any]:
    if signal == "HOLD":
        return {"status": "skipped", "reason": "no_trade_signal", "signal": signal}

    side = "BUY" if signal == "BUY" else "SELL"
    amount_quote_eur = max(0.0, budget_eur * size_fraction)
    estimated_fee_bps = DRY_RUN_FEE_RATE * 10000.0
    return {
        "status": "simulated",
        "side": side,
        "signal": signal,
        "ticker": ticker,
        "market": ticker,
        "reference_price": round(price, 2),
        "size_fraction": size_fraction,
        "amount_quote_eur": round(amount_quote_eur, 2),
        "fee_rate": DRY_RUN_FEE_RATE,
        "estimated_fee_bps": estimated_fee_bps,
    }
