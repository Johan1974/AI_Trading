"""Markets REST endpoints (lazy import van ``app.trading_core``)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from app.services.market_scanner import MarketScanner
from app.services.state import STATE, append_event

router = APIRouter(tags=["markets"])


@router.get("/markets/active")
def markets_active(min_volume_eur: float | None = Query(default=None)) -> dict[str, Any]:
    import app.trading_core as main_mod

    if min_volume_eur is not None:
        scanner = MarketScanner(min_volume_eur=min_volume_eur)
        markets = scanner.fetch_active_pairs()
        return {"markets": markets, "min_volume_eur": min_volume_eur}
    if not STATE.get("active_markets"):
        try:
            main_mod._refresh_active_markets_cache()
        except Exception as exc:
            print(f"[MARKETS] /markets/active refresh mislukt: {exc}")
    return {
        "markets": STATE.get("active_markets", []),
        "min_volume_eur": main_mod.MARKET_SCANNER.min_volume_eur,
    }


@router.post("/markets/select")
def market_select(market: str = Query(...)) -> dict[str, Any]:
    import app.trading_core as main_mod

    if not STATE.get("active_markets"):
        try:
            main_mod._refresh_active_markets_cache()
        except Exception as exc:
            print(f"[MARKETS] /markets/select refresh mislukt: {exc}")
    target = market.upper()
    active = STATE.get("active_markets", [])
    if not any(x.get("market") == target for x in active):
        raise HTTPException(status_code=400, detail=f"Market {target} not in active filtered list.")
    STATE["selected_market"] = target
    append_event({"ts": datetime.utcnow().isoformat(), "type": "market_select", "market": target})
    return {"selected_market": target}


@router.get("/markets/selected")
def market_selected() -> dict[str, str]:
    return {"selected_market": STATE.get("selected_market", "BTC-EUR")}
