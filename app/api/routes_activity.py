"""Activity snapshot REST (lazy import van ``app.trading_core``)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from core.risk_manager import risk_profile_dict
from app.services.state import STATE

from app.settings import LIVE_MODE

router = APIRouter(tags=["activity"])


@router.get("/activity")
def activity() -> dict[str, Any]:
    import app.trading_core as m

    lec = STATE.get("last_engine_cycle")
    last_engine_tick_utc = lec.get("ts") if isinstance(lec, dict) else None
    return {
        "mode": "live" if LIVE_MODE else "paper",
        "bot_status": STATE.get("bot_status", "running"),
        "selected_market": STATE.get("selected_market", "BTC-EUR"),
        "active_markets_count": len(STATE.get("active_markets", [])),
        "started_at": STATE["started_at"],
        "last_prediction": STATE["last_prediction"],
        "last_scores": STATE.get("last_scores"),
        "last_order": STATE["last_order"],
        "paper_portfolio": STATE["paper_portfolio"],
        "events": STATE["events"],
        "fear_greed": STATE.get("fear_greed") or {},
        "risk_profile": risk_profile_dict(),
        "last_engine_tick_utc": last_engine_tick_utc,
        "whale_panic_cooldowns": m._whale_panic_cooldowns_payload(),
        "elite_ai_signals": m._elite_ai_signals_payload(),
        "scanner_intel_feed": (
            list(STATE.get("scanner_intel_feed") or [])[-25:]
            if isinstance(STATE.get("scanner_intel_feed"), list)
            else []
        ),
        "allocation_snapshot": m._allocation_snapshot_for_activity(),
    }
