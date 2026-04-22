"""Bot control REST endpoints (lazy import van ``app.trading_core``)."""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter

from app.services.state import STATE, append_event

router = APIRouter(tags=["bot"])


@router.post("/bot/pause")
def pause_bot() -> dict[str, str]:
    import app.trading_core as main_mod

    STATE["bot_status"] = "paused"
    append_event({"ts": datetime.utcnow().isoformat(), "type": "bot_status", "status": "paused"})
    main_mod.TELEGRAM.send_bot_status("paused")
    return {"bot_status": "paused"}


@router.post("/bot/resume")
def resume_bot() -> dict[str, str]:
    import app.trading_core as main_mod

    STATE["bot_status"] = "running"
    append_event({"ts": datetime.utcnow().isoformat(), "type": "bot_status", "status": "running"})
    main_mod.TELEGRAM.send_bot_status("running")
    return {"bot_status": "running"}


@router.post("/bot/panic")
def panic_stop() -> dict[str, str]:
    import app.trading_core as main_mod

    STATE["bot_status"] = "panic_stop"
    append_event({"ts": datetime.utcnow().isoformat(), "type": "bot_status", "status": "panic_stop"})
    main_mod.TELEGRAM.send_bot_status("panic_stop")
    return {"bot_status": "panic_stop"}
