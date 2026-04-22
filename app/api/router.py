"""Kleine, state-only routes zonder zware main-implementaties (uitbreidbaar)."""

from __future__ import annotations

from fastapi import APIRouter

from app.services.state import STATE
from app.settings import LIVE_MODE

router = APIRouter(tags=["meta"])


@router.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "mode": "live" if LIVE_MODE else "paper",
        "bot_status": str(STATE.get("bot_status", "running")),
        "cmc_ok": "yes" if bool((STATE.get("cmc_metrics") or {}).get("ok")) else "no",
    }
