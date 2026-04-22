"""
Runtime-modus (PAPER/LIVE) en validatie — los van FastAPI zodat tests/worker-imports lichter blijven.
"""

from __future__ import annotations

import os


def _trading_mode_raw() -> str:
    return str(os.getenv("TRADING_MODE", "PAPER") or "PAPER").strip().upper()


TRADING_MODE = _trading_mode_raw()
LIVE_MODE = TRADING_MODE == "LIVE"


def validate_mode_configuration() -> None:
    if TRADING_MODE not in {"PAPER", "LIVE"}:
        raise RuntimeError("TRADING_MODE moet PAPER of LIVE zijn.")
    if not LIVE_MODE:
        return
    missing = [
        name
        for name in ("BITVAVO_KEY_TRADE", "BITVAVO_SECRET_TRADE")
        if not os.getenv(name, "").strip()
    ]
    if missing:
        raise RuntimeError(
            f"TRADING_MODE=LIVE maar ontbrekende variabelen in vault: {', '.join(missing)}"
        )
