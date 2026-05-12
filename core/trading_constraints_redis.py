"""
Redis-persisted trading constraints (portal POST → worker reads).

Key: ``trading:constraints`` (JSON). Used for position sizing and optional
equal-weight slot percentage (replaces env-only 12,5% default when set).
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

TRADING_CONSTRAINTS_KEY = "trading:constraints"


def _redis_url() -> str:
    host = str(os.getenv("REDIS_HOST", "redis")).strip()
    port = str(os.getenv("REDIS_PORT", "6379")).strip()
    url = str(os.getenv("REDIS_URL", f"redis://{host}:{port}/0")).strip()
    if not url:
        url = f"redis://{host}:{port}/0"
    return url


def read_trading_constraints() -> dict[str, Any]:
    """Return parsed JSON dict from Redis, or {} if missing/unreadable."""
    try:
        import redis

        r = redis.Redis.from_url(
            _redis_url(),
            decode_responses=True,
            socket_connect_timeout=1.5,
            socket_timeout=1.5,
        )
        try:
            raw = r.get(TRADING_CONSTRAINTS_KEY)
        finally:
            r.close()
    except Exception:
        return {}
    if not raw:
        return {}
    try:
        out = json.loads(raw)
        return out if isinstance(out, dict) else {}
    except Exception:
        return {}


def write_trading_constraints(data: dict[str, Any]) -> None:
    """Persist full constraints document (replace key)."""
    import redis

    r = redis.Redis.from_url(
        _redis_url(),
        decode_responses=True,
        socket_connect_timeout=2.0,
        socket_timeout=2.0,
    )
    try:
        r.set(TRADING_CONSTRAINTS_KEY, json.dumps(data, default=str))
    finally:
        r.close()


def elite_slot_pct_from_constraints() -> float | None:
    """If set, overrides ELITE_EQUAL_WEIGHT_SLOT_PCT for allocation caps."""
    d = read_trading_constraints()
    if not d or d.get("elite_slot_equity_pct") is None:
        return None
    try:
        return float(d["elite_slot_equity_pct"])
    except (TypeError, ValueError):
        return None


def merge_position_sizing_post(
    *,
    tab: str,
    value: float,
    equity: float | None,
) -> dict[str, Any]:
    """
    Bouwt het Redis-document voor position sizing.

    tab: ``fixed`` | ``pct`` (of ``fixed_eur`` / ``percentage``).
    """
    prev = read_trading_constraints()
    tab_l = str(tab or "").strip().lower()
    max_trade_default = float(os.getenv("RISK_MAX_TRADE_EQUITY_PCT", "10") or 10)
    base_default = float(os.getenv("RISK_BASE_TRADE_EUR", "100") or 100)

    elite_slot: float | None = None
    if tab_l in ("fixed", "fixed_eur", "eur", "euro"):
        sizing_mode = "fixed_eur"
        base_trade_eur = max(1.0, min(1_000_000.0, float(value)))
        try:
            max_trade_equity_pct = float(prev.get("max_trade_equity_pct", max_trade_default))
        except (TypeError, ValueError):
            max_trade_equity_pct = max_trade_default
        max_trade_equity_pct = max(0.05, min(100.0, max_trade_equity_pct))
        eqv = float(equity) if equity is not None and float(equity) > 0 else 0.0
        if eqv > 0:
            elite_slot = min(100.0, max(0.5, (base_trade_eur / eqv) * 100.0))
        else:
            raw_es = prev.get("elite_slot_equity_pct")
            try:
                elite_slot = float(raw_es) if raw_es is not None else None
            except (TypeError, ValueError):
                elite_slot = None
    elif tab_l in ("pct", "percentage", "equity_pct", "%"):
        sizing_mode = "equity_pct"
        max_trade_equity_pct = max(0.1, min(100.0, float(value)))
        try:
            base_trade_eur = float(prev.get("base_trade_eur", base_default))
        except (TypeError, ValueError):
            base_trade_eur = base_default
        base_trade_eur = max(1.0, base_trade_eur)
        elite_slot = max_trade_equity_pct
    else:
        raise ValueError("position_sizing_tab must be 'fixed' or 'pct'")

    out: dict[str, Any] = {
        "sizing_mode": sizing_mode,
        "base_trade_eur": base_trade_eur,
        "max_trade_equity_pct": max_trade_equity_pct,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    if elite_slot is not None:
        out["elite_slot_equity_pct"] = elite_slot
    elif "elite_slot_equity_pct" in prev:
        out["elite_slot_equity_pct"] = prev.get("elite_slot_equity_pct")
    return out


def apply_paper_reset_allocation_constraints(*, equity_eur: float) -> dict[str, Any]:
    """
    Na paper portfolio-reset: max **10% van equity** per trade-order via Redis
    (``sizing_mode`` = ``equity_pct``, ``max_trade_equity_pct`` = 10).
    Bij € 1.000 equity is dat **€ 100** per inleg (mits Redis-write lukt).
    """
    eq = max(1.0, float(equity_eur))
    merged = merge_position_sizing_post(tab="pct", value=10.0, equity=eq)
    write_trading_constraints(merged)
    return merged
