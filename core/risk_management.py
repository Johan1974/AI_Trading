"""
Whale-driven panic protection: extreme exchange inflow bursts → forced exit + buy cooldown.

News-based inflow events (same pipeline as `core.social_engine` whale radar) are ingested here;
panic triggers only on held Elite positions when configured thresholds are met.

Equal-weight Elite portfolio: maximaal één slot per munt (default 12,5% equity per slot, 8 slots).
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta
from typing import Any

from app.datetime_util import UTC

from core.risk_manager import position_value_eur

SKIP_MAX_PORTFOLIO_ALLOCATION_LOG = "SKIP: Maximum portfolio allocation reached."

# Panic: >3 large inflows in 10 minutes = at least 4 events (strict interpretation of "meer dan 3").
_WHALE_PANIC_WINDOW_SEC = float(os.getenv("WHALE_PANIC_WINDOW_SEC", "600") or 600.0)
_WHALE_PANIC_MIN_INFLOWS = int(os.getenv("WHALE_PANIC_MIN_INFLOWS", "4") or 4)
_WHALE_PANIC_MIN_USD = float(os.getenv("WHALE_PANIC_INFLOW_MIN_USD", "5000000") or 5_000_000.0)
_WHALE_PANIC_COOLDOWN_SEC = float(os.getenv("WHALE_PANIC_COOLDOWN_SEC", "3600") or 3600.0)
_WHALE_PANIC_REARM_SEC = float(os.getenv("WHALE_PANIC_REARM_SEC", "120") or 120.0)
_LOG_RETENTION_SEC = float(os.getenv("WHALE_PANIC_LOG_RETENTION_SEC", "900") or 900.0)

_ELITE_EQ_DEFAULT_SLOT_PCT = 12.5
_ELITE_EQ_DEFAULT_SLOTS = 8


def elite_equal_weight_enabled() -> bool:
    return str(os.getenv("ELITE_EQUAL_WEIGHT_ENABLED", "1")).strip().lower() in {"1", "true", "yes", "on"}


def elite_equal_weight_slot_pct() -> float:
    """Maximaal equity-percentage per Elite-munt (default 12,5% = 1/8)."""
    v = float(os.getenv("ELITE_EQUAL_WEIGHT_SLOT_PCT", str(_ELITE_EQ_DEFAULT_SLOT_PCT)) or _ELITE_EQ_DEFAULT_SLOT_PCT)
    return max(0.5, min(100.0, v))


def elite_equal_weight_slot_count() -> int:
    n = int(os.getenv("ELITE_EQUAL_WEIGHT_SLOT_COUNT", str(_ELITE_EQ_DEFAULT_SLOTS)) or _ELITE_EQ_DEFAULT_SLOTS)
    return max(1, min(32, n))


def total_crypto_notional_eur(wallet: dict[str, Any], mark_overrides: dict[str, float] | None = None) -> float:
    """Som van markt-waarde open crypto-posities (EUR), gebruikt last_prices_by_market + overrides."""
    mark_overrides = mark_overrides or {}
    pbm = wallet.get("position_by_market") if isinstance(wallet.get("position_by_market"), dict) else {}
    lp = wallet.get("last_prices_by_market") if isinstance(wallet.get("last_prices_by_market"), dict) else {}
    total = 0.0
    for mk, qv in pbm.items():
        q = float(qv or 0.0)
        if q <= 1e-12:
            continue
        mku = str(mk).strip().upper()
        px = float(mark_overrides.get(mku) or lp.get(mku, 0.0) or 0.0)
        if px <= 0 and mku == str(wallet.get("position_symbol") or "").strip().upper():
            px = float(wallet.get("last_price") or 0.0)
        total += q * px
    return max(0.0, total)


def allocation_snapshot(wallet: dict[str, Any], equity: float | None = None) -> dict[str, Any]:
    """
    UI / Executive snapshot: slots bezet + gewicht per actieve munt.
    `weight_pct` is positiewaarde t.o.v. equity (niet t.o.v. alleen crypto).
    """
    eq = float(equity if equity is not None else wallet.get("equity", 0.0) or 0.0)
    eq = max(1e-9, eq)
    slot_pct = elite_equal_weight_slot_pct()
    max_slots = elite_equal_weight_slot_count()
    pbm = wallet.get("position_by_market") if isinstance(wallet.get("position_by_market"), dict) else {}
    lp = wallet.get("last_prices_by_market") if isinstance(wallet.get("last_prices_by_market"), dict) else {}
    lines: list[dict[str, Any]] = []
    slots_used = 0
    for mk, qv in sorted(pbm.items(), key=lambda x: str(x[0])):
        q = float(qv or 0.0)
        if q <= 1e-12:
            continue
        mku = str(mk).strip().upper()
        px = float(lp.get(mku, 0.0) or 0.0)
        if px <= 0 and mku == str(wallet.get("position_symbol") or "").strip().upper():
            px = float(wallet.get("last_price") or 0.0)
        pv = q * px
        w_pct = (pv / eq) * 100.0 if eq > 0 else 0.0
        base = mku.split("-", 1)[0] if "-" in mku else mku
        lines.append(
            {
                "market": mku,
                "coin": base,
                "weight_pct": round(w_pct, 2),
                "notional_eur": round(pv, 2),
                "in_position": True,
            }
        )
        if w_pct >= 0.5:
            slots_used += 1
    markets_in_position = [str(x.get("market") or "") for x in lines if str(x.get("market") or "")]
    return {
        "equal_weight_enabled": elite_equal_weight_enabled(),
        "slot_pct": round(slot_pct, 2),
        "max_slots": int(max_slots),
        "slots_used": int(min(slots_used, max_slots)),
        "slots_label": f"{int(min(slots_used, max_slots))}/{int(max_slots)}",
        "lines": lines,
        "markets_in_position": markets_in_position,
        "summary": f"Allocatie: {int(min(slots_used, max_slots))}/{int(max_slots)} slots bezet (max {slot_pct:.1f}% per munt)",
    }


def apply_equal_weight_buy_fraction_cap(
    *,
    equity: float,
    cash: float,
    wallet: dict[str, Any],
    market: str,
    live_price: float,
    size_fraction: float,
    fee_rate: float,
) -> tuple[float, str | None]:
    """
    Fixed-percentage equal-weight BUY-cap: per munt max slot_pct van equity; max `slot_count` posities;
    geen extra BUY op dezelfde munt zolang de slot al (bijna) vol is.
    Retourneert (nieuwe size_fraction, reason_code of None).
    """
    if not elite_equal_weight_enabled():
        return float(size_fraction), None
    eq = max(1e-9, float(equity))
    fee_r = max(0.0, float(fee_rate))
    slot_pct = elite_equal_weight_slot_pct() / 100.0
    slot_cap_eur = eq * slot_pct
    mku = str(market or "").strip().upper()
    px_m = float(live_price or 0.0)
    if px_m <= 0:
        return float(size_fraction), None
    pv_m = position_value_eur(wallet, px_m, mku)
    tol_eur = max(1.0, eq * 0.0005)
    if pv_m >= slot_cap_eur - tol_eur:
        return 0.0, "equal_weight_slot_full"

    pbm = wallet.get("position_by_market") if isinstance(wallet.get("position_by_market"), dict) else {}
    open_markets = [str(k).strip().upper() for k, q in pbm.items() if float(q or 0.0) > 1e-12]
    if mku not in open_markets and len(open_markets) >= elite_equal_weight_slot_count():
        return 0.0, "equal_weight_max_distinct_slots"

    room_slot_eur = max(0.0, slot_cap_eur - pv_m)
    max_cash_quote = max(0.0, float(cash) / max(1e-12, (1.0 + fee_r)))
    desired_quote = max(0.0, float(size_fraction)) * eq
    allowed_quote = min(desired_quote, room_slot_eur, max_cash_quote)
    new_frac = min(1.0, allowed_quote / eq)

    min_quote = max(1.0, float(os.getenv("EQUAL_WEIGHT_MIN_ORDER_EUR", "5") or 5.0))
    if desired_quote >= min_quote and allowed_quote + 1e-9 < min_quote:
        total_pv = total_crypto_notional_eur(wallet, {mku: px_m})
        deploy_ratio = total_pv / eq if eq > 0 else 0.0
        slot_need = slot_cap_eur * (1.0 + fee_r)
        if float(cash) + 1e-6 < slot_need and deploy_ratio >= (1.0 - slot_pct - 0.03):
            return 0.0, "skip_max_portfolio_allocation"

    if new_frac <= 1e-12 and desired_quote > 1e-6:
        if room_slot_eur <= tol_eur:
            return 0.0, "equal_weight_slot_full"
        if max_cash_quote <= min_quote:
            total_pv2 = total_crypto_notional_eur(wallet, {mku: px_m})
            if (total_pv2 / eq if eq > 0 else 0.0) >= 0.85:
                return 0.0, "skip_max_portfolio_allocation"

    return new_frac, None


def _now_ts() -> float:
    return time.time()


def _parse_event_ts(published_at: str | None) -> float:
    s = str(published_at or "").strip()
    if not s:
        return _now_ts()
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt.timestamp()
    except Exception:
        return _now_ts()


def ingest_whale_inflow_events_for_panic(state: dict[str, Any], moves: list[dict[str, Any]] | None) -> None:
    """Append elite inflow moves above panic USD threshold to a rolling log (deduped)."""
    if not isinstance(state, dict) or not moves:
        return
    log: list[dict[str, Any]] = state.setdefault("whale_inflow_panic_log", [])
    seen: set[str] = state.setdefault("whale_inflow_seen_keys", set())  # type: ignore[assignment]
    if not isinstance(seen, set):
        seen = set()
        state["whale_inflow_seen_keys"] = seen

    now = _now_ts()
    for m in moves:
        if not isinstance(m, dict):
            continue
        if str(m.get("direction") or "").lower() != "inflow":
            continue
        usd = float(m.get("usd_notional_est", 0.0) or 0.0)
        if usd < _WHALE_PANIC_MIN_USD:
            continue
        mk = str(m.get("market") or "").upper()
        if not mk:
            continue
        ts = _parse_event_ts(str(m.get("published_at") or ""))
        headline = str(m.get("headline") or "")[:160]
        key = f"{mk}|{int(ts)}|{headline}"
        if key in seen:
            continue
        seen.add(key)
        log.append({"market": mk, "ts": ts, "usd": usd, "headline": headline})
        if len(seen) > 2000:
            seen.clear()

    cutoff = now - _LOG_RETENTION_SEC
    pruned = [e for e in log if isinstance(e, dict) and float(e.get("ts", 0.0) or 0.0) >= cutoff]
    log[:] = pruned

    counts: dict[str, int] = {}
    for e in pruned:
        if not isinstance(e, dict):
            continue
        mk = str(e.get("market") or "").upper()
        if not mk:
            continue
        if now - float(e.get("ts", 0.0) or 0.0) > _WHALE_PANIC_WINDOW_SEC:
            continue
        if float(e.get("usd", 0.0) or 0.0) < _WHALE_PANIC_MIN_USD:
            continue
        counts[mk] = int(counts.get(mk, 0) or 0) + 1

    by_mkt: dict[str, dict[str, Any]] = {
        mk: {
            "active": int(c) >= _WHALE_PANIC_MIN_INFLOWS,
            "count": int(c),
            "window_sec": int(_WHALE_PANIC_WINDOW_SEC),
            "min_usd": _WHALE_PANIC_MIN_USD,
        }
        for mk, c in counts.items()
    }
    state["whale_danger_by_market"] = by_mkt


def whale_inflow_burst_count(state: dict[str, Any], market: str) -> int:
    mku = str(market or "").upper()
    now = _now_ts()
    log = state.get("whale_inflow_panic_log") if isinstance(state.get("whale_inflow_panic_log"), list) else []
    n = 0
    for e in log:
        if not isinstance(e, dict):
            continue
        if str(e.get("market") or "").upper() != mku:
            continue
        if now - float(e.get("ts", 0.0) or 0.0) > _WHALE_PANIC_WINDOW_SEC:
            continue
        if float(e.get("usd", 0.0) or 0.0) < _WHALE_PANIC_MIN_USD:
            continue
        n += 1
    return n


def whale_danger_zone_for_market(state: dict[str, Any], market: str) -> dict[str, Any]:
    """Dashboard / history API: danger when inflow burst threshold reached for this pair."""
    mku = str(market or "").upper()
    row = (state.get("whale_danger_by_market") or {}).get(mku) if isinstance(state.get("whale_danger_by_market"), dict) else None
    if isinstance(row, dict) and bool(row.get("active")):
        return {
            "active": True,
            "inflow_count": int(row.get("count", 0) or 0),
            "window_minutes": int(_WHALE_PANIC_WINDOW_SEC // 60),
            "min_usd": float(row.get("min_usd", _WHALE_PANIC_MIN_USD) or _WHALE_PANIC_MIN_USD),
            "label": "Whale Danger Zone",
        }
    c = whale_inflow_burst_count(state, mku)
    return {
        "active": c >= _WHALE_PANIC_MIN_INFLOWS,
        "inflow_count": c,
        "window_minutes": int(_WHALE_PANIC_WINDOW_SEC // 60),
        "min_usd": _WHALE_PANIC_MIN_USD,
        "label": "Whale Danger Zone",
    }


def whale_panic_should_force_sell(state: dict[str, Any], held_market: str) -> tuple[bool, str]:
    """True if held market has enough large inflows inside the window (independent of cooldown)."""
    mku = str(held_market or "").upper()
    if not mku:
        return False, ""
    c = whale_inflow_burst_count(state, mku)
    if c >= _WHALE_PANIC_MIN_INFLOWS:
        return True, f"whale_inflow_burst_{c}_in_{int(_WHALE_PANIC_WINDOW_SEC)}s"
    return False, ""


def market_blocked_by_whale_panic_cooldown(state: dict[str, Any], market: str) -> bool:
    mku = str(market or "").upper()
    cd = state.get("whale_panic_cooldown_until") if isinstance(state.get("whale_panic_cooldown_until"), dict) else {}
    until = float(cd.get(mku, 0.0) or 0.0)
    return until > _now_ts()


def set_whale_panic_cooldown(state: dict[str, Any], market: str) -> None:
    mku = str(market or "").upper()
    if not mku:
        return
    cd = state.setdefault("whale_panic_cooldown_until", {})
    if not isinstance(cd, dict):
        cd = {}
        state["whale_panic_cooldown_until"] = cd
    cd[mku] = _now_ts() + _WHALE_PANIC_COOLDOWN_SEC


def _last_panic_sell_ts(state: dict[str, Any], market: str) -> float:
    mku = str(market or "").upper()
    mp = state.get("whale_panic_last_sell_ts") if isinstance(state.get("whale_panic_last_sell_ts"), dict) else {}
    return float(mp.get(mku, 0.0) or 0.0) if isinstance(mp, dict) else 0.0


def record_whale_panic_sell_fired(state: dict[str, Any], market: str) -> None:
    mku = str(market or "").upper()
    mp = state.setdefault("whale_panic_last_sell_ts", {})
    if not isinstance(mp, dict):
        mp = {}
        state["whale_panic_last_sell_ts"] = mp
    mp[mku] = _now_ts()


def can_fire_whale_panic_sell(state: dict[str, Any], held_market: str) -> bool:
    """Debounce repeated panic sells while headlines remain hot."""
    mku = str(held_market or "").upper()
    if not mku:
        return False
    last = _last_panic_sell_ts(state, mku)
    return (_now_ts() - last) >= _WHALE_PANIC_REARM_SEC
