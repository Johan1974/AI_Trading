"""Activity snapshot REST (lazy import van ``app.trading_core``)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from core.risk_manager import risk_profile_dict
from app.services.state import STATE

from app.services.prediction_ui import (
    prediction_signal_allowed_by_rl,
    tenant_rl_decision_for_symbol,
    trade_confidence_threshold_01,
)
from app.settings import LIVE_MODE

router = APIRouter(tags=["activity"])


def _activity_trades_merged() -> list[dict[str, Any]]:
    """STATE trades (EVENT rows) + ACTIVE_LOT rows from open wallet positions."""
    base: list[dict[str, Any]] = list(STATE.get("trades") or [])
    try:
        import app.trading_core as _tc
        pm = getattr(_tc, "PAPER_MANAGER", None)
        if pm is not None:
            lots = pm.round_trip_ledger(limit=50)
            active = [r for r in lots if str(r.get("status") or "").upper() == "ACTIVE"]
            # Deduplicate: active lots keyed by market; EVENT rows keyed by market too.
            active_markets = {str(r.get("market") or "").upper() for r in active}
            filtered_base = [r for r in base if str(r.get("market") or "").upper() not in active_markets]
            return active + filtered_base
    except Exception:
        pass
    return base


@router.get("/activity/ping")
def activity_ping() -> dict[str, Any]:
    """Liveness/readiness: alleen ``STATE`` (geen ``trading_core``, allocatie, replay of system_stats)."""
    lec = STATE.get("last_engine_cycle")
    tick = lec.get("ts") if isinstance(lec, dict) else None
    return {
        "ok": True,
        "bot_status": STATE.get("bot_status", "running"),
        "selected_market": STATE.get("selected_market", "BTC-EUR"),
        "last_engine_tick_utc": tick,
    }


def _build_worker_calc_hints() -> list[str]:
    """Korte UI-teksten: wat de worker / laatste cyclus deed (voor 'Actieve berekeningen')."""
    out: list[str] = []
    ls = STATE.get("last_scores") if isinstance(STATE.get("last_scores"), dict) else {}
    lp = STATE.get("last_prediction") if isinstance(STATE.get("last_prediction"), dict) else {}
    lo = STATE.get("last_order") if isinstance(STATE.get("last_order"), dict) else {}
    lec = STATE.get("last_engine_cycle") if isinstance(STATE.get("last_engine_cycle"), dict) else {}

    tr = ls.get("technical_predicted_return_pct")
    if tr is not None:
        try:
            out.append(f"RSI/techniek: verwacht {float(tr):+.2f}% (korte horizon)")
        except (TypeError, ValueError):
            pass

    js = str(ls.get("judge_signal") or "").upper()
    if js:
        comp = ls.get("judge_composite_score")
        out.append(f"Sentiment check: judge → {js}" + (f" (Δ {comp})" if comp is not None else ""))

    ss = ls.get("sentiment_score")
    if ss is not None:
        try:
            out.append(f"Nieuwslaag: sentiment-score {float(ss):.3f}")
        except (TypeError, ValueError):
            pass

    sig = str(lp.get("signal") or "").upper()
    if sig:
        _tk = str(lp.get("ticker") or "").strip().upper().replace("/", "-")
        _dec = tenant_rl_decision_for_symbol(STATE if isinstance(STATE, dict) else {}, _tk) if _tk else None
        _th = trade_confidence_threshold_01()
        if prediction_signal_allowed_by_rl(sig, _dec, _th):
            out.append(f"Voorspelling: {sig} · {_tk}")

    rd = lo.get("risk_decision") if isinstance(lo.get("risk_decision"), dict) else {}
    rs = str(rd.get("reason") or "").strip()
    if rs and rs != "approved":
        out.append(f"Risk gate: {rs}")

    pair = str(lec.get("pair") or "").strip().upper()
    if pair:
        out.append(f"Worker: laatste cyclus {pair}")

    return out[-12:]


@router.get("/activity")
def activity() -> dict[str, Any]:
    import app.trading_core as m

    lec = STATE.get("last_engine_cycle")
    last_engine_tick_utc = lec.get("ts") if isinstance(lec, dict) else None
    return {
        "mode": "live" if LIVE_MODE else "paper",
        "bot_status": STATE.get("bot_status", "running"),
        "selected_market": STATE.get("selected_market", "BTC-EUR"),
        "active_markets": STATE.get("active_markets", []),
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
        "rl_last_decision": STATE.get("rl_last_decision") if isinstance(STATE.get("rl_last_decision"), dict) else {},
        "rl_inference_greedy": bool(STATE.get("rl_inference_greedy")),
        "rl_multi_decisions": STATE.get("rl_multi_decisions") if isinstance(STATE.get("rl_multi_decisions"), dict) else {},
        "worker_calc_hints": _build_worker_calc_hints(),
        "worker_calc_hints_by_market": m._worker_calc_hints_by_market_for_redis(),
        "trades": _activity_trades_merged(),
        "system_stats": m._system_stats_payload_for_websocket(),
        "cockpit_log_tail": list(STATE.get("cockpit_log_tail") or [])[-100:]
        if isinstance(STATE.get("cockpit_log_tail"), list)
        else [],
        "rl_replay": m.replay_stats_for_activity(),
        "rl_optimizer": _rl_optimizer_activity_payload(m),
    }


def _rl_optimizer_activity_payload(mod: Any) -> dict[str, Any]:
    agent = getattr(mod, "RL_AGENT", None)
    st = getattr(agent, "last_training_stats", None) if agent else None
    if isinstance(st, dict) and st.get("global_step_count"):
        return {
            "learning_rate": st.get("learning_rate"),
            "global_step_count": st.get("global_step_count"),
            "exploration_rate_pct": st.get("exploration_rate_pct"),
        }
    return STATE.get("rl_optimizer_stats") or {"learning_rate": None, "global_step_count": 0, "exploration_rate_pct": None}
