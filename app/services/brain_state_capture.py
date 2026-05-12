"""
Snapshot van AI-/brain-state voor persistentie bij trades (offline training).
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from app.datetime_util import UTC


def _summarize_multi(decisions: Any, *, max_pairs: int = 24) -> dict[str, Any]:
    if not isinstance(decisions, dict):
        return {}
    out: dict[str, Any] = {}
    for i, (k, v) in enumerate(decisions.items()):
        if i >= max_pairs:
            out["_truncated_after"] = max_pairs
            break
        if not isinstance(v, dict):
            continue
        mk = str(k).strip().upper().replace("/", "-")
        out[mk] = {
            "prob_buy": v.get("prob_buy"),
            "prob_hold": v.get("prob_hold"),
            "prob_sell": v.get("prob_sell"),
            "action": v.get("action") or v.get("action_name"),
            "policy_status": v.get("policy_status"),
            "confidence": v.get("confidence"),
        }
    return out


def capture_brain_state_dict() -> dict[str, Any]:
    from app.services.state import STATE

    rl_last = STATE.get("rl_last_decision")
    obs_tail = STATE.get("rl_last_observation")
    obs_summary: Any = None
    if isinstance(obs_tail, list):
        obs_summary = {"len": len(obs_tail), "tail": [float(x) for x in obs_tail[-12:]]}
    elif hasattr(obs_tail, "tolist"):
        try:
            flat = obs_tail.tolist()
            if isinstance(flat, list):
                obs_summary = {"len": len(flat), "tail": [float(x) for x in flat[-12:]]}
        except Exception:
            obs_summary = str(type(obs_tail).__name__)

    return {
        "ts_utc": datetime.now(UTC).isoformat(),
        "selected_market": STATE.get("selected_market"),
        "decision_threshold": STATE.get("decision_threshold"),
        "decision_threshold_regime_boost": STATE.get("decision_threshold_regime_boost"),
        "regime_high_volatility": STATE.get("regime_high_volatility"),
        "regime_atr_14": STATE.get("regime_atr_14"),
        "regime_atr_mean_24": STATE.get("regime_atr_mean_24"),
        "market_regime": STATE.get("market_regime") or {},
        "rl_last_decision": rl_last if isinstance(rl_last, dict) else None,
        "rl_multi_decisions_summary": _summarize_multi(STATE.get("rl_multi_decisions")),
        "rl_last_observation_summary": obs_summary,
        "last_scores": STATE.get("last_scores"),
        "signal_markers_tail": (STATE.get("signal_markers") or [])[-5:]
        if isinstance(STATE.get("signal_markers"), list)
        else None,
    }


def capture_brain_state_json(*, max_chars: int = 120_000) -> str:
    raw = json.dumps(capture_brain_state_dict(), ensure_ascii=False, default=str)
    if len(raw) <= max_chars:
        return raw
    return raw[: max_chars - 30] + "\n…[truncated brain_state_json]"
