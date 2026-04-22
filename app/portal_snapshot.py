"""
Bestand: app/portal_snapshot.py
Relatief pad: ./app/portal_snapshot.py
Functie: Redis-snapshot tussen worker- en portal-container. Geen file I/O om Disk Wait te voorkomen.
"""

from __future__ import annotations

import json
import os
from typing import Any
import redis

from app.services.state import get_tenant_state

# Alleen deze specifieke keys mogen vanuit de worker naar de portal gesynchroniseerd worden.
# Dit voorkomt dat gigantische (per ongeluk gecachete) objecten of dataframes de file opblazen naar gigabytes.
PORTAL_SYNC_KEYS = {
    "bot_status", "active_markets", "scanner_selected", "selected_market",
    "paper_portfolio", "last_engine_cycle", "whale_panic_cooldown_until",
    "last_prediction", "last_order", "fear_greed", "news_insights",
    "last_scores", "rl_multi_decisions", "whale_radar_moves", "whale_flow_by_market",
    "social_momentum_by_market", "social_buzz_summary", "scanner_intel_feed",
    "signal_markers", "news_lag_history", "rl_shared_buffer", "cmc_metrics", "events",
    "macro_context", "whale_watch", "auto_opt_exploration_eps",
    "auto_opt_risk_cap_pct", "auto_opt_train_chunk_steps",
    "decision_threshold", "stop_loss_pct", "started_at"
}

def _get_redis_client() -> redis.Redis:
    url = str(os.getenv("REDIS_URL", "redis://localhost:6379")).strip()
    return redis.Redis.from_url(url, decode_responses=True)


def _trim_large_data(v: Any, max_len: int = 50) -> Any:
    """Recursief inkorten van lijsten en DataFrames tot max_len items."""
    if type(v).__name__ == "DataFrame":
        try:
            return v.tail(max_len).to_dict(orient="records")
        except Exception:
            pass
    if isinstance(v, list):
        trimmed = v[-max_len:] if len(v) > max_len else v
        return [_trim_large_data(item, max_len) for item in trimmed]
    if isinstance(v, dict):
        return {k: _trim_large_data(val, max_len) for k, val in v.items()}
    return v


def _json_safe_dict(d: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        try:
            # Pre-emptieve trimming: standaardlijsten voor de UI inperken tot max 50 items.
            if k in {"events", "news_insights", "signal_markers", "news_lag_history", "scanner_intel_feed", "rl_shared_buffer", "paper_portfolio"}:
                v = _trim_large_data(v, max_len=50)

            v_str = json.dumps(v, default=str)
            v_size = len(v_str)
            
            # Veiligheidsnet: als de key/waarde alsnog veel te zwaar weegt (> 250KB), trim agressief naar 10 items.
            if v_size > 250_000:
                v = _trim_large_data(v, max_len=10)
                v_str = json.dumps(v, default=str)
            
            out[k] = json.loads(v_str)
        except (TypeError, ValueError):
            continue
    return out


def write_worker_portal_snapshot(*, extras: dict[str, Any] | None = None) -> None:
    raw_state = dict(get_tenant_state())

    filtered_state = {k: v for k, v in raw_state.items() if k in PORTAL_SYNC_KEYS}
    
    # Portfolio limiet afdwingen voor de Redis payload (max 500 items)
    if "paper_portfolio" in filtered_state and isinstance(filtered_state["paper_portfolio"], dict):
        pf_hist = filtered_state["paper_portfolio"].get("history")
        if isinstance(pf_hist, list):
            filtered_state["paper_portfolio"]["history"] = pf_hist[-500:]

    tenant = _json_safe_dict(filtered_state)
    ext = _json_safe_dict(dict(extras or {}))
    
    blob = {"v": 1, "tenant": tenant, "extras": ext}

    json_str = json.dumps(blob, separators=(",", ":"), ensure_ascii=False)
        
    try:
        r = _get_redis_client()
        r.set("ai_trading_snapshot", json_str)
    except Exception as e:
        pass


def read_worker_portal_snapshot() -> dict[str, Any] | None:
    try:
        r = _get_redis_client()
        data_str = r.get("ai_trading_snapshot")
        if not data_str:
            return None
        data = json.loads(data_str)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


PORTAL_MERGE_SKIP = frozenset({"ws_connections", "last_ws_heartbeat_ts"})


def apply_worker_snapshot_to_portal(blob: dict[str, Any]) -> None:
    if not isinstance(blob, dict):
        return
    tenant = blob.get("tenant")
    if not isinstance(tenant, dict):
        return
    s = get_tenant_state()
    for k, v in tenant.items():
        if k in PORTAL_MERGE_SKIP:
            continue
        s[k] = v
    extras = blob.get("extras")
    if isinstance(extras, dict):
        bw = extras.get("brain_ws")
        if isinstance(bw, dict) and bw:
            s["_portal_brain_ws"] = bw
