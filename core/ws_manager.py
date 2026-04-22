"""
WebSocket helpers: compact JSON payloads, training-monitor trimming,
en Elite-8 lite-regels (prijs + AI-status) naast volledige brain-data voor focus-markt.
"""

from __future__ import annotations

from typing import Any

WS_TRIM_CHART_POINTS = 400


def trim_training_monitor(tm: dict[str, Any], max_pts: int = WS_TRIM_CHART_POINTS) -> dict[str, Any]:
    if not isinstance(tm, dict):
        return {}
    out = dict(tm)
    for key in ("reward", "reward_normalized", "loss", "policy_entropy", "episode_length"):
        v = out.get(key)
        if isinstance(v, list) and len(v) > max_pts:
            out[key] = v[-max_pts:]
    nl = out.get("network_logs")
    if isinstance(nl, dict):
        nl2 = dict(nl)
        for nk, nv in list(nl2.items()):
            if isinstance(nv, list) and len(nv) > max_pts:
                nl2[nk] = nv[-max_pts:]
        out["network_logs"] = nl2
    return out


def compact_system_stats(d: dict[str, Any]) -> dict[str, Any]:
    """Korte keys voor WebSocket (topic → t)."""
    return {
        "t": "system_stats",
        "c": d.get("cpu_pct"),
        "r": d.get("ram_pct"),
        "d": d.get("disk_pct"),
        "g": d.get("gpu_util_pct"),
        "gm": d.get("gpu_mem_util_pct"),
        "ge": d.get("gpu_util_effective"),
        "vu": d.get("vram_used_mb"),
        "vt": d.get("vram_total_mb"),
        "gk": 1 if d.get("gpu_ok") else 0,
        "gn": d.get("gpu_name") or "",
        "gi": d.get("gpu_index", -1),
    }


def elite_lite_rows(
    focus: str,
    signals: list[dict[str, Any]],
    active_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Zeven 'achtergrond'-munten: alleen prijs + state + korte action."""
    focus_u = str(focus or "").upper()
    px: dict[str, float] = {}
    for r in active_rows or []:
        if not isinstance(r, dict):
            continue
        mk = str(r.get("market") or "").upper()
        if mk:
            try:
                px[mk] = float(r.get("last_price") or 0.0)
            except (TypeError, ValueError):
                px[mk] = 0.0
    out: list[dict[str, Any]] = []
    for s in signals or []:
        if not isinstance(s, dict):
            continue
        mk = str(s.get("market") or "").upper()
        if not mk or mk == focus_u:
            continue
        out.append(
            {
                "m": mk,
                "p": round(float(px.get(mk, 0.0)), 8),
                "s": str(s.get("state") or "neutral"),
                "a": str(s.get("action") or "")[:8],
            }
        )
    return out


def build_brain_ws_wire_payload(
    *,
    focus_market: str,
    training_monitor: dict[str, Any],
    feature_weights: dict[str, Any],
    feature_weights_policy: dict[str, Any],
    rl_observation: dict[str, Any],
    social_buzz: dict[str, Any],
    lite_elite: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "t": "brain_stats",
        "v": 1,
        "f": str(focus_market or "").upper(),
        "tm": trim_training_monitor(training_monitor),
        "fw": feature_weights or {},
        "fwp": feature_weights_policy or {},
        "rl": rl_observation or {},
        "sb": social_buzz or {},
        "L": lite_elite,
    }
