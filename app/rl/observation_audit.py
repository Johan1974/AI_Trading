"""
Tijdelijke / optionele RL input-audit: heartbeat logging, NaN-checks, bron-leeg waarschuwingen.
Zet RL_DATA_AUDIT_HEARTBEAT=1 voor console [DATA AUDIT] elke engine-cycle (run_paper_cycle).
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
from core.analytics import normalize_feature_weights


def _heartbeat_enabled() -> bool:
    return str(os.getenv("RL_DATA_AUDIT_HEARTBEAT", "0")).strip().lower() in ("1", "true", "yes", "on")


def log_rl_data_audit_heartbeat(
    last_row: dict[str, Any],
    *,
    portal_sentiment: float | None = None,
    news_article_count: int | None = None,
) -> None:
    """Logt dezelfde numerieke bronnen als de RL observation state-features (één rij)."""
    if not _heartbeat_enabled():
        return
    pa = float(last_row.get("price_action", 0) or 0)
    rsi = float(last_row.get("rsi_14", 0) or 0)
    macd = float(last_row.get("macd", 0) or 0)
    sent = float(last_row.get("sentiment_score", 0) or 0)
    whale = float(last_row.get("whale_pressure", 0) or 0)
    dom = float(last_row.get("btc_dominance_pct", 0) or 0)
    fin_line = ""
    if portal_sentiment is not None and np.isfinite(portal_sentiment):
        fin_line = f"\n  - Portal sentiment (FinBERT/judge-pad): [{float(portal_sentiment):.4f}]"
    news_line = ""
    if news_article_count is not None:
        news_line = f"\n  - NewsAPI artikelen in frame-window: [{int(news_article_count)}]"
    vec = np.array([pa, rsi, macd, sent, whale, dom], dtype=float)
    if not np.all(np.isfinite(vec)):
        print(
            "WARNING: RL observation audit found NaN/Inf in core fields "
            f"(price_action, rsi, macd, sentiment, whale, dominance): {vec!r}"
        )
    print(
        "[DATA AUDIT] Input Vector:\n"
        f"  - Price Action: [{pa:.6f}]\n"
        f"  - RSI/MACD: [{rsi:.4f} / {macd:.6f}]\n"
        f"  - News Sentiment (RL frame, hourly merge / proxy): [{sent:.4f}]{news_line}\n"
        f"  - Whale Pressure: [{whale:.4f}]\n"
        f"  - BTC Dominance: [{dom:.4f}]{fin_line}"
    )


def warn_if_news_sentiment_empty(news_article_count: int | None, news_api_key_present: bool) -> None:
    """Console waarschuwing als er geen nieuws-artikelen zijn om sentiment te voeden."""
    if news_article_count is None:
        return
    if news_article_count > 0:
        return
    if not news_api_key_present:
        print("WARNING: News Sentiment source is empty!")
        return
    print("WARNING: News Sentiment source is empty!")


def warn_if_whale_source_likely_empty(cc_key_present: bool) -> None:
    if not cc_key_present:
        print("WARNING: Whale Pressure source may be empty (no CryptoCompare API key).")


def merge_feature_weights_for_brain(
    state_decision: dict[str, Any] | None,
    agent_last_weights: dict[str, float] | None,
) -> dict[str, float]:
    """Zorgt dat de Brain WS nooit stil blijft door lege STATE na een mislukte RL-tick."""
    fw: dict[str, float] = {}
    if isinstance(state_decision, dict):
        raw = state_decision.get("feature_weights")
        if isinstance(raw, dict) and raw:
            fw = {str(k): float(v) for k, v in raw.items() if isinstance(v, (int, float))}
    if fw:
        return normalize_feature_weights(fw, method="softmax")
    if isinstance(agent_last_weights, dict) and agent_last_weights:
        raw = {str(k): float(v) for k, v in agent_last_weights.items() if isinstance(v, (int, float))}
        return normalize_feature_weights(raw, method="softmax")
    return {}


def bar_values_from_obs_and_weights(
    feature_weights: dict[str, float],
    rl_observation: dict[str, float] | None,
) -> dict[str, float]:
    """
    Balken = policy feature-mix × amplitude van de ruwe RL-input (zelfde idee als decide(): |obs|×netgewicht).
    Als rl_observation ontbreekt: ongewijzigde weights.
    """
    if not feature_weights:
        return {}
    if not isinstance(rl_observation, dict) or not rl_observation:
        return normalize_feature_weights(dict(feature_weights), method="softmax")
    out: dict[str, float] = {}
    for k, w in feature_weights.items():
        wv = float(w) if isinstance(w, (int, float)) else 0.0
        ov = float(rl_observation.get(str(k), 0.0) or 0.0)
        out[str(k)] = wv * (1.0 + abs(ov))
    s = sum(max(0.0, v) for v in out.values())
    if s <= 1e-12:
        return normalize_feature_weights(dict(feature_weights), method="softmax")
    norm = {k: max(0.0, v) / s for k, v in out.items()}
    return normalize_feature_weights(norm, method="minmax")
