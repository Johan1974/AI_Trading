"""
UI-consistent voorspelling: grafiek-overlay en hint-tekst volgen RL-policy,
niet losse RSI/sentiment tegenstrijdig met PPO-kansen.
"""

from __future__ import annotations

import os
from typing import Any


def trade_confidence_threshold_01() -> float:
    """Zelfde drempel als RL-action gate (0..1)."""
    try:
        from app.services.state import STATE
        from core.risk_manager import sentiment_buy_threshold_offset

        raw = STATE.get("decision_threshold")
        _env_th = float(os.getenv("RL_ACTION_MIN_CONFIDENCE", "0.20") or 0.20)
        if raw is None or raw == "":
            th = _env_th
        else:
            th = float(raw)
        if not th == th or th < 0:
            th = _env_th
        if th > 1.0 + 1e-6:
            th = th / 100.0
        boost = float(STATE.get("decision_threshold_regime_boost") or 0.0)
        if not boost == boost or boost < 0:
            boost = 0.0
        boost = min(float(boost), 0.25)
        last_scores = STATE.get("last_scores") or {}
        agg_sentiment = float(last_scores.get("sentiment_score") or 0.0)
        sentiment_offset = sentiment_buy_threshold_offset(agg_sentiment)
        return max(0.0, min(1.0, th + boost + sentiment_offset))
    except Exception:
        return max(0.0, min(1.0, float(os.getenv("RL_ACTION_MIN_CONFIDENCE", "0.20") or 0.20)))


def rl_decision_usable(dec: dict[str, Any] | None) -> bool:
    if not isinstance(dec, dict):
        return False
    if dec.get("analysis_unavailable") or str(dec.get("policy_status") or "").lower() == "unavailable":
        return False
    try:
        pb = float(dec.get("prob_buy") or 0.0)
        ph = float(dec.get("prob_hold") or 0.0)
        ps = float(dec.get("prob_sell") or 0.0)
    except (TypeError, ValueError):
        return False
    s = pb + ph + ps
    return s > 1e-9 and max(pb, ph, ps) > 1e-9


def tenant_rl_decision_for_symbol(tenant: dict[str, Any] | None, symbol_u: str) -> dict[str, Any] | None:
    """
    Zoekt RL-besluit voor ``symbol_u`` (normalisatie BTC-EUR) in ``tenant.rl_multi_decisions``
    of valt terug op rl_last_decision als die markt expliciet matcht.
    """
    if not isinstance(tenant, dict) or not symbol_u:
        return None
    mku = str(symbol_u).strip().upper().replace("/", "-")
    multi = tenant.get("rl_multi_decisions")
    if isinstance(multi, dict):
        proc = None
        for k, v in multi.items():
            if str(k).strip().upper().replace("/", "-") == mku:
                proc = v
                break
        if proc is not None:
            try:
                import app.trading_core as tc

                return tc._rl_decision_as_dict_with_fallback(proc)  # type: ignore[attr-defined]
            except Exception:
                pass
    gl = tenant.get("rl_last_decision")
    if isinstance(gl, dict):
        t = str(gl.get("ticker") or gl.get("market") or "").strip().upper().replace("/", "-")
        if t == mku:
            try:
                import app.trading_core as tc

                return tc._rl_decision_as_dict_with_fallback(gl)  # type: ignore[attr-defined]
            except Exception:
                pass
    return None


def prediction_signal_allowed_by_rl(sig: str, dec: dict[str, Any] | None, th: float) -> bool:
    """
    BUY/SELL-tekst alleen tonen als dezelfde RL-kans boven drempel wint;
    anders false zodat de UI niet voorloopt op 0%%-balkjes.
    """
    s = str(sig or "").strip().upper()
    if s not in {"BUY", "SELL"}:
        return True
    if not rl_decision_usable(dec):
        return False
    try:
        pb = float(dec.get("prob_buy") or 0.0)
        ph = float(dec.get("prob_hold") or 0.0)
        ps = float(dec.get("prob_sell") or 0.0)
    except (TypeError, ValueError):
        return False
    mx = max(pb, ph, ps)
    if mx < th - 1e-9:
        return False
    if s == "BUY":
        return pb >= th - 1e-9 and pb + 1e-9 >= max(ph, ps)
    if s == "SELL":
        return ps >= th - 1e-9 and ps + 1e-9 >= max(ph, pb)
    return False


def _classic_last_prediction_series(lp: dict[str, Any] | None, current_price: float) -> tuple[list[float], float, float]:
    """Monotone interpolatie latest → next (legacy RSI/techniek-pad)."""
    pc = float(current_price or 0.0)
    if not isinstance(lp, dict):
        return [], pc, pc
    lc = float(lp.get("latest_close") or 0.0)
    pn = float(lp.get("predicted_next_close") or 0.0)
    if lc <= 0.0 and current_price > 0:
        lc = float(current_price)
    if pn <= 0.0 or lc <= 0.0:
        fb = lc if lc > 0 else pc
        return [], fb, fb
    n = max(3, min(48, int(os.getenv("PREDICTION_CHART_STEPS", "16") or 16)))
    series = [round(lc + (pn - lc) * (i / (n - 1)), 6) for i in range(n)]
    return series, lc, pn


def build_overlay_prices_from_rl_or_fallback(
    current_price: float,
    ai_decision: dict[str, Any] | None,
    lp: dict[str, Any] | None,
) -> tuple[list[float], float, float]:
    """
    Gecombineerde RL-richting voor de overlay: verwachte rendements-component +
    (prob_buy - prob_sell) bias. Zonder bruikbare RL of onder de drempel: lege serie
    (geen RSI-tegenstrijdige lijn).
    """
    pc = float(current_price or 0.0)
    if pc <= 0.0:
        pc = 0.01
    th = trade_confidence_threshold_01()
    dec = ai_decision if isinstance(ai_decision, dict) else {}
    if rl_decision_usable(dec):
        try:
            pb = float(dec.get("prob_buy") or 0.0)
            ph = float(dec.get("prob_hold") or 0.0)
            ps = float(dec.get("prob_sell") or 0.0)
        except (TypeError, ValueError):
            return _classic_last_prediction_series(lp, pc)
        mx = max(pb, ph, ps)
        if mx < th - 1e-9:
            return [], pc, pc
        bias = pb - ps
        try:
            exp = float(dec.get("expected_reward_pct") or 0.0)
        except (TypeError, ValueError):
            exp = 0.0
        delta = bias * 0.022 + (exp / 100.0) * 0.38
        delta = max(-0.035, min(0.035, float(delta)))
        lc = pc
        pn = lc * (1.0 + delta)
        if pn <= 0:
            pn = lc
        n = max(3, min(48, int(os.getenv("PREDICTION_CHART_STEPS", "16") or 16)))
        series = [round(lc + (pn - lc) * (i / (n - 1)), 6) for i in range(n)]
        return series, lc, pn
    # Geen bruikbare RL-beslissing → geen prediction-lijn (beter dan misleidende RSI-fallback)
    return [], pc, pc
