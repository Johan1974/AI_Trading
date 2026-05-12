"""
Bestand: app/services/market_regime.py
Functie: Marktregime-detectie — BULL / BEAR / RANGING / VOLATILE.

Scoring op basis van genormaliseerde RL-features uit de laatste rij van de preprocessor.
Slaat resultaat op in STATE en past decision_threshold_regime_boost aan.
"""

from __future__ import annotations

from typing import Any

from app.services.state import STATE

# Drempels voor regime-classificatie
_BULL_THRESHOLD = 3    # ≥ 3 van 6 mogelijke bull-punten
_BEAR_THRESHOLD = 3    # ≥ 3 van 6 mogelijke bear-punten

# Threshold-aanpassingen per regime (optelt bij base decision_threshold)
_REGIME_BOOST: dict[str, float] = {
    "VOLATILE": 0.05,   # voorzichtiger bij hoge volatiliteit
    "BEAR":     0.03,   # conservatiever in bearmarkt
    "RANGING":  0.00,   # geen aanpassing
    "BULL":    -0.02,   # iets agressiever in bullmarkt
}


def _score_features(last: dict[str, Any]) -> tuple[int, int]:
    """Bereken bull/bear score (elk 0–6) vanuit genormaliseerde RL-features."""
    bull = 0
    bear = 0

    # ema_gap_pct: tanh(zscore), positief = boven EMA
    eg = float(last.get("ema_gap_pct") or 0.0)
    if eg > 0.01:
        bull += 2
    elif eg < -0.01:
        bear += 2

    # rsi_14: minmax [0,1], 0.5 = neutraal (raw RSI 50)
    rsi = float(last.get("rsi_14") or 0.5)
    if rsi > 0.55:
        bull += 1
    elif rsi < 0.45:
        bear += 1

    # bollinger_position: [0,1], 0.5 = midden band
    bp = float(last.get("bollinger_position") or 0.5)
    if bp > 0.65:
        bull += 1
    elif bp < 0.35:
        bear += 1

    # macd: genormaliseerd, positief = bullish crossover
    macd = float(last.get("macd") or 0.0)
    if macd > 0.01:
        bull += 1
    elif macd < -0.01:
        bear += 1

    # price_action: genormaliseerde recente prijsbeweging
    pa = float(last.get("price_action") or 0.0)
    if pa > 0.02:
        bull += 1
    elif pa < -0.02:
        bear += 1

    return bull, bear


def refresh_market_regime_from_last_row(last: dict[str, Any] | None) -> None:
    """
    Classificeert het marktregime op basis van de laatste preprocessor-rij.
    Werkt STATE bij en past decision_threshold_regime_boost aan.
    """
    if not isinstance(last, dict):
        return

    # --- Bestaande ATR-volatiliteitscheck ---
    try:
        atr = float(last.get("atr_14") or 0.0)
        atr_m = float(last.get("atr_mean_24") or 0.0)
    except (TypeError, ValueError):
        atr, atr_m = 0.0, 0.0
    high_vol = bool(atr_m > 0.0 and atr > atr_m and (atr - atr_m) / atr_m > 0.05)

    STATE["regime_atr_14"] = atr
    STATE["regime_atr_mean_24"] = atr_m
    STATE["regime_high_volatility"] = high_vol

    # --- Bull/Bear scoring ---
    bull_score, bear_score = _score_features(last)
    max_score = 6
    net = bull_score - bear_score

    if high_vol:
        regime = "VOLATILE"
    elif bull_score >= _BULL_THRESHOLD and bull_score > bear_score:
        regime = "BULL"
    elif bear_score >= _BEAR_THRESHOLD and bear_score > bull_score:
        regime = "BEAR"
    else:
        regime = "RANGING"

    confidence = round(max(bull_score, bear_score) / max_score, 3)

    boost = _REGIME_BOOST.get(regime, 0.0)
    STATE["decision_threshold_regime_boost"] = boost
    STATE["market_regime"] = {
        "regime": regime,
        "confidence": confidence,
        "bull_score": bull_score,
        "bear_score": bear_score,
        "net_score": net,
        "high_vol": high_vol,
        "boost": boost,
    }
