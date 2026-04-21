"""
Feature preprocessing for RL observation space normalization and signal integrity.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

SIGNAL_STRENGTH_MULTIPLIER = float(1.5)


def _safe_minmax(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").astype(float)
    lo = float(np.nanpercentile(s, 1)) if np.isfinite(s).any() else 0.0
    hi = float(np.nanpercentile(s, 99)) if np.isfinite(s).any() else 1.0
    if not np.isfinite(lo):
        lo = 0.0
    if not np.isfinite(hi) or hi <= lo + 1e-12:
        hi = lo + 1.0
    out = (s - lo) / (hi - lo)
    return out.clip(0.0, 1.0).fillna(0.0)


def _safe_zscore_tanh(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").astype(float)
    mu = float(np.nanmean(s)) if np.isfinite(s).any() else 0.0
    sigma = float(np.nanstd(s)) if np.isfinite(s).any() else 1.0
    if not np.isfinite(sigma) or sigma <= 1e-12:
        sigma = 1.0
    z = (s - mu) / sigma
    return pd.Series(np.tanh(z), index=s.index).replace([np.inf, -np.inf], 0.0).fillna(0.0)


def forward_fill_dead_signal(
    series: pd.Series,
    *,
    last_value: float | None = None,
    treat_zero_as_dead: bool = True,
) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").astype(float)
    if treat_zero_as_dead:
        s = s.where(np.abs(s) > 1e-12, np.nan)
    s = s.replace([np.inf, -np.inf], np.nan).ffill()
    if last_value is not None and np.isfinite(float(last_value)):
        s = s.fillna(float(last_value))
    return s.fillna(0.0)


def normalize_rl_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Core range guarantees for the RL observation space.
    out["price_action"] = _safe_zscore_tanh(out.get("price_action", 0.0))
    out["volatility_24"] = _safe_minmax(out.get("volatility_24", 0.0))
    out["volume_change"] = _safe_zscore_tanh(out.get("volume_change", 0.0))
    out["sentiment_score"] = _safe_zscore_tanh(out.get("sentiment_score", 0.0))
    out["news_confidence"] = _safe_minmax(out.get("news_confidence", 0.0))
    out["social_volume"] = _safe_minmax(out.get("social_volume", 0.0))
    out["fear_greed_score"] = _safe_minmax(out.get("fear_greed_score", 0.5))
    out["btc_dominance_pct"] = _safe_minmax(out.get("btc_dominance_pct", 50.0))
    out["whale_pressure"] = _safe_zscore_tanh(out.get("whale_pressure", 0.0))
    out["macro_volatility_window"] = _safe_minmax(out.get("macro_volatility_window", 0.0))
    out["bollinger_width"] = _safe_minmax(out.get("bollinger_width", 0.0))
    out["bollinger_position"] = _safe_minmax(out.get("bollinger_position", 0.5))
    out["orderbook_imbalance"] = _safe_zscore_tanh(out.get("orderbook_imbalance", 0.0))
    out["macd"] = _safe_zscore_tanh(out.get("macd", 0.0))
    out["rsi_14"] = _safe_minmax(out.get("rsi_14", 50.0))
    out["ema_gap_pct"] = _safe_zscore_tanh(out.get("ema_gap_pct", 0.0))
    # Keep News/Whale channels visible in the observation space.
    for signal_col in ("sentiment_score", "news_confidence", "whale_pressure"):
        out[signal_col] = (pd.to_numeric(out.get(signal_col, 0.0), errors="coerce").fillna(0.0) * SIGNAL_STRENGTH_MULTIPLIER)
    out["sentiment_score"] = out["sentiment_score"].clip(-1.0, 1.0)
    out["whale_pressure"] = out["whale_pressure"].clip(-1.0, 1.0)
    out["news_confidence"] = out["news_confidence"].clip(0.0, 1.0)
    for col in [
        "price_action",
        "volatility_24",
        "volume_change",
        "sentiment_score",
        "news_confidence",
        "social_volume",
        "fear_greed_score",
        "btc_dominance_pct",
        "whale_pressure",
        "macro_volatility_window",
        "bollinger_width",
        "bollinger_position",
        "orderbook_imbalance",
        "macd",
        "rsi_14",
        "ema_gap_pct",
    ]:
        out[col] = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return out


def attention_gate_weights(obs_features: np.ndarray, temperature: float = 0.7) -> np.ndarray:
    x = np.abs(np.asarray(obs_features, dtype=np.float32))
    t = max(1e-3, float(temperature))
    logits = x / t
    logits = logits - float(np.max(logits)) if logits.size else logits
    exp = np.exp(logits)
    denom = float(np.sum(exp))
    if denom <= 1e-12:
        return np.ones_like(x, dtype=np.float32) / max(1, x.size)
    return (exp / denom).astype(np.float32)

