"""
Analytics helpers for strategy/feature-weight rendering.
"""

from __future__ import annotations

import numpy as np


def normalize_feature_weights(
    weights: dict[str, float],
    *,
    method: str = "softmax",
) -> dict[str, float]:
    if not weights:
        return {}
    keys = [str(k) for k in weights.keys()]
    vals = np.array([float(weights.get(k, 0.0) or 0.0) for k in keys], dtype=float)
    vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
    vals = np.maximum(vals, 0.0)

    if str(method).lower() == "minmax":
        vmin = float(np.min(vals)) if vals.size else 0.0
        vmax = float(np.max(vals)) if vals.size else 0.0
        span = vmax - vmin
        scaled = (vals - vmin) / span if span > 1e-12 else np.ones_like(vals)
        denom = float(np.sum(scaled))
        norm = scaled / denom if denom > 1e-12 else np.full_like(vals, 1.0 / max(1, len(vals)))
    else:
        vmax = float(np.max(vals)) if vals.size else 0.0
        exps = np.exp(np.clip(vals - vmax, -50.0, 50.0))
        denom = float(np.sum(exps))
        norm = exps / denom if denom > 1e-12 else np.full_like(vals, 1.0 / max(1, len(vals)))

    return {k: float(v) for k, v in zip(keys, norm.tolist())}
