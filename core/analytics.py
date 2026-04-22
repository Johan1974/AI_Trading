"""
Analytics helpers for strategy/feature-weight rendering.
"""

from __future__ import annotations

import math
from typing import Any


def normalize_feature_weights(
    weights: dict[str, float],
    *,
    method: str = "softmax",
) -> dict[str, float]:
    if not weights:
        return {}
    keys = [str(k) for k in weights.keys()]
    vals = [max(0.0, float(weights.get(k, 0.0) or 0.0)) for k in keys]
    vals = [0.0 if (math.isnan(v) or math.isinf(v)) else v for v in vals]

    if str(method).lower() == "minmax":
        vmin = min(vals) if vals else 0.0
        vmax = max(vals) if vals else 0.0
        span = vmax - vmin
        if span > 1e-12:
            scaled = [(v - vmin) / span for v in vals]
        else:
            scaled = [1.0] * len(vals)
        denom = sum(scaled)
        if denom > 1e-12:
            norm = [s / denom for s in scaled]
        else:
            n = max(1, len(vals))
            norm = [1.0 / n] * len(vals)
    else:
        vmax = max(vals) if vals else 0.0
        exps = [math.exp(min(50.0, max(-50.0, v - vmax))) for v in vals]
        denom = sum(exps)
        if denom > 1e-12:
            norm = [e / denom for e in exps]
        else:
            n = max(1, len(vals))
            norm = [1.0 / n] * len(vals)

    return {k: float(v) for k, v in zip(keys, norm)}
