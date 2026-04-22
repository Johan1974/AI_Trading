"""
Whale radar & social/whale context for trading and ledger.

Uses CryptoCompare news (same family as `WhaleWatcherService`) to detect **large notional moves**
(≥ `WHALE_MOVE_MIN_USD`, default 1M USD) and classify **exchange inflow** vs **wallet outflow** heuristically.
"""

from __future__ import annotations

import math
import os
import re
from datetime import datetime, timedelta
from typing import Any, Sequence

import requests

from app.datetime_util import UTC

WHALE_MOVE_MIN_USD = float(os.getenv("WHALE_MOVE_MIN_USD", "1000000") or 1_000_000.0)

_INFLOW_PAT = re.compile(
    r"\b(exchange|binance|coinbase|kraken|okx|kucoin|gemini|bybit|deposit|inflow|sent to exchange|moved to exchange)\b",
    re.I,
)
_OUTFLOW_PAT = re.compile(
    r"\b(cold\s*storage|cold\s*wallet|wallet|withdraw|withdrawal|outflow|off\s*exchange|self[- ]custody|accumulation)\b",
    re.I,
)
_MONEY_PAT = re.compile(
    r"(?:\$|USD\s*|US\s*\$|EUR\s*)?\s*(\d+(?:[.,]\d+)?)\s*(million|billion|bn|thousand|k\b)?",
    re.I,
)


def _parse_usd_notional(blob: str) -> float:
    """Rough USD notional from headline+body snippets."""
    t = str(blob or "")
    best = 0.0
    for m in _MONEY_PAT.finditer(t):
        try:
            raw = str(m.group(1) or "").replace(",", ".")
            val = float(raw)
        except ValueError:
            continue
        unit = str(m.group(2) or "").lower()
        mult = 1.0
        if "billion" in unit or unit == "bn":
            mult = 1_000_000_000.0
        elif "million" in unit or unit == "m":
            mult = 1_000_000.0
        elif "thousand" in unit or unit == "k":
            mult = 1_000.0
        # Bare "million btc" style already captured by million in unit
        usd = val * mult
        # If text says EUR, scale lightly to USD for threshold compare
        if re.search(r"\bEUR\b", t[m.start() : m.end() + 8], re.I):
            usd *= 1.08
        best = max(best, usd)
    return float(best)


def _direction_from_text(blob: str) -> str:
    s = str(blob or "")
    inf = bool(_INFLOW_PAT.search(s))
    outf = bool(_OUTFLOW_PAT.search(s))
    if inf and not outf:
        return "inflow"
    if outf and not inf:
        return "outflow"
    if inf and outf:
        return "mixed"
    return "unknown"


def _mentions_elite_base(blob: str, bases: set[str]) -> str | None:
    u = str(blob or "").upper()
    for b in bases:
        if not b:
            continue
        if re.search(rf"\b{re.escape(b)}\b", u):
            return b
    if "BITCOIN" in u and "BTC" in bases:
        return "BTC"
    if "ETHEREUM" in u and "ETH" in bases:
        return "ETH"
    return None


def fetch_cryptocompare_news_rows(api_key: str | None, limit: int = 120) -> list[dict[str, Any]]:
    key = str(api_key or "").strip()
    headers: dict[str, str] = {}
    if key:
        headers["authorization"] = f"Apikey {key}"
    try:
        resp = requests.get(
            "https://min-api.cryptocompare.com/data/v2/news/",
            params={"lang": "EN"},
            headers=headers,
            timeout=14,
        )
        if resp.status_code != 200:
            return []
        payload = resp.json() if resp.content else {}
        rows = payload.get("Data") if isinstance(payload, dict) else []
        if not isinstance(rows, list):
            return []
        out: list[dict[str, Any]] = []
        for row in rows[: max(1, min(int(limit), 200))]:
            if not isinstance(row, dict):
                continue
            title = str(row.get("title") or "")
            body = str(row.get("body") or "")[: 2000]
            ts = row.get("published_on")
            out.append({"title": title, "body": body, "published_on": ts})
        return out
    except Exception:
        return []


def scan_whale_moves_from_news(
    articles: list[dict[str, Any]],
    elite_markets: list[str],
) -> list[dict[str, Any]]:
    bases = {str(m.split("-", 1)[0]).upper() for m in elite_markets if m and "-" in str(m)}
    bases |= {str(m).upper() for m in elite_markets if m and "-" not in str(m)}
    bases.discard("")
    moves: list[dict[str, Any]] = []
    cutoff = datetime.now(UTC) - timedelta(hours=6)
    for a in articles:
        title = str(a.get("title") or "")
        body = str(a.get("body") or "")[: 1500]
        blob = f"{title}\n{body}"
        usd = _parse_usd_notional(blob)
        if usd < WHALE_MOVE_MIN_USD:
            continue
        base_hit = _mentions_elite_base(blob, bases)
        if not base_hit:
            continue
        direction = _direction_from_text(blob)
        ts = a.get("published_on")
        try:
            pub_iso = datetime.fromtimestamp(int(ts), tz=UTC).isoformat() if ts is not None else ""
        except Exception:
            pub_iso = ""
        moves.append(
            {
                "symbol": base_hit,
                "market": f"{base_hit}-EUR",
                "direction": direction,
                "usd_notional_est": round(usd, 0),
                "headline": title[: 220],
                "published_at": pub_iso,
            }
        )
    moves.sort(key=lambda x: str(x.get("published_at") or ""), reverse=True)
    return moves


def refresh_whale_radar_state(
    *,
    state: dict[str, Any],
    cryptocompare_key: str | None,
    elite_markets: list[str],
) -> list[dict[str, Any]]:
    rows = fetch_cryptocompare_news_rows(cryptocompare_key, limit=140)
    moves = scan_whale_moves_from_news(rows, elite_markets)
    state["whale_radar_moves"] = moves[:12]
    by_mkt: dict[str, Any] = {}
    for m in moves[:20]:
        mk = str(m.get("market") or "").upper()
        if not mk:
            continue
        prev = by_mkt.get(mk, {})
        prev_st = float(prev.get("strength", 0.0) or 0.0)
        st = min(1.0, 0.35 + math.tanh(float(m.get("usd_notional_est", 0.0) or 0.0) / 2_500_000.0) * 0.5)
        if st > prev_st:
            by_mkt[mk] = {
                "bias": str(m.get("direction") or "unknown"),
                "strength": st,
                "headline": str(m.get("headline") or "")[: 200],
                "usd_notional_est": float(m.get("usd_notional_est", 0.0) or 0.0),
            }
    state["whale_flow_by_market"] = by_mkt
    try:
        from core.risk_management import ingest_whale_inflow_events_for_panic

        ingest_whale_inflow_events_for_panic(state, moves)
    except Exception:
        pass
    return moves[:12]


def whale_bias_for_market(market: str, state: dict[str, Any]) -> dict[str, Any]:
    mku = str(market or "").upper()
    row = (state.get("whale_flow_by_market") or {}).get(mku) if isinstance(state.get("whale_flow_by_market"), dict) else None
    if not isinstance(row, dict):
        return {"bias": "neutral", "strength": 0.0, "headline": ""}
    return {
        "bias": str(row.get("bias") or "unknown"),
        "strength": float(row.get("strength", 0.0) or 0.0),
        "headline": str(row.get("headline") or ""),
    }


def format_ledger_social_whale_context(market: str, state: dict[str, Any]) -> str:
    """One-line status for ledger: social buzz + whale flow."""
    mku = str(market or "").upper()
    parts: list[str] = []
    buzz = state.get("social_buzz_summary") if isinstance(state.get("social_buzz_summary"), dict) else {}
    for ln in buzz.get("lines") or []:
        if not isinstance(ln, dict):
            continue
        if str(ln.get("market") or "").upper() == mku:
            vel = float(ln.get("velocity_pct_1h", 0.0) or 0.0)
            if abs(vel) >= 5.0:
                parts.append(f"🔥 Social Buzz {vel:+.0f}%")
            if bool(ln.get("high_interest")):
                parts.append("🔥 High interest")
            break
    wf = whale_bias_for_market(mku, state)
    bias = str(wf.get("bias") or "neutral")
    st = float(wf.get("strength", 0.0) or 0.0)
    if bias == "inflow" and st >= 0.25:
        parts.append("🐳 Large inflow (exchange)")
    elif bias == "outflow" and st >= 0.25:
        parts.append("💎 Heavy outflow (bullish)")
    elif bias == "mixed" and st >= 0.25:
        parts.append("🐳 Mixed whale flow")
    if not parts:
        return "—"
    return " · ".join(parts)[: 280]


def build_trade_decision_context(market: str, state: dict[str, Any]) -> dict[str, Any]:
    """Bundle for RL decide(): whale bias + conflict hints."""
    wf = whale_bias_for_market(market, state)
    buzz_line = ""
    buzz = state.get("social_buzz_summary") if isinstance(state.get("social_buzz_summary"), dict) else {}
    for ln in buzz.get("lines") or []:
        if isinstance(ln, dict) and str(ln.get("market") or "").upper() == str(market).upper():
            buzz_line = str(ln.get("headline") or "")
            break
    return {
        "market": str(market).upper(),
        "whale_bias": str(wf.get("bias") or "neutral"),
        "whale_strength": float(wf.get("strength", 0.0) or 0.0),
        "whale_headline": str(wf.get("headline") or ""),
        "social_headline": buzz_line,
    }


def apply_whale_attention_blend(
    gate: Sequence[float] | Any,
    feature_names: list[str],
    *,
    whale_weight: float | None = None,
) -> Any:
    """Blend attention gate so whale_pressure channel gets fixed mass (default 0.25).

    Geen numpy in deze module: portal kan ``trading_core`` laden. RL-worker zet het
    resultaat om met ``np.asarray(..., dtype=np.float32)``.
    """
    w = float(os.getenv("WHALE_ATTENTION_WEIGHT", "0.25") if whale_weight is None else whale_weight)
    w = max(0.0, min(0.6, w))
    try:
        idx = int(feature_names.index("whale_pressure"))
    except ValueError:
        return gate
    seq = list(gate) if hasattr(gate, "__iter__") and not isinstance(gate, (str, bytes)) else []
    g = [float(x) for x in seq]
    if not g:
        return g
    n = len(g)
    wh = [0.0] * n
    wh[idx] = 1.0
    out = [(1.0 - w) * g[i] + w * wh[i] for i in range(n)]
    s = sum(out)
    if s <= 1e-12:
        return g
    return [float(x / s) for x in out]
