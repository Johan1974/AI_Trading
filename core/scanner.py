"""
Dynamic Elite-8 scanner for Bitvavo EUR pairs.

Milestone: Historical Potential & Regime Filtering
- Anchors (Top-3): BTC-EUR, ETH-EUR, SOL-EUR.
- Next-5 movers: most volatile among liquid + large-cap candidates that pass quality filters:
  - Daily trend: latest close > SMA-200
  - 30d momentum: positive
  - Longevity: >= 365 daily candles available (~1 year listed)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import requests


@dataclass
class ScanResult:
    selected_top: list[dict[str, Any]]
    candidates_top: list[dict[str, Any]]


# Retained as curated large-cap reference universe for fallback filtering.
PILLAR_ROTATION_BASES: tuple[str, ...] = (
    "ADA",
    "AVAX",
    "XRP",
    "LINK",
    "DOT",
    "ATOM",
    "NEAR",
    "ARB",
    "OP",
    "LTC",
    "DOGE",
    "BCH",
    "APT",
    "FIL",
    "INJ",
    "TIA",
    "SEI",
    "SUI",
)

# Fallback if CoinGecko is unavailable: bases generally above ~500M EUR market cap (curated, conservative).
FALLBACK_LARGE_CAP_BASES: frozenset[str] = frozenset(
    {
        "BTC",
        "ETH",
        "SOL",
        "XRP",
        "ADA",
        "AVAX",
        "LINK",
        "DOT",
        "ATOM",
        "NEAR",
        "ARB",
        "OP",
        "LTC",
        "DOGE",
        "BCH",
        "APT",
        "FIL",
        "INJ",
        "TIA",
        "SEI",
        "SUI",
        "MATIC",
        "POL",
        "TRX",
        "UNI",
        "AAVE",
        "CRV",
        "MKR",
        "SNX",
        "COMP",
        "GRT",
        "FET",
        "RENDER",
        "IMX",
        "STX",
    }
)

EXCLUDED_BASES: frozenset[str] = frozenset({"USDT", "USDC", "DAI", "TUSD", "USDE", "EURC", "BUSD"})


def _base_from_market(market: str) -> str:
    m = str(market or "").upper()
    if "-" in m:
        return m.split("-", 1)[0]
    return m


def _fetch_coingecko_market_cap_eur_by_symbol() -> dict[str, float]:
    """Map BASE (upper) -> market_cap in EUR. Empty on failure."""
    out: dict[str, float] = {}
    try:
        url = "https://api.coingecko.com/api/v3/coins/markets"
        params = {
            "vs_currency": "eur",
            "order": "market_cap_desc",
            "per_page": 250,
            "page": 1,
            "sparkline": "false",
        }
        resp = requests.get(url, params=params, timeout=25)
        if resp.status_code != 200:
            return out
        data = resp.json()
        if not isinstance(data, list):
            return out
        for row in data:
            if not isinstance(row, dict):
                continue
            sym = str(row.get("symbol") or "").upper().strip()
            cap = row.get("market_cap")
            if not sym or cap is None:
                continue
            try:
                out[sym] = max(float(cap), 0.0)
            except (TypeError, ValueError):
                continue
    except Exception:
        return {}
    return out


def _enrich_with_4h_volatility(
    base_url: str,
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    now = datetime.now(timezone.utc)
    since = int((now - timedelta(hours=4)).timestamp() * 1000)
    ranked: list[dict[str, Any]] = []
    for row in rows:
        market = str(row["market"])
        try:
            candles_resp = requests.get(
                f"{base_url}/v2/{market}/candles",
                params={"interval": "1m", "start": since, "limit": 240},
                timeout=20,
            )
            candles_resp.raise_for_status()
            candles = candles_resp.json()
            if not isinstance(candles, list) or not candles:
                continue
            highs = [float(c[2]) for c in candles if isinstance(c, list) and len(c) >= 4]
            lows = [float(c[3]) for c in candles if isinstance(c, list) and len(c) >= 4]
            if not highs or not lows:
                continue
            high_4h = max(highs)
            low_4h = min(lows)
            move_pct = 0.0 if low_4h <= 0 else ((high_4h - low_4h) / low_4h) * 100.0
            ranked.append(
                {
                    **row,
                    "high_4h": high_4h,
                    "low_4h": low_4h,
                    "move_pct_4h": round(move_pct, 3),
                }
            )
        except Exception:
            continue
    ranked.sort(key=lambda x: x.get("move_pct_4h", 0.0), reverse=True)
    return ranked


def _fetch_daily_candles(base_url: str, market: str, days: int = 420) -> list[list[Any]]:
    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)
    try:
        resp = requests.get(
            f"{base_url}/v2/{market}/candles",
            params={"interval": "1d", "start": start_ms, "end": end_ms, "limit": min(1000, days + 10)},
            timeout=25,
        )
        resp.raise_for_status()
        rows = resp.json()
        if not isinstance(rows, list):
            return []
        return [r for r in rows if isinstance(r, list) and len(r) >= 5]
    except Exception:
        return []


def _enrich_with_quality(base_url: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach health metrics and selection reason to each row."""
    out: list[dict[str, Any]] = []
    for row in rows:
        market = str(row.get("market", "")).upper()
        daily = _fetch_daily_candles(base_url, market, days=420)
        closes = [float(r[4]) for r in daily if len(r) >= 5]
        listing_days = len(closes)
        longevity_ok = listing_days >= 365
        sma_200 = 0.0
        trend_ok = False
        if len(closes) >= 200:
            window = closes[-200:]
            sma_200 = float(sum(window) / max(1, len(window)))
            trend_ok = closes[-1] > sma_200
        momentum_30d = 0.0
        momentum_ok = False
        if len(closes) >= 31 and closes[-31] > 0:
            momentum_30d = ((closes[-1] - closes[-31]) / closes[-31]) * 100.0
            momentum_ok = momentum_30d > 0.0
        quality_score = int(trend_ok) + int(momentum_ok) + int(longevity_ok)
        pass_quality = quality_score >= 3
        reasons: list[str] = []
        if trend_ok:
            reasons.append("SMA-200 support")
        if momentum_ok:
            reasons.append("Positive 30d momentum")
        if longevity_ok:
            reasons.append("1y listed")
        if not reasons:
            reasons = ["Rejected: long-term downtrend / weak history"]
        out.append(
            {
                **row,
                "quality_score": quality_score,
                "quality_trend_ok": trend_ok,
                "quality_momentum_ok": momentum_ok,
                "quality_longevity_ok": longevity_ok,
                "sma_200": round(sma_200, 8) if sma_200 > 0 else 0.0,
                "momentum_30d_pct": round(momentum_30d, 3),
                "listing_days": listing_days,
                "is_long_term_downtrend": bool(not trend_ok),
                "quality_reasons": reasons,
                "selection_reason": " + ".join(reasons),
                "passes_quality": pass_quality,
            }
        )
    return out


class DynamicVolatilityScanner:
    def __init__(self, base_url: str = "https://api.bitvavo.com") -> None:
        self.base_url = base_url.rstrip("/")

    def scan(self, top_volume_count: int = 30, elite_count: int = 8) -> ScanResult:
        top_volume_count = int(top_volume_count)
        elite_count = max(1, int(elite_count))
        top_pool = max(50, int(os.getenv("SCANNER_TOP_VOLUME_POOL", "50") or 50))
        mover_min_vol = float(os.getenv("SCANNER_MOVER_MIN_VOLUME_EUR", "1000000") or 1_000_000.0)
        min_mcap_eur = float(os.getenv("SCANNER_MIN_MARKET_CAP_EUR", "500000000") or 500_000_000.0)

        markets_resp = requests.get(f"{self.base_url}/v2/markets", timeout=20)
        markets_resp.raise_for_status()
        markets_data = markets_resp.json()
        ticker_resp = requests.get(f"{self.base_url}/v2/ticker/24h", timeout=20)
        ticker_resp.raise_for_status()
        ticker_data = ticker_resp.json()
        ticker_map = {
            str(row.get("market")): row
            for row in ticker_data
            if isinstance(row, dict) and row.get("market")
        }

        eur_pairs: list[dict[str, Any]] = []
        for market in markets_data:
            if not isinstance(market, dict):
                continue
            symbol = str(market.get("market", "")).upper()
            if not symbol.endswith("-EUR") or str(market.get("status", "")) != "trading":
                continue
            tk = ticker_map.get(symbol, {})
            eur_pairs.append(
                {
                    "market": symbol,
                    "volume_quote_24h": float(tk.get("volumeQuote", 0.0) or 0.0),
                    "last_price": float(tk.get("last", 0.0) or 0.0),
                    "price_change_pct_24h": float(tk.get("priceChangePercentage", 0.0) or 0.0),
                }
            )
        eur_pairs.sort(key=lambda x: x["volume_quote_24h"], reverse=True)
        top_by_volume = eur_pairs[: max(50, top_pool)]

        cap_by_base = _fetch_coingecko_market_cap_eur_by_symbol()
        use_cap = bool(cap_by_base)

        def passes_mcap(row: dict[str, Any]) -> bool:
            base = _base_from_market(str(row.get("market", "")))
            if use_cap:
                cap = float(cap_by_base.get(base, 0.0) or 0.0)
                return cap >= min_mcap_eur
            return base in FALLBACK_LARGE_CAP_BASES

        # --- Anchors: BTC, ETH, SOL (always present when available) ---
        anchor_set = {"BTC-EUR", "ETH-EUR", "SOL-EUR"}
        anchors: list[dict[str, Any]] = []
        by_market = {str(r["market"]).upper(): dict(r) for r in top_by_volume}

        for pm in ("BTC-EUR", "ETH-EUR", "SOL-EUR"):
            row = by_market.get(pm)
            if row:
                r = dict(row)
                r["is_pillar"] = True
                r["pillar_kind"] = "core"
                anchors.append(r)
        anchors = _enrich_with_quality(self.base_url, anchors)
        for a in anchors:
            a["selection_reason"] = f"Anchor: {a.get('selection_reason') or 'Core market'}"

        anchor_markets = {str(p["market"]).upper() for p in anchors}

        # --- Next-5 Movers: must pass historical quality + mcap + liquidity ---
        mover_candidates = [r for r in top_by_volume if str(r["market"]).upper() not in anchor_markets]
        mover_candidates = [r for r in mover_candidates if float(r.get("volume_quote_24h", 0.0) or 0.0) >= mover_min_vol]
        mover_candidates = [r for r in mover_candidates if _base_from_market(str(r.get("market", ""))) not in EXCLUDED_BASES]
        mover_candidates = [r for r in mover_candidates if passes_mcap(r)]
        mover_candidates = _enrich_with_quality(self.base_url, mover_candidates)
        mover_candidates = [r for r in mover_candidates if bool(r.get("passes_quality")) and not bool(r.get("is_long_term_downtrend"))]

        enriched = _enrich_with_4h_volatility(self.base_url, mover_candidates)
        movers = enriched[: max(0, elite_count - len(anchors))]
        for m in movers:
            m["is_pillar"] = False
            m["pillar_kind"] = "mover"
            m["selection_reason"] = (
                f"Chosen: {m.get('selection_reason', 'quality pass')} + "
                f"High volatility ({float(m.get('move_pct_4h', 0.0)):.2f}% 4h)"
            )

        selected = anchors + movers
        # Dedupe by market, preserve order
        seen: set[str] = set()
        deduped: list[dict[str, Any]] = []
        for r in selected:
            mk = str(r.get("market", "")).upper()
            if not mk or mk in seen:
                continue
            seen.add(mk)
            deduped.append(r)

        # Pad to elite_count from volume-ranked large-cap if short
        if len(deduped) < elite_count:
            for r in top_by_volume:
                mk = str(r["market"]).upper()
                if mk in seen:
                    continue
                if _base_from_market(mk) in EXCLUDED_BASES:
                    continue
                if not passes_mcap(r):
                    continue
                rr = _enrich_with_quality(self.base_url, [dict(r)])[0]
                if not bool(rr.get("passes_quality")):
                    continue
                rr.setdefault("move_pct_4h", 0.0)
                rr["is_pillar"] = False
                rr["pillar_kind"] = "fill"
                rr["selection_reason"] = f"Fallback quality pick: {rr.get('selection_reason', 'quality pass')}"
                deduped.append(rr)
                seen.add(mk)
                if len(deduped) >= elite_count:
                    break

        deduped = deduped[:elite_count]
        candidates = _enrich_with_quality(self.base_url, top_by_volume[: max(elite_count, top_volume_count)])
        candidates_top = _enrich_with_4h_volatility(self.base_url, candidates)
        return ScanResult(selected_top=deduped, candidates_top=candidates_top)
