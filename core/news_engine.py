"""
Social momentum & CryptoCompare social stats.

This module complements `app.services.news_engine.NewsEngineService` (RSS + news freshness).
It tracks CryptoCompare **social** metrics for Elite / quality-listed markets and derives:
- **Social velocity** (% change vs ~1h baseline of a composite activity score)
- **High interest** when velocity exceeds a threshold (default 300%)
- **Regime labels** vs short-term price drift for RL confirmation / divergence hints
"""

from __future__ import annotations

import math
import os
import time
from collections import deque
from datetime import datetime, timedelta
from typing import Any, Callable

import requests

from app.datetime_util import UTC


def market_allowed_for_social(row: dict[str, Any] | None) -> bool:
    """Only Elite anchors or scanner-passed quality movers get social signal wiring."""
    if not row or not isinstance(row, dict):
        return False
    if bool(row.get("is_pillar")):
        return True
    if bool(row.get("passes_quality")):
        return True
    return False


def _num(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def _deep_find(d: Any, *candidates: str, default: float = 0.0) -> float:
    """Breadth-first search for first numeric value matching candidate keys (case-insensitive)."""
    if not isinstance(d, dict):
        return default
    lower_map = {str(k).lower(): k for k in d.keys()}
    for name in candidates:
        k = lower_map.get(name.lower())
        if k is not None:
            return _num(d.get(k), default)
    stack: list[Any] = [d]
    seen: set[int] = set()
    while stack:
        cur = stack.pop()
        if not isinstance(cur, dict):
            continue
        cid = id(cur)
        if cid in seen:
            continue
        seen.add(cid)
        lm = {str(k).lower(): k for k in cur.keys()}
        for name in candidates:
            kk = lm.get(name.lower())
            if kk is not None:
                return _num(cur.get(kk), default)
        for v in cur.values():
            if isinstance(v, dict):
                stack.append(v)
    return default


def _parse_social_coin_blob(blob: dict[str, Any]) -> dict[str, float]:
    """Extract CC social fields defensively across possible payload shapes."""
    tw = blob.get("Twitter") if isinstance(blob.get("Twitter"), dict) else {}
    rd = blob.get("Reddit") if isinstance(blob.get("Reddit"), dict) else {}
    followers = _deep_find(tw, "followers", "followers_24hr", default=0.0)
    if followers <= 0.0:
        followers = _deep_find(blob, "twitter_followers", default=0.0)
    tw_change = _deep_find(
        tw,
        "followers_24hr_diff",
        "followers_24hour_diff",
        "followers_24h_diff",
        "followers_delta",
        "following_24hr_diff",
        default=0.0,
    )
    if abs(tw_change) < 1e-9:
        tw_change = _deep_find(blob, "twitter_followers_24hr_diff", default=0.0)
    reddit_posts_ph = _deep_find(rd, "posts_per_hour", "posts_per_h", "posts_per_hr", default=0.0)
    if reddit_posts_ph <= 0.0:
        reddit_posts_ph = _deep_find(blob, "reddit_posts_per_hour", default=0.0)
    reddit_active = _deep_find(rd, "active_users", "active_users_24h", "active", default=0.0)
    if reddit_active <= 0.0:
        reddit_active = _deep_find(blob, "reddit_active_users", default=0.0)
    return {
        "twitter_followers": followers,
        "twitter_followers_change_24h": tw_change,
        "reddit_posts_per_hour": reddit_posts_ph,
        "reddit_active_users": reddit_active,
    }


def _composite_activity(m: dict[str, float]) -> float:
    """Single scalar for velocity tracking (posts + active users + twitter growth)."""
    posts = max(0.0, m.get("reddit_posts_per_hour", 0.0))
    active = max(0.0, m.get("reddit_active_users", 0.0))
    tw_d = m.get("twitter_followers_change_24h", 0.0)
    return max(0.0, posts * 500.0 + active * 3.0 + max(0.0, tw_d) * 0.05)


class SocialVelocityTracker:
    """In-process history for composite social activity + news mention counts."""

    def __init__(self, maxlen: int = 48) -> None:
        self._composite: dict[str, deque[tuple[float, float]]] = {}
        self._news: dict[str, deque[tuple[float, float]]] = {}
        self._maxlen = max(8, int(maxlen))

    def _push(self, store: dict[str, deque[tuple[float, float]]], base: str, ts: float, val: float) -> None:
        q = store.setdefault(base.upper(), deque(maxlen=self._maxlen))
        q.append((ts, val))
        while len(q) > 1 and ts - q[0][0] > 7200:
            q.popleft()

    def push_api(self, base: str, composite: float) -> None:
        self._push(self._composite, base, time.time(), composite)

    def push_news(self, base: str, composite: float) -> None:
        self._push(self._news, base, time.time(), composite)

    def velocity_pct(self, base: str, composite_now: float, *, min_age_sec: float) -> tuple[float, float]:
        """Return (velocity_pct, baseline) using oldest sample at least min_age_sec old."""
        b = base.upper()
        q = self._composite.get(b)
        if not q or len(q) < 2:
            return 0.0, 0.0
        now = time.time()
        baseline = 0.0
        for ts, val in q:
            if now - ts >= min_age_sec:
                baseline = float(val)
        if baseline <= 1e-9:
            return 0.0, baseline
        return ((composite_now - baseline) / baseline) * 100.0, baseline

    def news_velocity_pct(self, base: str, news_now: float, *, min_age_sec: float) -> float:
        b = base.upper()
        q = self._news.get(b)
        if not q or len(q) < 2:
            return 0.0
        now = time.time()
        baseline = 0.0
        for ts, val in q:
            if now - ts >= min_age_sec:
                baseline = float(val)
        if baseline <= 1e-9:
            return 0.0
        return ((news_now - baseline) / baseline) * 100.0


_TRACKER = SocialVelocityTracker()


def _fetch_social_latest_map(api_key: str, bases: list[str]) -> dict[str, dict[str, Any]]:
    key = str(api_key or "").strip()
    if not key or not bases:
        return {}
    out: dict[str, dict[str, Any]] = {}
    chunk = max(1, min(30, int(os.getenv("SOCIAL_CC_FSYMS_CHUNK", "12") or 12)))
    url = "https://min-api.cryptocompare.com/data/social/coin/latest"
    for i in range(0, len(bases), chunk):
        part = bases[i : i + chunk]
        try:
            resp = requests.get(
                url,
                params={"fsyms": ",".join(part), "extraParams": "AI_Trading", "api_key": key},
                timeout=20,
            )
            if resp.status_code != 200:
                continue
            payload = resp.json() if resp.content else {}
            if str(payload.get("Response", "")).upper() != "SUCCESS":
                continue
            data = payload.get("Data")
            if isinstance(data, dict):
                for sym, blob in data.items():
                    if isinstance(blob, dict):
                        out[str(sym).upper()] = blob
        except Exception:
            continue
    return out


def _count_news_mentions(
    rows: list[dict[str, Any]],
    base: str,
    *,
    minutes: int = 60,
    now: datetime | None = None,
) -> int:
    now = now or datetime.now(UTC)
    cutoff = now - timedelta(minutes=max(5, int(minutes)))
    needle = str(base or "").upper().strip()
    if not needle:
        return 0
    n = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        ts = row.get("publishedAt") or row.get("published_at") or row.get("published_on")
        if not ts:
            continue
        try:
            dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            dt = dt.astimezone(UTC)
        except Exception:
            continue
        if dt < cutoff:
            continue
        blob = f"{row.get('title','')} {row.get('description','')} {row.get('summary','')}".upper()
        if needle in blob or f"{needle} " in blob or f" {needle}" in blob:
            n += 1
    return n


def _price_change_1h(
    price_hist: dict[str, Any],
    market: str,
    last_price: float,
) -> float:
    """Approx 1h price drift using snapshots stored each refresh (epoch seconds)."""
    mk = str(market).upper()
    q = price_hist.get(mk)
    if not isinstance(q, deque):
        q = deque(maxlen=120)
        price_hist[mk] = q
    now = time.time()
    q.append((now, float(last_price)))
    ref_px = 0.0
    for ts, px in q:
        if now - ts >= float(os.getenv("SOCIAL_PRICE_LOOKBACK_SEC", "3600") or 3600):
            ref_px = float(px)
    if ref_px <= 1e-12:
        return 0.0
    return (float(last_price) - ref_px) / ref_px


def _classify_regime(price_ret_1h: float, velocity_pct: float) -> str:
    pv = float(price_ret_1h)
    sv = float(velocity_pct)
    thr_p = float(os.getenv("SOCIAL_REGIME_PRICE_EPS", "0.0008") or 0.0008)
    thr_s = float(os.getenv("SOCIAL_REGIME_VELOCITY_EPS", "8.0") or 8.0)
    if pv > thr_p and sv > thr_s:
        return "bullish_correlated"
    if pv > thr_p and sv < -thr_s:
        return "divergent_risk"
    if pv < -thr_p and sv > thr_s:
        return "bearish_with_hype"
    return "neutral"


def refresh_social_momentum_state(
    *,
    state: dict[str, Any],
    fetch_fresh_news: Callable[[str | None, int], list[dict[str, Any]]],
    cryptocompare_key: str | None,
    active_markets: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Mutates `state` keys:
    - `social_momentum_by_market` (per MARKET-EUR)
    - `social_buzz_summary` (UI + WS)
    - `_social_price_hist` internal deque map
    """
    price_hist = state.setdefault("_social_price_hist", {})
    if not isinstance(price_hist, dict):
        price_hist = {}
        state["_social_price_hist"] = price_hist

    cc_key = str(cryptocompare_key or "").strip()
    news_rows: list[dict[str, Any]] = []
    try:
        news_rows = list(fetch_fresh_news(cc_key or None, 120) or [])
    except Exception:
        news_rows = []

    eligible = [m for m in active_markets if isinstance(m, dict) and market_allowed_for_social(m)]
    bases = []
    for m in eligible:
        mk = str(m.get("market", "")).upper()
        if "-" in mk:
            bases.append(mk.split("-", 1)[0])
    bases = sorted({b for b in bases if b})

    social_map = _fetch_social_latest_map(cc_key, bases) if cc_key else {}
    min_age = float(os.getenv("SOCIAL_VELOCITY_MIN_AGE_SEC", "3500") or 3500)
    hi_thr = float(os.getenv("SOCIAL_HIGH_INTEREST_PCT", "300") or 300.0)

    by_market: dict[str, Any] = {}
    buzz_lines: list[dict[str, Any]] = []

    for m in eligible:
        mk = str(m.get("market", "")).upper()
        base = mk.split("-", 1)[0] if "-" in mk else mk
        blob = social_map.get(base.upper(), {})
        parsed = _parse_social_coin_blob(blob) if isinstance(blob, dict) else {
            "twitter_followers": 0.0,
            "twitter_followers_change_24h": 0.0,
            "reddit_posts_per_hour": 0.0,
            "reddit_active_users": 0.0,
        }
        comp_api = _composite_activity(parsed)
        news_n = _count_news_mentions(news_rows, base, minutes=60)
        comp_news = float(news_n) * 40.0
        news_vel = _TRACKER.news_velocity_pct(base, comp_news, min_age_sec=min_age)
        _TRACKER.push_news(base, comp_news)

        vel_api, baseline = _TRACKER.velocity_pct(base, comp_api, min_age_sec=min_age)
        _TRACKER.push_api(base, comp_api)
        velocity_pct = max(vel_api, news_vel)
        high_interest = bool(velocity_pct >= hi_thr)

        last_px = _num(m.get("last_price"), 0.0)
        price_ret = _price_change_1h(price_hist, mk, last_px) if last_px > 0 else 0.0
        regime = _classify_regime(price_ret, velocity_pct)

        line_pct = round(float(velocity_pct), 1)
        text = f"[{base}] Social Buzz is up {line_pct}% in the last hour"
        if velocity_pct < 0:
            text = f"[{base}] Social Buzz is down {abs(line_pct)}% in the last hour"
        if high_interest:
            text = f"[{base}] HIGH INTEREST: social activity +{line_pct}% vs baseline (1h)"

        buzz_lines.append(
            {
                "market": mk,
                "symbol": base,
                "velocity_pct_1h": round(float(velocity_pct), 3),
                "high_interest": high_interest,
                "regime": regime,
                "headline": text,
                "reddit_posts_per_hour": round(parsed["reddit_posts_per_hour"], 4),
                "reddit_active_users": round(parsed["reddit_active_users"], 2),
                "twitter_followers_change_24h": round(parsed["twitter_followers_change_24h"], 2),
                "news_mentions_60m": int(news_n),
                "price_ret_1h_approx": round(float(price_ret), 6),
                "source": "cryptocompare+news" if cc_key else "news_only",
            }
        )

        by_market[mk] = {
            "enabled": True,
            "velocity_pct_1h": float(velocity_pct),
            "high_interest": high_interest,
            "regime": regime,
            "price_ret_1h_approx": float(price_ret),
            "reddit_posts_per_hour": float(parsed["reddit_posts_per_hour"]),
            "reddit_active_users": float(parsed["reddit_active_users"]),
            "twitter_followers_change_24h": float(parsed["twitter_followers_change_24h"]),
            "news_mentions_60m": int(news_n),
            "composite_api": float(comp_api),
            "composite_news": float(comp_news),
            "baseline_api": float(baseline),
            "ticker_line": text,
        }

    buzz_lines.sort(key=lambda r: abs(float(r.get("velocity_pct_1h") or 0.0)), reverse=True)
    state["social_momentum_by_market"] = by_market
    state["social_buzz_summary"] = {
        "updated_at": datetime.now(UTC).isoformat(),
        "lines": buzz_lines[:12],
        "markets": len(by_market),
    }
    return state["social_buzz_summary"]


def apply_social_overlay_to_rl_row(
    last: dict[str, float],
    ticker: str,
    social_by_market: dict[str, Any] | None,
    active_row: dict[str, Any] | None,
) -> tuple[dict[str, float], str]:
    """
    Light-touch adjustment on already-normalized RL features (post `normalize_rl_feature_frame`).
    Keeps observation dimension stable; boosts `social_volume` / nudges sentiment on regimes.
    """
    if not market_allowed_for_social(active_row):
        return last, ""
    snap = (social_by_market or {}).get(str(ticker).upper())
    if not isinstance(snap, dict) or not snap.get("enabled"):
        return last, ""

    out = dict(last)
    vel = float(snap.get("velocity_pct_1h", 0.0) or 0.0)
    regime = str(snap.get("regime") or "neutral")
    hi = bool(snap.get("high_interest"))

    boost = math.tanh(max(-3.0, min(3.0, vel / 120.0))) * float(os.getenv("SOCIAL_RL_SOCIAL_VOL_BOOST", "0.35") or 0.35)
    out["social_volume"] = float(min(1.0, max(-1.0, float(out.get("social_volume", 0.0) or 0.0) + boost)))

    sent = float(out.get("sentiment_score", 0.0) or 0.0)
    conf = float(out.get("news_confidence", 0.0) or 0.0)
    note_parts: list[str] = []
    if regime == "bullish_correlated":
        sent += float(os.getenv("SOCIAL_RL_SENTIMENT_BULL", "0.06") or 0.06)
        note_parts.append("Social confirms price (bullish correlation)")
    elif regime == "divergent_risk":
        conf *= float(os.getenv("SOCIAL_RL_CONF_DIVERGENCE_MULT", "0.88") or 0.88)
        sent -= float(os.getenv("SOCIAL_RL_SENTIMENT_DIV", "0.04") or 0.04)
        note_parts.append("Divergence: price up vs cooling social (fake-out risk)")
    elif regime == "bearish_with_hype":
        note_parts.append("Bearish tape but rising social (hype / relief risk)")
    if hi:
        note_parts.append("HIGH INTEREST social spike")
    out["sentiment_score"] = float(max(-1.0, min(1.0, sent)))
    out["news_confidence"] = float(max(0.0, min(1.0, conf)))

    note = " ".join(note_parts).strip()
    return out, note
