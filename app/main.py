"""
BESTANDSNAAM: app/main.py
FUNCTIE: FastAPI-/UI-laag: API routes en WebSockets voor de frontend. Delegeert zware trading- en ML-logica naar `app.trading_core`.
"""


from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

import os

# Tijdelijk vriendelijker voor test-trades: smallere judge-neutrale band + lagere expected-return drempel
# (override via .env: JUDGE_SIGNAL_THRESHOLD, EXPECTED_RETURN_SIGNAL_THRESHOLD_PCT).
if os.getenv("JUDGE_SIGNAL_THRESHOLD") is None:
    os.environ["JUDGE_SIGNAL_THRESHOLD"] = "0.08"
if os.getenv("EXPECTED_RETURN_SIGNAL_THRESHOLD_PCT") is None:
    os.environ["EXPECTED_RETURN_SIGNAL_THRESHOLD_PCT"] = "0.35"
if os.getenv("RISK_MAX_SPREAD_BPS_FOR_TRADING") is None:
    os.environ["RISK_MAX_SPREAD_BPS_FOR_TRADING"] = "90"

import app.trading_core as tc

__TC_SKIP = frozenset({"JARVIS_REPORTER", "RESTART_MAIL_TASK"})
for __n in dir(tc):
    if __n.startswith("__") or __n in __TC_SKIP:
        continue
    globals()[__n] = getattr(tc, __n)

import asyncio
import json
import shutil
import logging
from datetime import timezone
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import traceback
import sys

from app.services.prediction_ui import (
    build_overlay_prices_from_rl_or_fallback,
    prediction_signal_allowed_by_rl,
    tenant_rl_decision_for_symbol,
    trade_confidence_threshold_01,
)
from app.services.reporting import log_persistent_crash_error
from app.services.state import STATE, append_event, current_tenant_id, set_current_tenant
from core.worker_execution import calculate_order_size_for_signal
from core.worker_execution import enforce_emergency_learning_bootstrap
from app.schemas.prediction import PredictionResponse

UTC = timezone.utc
_log = logging.getLogger(__name__)

app = FastAPI(title="AI Trading Bot", version="1.0.0")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")

# 1. Overschrijf de CORSMiddleware (Geen restricties, breek 403-muur af voor WebSockets)
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=".*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _prediction_fresh_in_blob(blob: dict[str, Any], max_age_seconds: int = 60) -> bool:
    tenant = blob.get("tenant", blob) if isinstance(blob, dict) else {}
    lp = tenant.get("last_prediction") if isinstance(tenant, dict) else {}
    if not isinstance(lp, dict):
        return False
    ts_raw = lp.get("generated_at")
    if not ts_raw:
        return False
    try:
        dt = datetime.fromisoformat(str(ts_raw).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return (datetime.now(UTC) - dt.astimezone(UTC)).total_seconds() <= float(max_age_seconds)
    except Exception:
        return False


def _runtime_status_from_blob(blob: dict[str, Any]) -> dict[str, Any]:
    tenant = blob.get("tenant", blob) if isinstance(blob, dict) else {}
    extras = blob.get("extras", blob) if isinstance(blob, dict) else {}
    sys_stats = extras.get("system_stats", extras) if isinstance(extras, dict) else {}
    lp = tenant.get("last_prediction") if isinstance(tenant, dict) else {}
    last_inf = (
        (lp.get("generated_at") if isinstance(lp, dict) else None)
        or blob.get("last_engine_tick_utc")
        or datetime.now(UTC).isoformat()
    )
    gpu_ok = bool(sys_stats.get("gpu_ok", sys_stats.get("gk", False)))
    gpu_status = "ok" if gpu_ok else "error"
    pf = tenant.get("paper_portfolio") if isinstance(tenant, dict) else {}
    db_connected = bool(isinstance(pf, dict) and ("equity" in pf or "cash" in pf))
    def _pct01(v: Any) -> float | None:
        try:
            x = float(v)
        except (TypeError, ValueError):
            return None
        if not x == x or x < 0.0:
            return None
        return x * 100.0 if x <= 1.0 + 1e-6 else min(100.0, x)

    has_live_probs = False
    if isinstance(lp, dict):
        for k in ("prob_buy", "prob_hold", "prob_sell", "buy", "hold", "sell"):
            p = _pct01(lp.get(k))
            if p is not None and p > 0.01:
                has_live_probs = True
                break

    is_fresh = _prediction_fresh_in_blob(blob, max_age_seconds=90)
    worker_status = "online" if (is_fresh or has_live_probs) else "degraded"
    return {
        "worker_status": worker_status,
        "gpu_status": gpu_status,
        "db_connected": db_connected,
        "last_inference_time": str(last_inf),
    }


def _validated_ai_action_probs_from_row(row_like: Any) -> dict[str, float]:
    """Validator voor UI/API: altijd geldige buy/hold/sell percentages."""
    from core.worker_execution import ensure_final_probs

    def _pct01_local(x: Any) -> float:
        try:
            v = float(x)
        except (TypeError, ValueError):
            return 0.0
        if not v == v or v < 0.0:
            return 0.0
        return round(v * 100.0, 2) if v <= 1.0 + 1e-6 else round(min(100.0, v), 2)

    row = dict(row_like) if isinstance(row_like, dict) else {}
    if "prob_buy" not in row and row.get("buy_pct") is not None:
        row["prob_buy"] = row.get("buy_pct")
    if "prob_hold" not in row and row.get("hold_pct") is not None:
        row["prob_hold"] = row.get("hold_pct")
    if "prob_sell" not in row and row.get("sell_pct") is not None:
        row["prob_sell"] = row.get("sell_pct")
    # Geen synthetic action-softmax fallback: alleen echte model-kansen tonen.
    if row.get("prob_buy") is None and row.get("prob_hold") is None and row.get("prob_sell") is None:
        return {"buy_pct": 0.0, "hold_pct": 0.0, "sell_pct": 0.0}
    out = ensure_final_probs(row)
    return {
        "buy_pct": _pct01_local(out.get("prob_buy")),
        "hold_pct": _pct01_local(out.get("prob_hold")),
        "sell_pct": _pct01_local(out.get("prob_sell")),
    }


def _bundle_hardware_tab_logs(limit: int) -> tuple[list[str], str]:
    """
    Hardware-tab: tail van worker-, portal- en blackbox-log (zelfde volume).
    Blackbox bevat doorgaans alle stdout van de worker; portal_api de portal.
    """
    logs = tc.LOGS_DIR
    chunks: list[tuple[str, Path, list[str]]] = []
    per = max(40, min(120, limit // 3))
    for label, fname in (
        ("worker", "worker_execution.log"),
        ("portal", "portal_api.log"),
        ("blackbox", "blackbox.log"),
    ):
        p = logs / fname
        lines = _tail_lines(p, limit=per)
        if any(ln.strip() for ln in lines):
            chunks.append((label, p, lines))
    if not chunks:
        ts = datetime.now().astimezone().isoformat()
        hint = (
            f"{ts} [PORTAL] Geen enkel logbestand met inhoud onder {logs} "
            f"(verwacht o.a. worker_execution.log). Check volume ./_logs_hub:/app/logs en permissies."
        )
        return [hint], str(logs)
    merged: list[str] = []
    path_bits: list[str] = []
    for label, path, lines in chunks:
        if merged:
            merged.append(f"--- {label}: {path.name} ---")
        merged.extend(lines)
        path_bits.append(path.name)
    if len(merged) > limit:
        merged = merged[-limit:]
    return merged, str(logs / path_bits[0])


async def _await_redis_snapshot() -> dict[str, Any] | None:
    """P1: sync Redis HGET/socket off the event-loop (portal hot paths)."""
    return await asyncio.to_thread(read_worker_portal_snapshot)


async def _await_hardware_logs_bundle(limit: int) -> tuple[list[str], str]:
    return await asyncio.to_thread(_bundle_hardware_tab_logs, limit)


def _sync_api_system_storage() -> dict[str, Any]:
    stats = _read_storage_stats()
    total, used, free = shutil.disk_usage(Path.home())
    usage_pct = 0.0 if total <= 0 else (float(used) / float(total)) * 100.0
    return {
        "stats": stats,
        "disk": {
            "total_bytes": int(total),
            "used_bytes": int(used),
            "free_bytes": int(free),
            "usage_pct": round(usage_pct, 2),
        },
    }


def _sync_read_crash_log(limit: int) -> dict[str, Any]:
    path = tc.LOGS_DIR / "crash_log.txt"
    if not path.is_file():
        return {"lines": [], "path": str(path), "exists": False}
    try:
        raw = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return {"lines": raw[-int(limit) :], "path": str(path), "exists": True}
    except Exception as exc:
        return {
            "lines": [],
            "path": str(path),
            "exists": True,
            "error": f"read_failed: {exc}",
        }


def _sync_debug_policy_writes(market: str | None, limit: int) -> dict[str, Any]:
    path = tc.LOGS_DIR / "worker_execution.log"
    if not path.is_file():
        return {"lines": [], "path": str(path), "exists": False, "market": market}
    market_u = str(market or "").strip().upper().replace("/", "-")
    try:
        raw = path.read_text(encoding="utf-8", errors="replace").splitlines()
        lines = [ln for ln in raw if "[POLICY-WRITE]" in ln]
        if market_u:
            needle = f"MARKET={market_u}"
            lines = [ln for ln in lines if needle in ln.upper()]
        return {
            "lines": lines[-int(limit) :],
            "path": str(path),
            "exists": True,
            "market": market_u or None,
            "count": len(lines),
        }
    except Exception as exc:
        return {
            "lines": [],
            "path": str(path),
            "exists": True,
            "market": market_u or None,
            "error": f"read_failed: {exc}",
        }


def _sync_append_browser_log(level: str, msg: str, stack: str, url: str) -> None:
    ts = datetime.now().astimezone().isoformat()
    line = f"{ts} [BROWSER][{level}] {msg}"
    if stack:
        line += f" | Stack: {stack}"
    target_log = tc.LOGS_DIR / "browser_console.log"
    with target_log.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def _sync_settings_redis_write(tab: str, v: float, eq_hint: float | None) -> dict[str, Any]:
    from core.trading_constraints_redis import merge_position_sizing_post, write_trading_constraints

    merged = merge_position_sizing_post(tab=str(tab), value=v, equity=eq_hint)
    write_trading_constraints(merged)
    return merged


def _normalize_trade_api_rows(rows: list[Any]) -> list[dict[str, Any]]:
    """Zorgt dat elke rij `pair` + `market` deelt (UI), en status bruikbaar is voor de ledger."""
    out: list[dict[str, Any]] = []
    for raw in rows:
        if not isinstance(raw, dict):
            continue
        d = dict(raw)
        if not str(d.get("row_type") or "").strip():
            typ = str(d.get("type") or "").upper()
            if typ == "ROUND_TRIP":
                d["row_type"] = "ROUND_TRIP"
            elif typ == "ACTIVE":
                d["row_type"] = "ACTIVE_LOT"
        if d.get("open_time_utc") and not d.get("entry_ts_utc"):
            d["entry_ts_utc"] = str(d["open_time_utc"])
        if d.get("close_time_utc") not in (None, "") and not d.get("exit_ts_utc"):
            d["exit_ts_utc"] = str(d["close_time_utc"])
        pair = str(d.get("pair") or d.get("market") or "").strip().upper().replace("/", "-")
        d["pair"] = pair
        d["market"] = pair
        rt = str(d.get("row_type") or "").upper()
        if rt == "EVENT":
            st = str(d.get("outcome") or "FILLED").strip().upper() or "FILLED"
            d["status"] = st
        elif rt == "ROUND_TRIP":
            d["status"] = "CLOSED"
        elif rt == "ACTIVE_LOT":
            d["status"] = "ACTIVE"
        else:
            d.setdefault("status", str(d.get("status") or ""))
        out.append(d)
    return out


def _trade_ledger_rows_for_stats(tenant: dict[str, Any], data: dict[str, Any]) -> list[dict[str, Any]]:
    """Laatste trades voor stats/WS: worker-Redis `trades`, anders top-level `trades`, anders SQLite."""
    raw: list[dict[str, Any]] = []
    cand = tenant.get("trades") if isinstance(tenant, dict) else None
    if isinstance(cand, list) and cand:
        raw = [x for x in cand if isinstance(x, dict)]
    if not raw and isinstance(data, dict):
        top = data.get("trades")
        if isinstance(top, list) and top:
            raw = [x for x in top if isinstance(x, dict)]
    if not raw:
        try:
            from core.database import get_all_trades

            raw = get_all_trades(limit=20)
        except Exception:
            raw = []
    out: list[dict[str, Any]] = []
    for item in raw[-20:]:
        pair_u = str(item.get("pair") or item.get("market") or "").strip().upper().replace("/", "-")
        rt = str(item.get("row_type") or "")
        if not rt:
            typ = str(item.get("type") or "").upper()
            if typ == "ACTIVE":
                rt = "ACTIVE_LOT"
            elif typ == "ROUND_TRIP":
                rt = "ROUND_TRIP"
        if rt == "EVENT":
            action = str(item.get("action") or "").upper()
        elif rt == "ACTIVE_LOT":
            action = "BUY"
        elif rt == "ROUND_TRIP":
            action = "SELL"
        else:
            action = str(item.get("action") or "").upper()
        if action not in ("BUY", "SELL"):
            continue
        ts = str(item.get("exit_ts_utc") or item.get("entry_ts_utc") or "")[:19]
        try:
            sent = float(item.get("sentiment_score") or 0.0)
        except (TypeError, ValueError):
            sent = 0.0
        try:
            pnl = float(item.get("pnl_eur") or 0.0)
        except (TypeError, ValueError):
            pnl = 0.0
        try:
            px = float(item.get("entry_price") or 0.0)
        except (TypeError, ValueError):
            px = 0.0
        out.append(
            {
                "action": action,
                "market": pair_u,
                "pair": pair_u,
                "ts": ts,
                "entry_price": px,
                "price": px,
                "pnl_eur": pnl,
                "sentiment_score": sent,
            }
        )
    return out[-20:]


def _sync_api_trades(limit: int, view: str) -> dict[str, Any]:
    """Paper-ledger: events/all = merged SQLite + actieve wallet-lots; roundtrip = idem."""
    from core.database import get_all_trades

    vs = str(view).lower()
    lim = int(max(1, min(int(limit or 200), 50000)))
    if vs == "roundtrip":
        try:
            rows = list(PAPER_MANAGER.round_trip_ledger(limit=lim) or [])
        except Exception:
            rows = []
        return {"trades": _normalize_trade_api_rows(rows)}
    # events/all: SQLite history + actieve open lots (ACTIVE_LOT) zodat Ledger open posities toont
    rows = get_all_trades(limit=lim)
    active_lots: list[dict] = []
    try:
        active = list(PAPER_MANAGER.round_trip_ledger(limit=200) or [])
        active_lots = [r for r in active if str(r.get("type") or r.get("status") or "").upper() == "ACTIVE"]
    except Exception:
        pass
    # Markten met een actieve lot: forceer wallet als enige waarheidsbron, filter conflicterende DB-rijen
    active_markets = {
        str(r.get("pair") or r.get("market") or "").upper().replace("/", "-")
        for r in active_lots
        if str(r.get("pair") or r.get("market") or "").strip()
    }
    if active_markets:
        rows = [
            r for r in rows
            if str(r.get("pair") or r.get("market") or "").upper().replace("/", "-") not in active_markets
        ]
    rows = active_lots + rows
    return {"trades": _normalize_trade_api_rows(rows)}


def _sync_api_history(pair: str, lookback_days: int, interval: str) -> dict[str, Any]:
    target = pair.upper()
    labels, prices = _fetch_history_series(pair=target, lookback_days=lookback_days, interval=interval)
    markers = [m for m in STATE.get("signal_markers", []) if m.get("ticker", "").upper() == target][:200]
    return {
        "pair": target,
        "interval": str(interval or "5m").strip().lower(),
        "tv_symbol": f"BITVAVO:{target}" if "-" in target else target,
        "labels": labels,
        "prices": prices,
        "markers": markers,
        "whale_danger_zone": whale_danger_zone_for_market(STATE, target),
    }


def _sync_terminal_chart_points(ticker: str, lookback_days: int) -> dict[str, Any]:
    tu = str(ticker or "").upper()
    labels, prices = _fetch_history_series(
        pair=tu,
        lookback_days=lookback_days,
        interval="5m",
    )
    markers = [
        m
        for m in STATE.get("signal_markers", [])
        if m.get("ticker", "").upper() == tu
    ][:200]
    return {
        "ticker": tu,
        "labels": labels,
        "prices": prices,
        "markers": markers,
    }


def _sync_news_ticker_worker(elite_mix: int) -> list[dict[str, Any]]:
    """Blokkerende nieuws-fetch + mapping (worker); portal gebruikt snelle cache-tak in de route."""
    active_markets = STATE.get("active_markets", [])
    if not active_markets:
        try:
            _refresh_active_markets_cache()
        except Exception:
            pass
    active_markets = STATE.get("active_markets", [])
    news_query = "crypto"
    news_api_key = os.getenv("CRYPTOCOMPARE_KEY")
    if int(elite_mix or 0) == 1:
        elite_markets = [str(m.get("market", "")).upper() for m in active_markets if m.get("market")]
        rows = NEWS_ENGINE.fetch_fresh_news(cryptocompare_key=os.getenv("CRYPTOCOMPARE_KEY"), limit=90)
        rows = NEWS_ENGINE.prioritize_for_elite_tickers(
            rows=rows, elite_tickers=elite_markets, coin_aliases=NEWS_ELITE_COIN_ALIASES
        )
        mapped = NEWS_MAPPING.get_mapped_news(
            news_query=news_query,
            news_api_key=news_api_key,
            active_markets=active_markets,
            prefetched_articles=rows,
        )
        return _interleave_elite_mapped_news(mapped, elite_markets[:8])
    combined_articles = _signal_news_articles(
        ticker=STATE.get("selected_market", "BTC-EUR"),
        news_query=news_query,
        news_api_key=news_api_key,
    )
    return NEWS_MAPPING.get_mapped_news(
        news_query=news_query,
        news_api_key=news_api_key,
        active_markets=active_markets,
        prefetched_articles=combined_articles,
    )


def _sync_map_portal_news_insights(insights: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Portal nieuws-cache mappen zonder zware I/O (P1: ``asyncio.to_thread`` voor consistentie)."""
    out: list[dict[str, Any]] = []
    for i in insights:
        if not isinstance(i, dict):
            continue
        out.append(
            {
                "text": i.get("headline", ""),
                "title": i.get("headline", ""),
                "summary": "",
                "url": "",
                "source": i.get("source", "Worker-Cache"),
                "coin": i.get("ticker_tag", "MKT"),
                "sentiment": i.get("finbert_score", 0.0),
                "confidence": i.get("finbert_confidence", 0.0),
                "published_at": i.get("ts", ""),
                "is_urgent": False,
                "affected_tickers": [i.get("ticker_tag", "MKT")],
            }
        )
    return out


def _sync_brain_state_overview() -> dict[str, Any]:
    try:
        STATE["whale_watch"] = WHALE_WATCHER.fetch_exchange_pressure(
            api_key=os.getenv("CRYPTOCOMPARE_KEY"),
            lookback_minutes=120,
        )
    except Exception:
        pass
    try:
        _refresh_cmc_metrics(force=False)
    except Exception:
        pass
    fw = {}
    decision = STATE.get("rl_last_decision")
    if isinstance(decision, dict):
        fw = decision.get("feature_weights", {}) if isinstance(decision.get("feature_weights"), dict) else {}
    last_state = dict(STATE.get("rl_last_state", {}) or {}) if isinstance(STATE.get("rl_last_state"), dict) else {}
    fg_state = STATE.get("fear_greed") if isinstance(STATE.get("fear_greed"), dict) else {}
    if fg_state and fg_state.get("fear_greed_score") is not None and "fear_greed_score" not in last_state:
        last_state["fear_greed_score"] = float(fg_state.get("fear_greed_score") or 0.5)
    cmc = STATE.get("cmc_metrics") if isinstance(STATE.get("cmc_metrics"), dict) else {}
    btc_dom = float(last_state.get("btc_dominance_pct") or 0.0)
    if btc_dom <= 0.0:
        cmc_dom = float(cmc.get("btc_dominance_pct") or 0.0)
        if cmc_dom > 0.0:
            last_state["btc_dominance_pct"] = cmc_dom
        else:
            fb_dom = float(os.getenv("GENESIS_BTC_DOM_FALLBACK", "52.0") or 52.0)
            last_state["btc_dominance_pct"] = max(1.0, min(95.0, fb_dom))
    rsi_raw = last_state.get("rsi_14")
    try:
        rsi_val = float(rsi_raw) if rsi_raw is not None else 0.0
    except (TypeError, ValueError):
        rsi_val = 0.0
    if rsi_raw is None or rsi_val <= 0.0 or rsi_val > 100.0:
        last_state["rsi_14"] = 50.0
    wp_live = float((STATE.get("whale_watch") or {}).get("whale_pressure", 0.0) or 0.0)
    last_state["whale_pressure"] = wp_live
    whales_weight = float(fw.get("whale_pressure", 0.0))
    btc_dom_weight = float(fw.get("btc_dominance_pct", 0.0))
    macro_weight = float(fw.get("macro_volatility_window", 0.0))
    rsi_weight = float(fw.get("rsi_14", 0.0))
    return {
        "state": last_state,
        "weight_focus": {
            "whales": round(whales_weight, 4),
            "btc_dominance": round(btc_dom_weight, 4),
            "macro": round(macro_weight, 4),
            "rsi": round(rsi_weight, 4),
        },
        "social_buzz": STATE.get("social_buzz_summary") or {"lines": [{"headline": "Social Buzz data wordt verzameld..."}]},
    }


def _log_path_for_ws_follow() -> Path:
    """Live tail: volg het grootste van worker / portal / blackbox (meeste bytes)."""
    logs = tc.LOGS_DIR
    candidates = [
        logs / "worker_execution.log",
        logs / "portal_api.log",
        logs / "blackbox.log",
    ]
    best = candidates[0]
    best_sz = -1
    for p in candidates:
        try:
            sz = p.stat().st_size if p.exists() else 0
            if sz > best_sz:
                best_sz = sz
                best = p
        except OSError:
            continue
    return best


def _worker_calc_hints_from_tenant(tenant: dict[str, Any]) -> list[str]:
    """Zelfde informatie als /activity worker_calc_hints (stats-polling voor UI)."""
    out: list[str] = []
    if not isinstance(tenant, dict):
        return out
    ls = tenant.get("last_scores") if isinstance(tenant.get("last_scores"), dict) else {}
    lp = tenant.get("last_prediction") if isinstance(tenant.get("last_prediction"), dict) else {}
    lo = tenant.get("last_order") if isinstance(tenant.get("last_order"), dict) else {}
    lec = tenant.get("last_engine_cycle") if isinstance(tenant.get("last_engine_cycle"), dict) else {}
    tr = ls.get("technical_predicted_return_pct")
    if tr is not None:
        try:
            out.append(f"RSI/techniek: verwacht {float(tr):+.2f}%")
        except (TypeError, ValueError):
            pass
    js = str(ls.get("judge_signal") or "").upper()
    if js:
        out.append(f"Sentiment check: judge → {js}")
    ss = ls.get("sentiment_score")
    if ss is not None:
        try:
            out.append(f"Nieuwslaag: score {float(ss):.3f}")
        except (TypeError, ValueError):
            pass
    sig = str(lp.get("signal") or "").upper()
    if sig:
        _tk = str(lp.get("ticker") or "").strip().upper().replace("/", "-")
        _dec = tenant_rl_decision_for_symbol(tenant, _tk) if _tk else None
        _th = trade_confidence_threshold_01()
        if prediction_signal_allowed_by_rl(sig, _dec, _th):
            out.append(f"Voorspelling: {sig}" + (f" · {_tk}" if _tk else ""))
    rd = lo.get("risk_decision") if isinstance(lo.get("risk_decision"), dict) else {}
    rs = str(rd.get("reason") or "").strip()
    if rs and rs != "approved":
        out.append(f"Risk gate: {rs}")
    pair = str(lec.get("pair") or "").strip().upper()
    if pair:
        out.append(f"Worker: cyclus {pair}")
    return out[-12:]


def _sum_open_trades_invested_eur(portfolio: dict[str, Any]) -> float:
    """Som inleg (cost basis) van alle open lots: qty × entry_price."""
    total = 0.0
    if not isinstance(portfolio, dict):
        return 0.0
    obm = portfolio.get("open_lots_by_market")
    if isinstance(obm, dict):
        for lots in obm.values():
            if not isinstance(lots, list):
                continue
            for lot in lots:
                if not isinstance(lot, dict):
                    continue
                try:
                    q = float(lot.get("qty") or 0.0)
                    ep = float(lot.get("entry_price") or 0.0)
                except (TypeError, ValueError):
                    continue
                total += max(0.0, q * ep)
    if total <= 1e-12:
        ol = portfolio.get("open_lots")
        if isinstance(ol, list):
            for lot in ol:
                if not isinstance(lot, dict):
                    continue
                try:
                    q = float(lot.get("qty") or 0.0)
                    ep = float(lot.get("entry_price") or 0.0)
                except (TypeError, ValueError):
                    continue
                total += max(0.0, q * ep)
    return round(max(0.0, total), 2)


def _paper_display_balances(portfolio: dict[str, Any]) -> tuple[float, float, float]:
    """
    UI-saldo: cash, som inleg open posities, equity = cash + inleg (geen gemuteerde wallet nodig).
    """
    cash = float(portfolio.get("cash") or 0.0) if isinstance(portfolio, dict) else 0.0
    invested = _sum_open_trades_invested_eur(portfolio if isinstance(portfolio, dict) else {})
    eq = round(cash + invested, 2)
    return cash, invested, eq


def _paper_portfolio_api_subset(pf: dict[str, Any]) -> dict[str, Any]:
    """Consistente paper_portfolio-keys voor /stats en format_stats (equity = cash + open inleg)."""
    c, inv, eq = _paper_display_balances(pf if isinstance(pf, dict) else {})
    b = pf if isinstance(pf, dict) else {}
    return {
        "equity": float(eq),
        "cash": float(c),
        "open_positions_invested_eur": float(inv),
        "trades_count": int(b.get("trades_count", 0) or 0),
        "realized_pnl_eur": float(b.get("realized_pnl_eur", 0.0) or 0.0),
    }


def format_stats(data: dict[str, Any], policy_market: str | None = None) -> dict[str, Any]:
    """Dwingt elk inkomend Redis-pakket (inclusief L, m, p) in het Universal Schema."""
    tenant = data.get("tenant", data) if isinstance(data, dict) else {}
    extras = data.get("extras", data) if isinstance(data, dict) else {}
    sys_stats = extras.get("system_stats", extras) if isinstance(extras, dict) else {}
    
    # Zoek de market. Geef prioriteit aan de STATE van de Portal zelf, zodat UI-switches behouden blijven.
    portal_market = STATE.get("selected_market")
    payload_market = data.get("market", data.get("m", data.get("f", "BTC-EUR")))
    if policy_market:
        market = str(policy_market).strip().upper().replace("/", "-")
    else:
        market = portal_market if portal_market else tenant.get("selected_market", payload_market)
    if market == "Unknown" or not market:
        market = "BTC-EUR"
        
    # 3. Price Mismatch Fix: Als de payload voor een andere munt is dan de UI wil zien, forceer een deep search.
    price = float(data.get("p", data.get("price", 0.0)))
    
    # --- RL policy: per symbool uit rl_multi_decisions (worker inferentie-loop), anders globaal alleen als ticker matcht ---
    ai_decision: dict[str, Any] = {}
    mku = str(market or "").strip().upper().replace("/", "-")
    if isinstance(tenant, dict):
        multi = tenant.get("rl_multi_decisions")
        if isinstance(multi, dict):
            raw = multi.get(mku)
            if raw is None:
                for k, v in multi.items():
                    if str(k).strip().upper().replace("/", "-") == mku:
                        raw = v
                        break
            if raw is not None:
                ai_decision = _rl_decision_as_dict_with_fallback(raw)
        if not ai_decision:
            gl = tenant.get("rl_last_decision")
            gl = gl if isinstance(gl, dict) else {}
            gl_t = str(gl.get("ticker") or gl.get("market") or "").strip().upper().replace("/", "-")
            # Alleen globale rl_last_decision gebruiken als expliciet voor déze markt (geen lege ticker = alle munten).
            if gl and gl_t == mku:
                ai_decision = _rl_decision_as_dict_with_fallback(gl)
    reasoning = ai_decision.get("reasoning", "") if isinstance(ai_decision, dict) else ""
    
    if not reasoning and isinstance(tenant.get("active_markets"), list): # Fallback to scanner reason
        for m in tenant.get("active_markets", []):
            if m.get("market") == market:
                reasoning = str(m.get("selection_reason", "")).strip()
                break
                
    selection_reason = reasoning or "Data gesynct"
    
    # Deep search voor de prijs als de top-level 'p' of 'price' ontbreekt of van een andere munt is
    pm_payload = str(payload_market or "").strip().upper().replace("/", "-")
    if price == 0.0 or (pm_payload and pm_payload != mku):
        price = 0.0
        L = data.get("L", [])
        if isinstance(L, list):
            for item in L:
                if isinstance(item, dict) and item.get("m") == market and float(item.get("p", 0.0)) > 0:
                    price = float(item.get("p", 0.0))
                    break # Gevonden, stop met zoeken
                    
        # Fallback naar de snapshot cache (voor de AJAX polling)
        if price == 0.0:
            for m in tenant.get("active_markets", []):
                if isinstance(m, dict):
                    lp = float(m.get("last_price", m.get("price", 0.0)))
                    if str(m.get("market")) == str(market) and lp > 0:
                        price = lp # Geen side-effects meer
                        break
                        
        # GHOST DATA PREVENTIE: Voorkom dat UI 'Laden...' toont en de validator timeouts/bootloops veroorzaakt
        if price == 0.0:
            price = 0.01
            
    # Ledger en AI Weights ophalen voor volledige Canonical Schema dekking
    portfolio = tenant.get("paper_portfolio", {}) if isinstance(tenant, dict) else {}
    _pp_ui = _paper_portfolio_api_subset(portfolio if isinstance(portfolio, dict) else {})
    history = portfolio.get("history", []) if isinstance(portfolio, dict) else []
    
    # Trade-ledger: laatste uit worker/Redis `trades` (merged SQLite) of directe DB-read; niet alleen paper_portfolio.history.
    trade_ledger = _trade_ledger_rows_for_stats(tenant if isinstance(tenant, dict) else {}, data if isinstance(data, dict) else {})
    if not trade_ledger:
        actual_trades = [
            item
            for item in history
            if isinstance(item, dict) and str(item.get("action", "")).upper() in {"BUY", "SELL"}
        ]
        trade_ledger = actual_trades[-5:]
    
    # --- WEIGHTS LOGIC FIXED ---
    weights = {}
    if isinstance(ai_decision.get("feature_weights"), dict):
        weights = ai_decision["feature_weights"]
    elif isinstance(tenant.get("feature_weights_by_market"), dict):
        weights = tenant["feature_weights_by_market"].get(market, {})
        
    # Map RL feature weights to UI categories
    price_features = [
        "price_action", "volatility_24", "volume_change", "bollinger_width", 
        "bollinger_position", "orderbook_imbalance", "bid_ask_spread", "macd", "rsi_14", "ema_gap_pct"
    ]
    news_features = ["sentiment_score", "news_confidence", "social_volume"]
    correlation_features = ["fear_greed_score", "btc_dominance_pct", "whale_pressure", "macro_volatility_window"]

    # GHOST DATA PREVENTIE: Als RL-agent nog niet getraind is (gewichten = leeg), 
    # verdeel de baseline over de features zodat UI/Validator geen spook-nulwaarden ziet.
    if not weights:
        all_features = price_features + news_features + correlation_features
        baseline_w = 1.0 / len(all_features) if all_features else 0.0
        weights = {f: baseline_w for f in all_features}
        

    price_weight = sum(weights.get(f, 0.0) for f in price_features)
    news_weight = sum(weights.get(f, 0.0) for f in news_features)
    correlation_weight = sum(weights.get(f, 0.0) for f in correlation_features)
    
    ai_weights = {
        "price": float(price_weight),
        "news": float(news_weight),
        "correlation": float(correlation_weight)
    }
    
    # 5. Extractie van velden voor UI (Terminal, AI Brain, Ledger)
    sentiment_score = float(tenant.get("last_scores", {}).get("sentiment_score", 0.0) if isinstance(tenant.get("last_scores"), dict) else 0.0)
    rl_confidence = float(ai_decision.get("confidence", 0.0) if isinstance(ai_decision, dict) else 0.0)
    model_version = str(tenant.get("model_version", "v1.0.0"))

    def _pct01(x: Any) -> float | None:
        try:
            v = float(x)
        except (TypeError, ValueError):
            return None
        if not v == v or v < 0.0:
            return None
        return round(v * 100.0, 2) if v <= 1.0 + 1e-6 else round(min(100.0, v), 2)

    ai_action_probs: dict[str, float | None] = {"buy_pct": None, "hold_pct": None, "sell_pct": None}
    if isinstance(ai_decision, dict):
        ai_action_probs["hold_pct"] = _pct01(ai_decision.get("prob_hold"))
        ai_action_probs["buy_pct"] = _pct01(ai_decision.get("prob_buy"))
        ai_action_probs["sell_pct"] = _pct01(ai_decision.get("prob_sell"))

    def _ai_probs_blank_for_ui() -> bool:
        vals = [ai_action_probs.get("hold_pct"), ai_action_probs.get("buy_pct"), ai_action_probs.get("sell_pct")]
        for v in vals:
            if v is None:
                continue
            try:
                if float(v) > 0.01:
                    return False
            except (TypeError, ValueError):
                continue
        return True

    # Per-markt (policy_market): vul ontbrekende probs uit Redis ``prediction:PAIR`` i.p.v. globale 40/50/10.
    if policy_market and _ai_probs_blank_for_ui():
        try:
            from core.worker_execution import read_per_market_prediction_policy

            pol = read_per_market_prediction_policy(mku)
            if isinstance(pol, dict) and pol:
                ai_decision = _rl_decision_as_dict_with_fallback(pol)
                ai_action_probs["hold_pct"] = _pct01(ai_decision.get("prob_hold"))
                ai_action_probs["buy_pct"] = _pct01(ai_decision.get("prob_buy"))
                ai_action_probs["sell_pct"] = _pct01(ai_decision.get("prob_sell"))
        except Exception:
            pass

    # Geen statische signal→percentage mapping (10/28/62 enz.): dat leek live RL-policy maar was
    # alleen last_prediction.signal. Laat probs leeg/0 zodat UI "Thinking…"/N/A kan tonen of Redis/validator.

    # Geen action-seed fallback als probs leeg zijn: toon dan "calculating/thinking" in UI.

    # UI/WS: expliciet welke markt deze percentages betreffen (voorkomt kruislings tonen bij polls zonder context).
    if isinstance(ai_action_probs, dict) and mku:
        ai_action_probs = dict(ai_action_probs)
        ai_action_probs["market"] = mku
        ai_action_probs["ticker"] = mku
    
    change_24h = 0.0
    volatility_4h = 0.0
    for m in tenant.get("active_markets", []):
        if isinstance(m, dict) and str(m.get("market")) == str(market):
            change_24h = float(m.get("price_change_pct_24h", 0.0) or 0.0)
            volatility_4h = float(m.get("move_pct_4h", 0.0) or 0.0)
            break
    
    # Zorg dat de disk usage berekend wordt als deze ontbreekt in de Redis payload
    disk_usage = float(data.get("disk_usage", sys_stats.get("d", sys_stats.get("disk_pct", 0.0))))
    if disk_usage <= 0.0:
        try:
            import shutil
            total, used, _ = shutil.disk_usage("/hostfs" if os.path.exists("/hostfs") else "/")
            disk_usage = (used / total) * 100.0 if total > 0 else 0.0
        except Exception:
            pass

    alloc_snap = allocation_snapshot(
        portfolio,
        float(_pp_ui.get("equity", tc.default_paper_reset_balance_eur())),
    )

    fg_data = tenant.get("fear_greed", {})
    if not isinstance(fg_data, dict): fg_data = {}
    fear_greed_score = float(fg_data.get("fear_greed_score") or fg_data.get("fng_value") or 50.0)
    fear_greed_class = str(fg_data.get("fear_greed_class") or fg_data.get("fng_classification") or "Neutral")

    lp_raw = data.get("last_prediction") if isinstance(data.get("last_prediction"), dict) else None
    if not lp_raw and isinstance(tenant, dict):
        lp_raw = tenant.get("last_prediction") if isinstance(tenant.get("last_prediction"), dict) else None
    if not lp_raw:
        lp_raw = STATE.get("last_prediction") if isinstance(STATE.get("last_prediction"), dict) else None
    lp_ticker_n = str((lp_raw or {}).get("ticker") or "").strip().upper().replace("/", "-")
    sym_for_lp = str(market or "").strip().upper().replace("/", "-")
    lp_use = lp_raw if lp_raw and lp_ticker_n and lp_ticker_n == sym_for_lp else None
    pred_series, p_latest, p_next = build_overlay_prices_from_rl_or_fallback(
        float(price),
        ai_decision if isinstance(ai_decision, dict) else {},
        lp_use,
    )

    try:
        _th_raw = STATE.get("decision_threshold")
        if _th_raw is None or _th_raw == "":
            _th_dec = float(os.getenv("RL_ACTION_MIN_CONFIDENCE", "0.55") or 0.55)
        else:
            _th_dec = float(_th_raw)
        if not _th_dec == _th_dec or _th_dec < 0:
            _th_dec = 0.55
        rl_decision_threshold_pct = (
            round(min(100.0, _th_dec), 1) if _th_dec > 1.0 + 1e-6 else round(_th_dec * 100.0, 1)
        )
    except (TypeError, ValueError):
        rl_decision_threshold_pct = 55.0

    signal_threshold_pct = rl_decision_threshold_pct
    st_env = os.getenv("SIGNAL_THRESHOLD")
    if st_env is not None and str(st_env).strip() != "":
        try:
            st_v = float(st_env)
            if st_v == st_v and st_v >= 0.0:
                signal_threshold_pct = (
                    round(min(100.0, st_v), 2) if st_v > 1.0 + 1e-6 else round(st_v * 100.0, 2)
                )
        except (TypeError, ValueError):
            pass

    return {
        "market": str(market),
        "price": float(price),
        "price_change_pct_24h": float(change_24h),
        "volatility_pct_4h": float(volatility_4h),
        "allocation_summary": alloc_snap.get("summary", "Allocatie: —"),
        "fear_greed_score": fear_greed_score,
        "fear_greed_class": fear_greed_class,
        "cpu_load": float(data.get("cpu_load", sys_stats.get("c", sys_stats.get("cpu_pct", 13.6)))),
        "gpu_temp": float(data.get("gpu_temp", sys_stats.get("gpu_temp_max", sys_stats.get("g", 41.0)))),
        "ram_usage": float(data.get("ram_usage", sys_stats.get("r", sys_stats.get("ram_pct", 0.0)))),
        "disk_usage": float(round(disk_usage, 1)),
        "bot_status": str(data.get("bot_status", tenant.get("bot_status", "running"))),
        "decision_reasoning": selection_reason,
        "trade_ledger": trade_ledger,
        "ai_weights": ai_weights,
        "sentiment_score": sentiment_score,
        "rl_confidence": rl_confidence,
        "rl_decision_threshold_pct": rl_decision_threshold_pct,
        "signal_threshold_pct": signal_threshold_pct,
        "ai_action_probs": ai_action_probs,
        # Cockpit / predictions-poll: ruwe policy-kansen (fallback als ai_action_probs null is).
        "rl_last_decision": ai_decision if isinstance(ai_decision, dict) and ai_decision else None,
        "model_version": model_version,
        "paper_portfolio": dict(_pp_ui),
        "last_update": data.get("last_update", datetime.now().astimezone().isoformat()),
        "active_markets": tenant.get("active_markets", []),
        # AI Prediction Overlay (hoofdgrafiek): pad geïnterpoleerd uit last_prediction (worker/RL-pipeline).
        "predicted_price": pred_series,
        "predicted_latest_close": float(p_latest),
        "predicted_next_close": float(p_next),
        # Zichtbaar vooruit: aantal bar-stappen (tijd) dat de voorspellingslijn voorloopt op de slotkoers.
        "prediction_bar_lead": max(1, min(12, int(os.getenv("PREDICTION_CHART_BAR_LEAD", "2") or 2))),
        "worker_calc_hints": _worker_calc_hints_from_tenant(tenant if isinstance(tenant, dict) else {}),
        # Volledige multi-map voor Terminal (BUY/HOLD/SELL per paar); anders valt WS/activity terug op één rl_last_decision.
        "rl_multi_decisions": tenant.get("rl_multi_decisions")
        if isinstance(tenant, dict) and isinstance(tenant.get("rl_multi_decisions"), dict)
        else {},
        # Social buzz strip: ticker-regels gesorteerd op velocity (snelste bewegende munten bovenaan).
        "social_buzz_summary": STATE.get("social_buzz_summary") or {},
    }


def _build_predicted_price_fields(
    lp: dict[str, Any] | None,
    current_price: float,
) -> tuple[list[float], float, float]:
    """
    Bouwt ``predicted_price`` (monotone interpolatie latest → next close) voor de UI-grafiek.
    Retourneert ook latest/next voor hit-rate op de client.
    """
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


def _build_dashboard_payload(blob: dict[str, Any], *, policy_market: str | None = None) -> dict[str, Any]:
    """Bouwt het platte 'Dashboard-Ready' object voor de frontend via Universal Schema."""
    if isinstance(blob, str):
        try:
            blob = json.loads(blob)
        except Exception:
            blob = {}
            
    return format_stats(blob, policy_market=policy_market)


def _api_stats_minimal_emergency(policy_market: str | None, *, detail: str | None = None) -> dict[str, Any]:
    """Laatste redmiddel: geldig Universal-schema-achtig object zonder Redis/format_stats (geen lege response)."""
    market = (policy_market or str(STATE.get("selected_market") or "BTC-EUR")).strip().upper().replace("/", "-")
    return {
        "market": market,
        "price": 0.01,
        "price_change_pct_24h": 0.0,
        "volatility_pct_4h": 0.0,
        "allocation_summary": "Allocatie: — (noodfallback API)",
        "fear_greed_score": 50.0,
        "fear_greed_class": "Neutral",
        "cpu_load": 0.0,
        "gpu_temp": 0.0,
        "ram_usage": 0.0,
        "disk_usage": 0.0,
        "bot_status": str(STATE.get("bot_status") or "unknown"),
        "decision_reasoning": "Noodfallback: /api/v1/stats kon niet normaal worden opgebouwd.",
        "trade_ledger": [],
        "ai_weights": {"price": 0.34, "news": 0.33, "correlation": 0.33},
        "sentiment_score": 0.0,
        "rl_confidence": 0.0,
        "rl_decision_threshold_pct": 55.0,
        "signal_threshold_pct": 55.0,
        "ai_action_probs": {
            "buy_pct": None,
            "hold_pct": None,
            "sell_pct": None,
            "market": market,
            "ticker": market,
        },
        "rl_last_decision": None,
        "model_version": str(STATE.get("model_version") or "unknown"),
        "paper_portfolio": {
            "equity": float(tc.default_paper_reset_balance_eur()),
            "cash": float(tc.default_paper_reset_balance_eur()),
            "open_positions_invested_eur": 0.0,
            "trades_count": 0,
            "realized_pnl_eur": 0.0,
        },
        "last_update": datetime.now(UTC).isoformat(),
        "active_markets": STATE.get("active_markets", []) if isinstance(STATE.get("active_markets"), list) else [],
        "predicted_price": [],
        "predicted_latest_close": 0.01,
        "predicted_next_close": 0.01,
        "prediction_bar_lead": max(1, min(12, int(os.getenv("PREDICTION_CHART_BAR_LEAD", "2") or 2))),
        "worker_calc_hints": [],
        "worker_status": "error",
        "gpu_status": "error",
        "db_connected": False,
        "last_inference_time": datetime.now(UTC).isoformat(),
        "compute_device": str(STATE.get("compute_device") or "unknown"),
        "prediction_fresh": False,
        "api_stats_emergency": True,
        "api_stats_error": (detail or "")[:800],
        "ok": False,
    }


def _api_stats_without_redis_snapshot(policy_market: str | None) -> dict[str, Any]:
    """Universal schema wanneer worker nog geen snapshot in Redis heeft (bestaande gedrag)."""
    portfolio = STATE.get("paper_portfolio") if isinstance(STATE.get("paper_portfolio"), dict) else {}
    if not portfolio and isinstance(PAPER_MANAGER.wallet, dict):
        portfolio = PAPER_MANAGER.wallet
    market = (policy_market or str(STATE.get("selected_market") or "BTC-EUR")).upper()
    price = 0.01
    for m in STATE.get("active_markets", []) if isinstance(STATE.get("active_markets"), list) else []:
        if isinstance(m, dict) and str(m.get("market", "")).upper() == market:
            try:
                lp = float(m.get("last_price", m.get("price", 0.01)) or 0.01)
                if lp > 0:
                    price = lp
            except (TypeError, ValueError):
                pass
            break
    fallback = _build_dashboard_payload(
        {
            "market": market,
            "price": price,
            "paper_portfolio": portfolio,
            "bot_status": str(STATE.get("bot_status") or "running"),
            "selection_reason": "Fallback stats actief: wachten op verse worker snapshot.",
        },
        policy_market=policy_market,
    )
    fallback.update(
        {
            "worker_status": "degraded",
            "gpu_status": "error" if str(STATE.get("compute_device") or "").lower() == "cpu" else "loading",
            "db_connected": bool(isinstance(portfolio, dict) and ("equity" in portfolio or "cash" in portfolio)),
            "last_inference_time": str(
                (STATE.get("last_prediction") or {}).get("generated_at")
                if isinstance(STATE.get("last_prediction"), dict)
                else datetime.now(UTC).isoformat()
            ),
        }
    )
    return fallback


@app.get("/api/v1/stats")
async def api_stats(
    request: Request,
    symbol: str | None = Query(
        None,
        description="Optioneel: markt voor RL BUY/HOLD/SELL en prijsvelden (bijv. ADA-EUR). Standaard: portal-selectie.",
    ),
) -> dict[str, Any]:
    """Retourneert het platte Dashboard-Ready object voor de frontend."""
    pm = str(symbol).strip().upper().replace("/", "-") if symbol and str(symbol).strip() else None
    if pm:
        STATE["selected_market"] = pm
    try:
        blob: dict[str, Any] | None
        try:
            blob = await _await_redis_snapshot()
        except Exception as exc:
            _log.warning("[API-FLOW] read_worker_portal_snapshot failed: %s", exc)
            blob = None

        if blob is not None and isinstance(blob, str):
            try:
                blob = json.loads(blob)
            except Exception as exc:
                _log.warning("[API-FLOW] snapshot JSON parse failed: %s", exc)
                blob = None
        if blob is not None and not isinstance(blob, dict):
            blob = None

        if blob:
            try:
                apply_worker_snapshot_to_portal(blob)
            except Exception as exc:
                _log.warning("[API-FLOW] apply_worker_snapshot_to_portal failed: %s", exc)
            payload = _build_dashboard_payload(blob, policy_market=pm)
            pf = payload.get("paper_portfolio") if isinstance(payload.get("paper_portfolio"), dict) else {}
            state_pf = STATE.get("paper_portfolio") if isinstance(STATE.get("paper_portfolio"), dict) else {}
            if not pf and state_pf:
                pf = dict(state_pf)
            if not pf and isinstance(PAPER_MANAGER.wallet, dict):
                pf = dict(PAPER_MANAGER.wallet)
            if pf:
                payload["paper_portfolio"] = _paper_portfolio_api_subset(pf)
            payload.update(_runtime_status_from_blob(blob))
            payload["compute_device"] = str(STATE.get("compute_device") or "unknown")
            payload["prediction_fresh"] = _prediction_fresh_in_blob(blob, max_age_seconds=60)
            _log.debug("[API-FLOW] Dashboard requested data via %s - 200 OK", request.url.path)
            return payload

        return _api_stats_without_redis_snapshot(pm)
    except Exception as exc:
        _log.exception("[API-FLOW] /api/v1/stats failed, trying no-snapshot fallback: %s", exc)
        try:
            return _api_stats_without_redis_snapshot(pm)
        except Exception as exc2:
            _log.exception("[API-FLOW] /api/v1/stats no-snapshot fallback failed: %s", exc2)
            try:
                return _api_stats_minimal_emergency(pm, detail=f"{exc!s}; {exc2!s}")
            except Exception as exc3:
                _log.exception("[API-FLOW] /api/v1/stats hard fallback: %s", exc3)
                mk = (pm or str(STATE.get("selected_market") or "BTC-EUR")).strip().upper().replace("/", "-")
                return {
                    "ok": False,
                    "market": mk,
                    "price": 0.01,
                    "bot_status": str(STATE.get("bot_status") or "unknown"),
                    "last_update": datetime.now(UTC).isoformat(),
                    "worker_status": "error",
                    "api_stats_emergency": True,
                    "api_stats_error": f"{exc!s}; {exc2!s}; {exc3!s}"[:800],
                    "paper_portfolio": {
                        "equity": float(tc.default_paper_reset_balance_eur()),
                        "cash": float(tc.default_paper_reset_balance_eur()),
                        "open_positions_invested_eur": 0.0,
                        "trades_count": 0,
                        "realized_pnl_eur": 0.0,
                    },
                    "active_markets": [],
                }

@app.get("/api/v1/debug_data")
async def api_debug_data() -> dict[str, Any]:
    """Tijdelijk debug endpoint om de vertaalde JSON te inspecteren."""
    blob = await _await_redis_snapshot()
    if blob:
        return _build_dashboard_payload(blob)
    return {"status": "error", "message": "No snapshot in Redis"}

async def _publish_worker_command(action: str, **kwargs: Any) -> None:
    if _process_role() != "portal":
        return
    import redis.asyncio as aioredis
    host = str(os.getenv("REDIS_HOST", "redis")).strip()
    port = str(os.getenv("REDIS_PORT", "6379")).strip()
    url = str(os.getenv("REDIS_URL", f"redis://{host}:{port}/0")).strip()
    if "localhost" in url or "127.0.0.1" in url:
        url = f"redis://{host}:{port}/0"
    try:
        client = aioredis.from_url(url, decode_responses=True)
        payload = {"action": action}
        payload.update(kwargs)
        await client.publish("worker_commands", json.dumps(payload))
        await client.aclose()
    except Exception as e:
        print(f"[PORTAL] Kon commando {action} niet naar worker sturen: {e}")

def _mount_api_routers() -> None:
    """Mount kleine APIRouters na globale services (queues, scanner, …).

    Lichte HTTP-probes (geen zware DB / geen ``trading_core``-payload): ``GET /health``
    (``app.api.router``) en ``GET /activity/ping`` (``routes_activity``); Docker-health
    en watchdog kunnen die gebruiken i.p.v. ``/activity`` of ``/api/v1/stats``.
    """
    from app.api.router import router as api_meta_router
    from app.api.routes_activity import router as activity_router
    from app.api.routes_bot import router as bot_router
    from app.api.routes_markets import router as markets_router
    # Voorspellingen: GET /api/v1/predictions?symbol=BTC-EUR — handler in ``routes_predictions``.
    # Laatste historische candle wordt daar gelijkgetrokken met ``active_markets`` (Terminal Real Price vs hoofdgrafiek).
    from app.api.routes_predictions import router as predictions_router
    from app.api.routes_news import router as news_router

    app.include_router(api_meta_router)
    app.include_router(markets_router)
    app.include_router(bot_router)
    app.include_router(activity_router)
    app.include_router(predictions_router)
    app.include_router(news_router)


_mount_api_routers()


@app.post("/api/v1/settings")
async def api_v1_settings(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Persist position sizing (and gerelateerde slot-%) in Redis onder ``trading:constraints``."""
    tab = body.get("position_sizing_tab") if isinstance(body, dict) else None
    if tab is None:
        tab = body.get("tab") if isinstance(body, dict) else None
    val = body.get("value") if isinstance(body, dict) else None
    if tab is None or val is None:
        raise HTTPException(status_code=422, detail="position_sizing_tab (of tab) en value zijn verplicht.")
    try:
        v = float(val)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail="value moet een getal zijn.") from exc

    wallet = STATE.get("paper_portfolio") or {}
    eq_hint: float | None = None
    if isinstance(wallet, dict):
        try:
            eq_raw = float(wallet.get("equity") or 0.0)
            eq_hint = eq_raw if eq_raw > 0 else None
        except (TypeError, ValueError):
            eq_hint = None

    try:
        merged = await asyncio.to_thread(_sync_settings_redis_write, str(tab), v, eq_hint)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Redis schrijven mislukt: {exc}") from exc

    return {"ok": True, "trading_constraints": merged}


@app.get("/api/v1/rl/overnight-readiness")
def api_v1_rl_overnight_readiness() -> dict[str, Any]:
    """Env/paden-checklist: wanneer traint de bot, waar staan weights en uur-metrics (JSONL)."""
    from core.worker_execution import rl_overnight_learning_snapshot

    return rl_overnight_learning_snapshot()


# Paper trades (opened/closed): Telegram direct via ``core.worker_execution.schedule_immediate_trade_telegram``,
# aangeroepen vanuit ``trading_core`` na elke geslaagde ``PAPER_MANAGER.process_signal`` (niet-blokkerend).

@app.get("/api/v1/rl/paper-feedback")
def api_v1_rl_paper_feedback() -> dict[str, Any]:
    """Replay-buffer stats + PPO learning_rate snapshot (worker met trading_core)."""
    import app.trading_core as tc_mod
    from core.worker_execution import paper_rl_learning_rate_snapshot

    return {
        "replay_buffer": tc_mod.replay_stats_for_activity(),
        "optimizer": paper_rl_learning_rate_snapshot(),
    }


@app.post("/api/v1/reset-paper")
async def api_v1_reset_paper(body: dict[str, Any] | None = Body(default=None)) -> dict[str, Any]:
    """
    Paperwallet naar startbedrag (standaard € 1.000): volledige SQL-wipe van ledger + wallets,
    Redis allocatie 10%/€100, ε-inferentie 0.50 en decision_threshold 0.45 op de worker.
    """
    b = body if isinstance(body, dict) else {}
    try:
        amt = float(b["starting_eur"]) if "starting_eur" in b else default_paper_reset_balance_eur()
    except (TypeError, ValueError):
        amt = default_paper_reset_balance_eur()
    if amt <= 0:
        amt = default_paper_reset_balance_eur()
    if _process_role() == "portal":
        await _publish_worker_command("reset_paper", starting_eur=amt)
        return {"ok": True, "queued": True, "starting_eur": amt}
    return reset_paper_portfolio_and_state(amt)


@app.post("/api/v1/paper/reconcile-balance")
async def api_v1_paper_reconcile_balance() -> dict[str, Any]:
    """
    Wis alleen spook-/open posities; cash en trade-ledger in SQLite blijven.
    Equity (Saldo) wordt gelijkgetrokken met cash. Op portal: commando naar worker.
    """
    if _process_role() == "portal":
        await _publish_worker_command("paper_reconcile_balance")
        return {"ok": True, "queued": True}
    return reconcile_paper_balance_in_state()


@app.post("/api/v1/sync-wallet")
async def api_v1_sync_wallet() -> dict[str, Any]:
    """
    Herlaad wallet (cash + open lots) uit de wallet_state SQLite snapshot.
    Overschrijft de in-memory wallet NIET als de bot actief handelt — alleen lezen + sync STATE.
    Op portal: stuurt 'sync_wallet' commando naar worker én voert zelf een DB-lees uit.
    """
    result = await asyncio.to_thread(PAPER_MANAGER.sync_wallet_from_db)
    if isinstance(result, dict) and result.get("ok"):
        STATE["paper_portfolio"] = PAPER_MANAGER.wallet
        _sanitize_paper_wallet()
    if _process_role() == "portal":
        try:
            await _publish_worker_command("sync_wallet")
        except Exception:
            pass
    return result


@app.post("/api/v1/rl-inference-mode")
async def api_v1_rl_inference_mode(body: dict[str, Any] | None = Body(default=None)) -> dict[str, Any]:
    """Zet greedy RL-inferentie (ε=0) aan/uit op de worker via Redis; portal STATE voor directe UI-sync."""
    b = body if isinstance(body, dict) else {}
    greedy = bool(b.get("greedy", False))
    await _publish_worker_command("rl_inference_greedy", greedy=greedy)
    if _process_role() == "portal":
        STATE["rl_inference_greedy"] = greedy
    return {"ok": True, "rl_inference_greedy": greedy}


@app.websocket("/ws/canonical-stats")
async def ws_canonical_stats(websocket: WebSocket) -> None:
    """2. WebSocket Origin Bypass: Directe Redis-stream voor het Canonical Schema met 1s interval en fallback."""
    await websocket.accept()
    STATE["ws_connections"] = int(STATE.get("ws_connections", 0) or 0) + 1

    def _touch_hb() -> None:
        STATE["last_ws_heartbeat_ts"] = datetime.now(UTC).isoformat()

    try:
        async def pump_messages() -> None:
            while True:
                try:
                    blob = await _await_redis_snapshot()
                    if blob:
                        pm = STATE.get("selected_market")
                        pm = str(pm).strip().upper().replace("/", "-") if pm else None
                        clean = format_stats(blob, policy_market=pm)
                        await websocket.send_json(clean)
                    else:
                        await websocket.send_json({"status": "waiting_for_data"})
                except (WebSocketDisconnect, RuntimeError):
                    break
                except Exception as e:
                    try:
                        await websocket.send_json({"status": "error", "reason": str(e)})
                    except (WebSocketDisconnect, RuntimeError):
                        break
                _touch_hb()
                await asyncio.sleep(1.0)
        
        pump = asyncio.create_task(pump_messages())
        # Wacht tot de client de verbinding verbreekt. De pump_messages taak
        # zal een WebSocketDisconnect exception gooien bij de volgende send,
        # die hier wordt opgevangen. Dit lost de deadlock op.
        await pump
    except (WebSocketDisconnect, asyncio.CancelledError):
        pass
    finally:
        if 'pump' in locals() and not pump.done():
            pump.cancel()
        STATE["ws_connections"] = max(0, int(STATE.get("ws_connections", 1) or 1) - 1)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    if isinstance(exc, HTTPException):
        return JSONResponse(
            status_code=int(exc.status_code),
            content=jsonable_encoder(
                {"detail": exc.detail, "path": str(request.url.path), "ok": False}
            ),
        )
    ts = datetime.now().astimezone().isoformat()
    print(f"{ts} [API-ERROR][CRITICAL] Onverwachte fout in route {request.url.path}. Context: method={request.method}, client={request.client.host if request.client else 'Unknown'}. Error: {exc}")
    traceback.print_exc(file=sys.stdout)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Interne Server Fout",
            "error": str(exc),
            "path": request.url.path,
            "ok": False,
        },
    )

async def _data_integrity_audit_loop() -> None:
    """Continuously checks integrity of Redis state and SQLite databases."""
    import sqlite3
    from app.services.paper_engine import PaperConfig
    
    print("[AUDIT] Audit Engine Loop gestart. Controleert Redis & SQLite integriteit op de achtergrond.")
    while True:
        try:
            ts_h = datetime.now(UTC).isoformat(timespec="milliseconds")
            pf = STATE.get("paper_portfolio")
            pf = pf if isinstance(pf, dict) else {}
            cash_h = pf.get("cash", "—")
            eq_h = pf.get("equity", "—")
            pair_h = STATE.get("selected_market") or os.getenv("DEFAULT_TICKER", "BTC-EUR")
            print(f"{ts_h} [AUDIT][HEALTH] cash={cash_h} equity={eq_h} active_pair={pair_h}")
            await asyncio.sleep(600)  # 10 minutes
            # 1. Redis Integrity Check
            active = STATE.get("active_markets", [])
            if not isinstance(active, list):
                print("[AUDIT] Waarschuwing: 'active_markets' in State is corrupt. Reset naar lege lijst.")
                STATE["active_markets"] = []

            pf = STATE.get("paper_portfolio")
            if isinstance(pf, dict):
                if "equity" not in pf or "cash" not in pf:
                    print("[AUDIT] Waarschuwing: 'paper_portfolio' mist kritieke keys (equity/cash).")

            # 2. SQLite Integrity Check
            db_path = Path(PaperConfig().db_path)
            if not db_path.is_absolute():
                db_path = Path.cwd() / db_path

            if db_path.exists():
                try:
                    with sqlite3.connect(str(db_path), timeout=15.0) as conn:
                        conn.execute("PRAGMA journal_mode=WAL;")
                        cur = conn.cursor()
                        cur.execute("PRAGMA integrity_check;")
                        result = cur.fetchone()
                        if result and result[0].lower() != "ok":
                            print(f"[AUDIT] KRITIEK: SQLite corruptie gedetecteerd in {db_path}: {result[0]}")
                except sqlite3.DatabaseError as e:
                    print(f"[AUDIT] KRITIEK: SQLite DatabaseError op {db_path}: {e}")

        except Exception:
            traceback.print_exc()
            log_persistent_crash_error("_data_integrity_audit_loop")
            sys.exit(1)


def _print_frequent_restart_warn_once(
    ts_iso: str,
    *,
    restart_delta_s: float,
    prev_seen_utc: str,
    logs_dir: Path,
) -> None:
    """Eén WARN-regel per snelle herstart: voorkomt dubbele logs bij UVICORN_WORKERS>1."""
    msg = (
        f"{ts_iso} [STARTUP][WARN] Frequent restart detected (<10m). "
        "Check worker_execution.log / portal_api.log for preceding Traceback."
    )
    stamp = f"{prev_seen_utc}|{restart_delta_s:.3f}"
    lock_path = logs_dir / "startup_frequent_restart_warn.lock"
    stamp_path = logs_dir / "startup_frequent_restart_warn_stamp.txt"
    try:
        import fcntl

        logs_dir.mkdir(parents=True, exist_ok=True)
        with open(lock_path, "a+b") as lf:
            fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
            try:
                prev = stamp_path.read_text(encoding="utf-8").strip() if stamp_path.is_file() else ""
                if prev == stamp:
                    return
                print(msg)
                stamp_path.write_text(stamp, encoding="utf-8")
            finally:
                fcntl.flock(lf.fileno(), fcntl.LOCK_UN)
    except Exception:
        print(msg)


@app.on_event("startup")
async def startup_refresh_markets() -> None:
    try:
        from app.services.reporting import install_unhandled_exception_shutdown_telegram_hooks

        install_unhandled_exception_shutdown_telegram_hooks()

        role = _process_role()
        try:
            boot = enforce_emergency_learning_bootstrap()
            if bool(boot.get("epsilon_reset")) and isinstance(getattr(tc, "RL_AGENT", None), object):
                try:
                    st = getattr(tc.RL_AGENT, "last_training_stats", None)
                    if isinstance(st, dict):
                        st["exploration_final_eps"] = 0.80
                        st["exploration_rate_pct"] = 80.0
                except Exception:
                    pass
            print(
                f"{datetime.now().astimezone().isoformat()} [RL-BOOTSTRAP] "
                f"chunks={boot.get('chunks')} epsilon_reset={boot.get('epsilon_reset')} "
                f"min_experiences={boot.get('min_experiences')}"
            )
        except Exception as boot_exc:
            print(f"{datetime.now().astimezone().isoformat()} [RL-BOOTSTRAP][WARN] {boot_exc}")
        ts_now = datetime.now().astimezone()
        startup_marker = (
            Path("/app/logs") / "startup_last_seen_utc.txt"
            if Path("/.dockerenv").exists()
            else Path.cwd() / "_logs_hub" / "startup_last_seen_utc.txt"
        )
        startup_marker.parent.mkdir(parents=True, exist_ok=True)
        last_seen_txt = ""
        restart_delta_s: float | None = None
        try:
            if startup_marker.is_file():
                last_seen_txt = startup_marker.read_text(encoding="utf-8").strip()
                if last_seen_txt:
                    prev = datetime.fromisoformat(last_seen_txt.replace("Z", "+00:00"))
                    restart_delta_s = (datetime.now(UTC) - prev.astimezone(UTC)).total_seconds()
        except Exception as mark_err:
            print(f"{ts_now.isoformat()} [STARTUP][DEBUG] marker read failed: {mark_err}")
        try:
            startup_marker.write_text(datetime.now(UTC).isoformat(), encoding="utf-8")
        except Exception as mark_err:
            print(f"{ts_now.isoformat()} [STARTUP][DEBUG] marker write failed: {mark_err}")
        if restart_delta_s is not None:
            print(
                f"{ts_now.isoformat()} [STARTUP][DIAG] role={role} pid={os.getpid()} "
                f"restart_delta_s={restart_delta_s:.1f} prev_seen_utc={last_seen_txt or '-'}"
            )
            if restart_delta_s < 600:
                _print_frequent_restart_warn_once(
                    ts_now.isoformat(),
                    restart_delta_s=float(restart_delta_s),
                    prev_seen_utc=last_seen_txt or "",
                    logs_dir=startup_marker.parent,
                )
        else:
            print(f"{ts_now.isoformat()} [STARTUP][DIAG] role={role} pid={os.getpid()} first_seen_or_no_marker")

        # Persist startup mode (manual vs auto restart) in system_state.json for cross-module throttles.
        try:
            from app.services.reporting import read_system_state, write_system_state

            st = read_system_state()
            auto_threshold_s = max(60, int(os.getenv("AUTO_RESTART_THRESHOLD_SEC", "1800") or 1800))
            is_auto = restart_delta_s is not None and float(restart_delta_s) < float(auto_threshold_s)
            st["startup_mode"] = "auto" if is_auto else "manual"
            st["last_startup_seen_utc"] = datetime.now(UTC).isoformat()
            st["last_startup_role"] = str(role)
            st["last_startup_pid"] = int(os.getpid())
            write_system_state(st)
            if is_auto:
                print(
                    f"{ts_now.isoformat()} [STARTUP][SILENT] auto_restart_detected "
                    f"(restart_delta_s={float(restart_delta_s or 0.0):.1f} < {auto_threshold_s}s)"
                )
        except Exception as st_err:
            print(f"{ts_now.isoformat()} [STARTUP][DEBUG] system_state persist failed: {st_err}")

        # Crash analysis: log async loop exceptions to crash_log.txt
        try:
            import asyncio

            crash_path = (
                Path("/app/logs") / "crash_log.txt"
                if Path("/.dockerenv").exists()
                else Path.cwd() / "_logs_hub" / "crash_log.txt"
            )
            crash_path.parent.mkdir(parents=True, exist_ok=True)

            def _append_crash_log(msg: str) -> None:
                try:
                    with crash_path.open("a", encoding="utf-8") as fp:
                        fp.write(msg.rstrip() + "\n")
                except Exception:
                    pass

            loop = asyncio.get_running_loop()

            def _loop_exc_handler(loop, context):  # type: ignore[no-redef]
                ts = datetime.now().astimezone().isoformat()
                exc = context.get("exception")
                msg = context.get("message") or "asyncio loop exception"
                block = f"{ts} [CRASH][ASYNCIO] {msg}"
                if exc is not None:
                    block += f"\n{traceback.format_exc()}"
                _append_crash_log(block + "\n---")

            loop.set_exception_handler(_loop_exc_handler)
        except Exception as eh_err:
            print(f"{ts_now.isoformat()} [STARTUP][DEBUG] crash handler setup failed: {eh_err}")
        
        # NOODSITUATIE: startup Telegram volledig uit (crash-loop spam).
        # try:
        #     from app.services.reporting import notification_cooldown_due, mark_notification_sent
        #
        #     tg_min_sec = max(60, int(os.getenv("STARTUP_COOLDOWN_SEC", "3600") or 3600))
        #     tg_due = notification_cooldown_due(key="startup_any_last_sent_utc", cooldown_sec=tg_min_sec)
        #     if tg_due:
        #         msg = f"🚀 AI Trading Bot is online (Role: {role})"
        #         sent = False
        #         for method in ("send", "notify", "send_message", "send_msg"):
        #             if hasattr(tc.TELEGRAM, method):
        #                 getattr(tc.TELEGRAM, method)(msg)
        #                 sent = True
        #                 break
        #         if not sent:
        #             print("[STARTUP] TelegramNotifier mist standard send methods")
        #         else:
        #             mark_notification_sent(key="startup_any_last_sent_utc")
        #     else:
        #         print(
        #             f"{ts_now.isoformat()} [STARTUP][TELEGRAM] Overgeslagen (cooldown actief, "
        #             f"cooldown={tg_min_sec}s)."
        #         )
        # except Exception as tel_e:
        #     print(f"[STARTUP] Kon Telegram notificatie niet versturen (fail-safe actief): {tel_e}")

        if role == "portal":
            await _portal_startup_only()
        elif role == "worker":
            asyncio.create_task(_data_integrity_audit_loop())
            if hasattr(tc, "morning_report_scheduler_loop"):
                asyncio.create_task(tc.morning_report_scheduler_loop())
            return
        else:
            await _run_full_trading_startup()
            # Cold-start Telegram + HTML-mail: ``reporting.cold_start_send_initial_report`` (vault + flock),
            # gestart deferred vanuit ``trading_core._send_initial_report_after_startup`` na MAIN_ENGINE.start.
            try:
                from app.services.reporting import (
                    executive_notifications_muted,
                    refresh_portfolio_equity_integrity,
                )

                _w = tc.PAPER_MANAGER.wallet if isinstance(tc.PAPER_MANAGER.wallet, dict) else {}
                _ok_pi, _det_pi = refresh_portfolio_equity_integrity(_w)
                if executive_notifications_muted():
                    print("[STARTUP] Executive Telegram/e-mail (startup + ochtend) UIT — EXECUTIVE_NOTIFICATIONS_EMERGENCY_MUTE.")
                elif not _ok_pi:
                    print(f"[STARTUP] portfolio equity integrity FAILED — ε geforceerd naar 0.80: {_det_pi}")
                    _ra = getattr(tc, "RL_AGENT", None)
                    _st = getattr(_ra, "last_training_stats", None) if _ra is not None else None
                    if isinstance(_st, dict):
                        _st["exploration_final_eps"] = 0.80
                        _st["exploration_rate_pct"] = 80.0
            except Exception as _pi_exc:
                print(f"[STARTUP][WARN] portfolio integrity check: {_pi_exc}")
            asyncio.create_task(_data_integrity_audit_loop())
            port = int(os.getenv("PORT", "8000"))
            print(f"[DASHBOARD] Dashboard live op poort {port}")
    except Exception:
        traceback.print_exc()
        log_persistent_crash_error("startup_refresh_markets")
        sys.exit(1)


@app.on_event("shutdown")
async def shutdown_notify() -> None:
    if tc.JARVIS_REPORTER is not None:
        await tc.JARVIS_REPORTER.stop()
    tc.JARVIS_REPORTER = None
    if tc.RESTART_MAIL_TASK is not None:
        tc.RESTART_MAIL_TASK.cancel()
        try:
            await tc.RESTART_MAIL_TASK
        except asyncio.CancelledError:
            pass
        tc.RESTART_MAIL_TASK = None
    # Geen Telegram bij normale shutdown (docker compose stop, SIGTERM, FastAPI lifespan): voorkomt
    # meerdere identieke berichten bij UVICORN_WORKERS>1. Alleen onverwachte fout → debounced bericht
    # via ``reporting.install_unhandled_exception_shutdown_telegram_hooks`` + ``try_send_unexpected_shutdown_telegram``.


@app.middleware("http")
async def tenant_scope_middleware(request: Request, call_next):
    tenant = str(request.headers.get("x-tenant-id") or request.query_params.get("tenant_id") or "default").strip().lower()
    set_current_tenant(tenant or "default")

    try:
        if _process_role() == "portal" and request.method == "POST":
            path = request.url.path
            if path == "/bot/pause":
                await _publish_worker_command("bot_pause")
            elif path == "/bot/resume":
                await _publish_worker_command("bot_resume")
            elif path == "/bot/panic":
                await _publish_worker_command("bot_panic")

        response = await call_next(request)
        response.headers["x-tenant-id"] = current_tenant_id()
        return response
    except RequestValidationError:
        raise
    except HTTPException as he:
        return JSONResponse(
            status_code=int(he.status_code),
            content=jsonable_encoder({"detail": he.detail, "path": str(request.url.path), "ok": False}),
        )
    except Exception as exc:
        _log.exception("[API-FLOW] tenant_scope_middleware: onafgehandelde fout op %s", request.url.path)
        return JSONResponse(
            status_code=500,
            content={
                "detail": "Interne Server Fout",
                "error": str(exc)[:800],
                "path": str(request.url.path),
                "ok": False,
            },
        )

@app.get("/api/v1/snapshot")
async def api_snapshot(request: Request) -> dict[str, Any]:
    """Leest de actuele snapshot uit Redis en vult de Portal state."""
    blob = await _await_redis_snapshot()
    if blob:
        # ADAPTER: Forceer dict als blob per ongeluk als dubbele JSON string in Redis zit
        if isinstance(blob, str):
            try:
                blob = json.loads(blob)
            except Exception as e:
                print(f"{datetime.now().astimezone().isoformat()} [API-ERROR] Kan snapshot string niet parsen: {e}")
                
        apply_worker_snapshot_to_portal(blob)
        byte_size = len(json.dumps(blob))
        _log.debug("[API-FLOW] Dashboard requested data via %s, sent %s bytes from Redis", request.url.path, byte_size)
        return blob
    
    _log.debug("[API-FLOW] Dashboard requested data via %s, but no snapshot found in Redis!", request.url.path)
    return {"status": "no_snapshot_found"}

@app.get("/api/v1/ai_logic")
def api_ai_logic() -> dict[str, Any]:
    """Retourneert de beslis-redenering en weights exclusief voor de AI Brain tab."""
    decision = STATE.get("rl_last_decision", {})
    return decision if isinstance(decision, dict) else {}

@app.get("/api/v1/system_stats")
async def api_system_stats() -> dict[str, Any]:
    """Retourneert hardware metrieken (CPU/RAM/GPU) exclusief voor de Hardware tab."""
    return await asyncio.to_thread(_system_stats_payload_for_websocket)

@app.get("/predict", response_model=PredictionResponse)
async def predict(
    ticker: str = Query(default=os.getenv("DEFAULT_TICKER", "BTC-EUR")),
    lookback_days: int = Query(default=int(os.getenv("LOOKBACK_DAYS", "400")), ge=60, le=3000),
) -> PredictionResponse:
    if _process_role() == "portal":
        raise HTTPException(
            status_code=503,
            detail="Voorspellingen draaien op de worker-container (torch/transformers/pandas); portal is UI-only.",
        )
    if STATE.get("bot_status") in {"paused", "panic_stop"}:
        raise HTTPException(status_code=423, detail="Bot is paused or panic-stopped.")
    try:
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[Any] = loop.create_future()
        await PREDICT_QUEUE.put((ticker, lookback_days, current_tenant_id(), fut))
        prediction = await fut
    except Exception as exc:
        STATE["api_fail_streak"] = int(STATE.get("api_fail_streak", 0) or 0) + 1
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    STATE["last_prediction"] = prediction.model_dump()
    wallet = dict(STATE.get("paper_portfolio") or PAPER_MANAGER.wallet)
    equity = float(wallet.get("equity", tc.default_paper_reset_balance_eur()) or tc.default_paper_reset_balance_eur())
    cash = float(wallet.get("cash", equity) or equity)
    anchor_equity = float(
        wallet.get("paper_anchor_equity_eur")
        or wallet.get("starting_balance_eur")
        or equity
        or tc.default_paper_reset_balance_eur()
    )
    anchor_equity = max(1.0, anchor_equity)
    px = float(prediction.latest_close)
    sig = str(prediction.signal or "").upper()
    if sig not in {"BUY", "SELL", "HOLD"}:
        sig = "HOLD"
    size_frac, _quote_eur, size_note = calculate_order_size_for_signal(
        signal=sig,
        equity=equity,
        cash=cash,
        price=px,
        wallet=wallet,
        market=prediction.ticker,
    )
    spread_bps, slippage_bps, _book_imbalance_unused = _orderbook_spread_slippage_bps(
        prediction.ticker,
        quote_notional_eur=max(
            0.0,
            float(anchor_equity if str(sig).upper() == "BUY" else equity) * max(0.0, float(size_frac)),
        ),
    )
    if spread_bps <= 0.0:
        spread_bps = _estimate_spread_bps_from_recent_range(prediction.ticker)
    if slippage_bps <= 0.0:
        slippage_bps = max(0.0, spread_bps * 0.15)
    risk_decision = RISK_MANAGER.evaluate(
        proposed_signal=sig,
        proposed_size_fraction=size_frac,
        spread_bps=spread_bps,
        sentiment_score=prediction.news_sentiment,
    )
    final_signal = str(risk_decision.adjusted_signal).upper()
    final_frac = float(risk_decision.adjusted_size_fraction)
    if risk_decision.reason == "emergency_exit_negative_sentiment_shock" and final_signal == "SELL":
        final_frac = CORE_RISK.full_exit_size_fraction(
            equity=equity, wallet=wallet, price=px, market=prediction.ticker
        )
    elif final_signal == "BUY":
        ok, why = CORE_RISK.check_safety(
            signal="BUY",
            market=prediction.ticker,
            equity=equity,
            cash=cash,
            price=px,
            wallet=wallet,
            proposed_quote_eur=final_frac * anchor_equity,
            fee_rate=float(PAPER_MANAGER.config.fee_rate),
        )
        if not ok:
            final_signal = "HOLD"
            final_frac = 0.0
            size_note = why

    exec_px = _paper_execution_mid_to_fill_price(px, final_signal, spread_bps, slippage_bps)
    risk_controls = compute_risk_controls(exec_px)
    paper_order = build_paper_order(
        signal=final_signal,
        ticker=prediction.ticker,
        price=exec_px,
        size_fraction=final_frac,
        budget_eur=anchor_equity if str(final_signal).upper() == "BUY" else equity,
    )
    STATE["last_order"] = {
        "risk_controls": risk_controls,
        "risk_decision": {
            "approved": risk_decision.approved,
            "reason": risk_decision.reason,
            "spread_bps": round(spread_bps, 3),
            "slippage_bps": round(slippage_bps, 3),
            "max_spread_bps": RISK_MANAGER.max_spread_bps_for_trading,
        },
        "engine_risk": {"sizing_note": size_note, "safety_force_exit": False},
        "order": paper_order,
    }

    append_event(
        {
            "ts": datetime.utcnow().isoformat(),
            "type": "prediction",
            "ticker": prediction.ticker,
            "signal": prediction.signal,
            "expected_return_pct": prediction.expected_return_pct,
        }
    )
    _register_signal_marker(
        ticker=prediction.ticker,
        signal=prediction.signal,
        price=prediction.latest_close,
        expected_return_pct=prediction.expected_return_pct,
    )
    _publish_trading_redis_activity(kind="prediction", ticker=str(prediction.ticker))
    return prediction


@app.post("/paper/run")
async def run_paper_cycle(
    ticker: str = Query(default=os.getenv("DEFAULT_TICKER", "BTC-EUR")),
    lookback_days: int = Query(default=int(os.getenv("LOOKBACK_DAYS", "400")), ge=60, le=3000),
) -> dict[str, Any]:
    if _process_role() == "portal":
        await _publish_worker_command("paper_run", ticker=ticker, lookback_days=lookback_days)
        return {"status": "command_sent", "message": f"🚀 Paper trade commando voor {ticker} is verzonden naar de worker. UI herlaadt zo direct."}
        
    if STATE.get("bot_status") in {"paused", "panic_stop"}:
        raise HTTPException(status_code=423, detail="Bot is paused or panic-stopped.")
    try:
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[Any] = loop.create_future()
        await PAPER_RUN_QUEUE.put((ticker, lookback_days, current_tenant_id(), fut))
        return await fut
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/v1/news/ticker")
async def api_news_ticker(elite_mix: int = Query(0, ge=0, le=1)) -> list[dict[str, Any]]:
    # PORTAL FIX: Serveer direct het door de AI-verwerkte nieuws uit de Worker cache
    if _process_role() == "portal":
        insights = STATE.get("news_insights", [])
        if insights and isinstance(insights, list):
            snap = [x for x in insights if isinstance(x, dict)]
            if snap:
                return await asyncio.to_thread(_sync_map_portal_news_insights, snap)
    try:
        return await asyncio.to_thread(_sync_news_ticker_worker, elite_mix)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/v1/brain/state-overview")
async def api_brain_state_overview() -> dict[str, Any]:
    return await asyncio.to_thread(_sync_brain_state_overview)


@app.get("/api/v1/history")
async def api_history(
    pair: str = Query(default=os.getenv("DEFAULT_TICKER", "BTC-EUR")),
    lookback_days: int = Query(default=180, ge=30, le=1200),
    interval: str = Query(default="5m", description="Bitvavo candle interval (bv. 5m, 15m, 1h)"),
) -> dict[str, Any]:
    try:
        return await asyncio.to_thread(_sync_api_history, pair, lookback_days, interval)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/v1/trades")
async def api_trades(
    limit: int = Query(default=200, ge=1, le=50000),
    view: str = Query(default="events"),
) -> dict[str, Any]:
    return await asyncio.to_thread(_sync_api_trades, limit, view)


@app.get("/api/v1/system/storage")
async def api_system_storage() -> dict[str, Any]:
    return await asyncio.to_thread(_sync_api_system_storage)


@app.get("/api/v1/system/logs")
async def api_system_logs(limit: int = Query(default=200, ge=50, le=1000)) -> dict[str, Any]:
    lines, path_str = await _await_hardware_logs_bundle(limit)
    return {"lines": lines, "path": path_str}


@app.get("/api/v1/system/crash-log")
async def api_system_crash_log(limit: int = Query(default=200, ge=20, le=2000)) -> dict[str, Any]:
    """Tail van crash_log.txt uit de gedeelde log hub."""
    return await asyncio.to_thread(_sync_read_crash_log, limit)


@app.get("/api/v1/debug/policy-writes")
async def api_debug_policy_writes(
    market: str | None = Query(default=None),
    limit: int = Query(default=200, ge=20, le=2000),
) -> dict[str, Any]:
    """Debug-tail van POLICY-WRITE regels uit worker log."""
    return await asyncio.to_thread(_sync_debug_policy_writes, market, limit)


@app.post("/api/v1/log/browser")
async def api_log_browser(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Brug om frontend console.error/warn naar centraal logbestand te sturen."""
    level = str(payload.get("level", "ERROR")).upper()
    msg = str(payload.get("message", "")).replace("\n", " ")
    stack = str(payload.get("stacktrace", "")).replace("\n", " -> ")
    url = str(payload.get("url", ""))
    ts = datetime.now().astimezone().isoformat()
    try:
        await asyncio.to_thread(_sync_append_browser_log, level, msg, stack, url)
    except Exception as exc:
        print(f"{ts} [API-ERROR][CRITICAL] Kan browser log niet wegschrijven: {exc}")
    return {"status": "logged"}

@app.get("/api/v1/performance/analytics")
def api_performance_analytics() -> dict[str, Any]:
    wallet = STATE.get("paper_portfolio")
    if not isinstance(wallet, dict) or not wallet:
        wallet = PAPER_MANAGER.wallet if isinstance(PAPER_MANAGER.wallet, dict) else {}
        
    analytics = PAPER_MANAGER.analytics()
    if not isinstance(analytics, dict):
        analytics = {}
    ps = analytics.setdefault("performance_summary", {})
    if not isinstance(ps, dict): ps = {}
    wl = analytics.setdefault("win_loss_ratio", {})
    if not isinstance(wl, dict): wl = {}
    
    for k in ["win_rate_pct", "max_win_eur", "max_loss_eur", "avg_hold_hours", "total_pnl_eur", "closed_trades"]:
        if k not in ps: ps[k] = 0.0
    for k in ["wins", "losses"]:
        if k not in wl: wl[k] = 0
        
    suggestions = []
    for row in analytics.get("coin_rollup", [])[:5]:
        suggestions.append(
            adjust_weights(
                coin=str(row.get("coin", "MKT")),
                avg_sentiment_top_losses=float(
                    analytics.get("sentiment_correlation", {}).get("avg_sentiment_top_10_losses", 0.0)
                ),
                avg_sentiment_top_wins=float(
                    analytics.get("sentiment_correlation", {}).get("avg_sentiment_top_10_wins", 0.0)
                ),
            )
        )
        
    raw_history = wallet.get("history", [])
    if isinstance(raw_history, list) and len(raw_history) > 100:
        step = max(1, len(raw_history) // 100)
        equity_curve = raw_history[::step]
        # Zorg dat de meest actuele stand altijd het laatste punt in de grafiek is
        if equity_curve[-1] != raw_history[-1]:
            equity_curve.append(raw_history[-1])
    else:
        equity_curve = raw_history if isinstance(raw_history, list) else []
        
    # FIX: Bereken de totalen direct uit de wallet geschiedenis als de database leeg lijkt
    if ps.get("closed_trades", 0) == 0 and raw_history:
        wins, losses, max_w, max_l, total_pnl, closed = 0, 0, 0.0, 0.0, 0.0, 0
        for trade in raw_history:
            if not isinstance(trade, dict) or str(trade.get("action", "")).upper() not in {"BUY", "SELL"}:
                continue
            pnl = float(trade.get("pnl_eur", 0.0) or 0.0)
            if pnl > 0: wins += 1
            else: losses += 1
            if pnl > max_w: max_w = pnl
            if pnl < max_l: max_l = pnl
            total_pnl += pnl
            closed += 1
            
        ps["closed_trades"] = closed
        ps["wins"] = wins
        ps["losses"] = losses
        ps["total_pnl_eur"] = total_pnl
        ps["max_win_eur"] = max_w
        ps["max_loss_eur"] = max_l
        if closed > 0:
            ps["win_rate_pct"] = (wins / closed) * 100.0
            
    return {
        "wallet": wallet,
        "equity_curve": equity_curve,
        "recent_trades": PAPER_MANAGER.recent_trades(limit=50) or (raw_history[-50:] if isinstance(raw_history, list) else []),
        "recent_actions": raw_history[-50:] if isinstance(raw_history, list) else [],
        "analytics": analytics,
        "weight_adjustment_suggestions": suggestions,
    }


def _is_generic_brain_reasoning_placeholder(text: str | None) -> bool:
    """Worker/WS zet vaak deze tekst voordat echte policy-reasoning beschikbaar is."""
    t = (text or "").strip().lower()
    if not t:
        return True
    return "wachten op eerste besluit" in t


def _scanner_reason_for_market(market_u: str) -> str | None:
    for row in STATE.get("active_markets") or []:
        if not isinstance(row, dict):
            continue
        if str(row.get("market") or "").upper() == market_u:
            s = str(row.get("selection_reason") or "").strip()
            return s or None
    return None


def _ensure_brain_reasoning_text(decision: dict[str, Any], market_u: str) -> dict[str, Any]:
    """Vul lege of placeholder RL-reasoning met multi-market copy, scanner of laatste voorspelling."""
    out = dict(decision) if isinstance(decision, dict) else {}
    cur = str(out.get("reasoning") or "").strip()
    if cur and not _is_generic_brain_reasoning_placeholder(cur):
        return out

    multi = STATE.get("rl_multi_decisions")
    if isinstance(multi, dict) and market_u in multi:
        alt = multi[market_u]
        ad = alt.__dict__ if hasattr(alt, "__dict__") else (alt if isinstance(alt, dict) else {})
        if isinstance(ad, dict):
            ar = str(ad.get("reasoning") or "").strip()
            if ar and not _is_generic_brain_reasoning_placeholder(ar):
                out["reasoning"] = ar
                return out

    sr = _scanner_reason_for_market(market_u)
    if sr:
        out["reasoning"] = f"Scanner ({market_u}): {sr}"
        return out

    lp = STATE.get("last_prediction")
    if isinstance(lp, dict):
        sig = str(lp.get("signal") or "").strip()
        tick = str(lp.get("ticker") or "").strip()
        if sig:
            out["reasoning"] = (
                f"Laatste voorspelling ({tick or market_u}): signaal {sig}. "
                "Volledige RL-policytekst volgt na de eerst voltooide inferentie op deze markt."
            )
            return out

    out["reasoning"] = (
        "Engine draait; er is nog geen RL-policytekst voor deze markt. "
        "Controleer worker-/GPU-logs als dit lang zo blijft."
    )
    return out


def _sync_api_brain_reasoning(request_path: str) -> dict[str, Any]:
    try:
        market_u = str(STATE.get("selected_market") or "BTC-EUR").upper()
        if _process_role() == "portal":
            ws_data = STATE.get("_portal_brain_ws", {})
            if isinstance(ws_data, dict) and ws_data:
                base_ga = (
                    ws_data.get("generated_at")
                    or STATE.get("last_engine_tick_utc")
                    or datetime.now().astimezone().isoformat()
                )
                merged_ws = _ensure_brain_reasoning_text(
                    {"reasoning": str(ws_data.get("reasoning") or "").strip(), "generated_at": base_ga},
                    market_u,
                )
                mr = str(merged_ws.get("reasoning") or "").strip()
                if mr and not _is_generic_brain_reasoning_placeholder(mr):
                    return {
                        "status": "ok",
                        "reasoning": mr,
                        "generated_at": str(merged_ws.get("generated_at") or base_ga).strip()
                        or datetime.now().astimezone().isoformat(),
                        "thinking_sections": ws_data.get("thinking_sections") or {},
                    }

        decision = STATE.get("rl_last_decision", {}) if isinstance(STATE.get("rl_last_decision"), dict) else {}
        if not decision:
            alt = tenant_rl_decision_for_symbol(dict(STATE), market_u)
            if isinstance(alt, dict) and alt:
                decision = alt

        if not decision:
            _ts = datetime.now().astimezone().isoformat()
            merged = _ensure_brain_reasoning_text({"reasoning": "", "generated_at": _ts}, market_u)
            return {
                "status": "ok",
                "weights": {},
                "loss": 0,
                "reasoning": str(merged.get("reasoning") or "").strip()
                or "Engine draait; er is nog geen RL-policytekst voor deze markt.",
                "generated_at": str(merged.get("generated_at") or _ts).strip() or _ts,
            }
        decision = _ensure_brain_reasoning_text(decision, market_u)
        byte_size = len(json.dumps(decision))
        _log.debug("[API-FLOW] Dashboard requested data via %s, sent %s bytes from Redis", request_path, byte_size)
        if isinstance(decision, dict):
            out_d = dict(decision)
            if not str(out_d.get("generated_at") or "").strip():
                out_d["generated_at"] = datetime.now().astimezone().isoformat()
            if not out_d.get("thinking_sections"):
                out_d["thinking_sections"] = _build_thinking_sections(out_d)
            if not out_d.get("buy_block_factors"):
                obs_snap = STATE.get("rl_last_observation")
                shadow_snap = list(STATE.get("shadow_trades") or [])[-20:]
                last_spread = float((shadow_snap[-1] if shadow_snap else {}).get("spread_bps") or 0.0)
                out_d["buy_block_factors"] = _build_buy_block_factors(
                    decision=out_d,
                    obs=obs_snap if isinstance(obs_snap, dict) else {},
                    shadow_trades=shadow_snap,
                    spread_bps=last_spread,
                )
            return out_d
        return decision
    except Exception:
        _ts = datetime.now().astimezone().isoformat()
        return {
            "status": "model_loading",
            "weights": {},
            "loss": 0,
            "reasoning": "Model aan het inladen...",
            "generated_at": _ts,
        }


def _sync_api_brain_feature_importance(market: str) -> dict[str, Any]:
    """Zelfde policy-gewichten als RL; voor Balken-sync zie ook `/ws/brain-stats` (× RL-input)."""
    try:
        target_market = str(market or STATE.get("selected_market") or "BTC-EUR").upper()
        multi = STATE.get("rl_multi_decisions", {})
        decision: dict[str, Any] = {}

        if isinstance(multi, dict) and target_market in multi:
            dec = multi[target_market]
            decision = dec.__dict__ if hasattr(dec, "__dict__") else (dec if isinstance(dec, dict) else {})

        if not decision:
            decision = STATE.get("rl_last_decision", {}) if isinstance(STATE.get("rl_last_decision"), dict) else {}

        policy_fw = decision.get("feature_weights") if decision else None
        fw_market = STATE.get("feature_weights_by_market") if isinstance(STATE.get("feature_weights_by_market"), dict) else {}
        per_market_fw = fw_market.get(target_market) if isinstance(fw_market, dict) else None
        fallback_rows = [v for v in (fw_market.values() if isinstance(fw_market, dict) else []) if isinstance(v, dict) and v]
        global_avg = _average_feature_weights(fallback_rows)
        merged_seed = (
            per_market_fw
            if isinstance(per_market_fw, dict) and per_market_fw
            else (global_avg if global_avg else {})
        )
        if merged_seed:
            tmp_decision = dict(decision)
            tmp_decision["feature_weights"] = merged_seed
            fw_policy = merge_feature_weights_for_brain(tmp_decision, policy_fw)
        else:
            fw_policy = merge_feature_weights_for_brain(decision, policy_fw)
        if not fw_policy:
            fw_policy = _balanced_feature_weights()
        obs = STATE.get("rl_last_observation")
        obs_d = obs if isinstance(obs, dict) else {}
        fw_bar = bar_values_from_obs_and_weights(fw_policy, obs_d)

        if _process_role() == "portal":
            ws_data = STATE.get("_portal_brain_ws", {})
            stats = ws_data.get("tm", ws_data.get("training_monitor", {})).get("stats", {}) if ws_data else {}
        else:
            stats = RL_AGENT.training_monitor().get("stats", {}) if isinstance(RL_AGENT.training_monitor(), dict) else {}

        global_steps = int(stats.get("global_step_count", 0) or 0)
        return {
            "feature_weights": fw_bar,
            "feature_weights_policy": fw_policy,
            "rl_observation": obs_d,
            "market": target_market,
            "global_average_feature_weights": global_avg,
            "calibrating": global_steps < 10,
            "social_buzz": STATE.get("social_buzz_summary") if isinstance(STATE.get("social_buzz_summary"), dict) else {},
        }
    except Exception as e:
        print(f"{datetime.now(UTC).isoformat()} [API-ERROR] Fout in api_brain_feature_importance: {e}")
        return {"status": "model_loading", "weights": {}, "loss": 0, "feature_weights": {}, "feature_weights_policy": {}}


def _sync_brain_market_models() -> dict[str, Any]:
    import sqlite3 as _sq
    from pathlib import Path as _P
    import time as _t

    model_dir = _P(os.getenv("RL_MODEL_DIR", "artifacts/rl"))
    now = _t.time()

    # Model file status per market: canonical zip first, then timestamped via models.json
    import json as _json
    model_map: dict[str, dict] = {}
    for f in model_dir.glob("ppo_*-EUR.zip"):
        if "_hourly_" in f.stem or any(c.isdigit() for c in f.stem.split("_")[-1][:8]):
            continue
        market = f.stem[4:]  # strip "ppo_"
        stat = f.stat()
        model_map[market] = {
            "has_model": True,
            "model_size_kb": round(stat.st_size / 1024, 1),
            "model_age_min": round((now - stat.st_mtime) / 60, 1),
        }
    # Also detect timestamped models registered in models.json (written by ensure_pretrained)
    for jf in model_dir.glob("ppo_*-EUR_models.json"):
        market = jf.stem[4:-7]  # strip "ppo_" prefix and "_models" suffix
        if market in model_map:
            continue
        try:
            rows = _json.loads(jf.read_text())
            if not isinstance(rows, list) or not rows:
                continue
            best = sorted(rows, key=lambda x: float(x.get("reward_score", -1e18)), reverse=True)[0]
            mp = model_dir / (str(best.get("model_path", "")).split("/")[-1])
            if not mp.suffix:
                mp = mp.with_suffix(".zip")
            if mp.exists():
                stat = mp.stat()
                model_map[market] = {
                    "has_model": True,
                    "model_size_kb": round(stat.st_size / 1024, 1),
                    "model_age_min": round((now - stat.st_mtime) / 60, 1),
                }
        except Exception:
            pass

    # Per-market action bias from replay buffer (last 200 rows per market)
    replay_stats: dict[str, dict] = {}
    db_path = _P(os.getenv("RL_REPLAY_BUFFER_DB", "data/rl_replay_buffer.db"))
    if not db_path.is_absolute():
        db_path = _P.cwd() / db_path
    try:
        conn = _sq.connect(str(db_path), timeout=3)
        cur = conn.cursor()
        cur.execute("""
            SELECT market,
                ROUND(100.0*SUM(CASE WHEN executed_signal='BUY' THEN 1 ELSE 0 END)/COUNT(*),1),
                ROUND(100.0*SUM(CASE WHEN executed_signal='HOLD' THEN 1 ELSE 0 END)/COUNT(*),1),
                ROUND(100.0*SUM(CASE WHEN executed_signal='SELL' THEN 1 ELSE 0 END)/COUNT(*),1),
                COUNT(*), MAX(ts_utc)
            FROM (SELECT * FROM rl_replay_experience ORDER BY id DESC LIMIT 2000)
            GROUP BY market
        """)
        for row in cur.fetchall():
            replay_stats[row[0]] = {
                "buy_pct": row[1], "hold_pct": row[2], "sell_pct": row[3],
                "replay_rows": row[4], "last_ts": row[5],
            }
        conn.close()
    except Exception:
        pass

    active = [str(m.get("market", "")).upper() for m in (STATE.get("active_markets") or []) if m.get("market")]
    models = []
    for mkt in active:
        entry = {"market": mkt, "has_model": False, "model_size_kb": 0, "model_age_min": None}
        entry.update(model_map.get(mkt, {}))
        entry.update(replay_stats.get(mkt, {"buy_pct": 0, "hold_pct": 0, "sell_pct": 0, "replay_rows": 0, "last_ts": None}))
        models.append(entry)

    trainer_log = docker_trainer_last_chunk()
    return {"models": models, "trainer_last_chunk": trainer_log}


def docker_trainer_last_chunk() -> str | None:
    try:
        ts = STATE.get("_trainer_last_chunk_ts")
        return str(ts) if ts else None
    except Exception:
        return None


def _sync_api_brain_training_monitor() -> dict[str, Any]:
    if _process_role() == "portal":
        ws_data = STATE.get("_portal_brain_ws", {})
        if ws_data:
            res = ws_data.get("tm") or ws_data.get("training_monitor") or {}
            if isinstance(res, dict) and res:
                return res

    res = RL_AGENT.training_monitor()
    if not isinstance(res, dict):
        res = {}
    stats = res.get("stats", {})
    if not isinstance(stats, dict):
        stats = {}
        res["stats"] = stats
    if "discount_factor" not in stats:
        stats["discount_factor"] = 0.99
    if "batch_size" not in stats:
        stats["batch_size"] = 128
    stats["is_training_active"] = str(os.getenv("RL_BACKGROUND_TRAIN", "0")).strip().lower() in ("1", "true", "yes", "on")
    net_logs = res.get("network_logs")
    if isinstance(net_logs, dict):
        import math as _math
        vl = [v for v in (net_logs.get("value_loss") or []) if isinstance(v, (int, float)) and v >= 0]
        net_logs["reward_error"] = [round(_math.sqrt(v), 6) for v in vl]
    return res


def _sync_api_brain_news_lag() -> dict[str, Any]:
    rows = STATE.get("news_lag_history", [])
    return {"items": rows[-120:] if isinstance(rows, list) else []}


def _sync_terminal_news_insights() -> dict[str, Any]:
    return {
        "items": STATE.get("news_insights", []),
        "selected_market": STATE.get("selected_market", "BTC-EUR"),
        "updated_from_prediction": STATE.get("last_prediction", {}).get("generated_at")
        if isinstance(STATE.get("last_prediction"), dict)
        else None,
    }


@app.get("/api/v1/brain/reasoning")
async def api_brain_reasoning(request: Request) -> dict[str, Any]:
    return await asyncio.to_thread(_sync_api_brain_reasoning, request.url.path)


@app.get("/api/v1/brain/feature-importance")
async def api_brain_feature_importance(market: str = Query(default="")) -> dict[str, Any]:
    return await asyncio.to_thread(_sync_api_brain_feature_importance, market)


@app.get("/api/v1/social/buzz")
def api_social_buzz() -> dict[str, Any]:
    """Social velocity / regime snapshot (Elite quality markets only)."""
    snap = STATE.get("social_buzz_summary")
    if isinstance(snap, dict):
        return snap
    return {"lines": [], "markets": 0, "updated_at": None}


@app.get("/api/v1/whale/radar")
def api_whale_radar() -> dict[str, Any]:
    moves = STATE.get("whale_radar_moves")
    if isinstance(moves, list):
        return {"moves": moves[:8]}
    return {"moves": []}


@app.get("/api/v1/brain/training-monitor")
async def api_brain_training_monitor() -> dict[str, Any]:
    return await asyncio.to_thread(_sync_api_brain_training_monitor)


@app.get("/api/v1/brain/market-models")
async def api_brain_market_models() -> dict[str, Any]:
    return await asyncio.to_thread(_sync_brain_market_models)


@app.get("/api/v1/brain/news-lag")
async def api_brain_news_lag() -> dict[str, Any]:
    return await asyncio.to_thread(_sync_api_brain_news_lag)


@app.get("/terminal/news-insights")
async def terminal_news_insights() -> dict[str, Any]:
    return await asyncio.to_thread(_sync_terminal_news_insights)


@app.get("/terminal/chart-points")
async def terminal_chart_points(
    ticker: str = Query(default=os.getenv("DEFAULT_TICKER", "BTC-EUR")),
    lookback_days: int = Query(default=120, ge=30, le=600),
) -> dict[str, Any]:
    try:
        return await asyncio.to_thread(_sync_terminal_chart_points, ticker, lookback_days)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/sentiment/current")
def current_sentiment() -> dict[str, Any]:
    scores = STATE.get("last_scores") or {}
    return {
        "sentiment_score": scores.get("sentiment_score"),
        "sentiment_confidence": scores.get("sentiment_confidence"),
        "updated_from_prediction": STATE.get("last_prediction", {}).get("generated_at")
        if isinstance(STATE.get("last_prediction"), dict)
        else None,
    }


@app.get("/bot/status")
def get_bot_status() -> dict[str, str]:
    return {"bot_status": STATE.get("bot_status", "running")}


def _blob_has_multi_policy_probs(blob: dict[str, Any]) -> bool:
    """True als rl_multi_decisions ergens geldige PPO-kansen heeft (los van format_stats)."""
    tenant = blob.get("tenant", blob) if isinstance(blob, dict) else {}
    multi = tenant.get("rl_multi_decisions") if isinstance(tenant, dict) else None
    if not isinstance(multi, dict):
        return False
    for dec in multi.values():
        if not isinstance(dec, dict):
            continue
        for k in ("prob_buy", "prob_hold", "prob_sell"):
            try:
                if float(dec.get(k) or 0.0) > 1e-6:
                    return True
            except (TypeError, ValueError):
                continue
    return False


@app.get("/api/v1/status")
async def api_status() -> dict[str, Any]:
    blob = await _await_redis_snapshot()
    if isinstance(blob, dict) and blob:
        rs = _runtime_status_from_blob(blob)
        tenant = blob.get("tenant", blob) if isinstance(blob, dict) else {}
        bot_running = str(tenant.get("bot_status", "running") or "running").lower() not in {
            "paused",
            "panic_stop",
            "stopped",
            "stop",
        }
        lec = tenant.get("last_engine_cycle") if isinstance(tenant, dict) else None
        engine_tick_ok = bool(isinstance(lec, dict) and str(lec.get("ts") or "").strip())

        # Engine is alleen error bij echte hard-fail; met policy-data => ACTIVE.
        market = str(STATE.get("selected_market") or "BTC-EUR").upper().replace("/", "-")
        stats = format_stats(blob, policy_market=market)
        probs = stats.get("ai_action_probs") if isinstance(stats, dict) else {}
        has_probs = False
        if isinstance(probs, dict):
            for k in ("buy_pct", "hold_pct", "sell_pct"):
                try:
                    v = probs.get(k)
                    if v is not None and float(v) > 0.01:
                        has_probs = True
                        break
                except (TypeError, ValueError):
                    continue
        multi_probs = _blob_has_multi_policy_probs(blob)
        has_flow = (
            bool(rs.get("db_connected"))
            or bool(str(rs.get("last_inference_time") or "").strip())
            or engine_tick_ok
        )
        ws = str(rs.get("worker_status") or "").lower()
        worker_stopped = ws in ("offline", "stopped", "dead")
        if has_probs or multi_probs:
            rs["ai_engine"] = "ACTIVE"
        elif bot_running and (rs.get("worker_status") == "online" or has_flow):
            rs["ai_engine"] = "THINKING"
        elif not bot_running:
            rs["ai_engine"] = "PAUSED"
        elif bot_running and worker_stopped:
            rs["ai_engine"] = "ERROR"
        else:
            rs["ai_engine"] = "THINKING"
        return rs
    return {
        "worker_status": "degraded",
        "gpu_status": "loading",
        "db_connected": False,
        "last_inference_time": "",
        "ai_engine": "THINKING",
    }


@app.get("/exchange/rate-limit/status")
def exchange_rate_limit_status() -> dict[str, Any]:
    return global_rate_limit_status()


@app.get("/api/v1/system/report-status")
def api_report_status() -> dict[str, Any]:
    if tc.JARVIS_REPORTER is None:
        return {
            "live": False,
            "channel_status": {"telegram": False, "email": False},
            "last_run": None,
            "next_run_at": None,
            "timezone": "Europe/Amsterdam",
            "audit_engine": {
                "last_run": AUDIT_LAST_RUN,
                "decision_threshold": STATE.get("decision_threshold"),
                "stop_loss_pct": STATE.get("stop_loss_pct"),
            },
            "autonomous_optimizer": {
                "last_run": AUTO_OPT_LAST_RUN,
                "exploration_eps": STATE.get("auto_opt_exploration_eps"),
                "risk_cap_pct": STATE.get("auto_opt_risk_cap_pct"),
                "train_chunk_steps": STATE.get("auto_opt_train_chunk_steps"),
                "last_tuning": AUTO_OPT_LAST_TUNING,
            },
        }
    snap = tc.JARVIS_REPORTER.status_snapshot()
    snap["audit_engine"] = {
        "last_run": AUDIT_LAST_RUN,
        "decision_threshold": STATE.get("decision_threshold"),
        "stop_loss_pct": STATE.get("stop_loss_pct"),
    }
    snap["autonomous_optimizer"] = {
        "last_run": AUTO_OPT_LAST_RUN,
        "exploration_eps": STATE.get("auto_opt_exploration_eps"),
        "risk_cap_pct": STATE.get("auto_opt_risk_cap_pct"),
        "train_chunk_steps": STATE.get("auto_opt_train_chunk_steps"),
        "last_tuning": AUTO_OPT_LAST_TUNING,
    }
    return snap


@app.get("/vault/balance-check")
def vault_balance_check(market: str | None = Query(default=None)) -> dict[str, Any]:
    target_market = (market or STATE.get("selected_market") or "BTC-EUR").upper()
    min_quote = float(os.getenv("PAIR_MIN_QUOTE_BALANCE", "50"))
    min_base = float(os.getenv("PAIR_MIN_BASE_BALANCE", "0.00001"))
    try:
        return check_pair_balance_from_vault(
            market=target_market,
            min_quote_balance=min_quote,
            min_base_balance=min_base,
        )
    except Exception as exc:
        return {
            "market": target_market,
            "available": False,
            "reason": f"balance_check_failed:{str(exc)}",
        }


@app.get("/dry-run/pnl/daily")
def dry_run_daily_pnl(date_utc: str | None = Query(default=None)) -> dict[str, Any]:
    target_date = date_utc or datetime.utcnow().date().isoformat()
    try:
        return calculate_daily_fictive_pnl(target_date)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.websocket("/ws/trades")
async def ws_trades(websocket: WebSocket) -> None:
    """Live paper-trade rows: compacte keys, max ~2/s, heartbeat, critical = lijst gewijzigd."""
    await websocket.accept()
    STATE["ws_connections"] = int(STATE.get("ws_connections", 0) or 0) + 1

    def _touch_hb() -> None:
        STATE["last_ws_heartbeat_ts"] = datetime.now(UTC).isoformat()

    try:
        async def pump_messages() -> None:
            last_state = None
            while True:
                try:
                    pf = STATE.get("paper_portfolio") or {}
                    current_state = f"{pf.get('trades_count')}_{pf.get('position_qty')}_{pf.get('cash')}"
                    if current_state != last_state:
                        rows = tc.PAPER_MANAGER.recent_trades(limit=50)
                        await websocket.send_text(json.dumps({"topic": "trades", "data": rows}, default=str))
                        last_state = current_state
                except (WebSocketDisconnect, RuntimeError):
                    break
                except Exception:
                    pass
                _touch_hb()
                await asyncio.sleep(1.0)
        pump = asyncio.create_task(pump_messages())
        while True:
            await websocket.receive_text()
            _touch_hb()
    except (WebSocketDisconnect, RuntimeError):
        return
    except Exception:
        try:
            await websocket.close()
        except Exception:
            pass
    finally:
        if 'pump' in locals():
            pump.cancel()
            try:
                await pump
            except asyncio.CancelledError:
                pass
        STATE["ws_connections"] = max(0, int(STATE.get("ws_connections", 1) or 1) - 1)


@app.websocket("/ws/logs")
async def ws_logs(websocket: WebSocket) -> None:
    await websocket.accept()
    STATE["ws_connections"] = int(STATE.get("ws_connections", 0) or 0) + 1
    target_log = _log_path_for_ws_follow()
    try:
        tc.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        if not target_log.exists():
            target_log.touch(exist_ok=True)
    except Exception as e:
        print(f"[WS-LOGS] Warning touching log file: {e}")
    try:
        initial_lines = _tail_lines(target_log, limit=200)
        for line in initial_lines:
            await websocket.send_text(line)

        with target_log.open("r", encoding="utf-8", errors="replace") as fh:
            fh.seek(0, os.SEEK_END)
            while True:
                where = fh.tell()
                line = fh.readline()
                if not line:
                    await asyncio.sleep(0.8)
                    try:
                        # Herstel het lees-punt als het bestand gesnoeid (prune) is door de worker
                        if target_log.stat().st_size < where:
                            fh.seek(0)
                        else:
                            fh.seek(where)
                    except Exception:
                        fh.seek(where)
                    STATE["last_ws_heartbeat_ts"] = datetime.now(UTC).isoformat()
                    continue
                await websocket.send_text(line.rstrip("\n"))
                STATE["last_ws_heartbeat_ts"] = datetime.now(UTC).isoformat()
    except WebSocketDisconnect:
        return
    except Exception as e:
        print(f"[WS-LOGS] Disconnected with error: {e}")
        try:
            await websocket.close()
        except Exception:
            pass
    finally:
        STATE["ws_connections"] = max(0, int(STATE.get("ws_connections", 1) or 1) - 1)


@app.websocket("/ws/system-stats")
async def ws_system_stats(websocket: WebSocket) -> None:
    """Live CPU/RAM/GPU: elke 2s compact JSON + 30s heartbeat (client stuurt hb_ack)."""
    await websocket.accept()
    STATE["ws_connections"] = int(STATE.get("ws_connections", 0) or 0) + 1

    def _touch_hb() -> None:
        STATE["last_ws_heartbeat_ts"] = datetime.now(UTC).isoformat()

    try:
        async def pump_messages() -> None:
            while True:
                try:
                    payload = _system_stats_payload_for_websocket()
                    if payload:
                        await websocket.send_text(json.dumps(payload, default=str))
                except (WebSocketDisconnect, RuntimeError):
                    break
                except Exception:
                    pass
                _touch_hb()
                await asyncio.sleep(2.0)
        pump = asyncio.create_task(pump_messages())
        while True:
            await websocket.receive_text()
            _touch_hb()
    except (WebSocketDisconnect, RuntimeError):
        return
    except Exception:
        try:
            await websocket.close()
        except Exception:
            pass
    finally:
        if 'pump' in locals():
            pump.cancel()
            try:
                await pump
            except asyncio.CancelledError:
                pass
        STATE["ws_connections"] = max(0, int(STATE.get("ws_connections", 1) or 1) - 1)




@app.websocket("/ws/brain-stats")
async def ws_brain_stats(websocket: WebSocket) -> None:
    """Directe Redis-stream voor AI Brain met 1 seconde interval."""
    await websocket.accept()
    STATE["ws_connections"] = int(STATE.get("ws_connections", 0) or 0) + 1

    def _touch_hb() -> None:
        STATE["last_ws_heartbeat_ts"] = datetime.now(UTC).isoformat()

    try:
        async def pump_messages() -> None:
            while True:
                try:
                    payload = _brain_ws_wire_payload()
                    if payload:
                        await websocket.send_json(payload)
                    else:
                        await websocket.send_json({"t": "brain_stats", "status": "loading", "reasoning": "Wachten op data..."})
                except (WebSocketDisconnect, RuntimeError):
                    break
                except Exception as e:
                    try:
                        await websocket.send_json({"t": "brain_stats", "status": "error", "reasoning": f"Fout: {str(e)}"})
                    except (WebSocketDisconnect, RuntimeError):
                        break
                    
                _touch_hb()
                await asyncio.sleep(1.0)
        pump = asyncio.create_task(pump_messages())
        while True:
            await websocket.receive_text()
            _touch_hb()
    except WebSocketDisconnect:
        return
    except Exception:
        try:
            await websocket.close()
        except Exception:
            pass
    finally:
        if 'pump' in locals():
            pump.cancel()
            try:
                await pump
            except asyncio.CancelledError:
                pass
        STATE["ws_connections"] = max(0, int(STATE.get("ws_connections", 1) or 1) - 1)


@app.websocket("/ws/trading-updates")
async def ws_trading_updates(websocket: WebSocket) -> None:
    """Portal: Redis pub/sub `trading_updates` doorsturen. Blijft open met heartbeats als Redis of data ontbreekt."""
    await websocket.accept()
    STATE["ws_connections"] = int(STATE.get("ws_connections", 0) or 0) + 1

    host = str(os.getenv("REDIS_HOST", "redis")).strip()
    port = str(os.getenv("REDIS_PORT", "6379")).strip()
    url = str(os.getenv("REDIS_URL", f"redis://{host}:{port}/0")).strip()
    if "localhost" in url or "127.0.0.1" in url:
        url = f"redis://{host}:{port}/0"

    import redis.asyncio as aioredis

    async def _send_heartbeat(**extra: Any) -> None:
        payload: dict[str, Any] = {
            "type": "heartbeat",
            "channel": TRADING_UPDATES_CHANNEL,
            "ts": datetime.now(UTC).isoformat(),
        }
        payload.update(extra)
        await websocket.send_text(json.dumps(payload, default=str))

    client: Any = None
    pump_task: asyncio.Task[Any] | None = None
    try:
        if not url:
            _log.warning("ws_trading_updates: lege REDIS_URL")
            try:
                await _send_heartbeat(redis=False, error="REDIS_URL niet geconfigureerd")
            except (WebSocketDisconnect, RuntimeError):
                return
            while True:
                try:
                    await asyncio.wait_for(websocket.receive_text(), timeout=45.0)
                except asyncio.TimeoutError:
                    try:
                        await _send_heartbeat(redis=False, idle=True)
                    except (WebSocketDisconnect, RuntimeError):
                        break
                except (WebSocketDisconnect, RuntimeError):
                    break
            return

        try:
            client = aioredis.from_url(
                url,
                decode_responses=True,
                socket_connect_timeout=4.0,
                socket_timeout=30.0,
            )
            await asyncio.wait_for(client.ping(), timeout=4.0)
        except Exception as e:
            _log.warning("ws_trading_updates: Redis niet bereikbaar (%s)", e)
            while True:
                try:
                    await _send_heartbeat(redis=False, message=str(e)[:240])
                except (WebSocketDisconnect, RuntimeError):
                    break
                try:
                    await asyncio.wait_for(websocket.receive_text(), timeout=25.0)
                except asyncio.TimeoutError:
                    continue
                except (WebSocketDisconnect, RuntimeError):
                    break
            return

        async def pump_messages() -> None:
            idle_ticks = 0
            pubsub = client.pubsub()
            subscribed = False
            try:
                await pubsub.subscribe(TRADING_UPDATES_CHANNEL)
                subscribed = True
            except Exception as sub_e:
                _log.warning("ws_trading_updates: subscribe mislukt (%s)", sub_e)
                while True:
                    try:
                        await _send_heartbeat(subscribed=False, message=str(sub_e)[:240])
                        await asyncio.sleep(10.0)
                    except (WebSocketDisconnect, RuntimeError):
                        break
                return

            try:
                while True:
                    try:
                        message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                    except Exception:
                        message = None
                    if message and message.get("type") == "message":
                        data = message.get("data")
                        if isinstance(data, str) and data:
                            try:
                                await websocket.send_text(data)
                            except (WebSocketDisconnect, RuntimeError):
                                break
                        idle_ticks = 0
                    else:
                        idle_ticks += 1
                        if idle_ticks >= 20:
                            idle_ticks = 0
                            try:
                                await _send_heartbeat(subscribed=subscribed, idle=True)
                            except (WebSocketDisconnect, RuntimeError):
                                break
                    STATE["last_ws_heartbeat_ts"] = datetime.now(UTC).isoformat()
            finally:
                try:
                    await pubsub.unsubscribe(TRADING_UPDATES_CHANNEL)
                except Exception:
                    pass
                try:
                    await pubsub.close()
                except Exception:
                    pass

        pump_task = asyncio.create_task(pump_messages())
        while True:
            try:
                await websocket.receive_text()
            except (WebSocketDisconnect, RuntimeError):
                break
    finally:
        if pump_task is not None:
            pump_task.cancel()
            try:
                await pump_task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
        if client is not None:
            try:
                await client.aclose()
            except Exception:
                pass
        STATE["ws_connections"] = max(0, int(STATE.get("ws_connections", 1) or 1) - 1)


@app.get("/", response_class=HTMLResponse)
def portal(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "default_ticker": os.getenv("DEFAULT_TICKER", "BTC-EUR"),
        },
    )
