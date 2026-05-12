"""
GET /api/v1/predictions/{symbol} — korte prijs-pad uit PPO ``model.predict`` + BitvavoTradingEnv,
met Redis-historie (optioneel) en portal-fallback naar ``last_prediction``.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import redis
from fastapi import APIRouter, HTTPException, Query

from app.services.prediction_ui import build_overlay_prices_from_rl_or_fallback, tenant_rl_decision_for_symbol

router = APIRouter(prefix="/api/v1", tags=["predictions"])
_log_pred = logging.getLogger(__name__)


def _int_action(act_arr: Any) -> int:
    """PPO ``predict``-return naar int zonder numpy (portal-image heeft geen numpy)."""
    if act_arr is None:
        return 0
    if hasattr(act_arr, "item"):
        try:
            return int(act_arr.item())
        except Exception:
            pass
    if isinstance(act_arr, (list, tuple)) and len(act_arr) > 0:
        try:
            return int(float(act_arr[0]))
        except Exception:
            return 0
    try:
        return int(float(act_arr))
    except Exception:
        return 0


def _redis_url() -> str:
    host = str(os.getenv("REDIS_HOST", "redis")).strip()
    port = str(os.getenv("REDIS_PORT", "6379")).strip()
    url = str(os.getenv("REDIS_URL", f"redis://{host}:{port}/0")).strip()
    if not url:
        url = f"redis://{host}:{port}/0"
    return url


def _redis_client() -> redis.Redis:
    return redis.Redis.from_url(
        _redis_url(),
        decode_responses=True,
        socket_connect_timeout=2.0,
        socket_timeout=2.0,
    )


def _load_history_from_redis(symbol: str) -> list[dict[str, Any]] | None:
    sym = str(symbol or "").strip().upper()
    if not sym:
        return None
    try:
        r = _redis_client()
        try:
            raw = r.get(f"market:{sym}:history")
        finally:
            r.close()
    except Exception:
        return None
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except Exception:
        return None
    return data if isinstance(data, list) else None


def _prediction_path_from_state(symbol: str, last_price: float) -> list[dict[str, Any]]:
    """
    Portal / geen model: gebruik dezelfde gecombineerde RL-overlaylogica als /stats,
    zodat chartlijn niet enkel op legacy last_prediction (RSI-pad) draait.
    """
    from app.services.state import STATE

    mku = str(symbol or "").strip().upper().replace("/", "-")
    lp = STATE.get("last_prediction") if isinstance(STATE.get("last_prediction"), dict) else {}
    lp_t = str(lp.get("ticker") or "").strip().upper().replace("/", "-")
    lp_use = lp if lp and lp_t == mku else None
    dec = tenant_rl_decision_for_symbol(STATE if isinstance(STATE, dict) else {}, mku)
    series, lc, _pn = build_overlay_prices_from_rl_or_fallback(float(last_price), dec, lp_use)
    if not series:
        return []
    now = datetime.now(timezone.utc)
    out: list[dict[str, Any]] = []
    for i, v in enumerate(series, start=1):
        t = (now + timedelta(minutes=i)).isoformat()
        out.append({"timestamp": t, "predicted_price": round(float(v if v > 0 else lc), 2)})
    return out


def _rollout_predicted_prices(symbol: str) -> list[dict[str, Any]]:
    """
    Vijf stappen PPO ``model.predict(obs, deterministic=True)`` + env.step op het RL-frame
    (laatste segment van candles), zodat elke stap de policy opnieuw evalueert.
    """
    import app.trading_core as tc
    from app.rl.data import build_rl_training_frame, fetch_bitvavo_historical_candles, patch_last_row_live_orderbook
    from app.rl.env import BitvavoTradingEnv

    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=int(os.getenv("PREDICTION_ROLLBACK_DAYS", "14") or 14))
    candles = fetch_bitvavo_historical_candles(str(symbol).upper(), "1h", start_dt, end_dt)
    if candles is None or getattr(candles, "empty", True):
        raise ValueError("Geen candles voor rollout.")

    cmc = {}
    if hasattr(tc, "_refresh_cmc_metrics"):
        try:
            cmc = tc._refresh_cmc_metrics()  # type: ignore[attr-defined]
        except Exception:
            cmc = {}
    rl_frame = build_rl_training_frame(
        candles,
        event_window_hours=24,
        market=str(symbol).upper(),
        news_query="crypto",
        news_api_key=os.getenv("CRYPTOCOMPARE_KEY"),
        cryptocompare_key=os.getenv("CRYPTOCOMPARE_KEY"),
        cmc_metrics=cmc if isinstance(cmc, dict) else {},
    )
    try:
        _qn = float(os.getenv("RL_ORDERBOOK_QUOTE_NOTIONAL_EUR", "500") or 500)
    except (TypeError, ValueError):
        _qn = 500.0
    spread_bps_r, _sl_r, book_imb_r = tc._orderbook_spread_slippage_bps(str(symbol).upper(), max(0.0, _qn))
    rl_frame = patch_last_row_live_orderbook(rl_frame, spread_bps_r, book_imb_r)
    from app.services.market_regime import refresh_market_regime_from_last_row

    refresh_market_regime_from_last_row(rl_frame.iloc[-1].to_dict())
    if len(rl_frame) < 12:
        raise ValueError("Te weinig RL-rijen voor rollout.")

    model = getattr(tc.RL_AGENT, "model", None)
    if model is None:
        raise RuntimeError("RL-model niet geladen.")

    env = BitvavoTradingEnv(data=rl_frame, max_trades=500)
    env.reset()
    start_idx = max(1, len(env.df) - 6)
    env.step_idx = int(start_idx)
    obs = env._build_observation()

    now = datetime.now(timezone.utc)
    out: list[dict[str, Any]] = []
    predict_calls = 0
    for i in range(5):
        act_arr, _states = model.predict(obs, deterministic=True)
        predict_calls += 1
        action = _int_action(act_arr)
        obs, _rew, terminated, truncated, _info = env.step(action)
        px = float(env._current_price())
        out.append(
            {
                "timestamp": (now + timedelta(minutes=i + 1)).isoformat(),
                "predicted_price": round(px, 2),
            }
        )
        if terminated or truncated:
            break
    if predict_calls <= 0:
        raise RuntimeError(f"Rollout stalled before model.predict for {symbol}")
    return out


_ROLL_CACHE: dict[str, tuple[float, list[dict[str, Any]]]] = {}
_ROLL_TTL_SEC = float(os.getenv("PREDICTION_ROLLOUT_CACHE_SEC", "12") or 12.0)

# Bitvavo-fallback voor Redis-miss: cache per symbool om parallelle Elite-polls te dempen (rate-limit + load).
_HISTORY_BV_CACHE: dict[str, tuple[float, list[dict[str, Any]]]] = {}
_HISTORY_BV_TTL_SEC = float(os.getenv("PREDICTION_HISTORY_BITVAVO_CACHE_SEC", "90") or 90.0)

# Korte HTTP-cache: coalesceert dubbele polls (app_core + module_feeds) binnen enkele seconden.
_PRED_PAYLOAD_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
_PRED_PAYLOAD_TTL_SEC = float(os.getenv("PREDICTIONS_HTTP_CACHE_SEC", "4") or 4.0)


def _rollout_cached(symbol: str) -> list[dict[str, Any]]:
    """Beperk zware RL+news frame-builds bij frequente UI-polls (anti-loop)."""
    sym = str(symbol or "").strip().upper()
    try:
        from core.worker_execution import min_observation_interval_sec

        min_ttl = max(min_observation_interval_sec(), _ROLL_TTL_SEC)
    except Exception:
        min_ttl = max(10.0, _ROLL_TTL_SEC)
    if min_ttl <= 0.0:
        min_ttl = 10.0
    now = time.monotonic()
    hit = _ROLL_CACHE.get(sym)
    if hit and (now - hit[0]) < min_ttl:
        return list(hit[1])
    _log_pred.info("Inference started for %s (new rollout)", sym)
    out = _rollout_predicted_prices(sym)
    _ROLL_CACHE[sym] = (now, list(out))
    return out


def get_rl_prediction(symbol: str, historical_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Projectie op basis van live RL-kansen (±3.5% geclamped, zelfde bron als policy-balken).
    Env-rollout vermeden: die geeft historische candle-closes terug als 'voorspelling'.
    """
    if not historical_data:
        raise ValueError("historical_data is leeg")
    last = historical_data[-1]
    last_price = float(last.get("price", 0) or 0.0)
    if last_price <= 0:
        raise ValueError("Ongeldige laatste prijs")

    return _prediction_path_from_state(symbol, last_price)


def _pct01_policy(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not v == v or v < 0.0:
        return None
    return round(v * 100.0, 2) if v <= 1.0 + 1e-6 else round(min(100.0, v), 2)


def _ai_probs_usable(ap: Any) -> bool:
    if not isinstance(ap, dict):
        return False
    for k in ("hold_pct", "buy_pct", "sell_pct"):
        v = ap.get(k)
        if v is None:
            continue
        try:
            if float(v) > 0.01:
                return True
        except (TypeError, ValueError):
            continue
    return False


def _validated_ai_action_probs(row_like: Any) -> dict[str, float]:
    """
    Validator voor AI-probs: normaliseert/fallbackt via worker-logic en geeft altijd
    buy/hold/sell percentages terug in 0..100.
    """
    from core.worker_execution import ensure_final_probs

    row_in = dict(row_like) if isinstance(row_like, dict) else {}
    # Frontend/API varianten kunnen buy_pct/hold_pct/sell_pct bevatten i.p.v. prob_*.
    if "prob_buy" not in row_in and row_in.get("buy_pct") is not None:
        row_in["prob_buy"] = row_in.get("buy_pct")
    if "prob_hold" not in row_in and row_in.get("hold_pct") is not None:
        row_in["prob_hold"] = row_in.get("hold_pct")
    if "prob_sell" not in row_in and row_in.get("sell_pct") is not None:
        row_in["prob_sell"] = row_in.get("sell_pct")
    # Forceer geen synthetic action-softmax als er geen model-probs zijn.
    if row_in.get("prob_buy") is None and row_in.get("prob_hold") is None and row_in.get("prob_sell") is None:
        return {"hold_pct": 0.0, "buy_pct": 0.0, "sell_pct": 0.0}
    row = ensure_final_probs(row_in)
    return {
        "hold_pct": _pct01_policy(row.get("prob_hold")) or 0.0,
        "buy_pct": _pct01_policy(row.get("prob_buy")) or 0.0,
        "sell_pct": _pct01_policy(row.get("prob_sell")) or 0.0,
    }


def _tag_ai_probs_symbol(out: dict[str, Any], sym: str) -> None:
    """Zet market/ticker op ai_action_probs zodat de UI geen ongetagde poll-data op de verkeerde munt toont."""
    if sym and isinstance(out.get("ai_action_probs"), dict):
        ap = dict(out["ai_action_probs"])
        ap.setdefault("market", sym)
        ap.setdefault("ticker", sym)
        out["ai_action_probs"] = ap


def _policy_stream_for_symbol(symbol: str) -> dict[str, Any]:
    """
    Policy per ``?symbol=`` — eerst Redis ``prediction:PAIR`` / ``pred:PAIR`` (zelfde als worker),
    daarna portal-snapshot via ``format_stats`` voor drempels en fallback.
    """
    sym = str(symbol or "").strip().upper().replace("/", "-")
    out: dict[str, Any] = {
        "ai_action_probs": None,
        "rl_confidence": None,
        "rl_last_decision": None,
        "signal_threshold_pct": None,
        "rl_decision_threshold_pct": None,
        "generated_at": None,
        "prediction_timestamp": None,
    }
    try:
        from app.main import format_stats
        from app.portal_snapshot import read_worker_portal_snapshot
        from core.worker_execution import read_per_market_prediction_policy, ensure_final_probs

        import app.trading_core as tc

        blob = read_worker_portal_snapshot()
        stats: dict[str, Any] = {}
        if isinstance(blob, dict) and blob:
            try:
                stats = format_stats(blob, policy_market=sym or None)
            except Exception:
                stats = {}
        if isinstance(stats, dict):
            out["signal_threshold_pct"] = stats.get("signal_threshold_pct")
            out["rl_decision_threshold_pct"] = stats.get("rl_decision_threshold_pct")

        pol = read_per_market_prediction_policy(sym)
        if isinstance(pol, dict) and pol:
            row = tc._rl_decision_as_dict_with_fallback(dict(pol))
            row = ensure_final_probs(row)
            if sym:
                row.setdefault("ticker", sym)
                row.setdefault("market", sym)
            out["rl_last_decision"] = row
            out["rl_confidence"] = float(row.get("confidence", 0.0) or 0.0) if row else 0.0
            out["generated_at"] = row.get("generated_at")
            out["prediction_timestamp"] = row.get("prediction_timestamp")
            out["ai_action_probs"] = _validated_ai_action_probs(row)
            if _ai_probs_usable(out["ai_action_probs"]):
                _tag_ai_probs_symbol(out, sym)
                return out

        # Fallback 1: worker snapshot (rl_multi_decisions) voor deze symbol.
        # Dit pad dekt gevallen waar per-market Redis policy nog niet is weggeschreven,
        # maar de worker-state al wel symbol-specifieke beslissingen bevat.
        if isinstance(blob, dict):
            tenant = blob.get("tenant", blob)
            if isinstance(tenant, dict):
                snap_dec = tenant_rl_decision_for_symbol(tenant, sym)
                if isinstance(snap_dec, dict) and snap_dec:
                    row_snap = ensure_final_probs(tc._rl_decision_as_dict_with_fallback(dict(snap_dec)))
                    row_snap.setdefault("ticker", sym)
                    row_snap.setdefault("market", sym)
                    out["rl_last_decision"] = row_snap
                    out["rl_confidence"] = float(row_snap.get("confidence", 0.0) or 0.0)
                    out["generated_at"] = row_snap.get("generated_at")
                    out["prediction_timestamp"] = row_snap.get("prediction_timestamp")
                    out["ai_action_probs"] = _validated_ai_action_probs(row_snap)
                    _tag_ai_probs_symbol(out, sym)
                    if _ai_probs_usable(out.get("ai_action_probs")):
                        return out

        if isinstance(stats, dict):
            # format_stats(policy_market=sym) zet ai_action_probs al per sym; niet overschrijven met een
            # willekeurige rl_last_decision van een andere markt.
            row_fb: dict[str, Any] = {}
            sap = stats.get("ai_action_probs")
            if isinstance(sap, dict):
                for _k in ("buy_pct", "hold_pct", "sell_pct"):
                    if sap.get(_k) is not None:
                        row_fb[_k] = sap[_k]
            cand = stats.get("rl_last_decision")
            if isinstance(cand, dict):
                t = str(cand.get("ticker") or cand.get("market") or "").strip().upper().replace("/", "-")
                if not t or t == sym:
                    for k in ("prob_buy", "prob_hold", "prob_sell", "confidence", "action", "reasoning"):
                        if k in cand:
                            row_fb[k] = cand[k]
                    out["rl_last_decision"] = cand
                else:
                    out["rl_last_decision"] = None
            else:
                out["rl_last_decision"] = None
            out["ai_action_probs"] = _validated_ai_action_probs(row_fb)
            out["rl_confidence"] = stats.get("rl_confidence")
            out["generated_at"] = out["generated_at"] or stats.get("last_update")
        _tag_ai_probs_symbol(out, sym)
        return out
    except Exception:
        return {}


def _prediction_is_fresh(policy_stream: dict[str, Any], max_age_seconds: int = 60) -> bool:
    now = datetime.now(timezone.utc)
    ts = policy_stream.get("prediction_timestamp")
    if ts is not None:
        try:
            age = now.timestamp() - float(ts)
            return age <= float(max_age_seconds)
        except (TypeError, ValueError):
            pass
    iso = policy_stream.get("generated_at")
    if iso:
        try:
            dt = datetime.fromisoformat(str(iso).replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return (now - dt).total_seconds() <= float(max_age_seconds)
        except Exception:
            return False
    return False


def _read_hit_rate(symbol: str) -> float | None:
    sym = str(symbol or "").strip().upper()
    for key in (f"predictions:{sym}:hit_rate", f"{sym}:hit_rate"):
        try:
            r = _redis_client()
            try:
                raw = r.get(key)
            finally:
                r.close()
        except Exception:
            continue
        if raw is None:
            continue
        try:
            return float(raw)
        except (TypeError, ValueError):
            continue
    return None


def _history_from_bitvavo(sym: str) -> list[dict[str, Any]]:
    from app.rl.data import fetch_bitvavo_historical_candles

    sym_u = str(sym or "").strip().upper()
    now_m = time.monotonic()
    hit = _HISTORY_BV_CACHE.get(sym_u)
    if hit and (now_m - hit[0]) < _HISTORY_BV_TTL_SEC:
        return list(hit[1])

    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=int(os.getenv("PREDICTION_HISTORY_FALLBACK_DAYS", "7") or 7))
    df = fetch_bitvavo_historical_candles(sym_u, "1h", start_dt, end_dt)
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        ts = row["timestamp"]
        ts_s = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
        rows.append({"timestamp": ts_s, "price": float(row["close"])})
    if rows:
        _HISTORY_BV_CACHE[sym_u] = (now_m, list(rows))
        _log_pred.info(
            "Fresh candles fetched for %s: count=%s latest_close=%s",
            sym_u,
            len(rows),
            rows[-1]["price"],
        )
    return rows


def align_prediction_history_last_price_from_state(symbol: str, history: list[dict[str, Any]]) -> None:
    """Laatste historische candle-prijs gelijk aan actieve markt-feed (Chart.js vs hoofdgrafiek / header)."""
    if not history or not isinstance(history, list):
        return
    sym_u = str(symbol or "").strip().upper().replace("/", "-")
    try:
        from app.services.state import STATE

        lp: float | None = None
        for m in STATE.get("active_markets") or []:
            if not isinstance(m, dict):
                continue
            mk = str(m.get("market", "")).strip().upper().replace("/", "-")
            if mk != sym_u:
                continue
            for key in ("last_price", "price", "last"):
                v = m.get(key)
                if v is None:
                    continue
                try:
                    ff = float(v)
                    if ff > 0:
                        lp = ff
                        break
                except (TypeError, ValueError):
                    pass
            break
        if lp is not None:
            last = history[-1]
            if isinstance(last, dict):
                last["price"] = float(lp)
    except Exception:
        return


def _predictions_payload(symbol: str) -> dict[str, Any]:
    sym = str(symbol or "").strip().upper().replace("/", "-")
    if not sym or "-" not in sym:
        raise HTTPException(status_code=400, detail="Ongeldig symbool (verwacht bv. BTC-EUR).")

    now_c = time.monotonic()
    c_hit = _PRED_PAYLOAD_CACHE.get(sym)
    if c_hit and _PRED_PAYLOAD_TTL_SEC > 0 and (now_c - c_hit[0]) < _PRED_PAYLOAD_TTL_SEC:
        cached = copy.deepcopy(c_hit[1])
        cached["current_time"] = datetime.now(timezone.utc).isoformat()
        return cached

    history = _load_history_from_redis(sym)
    degrade_reasons: list[str] = []
    if not history:
        try:
            history = _history_from_bitvavo(sym)
        except Exception as exc:
            _log_pred.warning("Bitvavo history fallback failed for %s: %s", sym, exc)
            degrade_reasons.append(f"bitvavo_unavailable:{exc}")
            history = []

    if not history:
        now_stub = datetime.now(timezone.utc)
        history = [{"timestamp": now_stub.isoformat(), "price": 0.01}]
        degrade_reasons.append("stub_history_no_redis_bitvavo")

    align_prediction_history_last_price_from_state(sym, history)

    try:
        future_forecast = get_rl_prediction(sym, history)
    except HTTPException:
        raise
    except Exception as exc:
        _log_pred.warning("get_rl_prediction failed for %s: %s", sym, exc)
        degrade_reasons.append(f"forecast_error:{exc}")
        future_forecast = []
        try:
            lp = float(history[-1].get("price", 0.01) or 0.01)
            future_forecast = _prediction_path_from_state(sym, max(lp, 1e-9))
        except Exception:
            future_forecast = []

    hit = _read_hit_rate(sym)
    mv = "RL-Agent-v2.1-Stable"
    try:
        from app.services.state import STATE

        mv = str(STATE.get("model_version") or mv)
    except Exception:
        pass

    policy_stream = _policy_stream_for_symbol(sym)
    is_fresh = _prediction_is_fresh(policy_stream, max_age_seconds=60)

    out: dict[str, Any] = {
        "symbol": sym,
        "current_time": datetime.now(timezone.utc).isoformat(),
        "historical": history[-20:],
        "predicted": future_forecast,
        # Alias voor oudere clients / docs
        "prediction_data": future_forecast,
        "accuracy_score": hit,
        "model_version": mv,
        "ai_action_probs": policy_stream.get("ai_action_probs"),
        "rl_confidence": policy_stream.get("rl_confidence"),
        "rl_last_decision": policy_stream.get("rl_last_decision"),
        "signal_threshold_pct": policy_stream.get("signal_threshold_pct"),
        "rl_decision_threshold_pct": policy_stream.get("rl_decision_threshold_pct"),
        "generated_at": policy_stream.get("generated_at"),
        "prediction_timestamp": policy_stream.get("prediction_timestamp"),
        "prediction_fresh": bool(is_fresh),
    }
    if not is_fresh:
        out["prediction_warning"] = (
            f"Prediction heartbeat for {sym} is stale/missing; serving latest cached symbol policy."
        )
    if degrade_reasons:
        extra = "; ".join(degrade_reasons)
        prev = out.get("prediction_warning")
        out["prediction_warning"] = f"{prev} | {extra}" if prev else f"Degraded response: {extra}"
        out["prediction_degraded"] = True
    ap = out.get("ai_action_probs") if isinstance(out.get("ai_action_probs"), dict) else {}
    out["buy"] = ap.get("buy_pct")
    out["hold"] = ap.get("hold_pct")
    out["sell"] = ap.get("sell_pct")
    ap = out.get("ai_action_probs")
    rld = out.get("rl_last_decision")
    prob_keys = ()
    if isinstance(rld, dict):
        prob_keys = (rld.get("prob_hold"), rld.get("prob_buy"), rld.get("prob_sell"))
    _log_pred.info(
        "GET /api/v1/predictions/%s — hist=%s pred_steps=%s ai_action_probs=%s rl_confidence=%s raw_probs=%s",
        sym,
        len(out.get("historical") or []),
        len(out.get("predicted") or []),
        ap,
        out.get("rl_confidence"),
        prob_keys,
    )
    if str(os.getenv("PREDICTIONS_API_LOG_FULL", "")).strip().lower() in ("1", "true", "yes"):
        try:
            _log_pred.info("GET /api/v1/predictions/%s full JSON: %s", sym, json.dumps(out, default=str)[:12000])
        except Exception as exc:
            _log_pred.warning("predictions full JSON log failed: %s", exc)
    try:
        _PRED_PAYLOAD_CACHE[sym] = (time.monotonic(), copy.deepcopy(out))
    except Exception:
        pass
    return out


def _predictions_safe_minimal_json(symbol: str, err: str) -> dict[str, Any]:
    """Altijd geldige JSON voor de Chart.js-terminal als ``_predictions_payload`` faalt."""
    sym = str(symbol or "").strip().upper().replace("/", "-") or "BTC-EUR"
    now = datetime.now(timezone.utc).isoformat()
    stub = [{"timestamp": now, "price": 0.01}]
    return {
        "symbol": sym,
        "ok": False,
        "error": str(err)[:600],
        "current_time": now,
        "historical": stub,
        "predicted": [],
        "prediction_data": [],
        "accuracy_score": None,
        "model_version": "fallback",
        "ai_action_probs": {"buy_pct": 0.0, "hold_pct": 0.0, "sell_pct": 0.0, "market": sym, "ticker": sym},
        "rl_confidence": 0.0,
        "rl_last_decision": None,
        "prediction_fresh": False,
        "prediction_degraded": True,
        "prediction_warning": f"degraded: {str(err)[:200]}",
        "buy": 0.0,
        "hold": 0.0,
        "sell": 0.0,
    }


_PRED_API_TIMEOUT = float(os.getenv("PREDICTIONS_API_TIMEOUT_SEC", "12") or 12.0)


async def _predictions_async(symbol: str) -> dict[str, Any]:
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(_predictions_payload, symbol),
            timeout=_PRED_API_TIMEOUT,
        )
    except asyncio.TimeoutError:
        _log_pred.warning("GET /api/v1/predictions timeout for %s (>%.0fs)", symbol, _PRED_API_TIMEOUT)
        return _predictions_safe_minimal_json(symbol, f"timeout after {_PRED_API_TIMEOUT:.0f}s")
    except HTTPException:
        raise
    except Exception as exc:
        _log_pred.exception("GET /api/v1/predictions failed: %s", exc)
        return _predictions_safe_minimal_json(symbol, str(exc))


@router.get("/predictions")
async def get_predictions_query(
    symbol: str = Query(..., min_length=5, description="Handelspaar, bijv. ADA-EUR"),
) -> dict[str, Any]:
    return await _predictions_async(symbol)


@router.get("/predictions/{symbol}")
async def get_predictions(symbol: str) -> dict[str, Any]:
    return await _predictions_async(symbol)
