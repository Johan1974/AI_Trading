"""
Worker-side order sizing: gebruikt ``trading:constraints`` in Redis via
``RiskManager`` / ``load_risk_engine_config`` (geen vaste 1/8 alleen op env).

Na ``reset_paper_portfolio_and_state`` schrijft de portal/worker via
``core.trading_constraints_redis.apply_paper_reset_allocation_constraints`` o.a.
``max_trade_equity_pct=10`` (10% van equity per order, dus €100 bij €1.000 start).

Paper: ``PAPER_ENFORCE_ONE_OPEN_TRADE_PER_PAIR`` + ``PAPER_ENFORCE_ONE_OPEN_TRADE_PER_BASE`` (defaults aan):
BUY-blokkade: zelfde **paar** én (Bitvavo EUR-paren) max. één open positie per **basis** (één BTC-EUR, één ETH-EUR, …).
Volledige DB-wipe + €1000:
``PaperTradeManager.reset_paper_account(..., full_environment_reset=True)`` + env ``RL_EXPLORATION_*`` (default ε≈0.10 na reset) / ``STATE['decision_threshold']``.

De paper-/live-cycle in ``app.trading_core`` roept deze helper aan zodat
ordergrootte altijd de actuele Redis-config volgt.

RL-logging: ``prob_*`` in Redis = policy-softmax (leidend voor UI); het veld ``action`` kan
bij exploratie afwijken — zie ``app.rl.agent_rl`` en env ``RL_INFERENCE_GREEDY`` (ε=0).
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_log_pred_store = logging.getLogger(__name__)


def align_ppo_total_timesteps(model: Any, total_timesteps: int) -> int:
    """
    Ceil ``total_timesteps`` naar een veelvoud van PPO ``n_steps`` (minstens één rollout).

    Een **floor**-afgeronde waarde (bv. 1650 → 1024) laat SB3 te weinig stappen t.o.v. de buffer vullen;
    dat gaf o.a. ``index 1024 is out of bounds for axis 0 with size 1024`` bij RL-BG chunks.
    """
    n_steps = int(getattr(model, "n_steps", 1024) or 1024)
    n_steps = max(1, n_steps)
    ts = int(max(n_steps, int(total_timesteps)))
    k = (ts + n_steps - 1) // n_steps
    return int(max(n_steps, k * n_steps))

from core.risk_manager import RiskManager, load_risk_engine_config


def _iron_buy_decision_lock_path() -> Path:
    raw = os.getenv("IRON_BUY_DECISION_LOCK_PATH", "").strip()
    if raw:
        return Path(raw).expanduser()
    db_raw = os.getenv("TRADE_HISTORY_DB_PATH", "data/database.db")
    p = Path(db_raw).expanduser()
    if not p.is_absolute():
        p = Path.cwd() / p
    p.parent.mkdir(parents=True, exist_ok=True)
    return p.parent / ".paper_iron_buy_decision.flock"


@contextmanager
def paper_iron_buy_decision_lock():
    """Globale file-lock rond paper BUY-beslissing (cross-thread/process)."""
    if str(os.getenv("PAPER_BUY_IRON_FLOCK", "1")).strip().lower() not in ("1", "true", "yes", "on"):
        yield
        return
    try:
        import fcntl
    except ImportError:
        yield
        return
    lp = _iron_buy_decision_lock_path()
    lp.parent.mkdir(parents=True, exist_ok=True)
    fp = open(lp, "a+b")
    try:
        fcntl.flock(fp.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(fp.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass
        try:
            fp.close()
        except Exception:
            pass


def _wallet_open_market_keys(wallet: dict[str, Any] | None) -> set[str]:
    s: set[str] = set()
    if not isinstance(wallet, dict):
        return s
    pbm = wallet.get("position_by_market") or {}
    if isinstance(pbm, dict):
        for mk, q in pbm.items():
            if float(q or 0) > 1e-12:
                s.add(str(mk).strip().upper().replace("/", "-"))
    obm = wallet.get("open_lots_by_market") or {}
    if isinstance(obm, dict):
        for mk, lots in obm.items():
            if not isinstance(lots, list):
                continue
            tq = sum(float(x.get("qty") or 0) for x in lots if isinstance(x, dict))
            if tq > 1e-12:
                s.add(str(mk).strip().upper().replace("/", "-"))
    sym = str(wallet.get("position_symbol") or "").strip().upper().replace("/", "-")
    if sym and float(wallet.get("position_qty") or 0) > 1e-12:
        s.add(sym)
    return s


def _sqlite_open_markets_for_tenant(db_file: Path, tenant_id: str) -> set[str]:
    if not db_file.is_file():
        return set()
    tid = str(tenant_id or "default").strip().lower() or "default"
    out: set[str] = set()
    try:
        con = sqlite3.connect(str(db_file))
        try:
            try:
                rows = con.execute(
                    "SELECT market FROM active_positions WHERE tenant_id = ?", (tid,)
                ).fetchall() or []
            except sqlite3.Error:
                rows = []
        finally:
            con.close()
    except Exception:
        return out
    for r in rows:
        if r and r[0]:
            out.add(str(r[0]).strip().upper().replace("/", "-"))
    return out


def paper_buy_blocked_if_open_position(*, wallet: dict[str, Any] | None, ticker: str) -> tuple[bool, str]:
    """
    BUY hard gate: weiger nieuwe koop als er al een OPEN positie is voor dit paar.
    Gebruikt ``has_active_paper_position_for_ticker`` (open_lots / position_by_market) plus
    Redis ``paper:open_pair:…`` (verwijderen als stale t.o.v. wallet) en strikte wallet-lot-telling.
    """
    from core.paper_open_guard import (
        redis_clear_open_pair,
        redis_has_open_pair_flag,
        strict_block_duplicate_log,
        wallet_open_lot_count,
    )
    from core.trading_engine import (
        has_active_paper_position_for_base_currency,
        has_active_paper_position_for_ticker,
        paper_base_currency_from_market,
    )

    mku = str(ticker or "").strip().upper().replace("/", "-")
    try:
        from app.services.state import current_tenant_id

        tid = str(current_tenant_id() or "default").strip().lower() or "default"
    except Exception:
        tid = "default"

    with paper_iron_buy_decision_lock():
        dbp = Path(os.getenv("TRADE_HISTORY_DB_PATH", "data/database.db")).expanduser()
        if not dbp.is_absolute():
            dbp = Path.cwd() / dbp
        open_union = _wallet_open_market_keys(wallet) | _sqlite_open_markets_for_tenant(dbp, tid)
        if mku in open_union:
            try:
                from app.services.reporting import format_skip_buy_open_pair_log

                msg = format_skip_buy_open_pair_log(mku)
            except Exception:
                msg = f"[SKIP] Trade geweigerd: er staat al een positie open voor {mku}"
            return True, msg

        if redis_has_open_pair_flag(tid, mku) and not has_active_paper_position_for_ticker(wallet, mku):
            redis_clear_open_pair(tid, mku)
        if wallet_open_lot_count(wallet, mku) > 1:
            return True, strict_block_duplicate_log(mku)

        if has_active_paper_position_for_ticker(wallet, mku):
            try:
                from app.services.reporting import format_skip_buy_open_pair_log

                msg = format_skip_buy_open_pair_log(mku)
            except Exception:
                pair_u = str(ticker or "").strip().upper().replace("/", "-")
                msg = f"[SKIP] Trade geweigerd: er staat al een positie open voor {pair_u}"
            return True, msg

        _one_base = str(os.getenv("PAPER_ENFORCE_ONE_OPEN_TRADE_PER_BASE", "1")).strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        if _one_base and has_active_paper_position_for_base_currency(wallet, mku):
            b = paper_base_currency_from_market(mku) or "?"
            return (
                True,
                f"[SKIP] Max 1 open positie per coin ({b}) op Bitvavo (EUR-paren); sluit eerst het openstaande {b}-EUR.",
            )
        return False, ""


def enforce_emergency_learning_bootstrap() -> dict[str, Any]:
    """
    Emergency bootstrap voor 'stilgevallen' RL-runs:
    - zet exploratie terug naar 0.80 als er geen training chunks zijn;
    - forceer vroege micro-train met lage drempel (default 10 ervaringen).
    """
    metrics_db_raw = os.getenv("RL_METRICS_DB_PATH", "data/rl_training_metrics.sqlite")
    chunks = -1
    try:
        from pathlib import Path

        p = Path(metrics_db_raw)
        if not p.is_absolute():
            p = Path.cwd() / p
        if p.exists():
            con = sqlite3.connect(str(p))
            try:
                chunks = int(con.execute("SELECT COUNT(*) FROM rl_training_chunks").fetchone()[0] or 0)
            finally:
                con.close()
        else:
            chunks = 0
    except Exception as exc:
        _log_pred_store.warning("learning bootstrap metrics-db probe failed: %s", exc)
        chunks = 0

    changed: dict[str, Any] = {"chunks": chunks, "epsilon_reset": False, "min_experiences": None}

    portfolio_math_hold = False
    try:
        from app.services.reporting import read_system_state

        st = read_system_state()
        if st.get("portfolio_equity_integrity_ok") is False:
            portfolio_math_hold = True
    except Exception:
        pass

    if chunks <= 0 or portfolio_math_hold:
        # User requested manual reset: epsilon terug naar exploratief niveau.
        os.environ["RL_EXPLORATION_FINAL_EPS"] = "0.80"
        # Forceer dat er niet pas laat micro-train gestart wordt.
        if not str(os.getenv("RL_MIN_EXPERIENCES_FOR_MICRO_TRAIN", "")).strip():
            os.environ["RL_MIN_EXPERIENCES_FOR_MICRO_TRAIN"] = "10"
        # Zorg dat on-close updates aanstaan.
        micro = int(os.getenv("RL_MICRO_UPDATE_STEPS_ON_CLOSE", "256") or 256)
        if micro <= 0:
            os.environ["RL_MICRO_UPDATE_STEPS_ON_CLOSE"] = "256"
        # Background loop sneller laten happen bij stilstand.
        if not str(os.getenv("RL_TRAIN_INTERVAL_SEC", "")).strip():
            os.environ["RL_TRAIN_INTERVAL_SEC"] = "60"
        changed["epsilon_reset"] = True
        changed["min_experiences"] = int(os.getenv("RL_MIN_EXPERIENCES_FOR_MICRO_TRAIN", "10") or 10)
    if portfolio_math_hold:
        changed["epsilon_reset"] = True
        changed["portfolio_math_hold_eps"] = True
    return changed


def schedule_immediate_trade_telegram(*, snap: dict[str, Any], market: str, execution_price: float) -> None:
    """
    Paper trade → Telegram (vault). Daemon-thread: faalt nooit naar de trading-loop door;
    gebruikt ``app.services.reporting.send_trade_execution_telegram``.
    """
    if not isinstance(snap, dict):
        return
    st = str(snap.get("status") or "").lower()
    if st not in ("opened", "closed"):
        return

    def _runner() -> None:
        try:
            from app.services import reporting

            reporting.send_trade_execution_telegram(snap, str(market), float(execution_price))
        except Exception as exc:
            _log_pred_store.debug("immediate trade telegram skipped: %s", exc)

    threading.Thread(target=_runner, daemon=True).start()


def rl_overnight_learning_snapshot() -> dict[str, Any]:
    """
    Checklist voor paper-overnight + RL: triggers, persistentie, buffers, uur-logs, greedy-test.
    Gebruikt env-flags en paden (geen zware imports); voor live RL-statistieken zie ``/api/v1/rl/paper-feedback``.
    """
    from pathlib import Path

    from app.services.rl_metrics_store import hourly_metrics_jsonl_path

    micro = int(os.getenv("RL_MICRO_UPDATE_STEPS_ON_CLOSE", "1024") or 0)
    min_exp_close = int(os.getenv("RL_MIN_EXPERIENCES_FOR_MICRO_TRAIN", "1") or 1)
    stall_sec = int(os.getenv("WATCHDOG_STALL_LIMIT_SEC", "1200") or 1200)
    bg = str(os.getenv("RL_BACKGROUND_TRAIN", "0")).strip().lower() in ("1", "true", "yes", "on")
    ep_h = int(os.getenv("RL_EPISODE_HOURS", "24") or 24)
    replay_db = Path(os.getenv("RL_REPLAY_DB_PATH", "data/rl_replay_buffer.db"))
    if not replay_db.is_absolute():
        replay_db = Path.cwd() / replay_db
    metrics_db = Path(os.getenv("RL_METRICS_DB_PATH", "data/rl_training_metrics.sqlite"))
    if not metrics_db.is_absolute():
        metrics_db = Path.cwd() / metrics_db
    hourly_file = hourly_metrics_jsonl_path()
    return {
        "training_triggers": {
            "on_paper_trade_close": {
                "enabled": micro > 0,
                "RL_MICRO_UPDATE_STEPS_ON_CLOSE": micro,
                "RL_MIN_EXPERIENCES_FOR_MICRO_TRAIN": min_exp_close,
                "implements": "RL_AGENT.online_update → SB3 PPO.learn (weights wél bijgewerkt bij success)",
                "note": "Bij 0 alleen replay-SQL logging, geen weight-update op close. Default micro=1024 + min_exp=1 → eerste SELL-close triggert volwaardige rollout. Timesteps worden naar boven afgerond op veelvouden van PPO n_steps (align_ppo_total_timesteps) om buffer index-fouten te voorkomen.",
            },
            "watchdog_auto_heal": {
                "WATCHDOG_STALL_LIMIT_SEC": stall_sec,
                "note": "Standaard 1200s (20 min) zonder engine/WS-tick voordat process-restart; override in .env.",
            },
            "trading_engine_scheduled": {
                "RL_EPISODE_HOURS": ep_h,
                "implements": "TradingEngine roept online_update aan na interval (standaard 24h) na elite-sweep.",
            },
            "optional_background_ppo": {
                "RL_BACKGROUND_TRAIN": bg,
                "implements": "Parallel PPO chunks + RL_CHECKPOINT_INTERVAL_SEC hourly .zip (zie trading_core._rl_background_training_loop).",
            },
        },
        "weight_persistence": {
            "RL_MODEL_DIR": os.getenv("RL_MODEL_DIR", "artifacts/rl"),
            "after_online_update_canonical_zip": str(os.getenv("RL_SAVE_CANONICAL_ZIP_AFTER_ONLINE_UPDATE", "1")),
            "RL_CANONICAL_SAVE_MIN_SEC": os.getenv("RL_CANONICAL_SAVE_MIN_SEC", "300"),
            "hourly_timestamps_zip": str(os.getenv("RL_HOURLY_MODEL_CHECKPOINT", "1")),
            "note": "PPO wordt als .zip op schijf gezet (SB3); canonical save throttled na micro-train; uur-checkpoints via RL_HOURLY_MODEL_CHECKPOINT.",
        },
        "experience_replay_buffer": {
            "path": str(replay_db),
            "implements": "SQLite append (paper_step + trade_close); géén SB3 ReplayBuffer — PPO shuffelt minibatches intern.",
            "note": "Geen harde 'max rows' in code; schijf groeit met trades. Voor RL-train gebruikt SB3 rollout data uit BitvavoTradingEnv, niet deze SQLite.",
        },
        "metrics_logging": {
            "sqlite_training_chunks": str(metrics_db),
            "hourly_jsonl": str(hourly_file),
            "RL_HOURLY_METRICS_INTERVAL_SEC": os.getenv("RL_HOURLY_METRICS_INTERVAL_SEC", "3600"),
            "RL_HOURLY_METRICS_FILE": os.getenv("RL_HOURLY_METRICS_FILE", ""),
        },
        "intelligence_test_tomorrow": {
            "greedy_inference": "POST /api/v1/rl-inference-mode met body {\"greedy\": true} (ε≈0 policy) of zet RL_INFERENCE_GREEDY via portal/worker_commands.",
            "compare": "Paper-PnL en cockpit logs met/zonder greedy; check rl_hourly_metrics.jsonl voor dalende avg_loss / reward-trend.",
            "reset_exploration_env": "RL_EXPLORATION_FINAL_EPS / global_step_count in RL_AGENT.last_training_stats (zie /api/v1/rl/paper-feedback).",
        },
    }


def persistent_crash_log_location() -> str:
    """Absoluut pad naar ``persistent_crash.log`` (zelfde als FastAPI audit-loop / startup)."""
    from app.services.reporting import persistent_crash_log_path

    return str(persistent_crash_log_path())


def run_initial_startup_report_cold_start() -> dict[str, Any]:
    """Zelfde cold-start rapport als na ``MAIN_ENGINE.start()`` (``cold_start_send_initial_report``, flock)."""
    # NOODSITUATIE: geen startup-rapport vanuit worker/CLI (crash-loop spam).
    # from app.services.reporting import cold_start_send_initial_report
    #
    # return cold_start_send_initial_report()
    return {"skipped": True, "reason": "STARTUP_AND_EXECUTIVE_NOTIFICATIONS_MUTED"}


def morning_report_vault_and_schedule_hint() -> dict[str, Any]:
    """Paden en env-keys voor ochtendrapport (Telegram/SMTP uit vault, geen secrets in code)."""
    from app.services.reporting import (
        initial_report_cold_start_lock_path,
        morning_report_marker_path,
        night_baseline_path,
        persistent_crash_log_path,
        resolve_vault_paths,
    )

    return {
        "vault_paths_tried": [str(p) for p in resolve_vault_paths()],
        "vault_keys_for_notifications": sorted(
            [
                "TELEGRAM_TOKEN",
                "TELEGRAM_CHAT_ID",
                "TELEGRAM_ENABLED",
                "SMTP_SERVER",
                "SMTP_PORT",
                "SMTP_USER",
                "SMTP_PASS",
                "EMAIL_RECEIVER",
                "PRIVATE_SMTP",
                "PRIVATE_PORT",
                "PRIVATE_EMAIL",
                "PRIVATE_PASS",
                "MORNING_REPORT_EMAIL_TO",
            ]
        ),
        "initial_report_cold_start_lock": str(initial_report_cold_start_lock_path()),
        "initial_report": (
            "cold_start_send_initial_report() via trading_core._send_initial_report_after_startup na MAIN_ENGINE.start; "
            "flock dedup bij multi-worker; INITIAL_REPORT_DELAY_SEC, SKIP_INITIAL_REPORT."
        ),
        "schedule": "08:00 Europe/Amsterdam + catch-up na opstart; asyncio-loop in trading_core / worker-tak in main.py.",
        "marker_file": str(morning_report_marker_path()),
        "night_baseline_file": str(night_baseline_path()),
        "persistent_crash_log": str(persistent_crash_log_path()),
        "persistent_crash_note": "ERROR+traceback bij fatale fouten in main.py (audit-loop/startup); overleeft restart:always.",
        "startup_telegram_throttle": {
            "STARTUP_COOLDOWN_SEC": os.getenv("STARTUP_COOLDOWN_SEC", "3600"),
            "DISABLE_STARTUP_BRIEFING_TELEGRAM": os.getenv("DISABLE_STARTUP_BRIEFING_TELEGRAM", "0"),
            "EXTENDED_STARTUP_TELEGRAM_ON_AUTO_RESTART": os.getenv("EXTENDED_STARTUP_TELEGRAM_ON_AUTO_RESTART", "0"),
            "note": "Jarvis Performance+Integrity en restart-audit delen flock+system_state; bij startup_mode=auto standaard stil tenzij EXTENDED_=1.",
        },
        "emergency_notification_mute": {
            "EXECUTIVE_NOTIFICATIONS_EMERGENCY_MUTE": "True in reporting.py — heropen executive sends met MUTE_EXECUTIVE_NOTIFICATIONS=0 (override noodsituatie); of zet constant op False",
            "PORTFOLIO_MATH_FATAL_EXIT": os.getenv("PORTFOLIO_MATH_FATAL_EXIT", "1"),
            "BTC_qty": "reporting.reporting_coerce_btc_qty_to_btc_base — satoshi ≥100000 → /1e8 vóór normalize",
            "ghost_positions": "reporting.sanitize_wallet_corrupt_positions_vs_cash — lot dropped als notional > cash",
            "over_allocation_strip": "reporting.strip_wallet_over_allocation_markets — na invalid_weight eerst corrupte markten wissen (PORTFOLIO_STRIP_OVERALLOCATION=1 default)",
            "PORTFOLIO_STRIP_ALLOCATION_RAW_WEIGHT_GT": os.getenv("PORTFOLIO_STRIP_ALLOCATION_RAW_WEIGHT_GT", "100.02"),
        },
        "portfolio_integrity": {
            "AUTO_REPAIR_BRIDGE": os.getenv("AUTO_REPAIR_BRIDGE", "1"),
            "AUTO_REPAIR_IMPOSSIBLE_REL_PCT": os.getenv("AUTO_REPAIR_IMPOSSIBLE_REL_PCT", "1000"),
            "repair_request_json": "/app/logs/repair_request.json (Docker) or ./_logs_hub/repair_request.json",
            "PORTFOLIO_EQUITY_MISMATCH_PCT": os.getenv("PORTFOLIO_EQUITY_MISMATCH_PCT", "5"),
            "PAPER_EQUITY_HARD_CAP_MULT": os.getenv("PAPER_EQUITY_HARD_CAP_MULT", "1.35"),
            "system_state_keys": ["portfolio_equity_integrity_ok", "portfolio_equity_integrity_detail"],
            "qty_normalization": "app.services.portfolio_qty.normalize_paper_base_qty (BTC satoshi / ETH wei heuristiek)",
            "sqlite_startup": "PaperTradeManager._sanitize_wallet_snapshot_from_sqlite — DB-RAW console + ghost wipe + persist",
            "allocation_invalid_weight": "core.risk_management.allocation_snapshot invalid_weight → reporting blokkeert sends",
            "epsilon_hold": "system_state ok=False → enforce_emergency_learning_bootstrap houdt RL_EXPLORATION_FINAL_EPS=0.80; auto-opt floor 0.80 in trading_core.",
        },
        "env": {
            "TRADING_VAULT_PATH": os.getenv("TRADING_VAULT_PATH", ""),
            "MORNING_REPORT_CATCHUP_FROM_HOUR": os.getenv("MORNING_REPORT_CATCHUP_FROM_HOUR", "8"),
            "MORNING_REPORT_CATCHUP_UNTIL_HOUR": os.getenv("MORNING_REPORT_CATCHUP_UNTIL_HOUR", "13"),
            "MORNING_REPORT_WINDOW_START_HOUR": os.getenv("MORNING_REPORT_WINDOW_START_HOUR", "22"),
            "MORNING_REPORT_WINDOW_END_HOUR": os.getenv("MORNING_REPORT_WINDOW_END_HOUR", "8"),
            "MORNING_REPORT_BASELINE_HOUR": os.getenv("MORNING_REPORT_BASELINE_HOUR", "23"),
        },
    }


def paper_rl_learning_rate_snapshot() -> dict[str, Any]:
    """
    Actieve PPO learning_rate staat in ``RL_AGENT.last_training_stats`` (SB3 ``lr_schedule`` na ``learn``).
    Constructor-default in ``app.rl.agent_rl`` ensure_pretrained: 3e-4 — overschrijf via env indien ondersteund.
    """
    try:
        from app.trading_core import RL_AGENT

        st = getattr(RL_AGENT, "last_training_stats", None)
        if not isinstance(st, dict):
            return {"note": "RL_AGENT.last_training_stats leeg (model nog niet geladen)."}
        return {
            "learning_rate": st.get("learning_rate"),
            "global_step_count": st.get("global_step_count"),
            "exploration_rate_pct": st.get("exploration_rate_pct"),
            "env_RL_LEARNING_RATE": os.getenv("RL_LEARNING_RATE") or os.getenv("PPO_LEARNING_RATE"),
            "note": "Micro-updates na paper-close: RL_MICRO_UPDATE_STEPS_ON_CLOSE (default 256 in trading_core).",
        }
    except Exception as exc:
        return {"error": str(exc), "note": "Alleen beschikbaar waar ``app.trading_core`` met RL geladen is."}


def min_observation_interval_sec() -> float:
    """Throttle voor worker-observatie/predict loops (standaard 12s, min 10s)."""
    raw = float(os.getenv("WORKER_OBSERVATION_INTERVAL_SEC", "12") or 12.0)
    return max(10.0, raw)


def is_paper_execution_paused() -> bool:
    """Zet env ``PAPER_EXECUTION_PAUSE=1`` om de synchrone paper-cycle direct te laten returnen (geen orders)."""
    v = str(os.getenv("PAPER_EXECUTION_PAUSE", "0")).strip().lower()
    return v in ("1", "true", "yes", "on")


def calculate_order_size_for_signal(
    *,
    signal: str,
    equity: float,
    cash: float,
    price: float,
    wallet: dict[str, Any],
    market: str,
) -> tuple[float, float, str]:
    """
    (size_fraction, quote_eur, note) — zelfde contract als ``RiskManager.calculate_trade_size``.
    BUY: harde vloer op beschikbaar cash (default €100) en order NOTIONAL + fee ≤ cash.
    """
    rm = RiskManager(load_risk_engine_config())
    sig_u = str(signal or "").upper()
    c = float(cash or 0.0)
    if sig_u == "BUY":
        try:
            floor = float(os.getenv("PAPER_MIN_CASH_FOR_NEW_BUY_EUR", "10") or 10.0)
        except (TypeError, ValueError):
            floor = 10.0
        if c < floor:
            return 0.0, 0.0, "blocked_cash_below_min_buy_floor"
    frac, quote, note = rm.calculate_trade_size(
        signal=signal,
        equity=equity,
        cash=cash,
        price=price,
        wallet=wallet,
        market=market,
    )
    if sig_u == "BUY" and quote > 0:
        try:
            fee_r = float(
                os.getenv("PAPER_TRADING_FEE_RATE", os.getenv("BITVAVO_TAKER_FEE", "0.0015")) or 0.0015
            )
        except (TypeError, ValueError):
            fee_r = 0.0015
        fee_r = max(0.0, fee_r)
        max_quote = c / (1.0 + fee_r) if c > 0 else 0.0
        if quote > max_quote + 1e-9:
            quote = max(0.0, max_quote)
            anchor = float(
                (wallet.get("paper_anchor_equity_eur") if isinstance(wallet, dict) else None)
                or (wallet.get("starting_balance_eur") if isinstance(wallet, dict) else None)
                or equity
            )
            anchor = max(1e-9, anchor)
            frac = min(1.0, quote / anchor)
            note = f"{note}|clamped_to_affordable_cash"
        if quote <= 1e-12:
            return 0.0, 0.0, "insufficient_cash_after_affordability_clamp"
    return frac, quote, note


def redis_prediction_policy_key(symbol: str) -> str:
    """Redis-sleutel per paar (default ``prediction:ADA-EUR``)."""
    sym = str(symbol or "").strip().upper().replace("/", "-")
    prefix = str(os.getenv("REDIS_PREDICTION_KEY_PREFIX", "prediction:")).strip()
    return f"{prefix}{sym}"


def ensure_final_probs(policy: dict[str, Any]) -> dict[str, Any]:
    """
    Zorg dat policy altijd expliciete final probs bevat:
    - prob_buy/prob_hold/prob_sell (0..1), som ≈ 1
    - buy/hold/sell (0..100)

    Ontbrekende of ongeldige tensors: normaliseer op ``action`` (zachte 3-weg) of uniform ⅓,
    zodat Redis/UI nooit een lege policy krijgen (geen hardcoded 40/50/10).
    """
    row = dict(policy) if isinstance(policy, dict) else {}

    def _f(v: Any) -> float | None:
        try:
            x = float(v)
            return x if x == x and x >= 0.0 else None
        except (TypeError, ValueError):
            return None

    for prob_key, alt_key, legacy_pct in (
        ("prob_buy", "buy_pct", "buy"),
        ("prob_hold", "hold_pct", "hold"),
        ("prob_sell", "sell_pct", "sell"),
    ):
        if row.get(prob_key) is None and row.get(alt_key) is not None:
            row[prob_key] = row[alt_key]
        if row.get(prob_key) is None and row.get(legacy_pct) is not None:
            lv = _f(row.get(legacy_pct))
            if lv is not None:
                row[prob_key] = lv / 100.0 if lv > 1.0 + 1e-9 else lv

    pb, ph, ps = _f(row.get("prob_buy")), _f(row.get("prob_hold")), _f(row.get("prob_sell"))
    status_l = str(row.get("policy_status") or "").strip().lower()
    unavailable = bool(row.get("analysis_unavailable")) or status_l in ("unavailable", "invalid", "error")
    for i, v in enumerate([pb, ph, ps]):
        if v is not None and v > 1.0 + 1e-9:
            if i == 0:
                pb = v / 100.0
            elif i == 1:
                ph = v / 100.0
            else:
                ps = v / 100.0

    # Als upstream expliciet "unavailable" meldt, ga nooit naar action_softmax fallback.
    if unavailable:
        pb = float(pb or 0.0)
        ph = float(ph or 0.0)
        ps = float(ps or 0.0)
        s_u = pb + ph + ps
        if s_u > 1e-9:
            pb, ph, ps = pb / s_u, ph / s_u, ps / s_u
        else:
            pb, ph, ps = 0.0, 0.0, 0.0
        row["policy_status"] = "unavailable"
        row["analysis_unavailable"] = True
        row["prob_buy"] = float(pb)
        row["prob_hold"] = float(ph)
        row["prob_sell"] = float(ps)
        row["buy"] = round(row["prob_buy"] * 100.0, 2)
        row["hold"] = round(row["prob_hold"] * 100.0, 2)
        row["sell"] = round(row["prob_sell"] * 100.0, 2)
        mx = max(float(pb), float(ph), float(ps))
        row["dominant_policy_prob"] = round(mx, 6)
        return row

    have_all = pb is not None and ph is not None and ps is not None
    if have_all:
        s0 = float(pb) + float(ph) + float(ps)
        if s0 > 1e-9:
            pb, ph, ps = float(pb) / s0, float(ph) / s0, float(ps) / s0
        else:
            have_all = False

    if not have_all:
        act = str(row.get("action", "") or "").strip().upper()
        row["policy_fallback_used"] = True
        if act == "BUY":
            pb, ph, ps = 0.50, 0.33, 0.17
        elif act == "SELL":
            pb, ph, ps = 0.17, 0.33, 0.50
        elif act == "HOLD":
            pb, ph, ps = 0.31, 0.38, 0.31
        else:
            pb, ph, ps = 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0
        row["policy_status"] = "action_softmax" if act in ("BUY", "SELL", "HOLD") else "uniform_prior"
        row["analysis_unavailable"] = False
    else:
        row["policy_fallback_used"] = False
        if not row.get("policy_status"):
            row["policy_status"] = "ok"
        row["analysis_unavailable"] = bool(row.get("analysis_unavailable", False))

    row["prob_buy"] = float(pb)
    row["prob_hold"] = float(ph)
    row["prob_sell"] = float(ps)
    row["buy"] = round(row["prob_buy"] * 100.0, 2)
    row["hold"] = round(row["prob_hold"] * 100.0, 2)
    row["sell"] = round(row["prob_sell"] * 100.0, 2)

    # Actie-confidence (0..1): zelfde schaal als agent_rl RL_ACTION_MIN_CONFIDENCE / reasoning-logs.
    # Niet verwarren met max(prob_buy, prob_hold, prob_sell) — dat zijn policy-marges voor de UI-balkjes.
    conf_v = _f(row.get("confidence"))
    if conf_v is not None:
        if conf_v > 1.0 + 1e-6:
            conf_v = conf_v / 100.0
        row["confidence"] = round(max(0.0, min(1.0, float(conf_v))), 4)
    else:
        act_u = str(row.get("action") or "").strip().upper()
        pick = {"HOLD": ph, "BUY": pb, "SELL": ps}.get(act_u)
        if pick is not None:
            row["confidence"] = round(max(0.0, min(1.0, float(pick))), 4)

    mx = max(float(pb), float(ph), float(ps))
    row["dominant_policy_prob"] = round(mx, 6)
    return row


def read_per_market_prediction_policy(symbol: str) -> dict[str, Any] | None:
    """Leest JSON policy per paar (``prediction:ADA-EUR``) — portal gebruikt dit voor symbool-specifieke UI."""
    try:
        import redis

        url = str(os.getenv("REDIS_URL", "")).strip()
        if not url:
            host = str(os.getenv("REDIS_HOST", "redis")).strip()
            port = str(os.getenv("REDIS_PORT", "6379")).strip()
            url = f"redis://{host}:{port}/0"
        if "localhost" in url or "127.0.0.1" in url:
            host = str(os.getenv("REDIS_HOST", "redis")).strip()
            port = str(os.getenv("REDIS_PORT", "6379")).strip()
            url = f"redis://{host}:{port}/0"
        r = redis.Redis.from_url(url, decode_responses=True, socket_connect_timeout=1.5, socket_timeout=1.5)
        try:
            sym = str(symbol or "").strip().upper().replace("/", "-")
            key_primary = redis_prediction_policy_key(symbol)
            raw = r.get(key_primary)
            if not raw and sym:
                raw = r.get(f"pred:{sym}")
        finally:
            try:
                r.close()
            except Exception:
                pass
        if not raw:
            return None
        out = json.loads(raw)
        return out if isinstance(out, dict) else None
    except Exception:
        return None


def write_per_market_prediction_policy(symbol: str, policy: dict[str, Any]) -> None:
    """Schrijft RL-policy snapshot per markt (JSON, TTL) voor portal/API-readers."""
    try:
        import redis

        url = str(os.getenv("REDIS_URL", "")).strip()
        if not url:
            host = str(os.getenv("REDIS_HOST", "redis")).strip()
            port = str(os.getenv("REDIS_PORT", "6379")).strip()
            url = f"redis://{host}:{port}/0"
        if "localhost" in url or "127.0.0.1" in url:
            host = str(os.getenv("REDIS_HOST", "redis")).strip()
            port = str(os.getenv("REDIS_PORT", "6379")).strip()
            url = f"redis://{host}:{port}/0"
        ttl = max(60, int(os.getenv("REDIS_POLICY_TTL_SEC", "7200") or 7200))
        r = redis.Redis.from_url(url, decode_responses=True, socket_connect_timeout=1.5, socket_timeout=1.5)
        try:
            sym = str(symbol or "").strip().upper().replace("/", "-")
            key = redis_prediction_policy_key(symbol)
            now_iso = datetime.now(timezone.utc).isoformat()
            now_ts = int(datetime.now(timezone.utc).timestamp())
            row = dict(policy) if isinstance(policy, dict) else {}
            row = ensure_final_probs(row)
            act_u = str(row.get("action") or "").strip().upper()
            if act_u in ("HOLD", "BUY", "SELL") and str(row.get("policy_status") or "") == "ok":
                try:
                    ph, pb, ps = float(row["prob_hold"]), float(row["prob_buy"]), float(row["prob_sell"])
                    triple = [("HOLD", ph), ("BUY", pb), ("SELL", ps)]
                    dom = max(triple, key=lambda x: x[1])[0]
                    if dom != act_u:
                        _log_pred_store.info(
                            "[%s] Policy-dominant (softmax): %s (H %.1f%% B %.1f%% S %.1f%%) — veld action=%s "
                            "(kan onder RL-exploratie afwijken; zet RL_INFERENCE_GREEDY=1 voor greedy).",
                            sym or key,
                            dom,
                            ph * 100.0,
                            pb * 100.0,
                            ps * 100.0,
                            act_u,
                        )
                except Exception:
                    pass
            row["generated_at"] = str(row.get("generated_at") or now_iso)
            row["prediction_timestamp"] = int(row.get("prediction_timestamp") or now_ts)
            payload = json.dumps(row, default=str)
            r.set(key, payload, ex=ttl)
            if sym:
                r.set(f"pred:{sym}", payload, ex=ttl)
            bh = row.get("prob_buy")
            hh = row.get("prob_hold")
            sh = row.get("prob_sell")
            latest_price = row.get("latest_close")

            def _pct_disp(v: Any) -> float:
                try:
                    f = float(v)
                except (TypeError, ValueError):
                    return float("nan")
                if f != f:
                    return float("nan")
                return f * 100.0 if f <= 1.0 + 1e-6 else min(100.0, f)

            b_, h_, s_ = _pct_disp(bh), _pct_disp(hh), _pct_disp(sh)
            _log_pred_store.info(
                "Saving prediction for %s: buy/hold/sell %% = %.0f/%.0f/%.0f (Redis keys %s, pred:%s)",
                sym or key,
                b_ if b_ == b_ else -1.0,
                h_ if h_ == h_ else -1.0,
                s_ if s_ == s_ else -1.0,
                key,
                sym or "",
            )
            probs = {"buy": b_, "hold": h_, "sell": s_}
            _log_pred_store.info("Latest candle close for %s received by AI: %s", sym or key, latest_price)
            _log_pred_store.info(f"New prediction calculated for {sym or key}: {probs}")
            try:
                _log_pred_store.info(
                    "[POLICY-WRITE] market=%s key=%s status=%s fallback=%s probs_raw=(buy=%s hold=%s sell=%s) probs_pct=(%.2f/%.2f/%.2f)",
                    sym or key,
                    key,
                    str(row.get("policy_status") or "unknown"),
                    bool(row.get("policy_fallback_used", False)),
                    row.get("prob_buy"),
                    row.get("prob_hold"),
                    row.get("prob_sell"),
                    b_ if b_ == b_ else -1.0,
                    h_ if h_ == h_ else -1.0,
                    s_ if s_ == s_ else -1.0,
                )
            except Exception:
                pass
        finally:
            try:
                r.close()
            except Exception:
                pass
    except Exception:
        pass
