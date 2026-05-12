"""
Experience replay buffer (SQLite) voor paper/RL-feedback.

Slaat per stap/markt op: uitgevoerd signaal, policy-softmax, greedy-actie,
RSI/EMA-gap/volume (uit last_scores), prijs. Bij trade-close: reward (PnL%) + logregel.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)

_DB_PATH = Path(os.getenv("RL_REPLAY_DB_PATH", "data/rl_replay_buffer.db"))


def _conn() -> sqlite3.Connection:
    path = _DB_PATH if _DB_PATH.is_absolute() else Path.cwd() / _DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(str(path), timeout=30.0)


def init_replay_db() -> None:
    with _conn() as c:
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS rl_replay_experience (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts_utc TEXT NOT NULL,
                market TEXT NOT NULL,
                executed_signal TEXT NOT NULL,
                policy_greedy TEXT,
                prob_hold REAL,
                prob_buy REAL,
                prob_sell REAL,
                explored INTEGER DEFAULT 0,
                rsi_14 REAL,
                ema_gap_pct REAL,
                volume_change REAL,
                price REAL,
                reward_pct REAL,
                kind TEXT NOT NULL,
                meta_json TEXT
            )
            """
        )
        c.execute("CREATE INDEX IF NOT EXISTS idx_replay_market_ts ON rl_replay_experience(market, ts_utc)")


def append_paper_experience(
    *,
    market: str,
    executed_signal: str,
    prob_hold: float,
    prob_buy: float,
    prob_sell: float,
    policy_greedy: str,
    explored: bool,
    rsi_14: float,
    ema_gap_pct: float,
    volume_change: float,
    price: float,
    meta: dict[str, Any] | None = None,
) -> None:
    init_replay_db()
    mkt = str(market or "").strip().upper().replace("/", "-")
    if not mkt:
        return
    ph, pb, ps = float(prob_hold), float(prob_buy), float(prob_sell)
    ssum = ph + pb + ps
    if ssum > 1e-12:
        ph, pb, ps = ph / ssum, pb / ssum, ps / ssum
    greedy = max(("HOLD", ph), ("BUY", pb), ("SELL", ps), key=lambda x: x[1])[0]
    if policy_greedy and policy_greedy.upper() in ("HOLD", "BUY", "SELL"):
        greedy = policy_greedy.upper()
    row_meta = dict(meta or {})
    row_meta["executed"] = str(executed_signal or "").upper()
    try:
        with _conn() as c:
            c.execute(
                """
                INSERT INTO rl_replay_experience (
                    ts_utc, market, executed_signal, policy_greedy,
                    prob_hold, prob_buy, prob_sell, explored,
                    rsi_14, ema_gap_pct, volume_change, price, reward_pct, kind, meta_json
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    datetime.utcnow().isoformat() + "Z",
                    mkt,
                    str(executed_signal or "").upper()[:8],
                    greedy,
                    ph,
                    pb,
                    ps,
                    1 if explored else 0,
                    float(rsi_14),
                    float(ema_gap_pct),
                    float(volume_change),
                    float(price),
                    None,
                    "paper_step",
                    json.dumps(row_meta, default=str)[: 8000],
                ),
            )
    except Exception as exc:
        _log.debug("replay append skip: %s", exc)


def append_trade_close(
    *,
    market: str,
    trade_num: int,
    reward_pct: float,
    pnl_eur: float,
    model_updated: bool,
) -> None:
    """Reward-rij + compacte logtekst voor cockpit / browser log."""
    init_replay_db()
    mkt = str(market or "").strip().upper().replace("/", "-")
    meta = {"pnl_eur": float(pnl_eur), "model_updated": bool(model_updated)}
    line = (
        f"Paper trade #{int(trade_num)} closed ({mkt}). Reward: {float(reward_pct):+.2f}% "
        f"(≈€{float(pnl_eur):+.2f}). "
        + ("Model micro-update uitgevoerd." if model_updated else "Geen PPO micro-update (RL_MICRO_UPDATE_STEPS_ON_CLOSE=0).")
    )
    try:
        with _conn() as c:
            c.execute(
                """
                INSERT INTO rl_replay_experience (
                    ts_utc, market, executed_signal, policy_greedy,
                    prob_hold, prob_buy, prob_sell, explored,
                    rsi_14, ema_gap_pct, volume_change, price, reward_pct, kind, meta_json
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    datetime.utcnow().isoformat() + "Z",
                    mkt,
                    "CLOSE",
                    "",
                    0.0,
                    0.0,
                    0.0,
                    0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    float(reward_pct),
                    "trade_close",
                    json.dumps({**meta, "log_line": line}, default=str)[: 8000],
                ),
            )
    except Exception as exc:
        _log.debug("replay close skip: %s", exc)


def replay_buffer_stats() -> dict[str, Any]:
    init_replay_db()
    try:
        with _conn() as c:
            n = int(c.execute("SELECT COUNT(*) FROM rl_replay_experience").fetchone()[0])
            last = c.execute(
                "SELECT ts_utc, kind, market, reward_pct FROM rl_replay_experience ORDER BY id DESC LIMIT 1"
            ).fetchone()
    except Exception:
        return {"rows": 0, "last": None}
    out = {"rows": n, "last": None}
    if last:
        out["last"] = {"ts_utc": last[0], "kind": last[1], "market": last[2], "reward_pct": last[3]}
    return out
