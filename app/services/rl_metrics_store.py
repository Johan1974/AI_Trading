"""
Persist RL training curve samples (reward, policy/value loss) to SQLite for overnight runs
and reload into RLAgentService for the AI Brain tab.
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from app.datetime_util import UTC

if TYPE_CHECKING:
    from app.rl.agent_rl import RLAgentService


def _db_path() -> Path:
    # Default naast trade_history; in Docker zet RL_METRICS_DB_PATH naar /app/storage/... (volume).
    raw = os.getenv("RL_METRICS_DB_PATH", "data/rl_training_metrics.sqlite")
    p = Path(raw)
    if not p.is_absolute():
        p = Path.cwd() / p
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_db_path()))
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def init_rl_metrics_db() -> None:
    with _conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS rl_training_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts_utc TEXT NOT NULL,
                pair TEXT NOT NULL,
                rewards_json TEXT NOT NULL,
                loss_json TEXT NOT NULL,
                value_loss_json TEXT NOT NULL,
                policy_entropy_json TEXT NOT NULL,
                episode_length_json TEXT NOT NULL,
                global_step INTEGER NOT NULL
            )
            """
        )


def append_training_chunk(
    pair: str,
    rewards: list[float],
    loss: list[float],
    value_loss: list[float],
    policy_entropy: list[float],
    episode_length: list[float],
    global_step: int,
) -> None:
    if not (rewards or loss or value_loss):
        return
    init_rl_metrics_db()
    ts = datetime.now(UTC).isoformat()
    with _conn() as conn:
        conn.execute(
            """
            INSERT INTO rl_training_chunks (
                ts_utc, pair, rewards_json, loss_json, value_loss_json,
                policy_entropy_json, episode_length_json, global_step
            ) VALUES (?,?,?,?,?,?,?,?)
            """,
            (
                ts,
                pair.upper(),
                json.dumps(rewards),
                json.dumps(loss),
                json.dumps(value_loss),
                json.dumps(policy_entropy),
                json.dumps(episode_length),
                int(global_step),
            ),
        )


def merge_historical_training_into_agent(agent: RLAgentService, max_points: int | None = None) -> None:
    """Load persisted chunks from SQLite into agent.last_training_progress / last_network_logs."""
    cap = max_points if max_points is not None else int(os.getenv("RL_TRAINING_CHART_MAX_POINTS", "8000") or 8000)
    cap = max(200, min(cap, 50000))
    path = _db_path()
    if not path.exists():
        init_rl_metrics_db()
        return
    init_rl_metrics_db()
    rewards: list[float] = []
    loss: list[float] = []
    value_loss: list[float] = []
    policy_entropy: list[float] = []
    episode_length: list[float] = []
    last_step = 0
    with _conn() as conn:
        rows = conn.execute(
            "SELECT rewards_json, loss_json, value_loss_json, policy_entropy_json, episode_length_json, global_step "
            "FROM rl_training_chunks ORDER BY id ASC"
        ).fetchall()
    for row in rows:
        try:
            rewards.extend(json.loads(row[0]))
            loss.extend(json.loads(row[1]))
            value_loss.extend(json.loads(row[2]))
            policy_entropy.extend(json.loads(row[3]))
            episode_length.extend(json.loads(row[4]))
            last_step = max(last_step, int(row[5] or 0))
        except Exception:
            continue
    if not rewards and not loss:
        return
    agent.last_training_progress = {
        "reward": rewards[-cap:],
        "loss": loss[-cap:],
        "policy_entropy": policy_entropy[-cap:],
        "episode_length": episode_length[-cap:],
    }
    agent.last_network_logs = {
        "approx_kl": agent.last_network_logs.get("approx_kl", []),
        "value_loss": value_loss[-cap:],
    }
    if last_step > int(agent.last_training_stats.get("global_step_count", 0) or 0):
        agent.last_training_stats["global_step_count"] = last_step
