#!/usr/bin/env python3
"""
Iron Lockdown: leeg active_positions + paper_trade_history, zet paper wallet op €1000 in SQLite,
Redis trading:constraints + snapshot paper_portfolio cash/equity, wis paper:* locks/flags.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

UTC = timezone.utc

DEFAULT_WALLET = {
    "cash": 1000.0,
    "equity": 1000.0,
    "starting_balance_eur": 1000.0,
    "paper_anchor_equity_eur": 1000.0,
    "position_qty": 0.0,
    "position_symbol": None,
    "avg_entry_price": 0.0,
    "last_price": None,
    "realized_pnl_eur": 0.0,
    "realized_pnl_pct": 0.0,
    "trades_count": 0,
    "wins": 0,
    "losses": 0,
    "history": [],
    "open_lots": [],
    "open_lots_by_market": {},
    "position_by_market": {},
    "last_prices_by_market": {},
}


def _resolve_db(cli: str) -> Path:
    raw = cli or str(os.getenv("TRADE_HISTORY_DB_PATH", "data/database.db")).strip()
    p = Path(raw).expanduser()
    return p if p.is_absolute() else Path.cwd() / p


def _sql_ignore(con: sqlite3.Connection, sql: str) -> None:
    try:
        con.execute(sql)
    except sqlite3.Error:
        pass


def _patch_redis_snapshot(cash: float, equity: float) -> None:
    try:
        import redis
    except ImportError:
        print("redis-py niet geïnstalleerd; oversla Redis snapshot-patch.")
        return
    host = str(os.getenv("REDIS_HOST", "127.0.0.1")).strip() or "127.0.0.1"
    port = int(os.getenv("REDIS_PORT", "6379") or 6379)
    url = str(os.getenv("REDIS_URL", f"redis://{host}:{port}/0")).strip()
    try:
        r = redis.Redis.from_url(url, decode_responses=True, socket_connect_timeout=2.0)
        try:
            raw = r.hget("worker_snapshot", "data") or r.get("ai_trading_snapshot")
            if not raw:
                print("Geen ai_trading_snapshot in Redis (worker mogelijk uit).")
                return
            blob = json.loads(raw)
            if not isinstance(blob, dict):
                return
            tenant = blob.get("tenant")
            if isinstance(tenant, dict):
                pf = tenant.get("paper_portfolio")
                if isinstance(pf, dict):
                    pf["cash"] = float(cash)
                    pf["equity"] = float(equity)
                    pf["open_lots_by_market"] = {}
                    pf["position_by_market"] = {}
                    pf["open_lots"] = []
                    pf["position_qty"] = 0.0
                    pf["position_symbol"] = None
                    tenant["paper_portfolio"] = pf
                    blob["tenant"] = tenant
            js = json.dumps(blob, separators=(",", ":"), ensure_ascii=False)
            r.hset("worker_snapshot", "data", js)
            r.set("ai_trading_snapshot", js)
            print("Redis ai_trading_snapshot: cash/equity → €1000, posities leeg.")
        finally:
            r.close()
    except Exception as exc:
        print(f"Redis patch mislukt ({exc}); start worker opnieuw na DB-reset.")


def _clear_paper_redis_keys() -> None:
    try:
        import redis
    except ImportError:
        return
    url = str(os.getenv("REDIS_URL", "")).strip()
    if not url:
        h = str(os.getenv("REDIS_HOST", "127.0.0.1")).strip() or "127.0.0.1"
        p = str(os.getenv("REDIS_PORT", "6379")).strip() or "6379"
        url = f"redis://{h}:{p}/0"
    try:
        r = redis.Redis.from_url(url, decode_responses=True, socket_connect_timeout=2.0)
        try:
            for key in r.scan_iter("paper:open_pair:*"):
                r.delete(key)
            for key in r.scan_iter("paper:buy_lock:*"):
                r.delete(key)
            print("Redis paper:open_pair:* en paper:buy_lock:* gewist.")
        finally:
            r.close()
    except Exception as exc:
        print(f"Redis key cleanup: {exc}")


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    ap = argparse.ArgumentParser(description="Iron Lockdown DB + Redis paper reset")
    ap.add_argument("--db", default="", help="SQLite pad (default TRADE_HISTORY_DB_PATH / data/database.db)")
    ap.add_argument("--tenant", default="default")
    ap.add_argument("--no-redis", action="store_true")
    args = ap.parse_args()
    db_path = _resolve_db(args.db)
    if not db_path.exists():
        print(f"Database niet gevonden: {db_path}")
        return 1
    tid = str(args.tenant or "default").strip().lower() or "default"
    snap = json.dumps(DEFAULT_WALLET, ensure_ascii=False)
    ts = datetime.now(UTC).isoformat()

    with sqlite3.connect(str(db_path)) as con:
        _sql_ignore(con, "DELETE FROM active_positions")
        _sql_ignore(con, "DELETE FROM paper_trade_history")
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS wallet_state (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tenant_id TEXT NOT NULL UNIQUE,
                snapshot_json TEXT NOT NULL,
                updated_ts_utc TEXT NOT NULL
            )
            """
        )
        con.execute(
            """
            INSERT INTO wallet_state (tenant_id, snapshot_json, updated_ts_utc)
            VALUES (?, ?, ?)
            ON CONFLICT(tenant_id) DO UPDATE SET
                snapshot_json = excluded.snapshot_json,
                updated_ts_utc = excluded.updated_ts_utc
            """,
            (tid, snap, ts),
        )
        con.commit()
        print(f"OK SQLite {db_path}: active_positions + paper_trade_history gewist; wallet_state → €1000 (tenant={tid})")

    if not args.no_redis:
        _clear_paper_redis_keys()
        try:
            from core.trading_constraints_redis import apply_paper_reset_allocation_constraints

            apply_paper_reset_allocation_constraints(equity_eur=1000.0)
            print("Redis trading:constraints → paper reset (10% sizing t.o.v. €1000).")
        except Exception as exc:
            print(f"trading:constraints: {exc}")
        _patch_redis_snapshot(1000.0, 1000.0)

    print("\nZet in .env / docker-compose.env o.a.:")
    print("  IRON_LOCKDOWN=1")
    print("  RL_EXPLORATION_INFERENCE_EPS=0.10")
    print("  RL_EXPLORATION_FINAL_EPS=0.10")
    print("  RL_EXPLORATION_MIN_EPS=0.10")
    print("Herstart worker + portal.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
