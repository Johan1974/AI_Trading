#!/usr/bin/env python3
"""
Harde paper-reset: SQLite-ledgers legen, wallet op vaste cash/equity, optioneel Redis FLUSHALL,
daarna trading:constraints opnieuw (10% equity ≈ €100 bij €1000).

Standaard-DB: ``TRADE_HISTORY_DB_PATH`` of ``data/database.db``.

Geen rijen in: trade_history, trade_events, paper_ledger_reset_batches, wallet_state (voor insert),
plus legacy-tabellen indien aanwezig (active_positions, paper_trade_history, closed_trades).
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def _wallet_snapshot(cash_equity: float) -> dict:
    bal = round(float(cash_equity), 2)
    return {
        "cash": bal,
        "equity": bal,
        "starting_balance_eur": bal,
        "paper_anchor_equity_eur": bal,
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


def _resolve_db_path(cli_db: str | None) -> Path:
    if cli_db:
        p = Path(cli_db).expanduser()
        return p if p.is_absolute() else Path.cwd() / p
    raw = str(os.getenv("TRADE_HISTORY_DB_PATH", "data/database.db")).strip()
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = Path.cwd() / p
    return p


def _sql_exec_ignore(conn: sqlite3.Connection, sql: str) -> None:
    try:
        conn.execute(sql)
    except sqlite3.Error:
        pass


def _count(conn: sqlite3.Connection, table: str) -> int:
    try:
        return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] or 0)
    except sqlite3.Error:
        return 0


def _redis_flushall_subprocess() -> bool:
    host = str(os.getenv("REDIS_HOST", "127.0.0.1")).strip() or "127.0.0.1"
    port = str(os.getenv("REDIS_PORT", "6379")).strip() or "6379"
    try:
        r = subprocess.run(
            ["redis-cli", "-h", host, "-p", port, "FLUSHALL"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if r.returncode == 0:
            return True
        sys.stderr.write(f"[redis-cli FLUSHALL] exit {r.returncode}: {r.stderr or r.stdout}\n")
    except FileNotFoundError:
        sys.stderr.write("[redis-cli] niet gevonden; overslaan (gebruik redis-py pad hieronder).\n")
    except Exception as exc:
        sys.stderr.write(f"[redis-cli FLUSHALL] {exc}\n")
    return False


def _redis_flushall_python() -> bool:
    try:
        import redis

        url = str(os.getenv("REDIS_URL", "")).strip()
        if not url:
            h = str(os.getenv("REDIS_HOST", "127.0.0.1")).strip() or "127.0.0.1"
            p = str(os.getenv("REDIS_PORT", "6379")).strip() or "6379"
            url = f"redis://{h}:{p}/0"
        r = redis.Redis.from_url(url, decode_responses=True, socket_connect_timeout=3.0, socket_timeout=3.0)
        try:
            r.flushall()
            return True
        finally:
            r.close()
    except Exception as exc:
        sys.stderr.write(f"[redis FLUSHALL via redis-py] {exc}\n")
        return False


def _apply_constraints_after_reset(equity_eur: float) -> None:
    try:
        from core.trading_constraints_redis import apply_paper_reset_allocation_constraints

        merged = apply_paper_reset_allocation_constraints(equity_eur=float(equity_eur))
        print(f"Redis trading:constraints na reset: {json.dumps(merged, default=str)}")
    except Exception as exc:
        sys.stderr.write(f"Waarschuwing: constraints niet geschreven ({exc}). Start worker/portal opnieuw.\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="Harde SQLite paper-reset + Redis + 10% sizing.")
    ap.add_argument("--db", help="Pad naar SQLite trade-history (default: env TRADE_HISTORY_DB_PATH of data/database.db)")
    ap.add_argument("--balance", type=float, default=1000.0, help="cash en equity in wallet_state (default 1000)")
    ap.add_argument("--tenant", default="default", help="tenant_id voor wallet_state INSERT")
    ap.add_argument("--no-redis", action="store_true", help="Geen FLUSHALL en geen constraints-write")
    ap.add_argument("--redis-only", action="store_true", help="Alleen Redis FLUSHALL + constraints (geen SQLite)")
    args = ap.parse_args()

    bal = round(float(args.balance), 2)
    tid = str(args.tenant or "default").strip().lower() or "default"
    ts = datetime.now(timezone.utc).isoformat()

    if not args.no_redis and not args.redis_only:
        ok_cli = _redis_flushall_subprocess()
        if not ok_cli:
            _redis_flushall_python()
        _apply_constraints_after_reset(bal)
    elif args.redis_only:
        ok_cli = _redis_flushall_subprocess()
        if not ok_cli:
            _redis_flushall_python()
        _apply_constraints_after_reset(bal)
        print("Alleen Redis uitgevoerd (--redis-only).")
        return 0

    db_path = _resolve_db_path(args.db)
    try:
        db_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"FOUT: kan map niet aanmaken voor database: {db_path.parent} ({exc})", file=sys.stderr)
        return 1

    snap = json.dumps(_wallet_snapshot(bal), separators=(",", ":"))

    try:
        conn_cm = sqlite3.connect(str(db_path))
    except sqlite3.OperationalError as exc:
        print(
            f"FOUT: kan database niet openen: {db_path} ({exc}). "
            f"Tip: map schrijfbaar maken of `--db /pad/naar/jouw/trade_history.db`.",
            file=sys.stderr,
        )
        return 1

    with conn_cm as conn:
        conn.execute("PRAGMA foreign_keys=OFF")
        for tbl in (
            "trade_history",
            "trade_events",
            "paper_ledger_reset_batches",
            "wallet_state",
        ):
            _sql_exec_ignore(conn, f"DELETE FROM {tbl}")
        for legacy in ("active_positions", "paper_trade_history", "closed_trades"):
            _sql_exec_ignore(conn, f"DELETE FROM {legacy}")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS wallet_state (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tenant_id TEXT NOT NULL UNIQUE,
                snapshot_json TEXT NOT NULL,
                updated_ts_utc TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            INSERT INTO wallet_state (tenant_id, snapshot_json, updated_ts_utc)
            VALUES (?, ?, ?)
            ON CONFLICT(tenant_id) DO UPDATE SET
                snapshot_json = excluded.snapshot_json,
                updated_ts_utc = excluded.updated_ts_utc
            """,
            (tid, snap, ts),
        )
        conn.commit()

        checks = ["trade_history", "trade_events", "wallet_state"]
        for t in checks:
            n = _count(conn, t)
            print(f"COUNT {t} = {n}")
        for leg in ("active_positions", "paper_trade_history", "closed_trades"):
            try:
                n = _count(conn, leg)
                print(f"COUNT {leg} = {n}")
            except Exception:
                pass

    print(f"OK: {db_path} — wallet tenant={tid} cash=equity={bal:.2f} EUR")
    if args.no_redis:
        print("(Redis ongewijzigd door --no-redis)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
