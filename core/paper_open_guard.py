"""
Harde garantie: maximaal één open paper-positie per markt (tenant + pair).

- SQLite ``active_positions`` met UNIQUE(tenant_id, market) + BEGIN IMMEDIATE bij BUY-reserve.
- Optionele Redis SETNX (``paper:buy_lock:…``) tegen races tussen processen.
- Wallet ``open_lots_by_market``: meerdere lots voor hetzelfde paar worden samengevoegd (gewogen entry).
"""

from __future__ import annotations

import contextlib
import os
import sqlite3
import threading
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

from app.datetime_util import UTC

# Voorkomt dat twee worker-threads tegelijk door de BUY-/active_positions-check glippen.
_paper_buy_thread_lock = threading.Lock()


def paper_buy_thread_lock() -> threading.Lock:
    return _paper_buy_thread_lock


def _flock_db_path(db_file: str | os.PathLike[str]) -> Path:
    p = Path(db_file)
    if not p.is_absolute():
        p = Path.cwd() / p
    return p


@contextlib.contextmanager
def paper_buy_flock_for_pair(db_file: str | os.PathLike[str], tenant_id: str, market: str) -> Iterator[None]:
    """
    Bestand-lock per tenant+paar naast de SQLite-DB: serialiseert BUY tussen meerdere worker-processen
    (threading.Lock alleen binnen één proces).
    """
    if str(os.getenv("PAPER_BUY_FLOCK", "1")).strip().lower() not in ("1", "true", "yes", "on"):
        yield
        return
    try:
        import fcntl
    except ImportError:
        yield
        return
    tid = str(tenant_id or "default").strip().lower() or "default"
    mku = str(market or "").strip().upper().replace("/", "-") or "x"
    safe = "".join(c if c.isalnum() else "_" for c in f"{tid}_{mku}")[:200]
    base = _flock_db_path(db_file)
    base.parent.mkdir(parents=True, exist_ok=True)
    lock_path = base.parent / f".paper_buy_{safe}.flock"
    fp = open(lock_path, "a+b")
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


@contextlib.contextmanager
def paper_buy_serialized(db_file: str | os.PathLike[str], tenant_id: str, market: str) -> Iterator[None]:
    """Cross-process (flock) + in-process (thread lock) voor één BUY-traject."""
    with paper_buy_flock_for_pair(db_file, tenant_id, market):
        with _paper_buy_thread_lock:
            yield


def can_open_position_sqlite(
    conn: sqlite3.Connection, tenant_id: str, market: str
) -> bool:
    """
    True als er (nog) geen rij in ``active_positions`` is voor dit paar.
    Caller moet ``BEGIN IMMEDIATE`` of exclusieve lock al hebben indien nodig.
    """
    tid = str(tenant_id or "default").strip().lower() or "default"
    mku = str(market or "").strip().upper().replace("/", "-")
    ensure_active_positions_ddl(conn)
    row = conn.execute(
        "SELECT 1 FROM active_positions WHERE tenant_id = ? AND market = ? LIMIT 1",
        (tid, mku),
    ).fetchone()
    return row is None


def has_active_position_for_base_sqlite(conn: sqlite3.Connection, tenant_id: str, market: str) -> bool:
    """True als er al een ``active_positions``-rij is met dezelfde basis (Bitvavo: *-EUR; max 1 open per coin)."""
    tid = str(tenant_id or "default").strip().lower() or "default"
    mku = str(market or "").strip().upper().replace("/", "-")
    base = mku.split("-", 1)[0].strip() if "-" in mku else mku
    if not base:
        return False
    ensure_active_positions_ddl(conn)
    rows = conn.execute("SELECT market FROM active_positions WHERE tenant_id = ?", (tid,)).fetchall() or []
    for row in rows:
        mk = str(row[0] or "").strip().upper().replace("/", "-")
        b = mk.split("-", 1)[0].strip() if "-" in mk else mk
        if b == base:
            return True
    return False


def strict_block_duplicate_log(pair: str) -> str:
    p = str(pair or "").strip().upper().replace("/", "-") or "?"
    return f"[BLOCK] Dubbele positie gedetecteerd voor {p}. Order geannuleerd."


def _redis_client():
    try:
        import redis
    except ImportError:
        return None
    try:
        host = str(os.getenv("REDIS_HOST", "redis")).strip()
        port = int(os.getenv("REDIS_PORT", "6379") or 6379)
        url = str(os.getenv("REDIS_URL", f"redis://{host}:{port}/0")).strip()
        if not url:
            url = f"redis://{host}:{port}/0"
        return redis.Redis.from_url(url, decode_responses=True)
    except Exception:
        return None


def redis_acquire_buy_lock(tenant_id: str, market: str, ttl_sec: int = 45) -> bool:
    """True = lock verkregen. Bij geen Redis: True (SQLite blijft bron van waarheid)."""
    r = _redis_client()
    if r is None:
        return True
    tid = str(tenant_id or "default").strip().lower() or "default"
    mku = str(market or "").strip().upper().replace("/", "-")
    if not mku:
        return True
    key = f"paper:buy_lock:{tid}:{mku}"
    try:
        ok = bool(r.set(key, "1", nx=True, ex=int(max(5, ttl_sec))))
        return ok
    except Exception:
        return True


def redis_release_buy_lock(tenant_id: str, market: str) -> None:
    r = _redis_client()
    if r is None:
        return
    tid = str(tenant_id or "default").strip().lower() or "default"
    mku = str(market or "").strip().upper().replace("/", "-")
    if not mku:
        return
    try:
        r.delete(f"paper:buy_lock:{tid}:{mku}")
    except Exception:
        pass


def redis_mark_open_pair(tenant_id: str, market: str) -> None:
    r = _redis_client()
    if r is None:
        return
    tid = str(tenant_id or "default").strip().lower() or "default"
    mku = str(market or "").strip().upper().replace("/", "-")
    if not mku:
        return
    try:
        r.set(f"paper:open_pair:{tid}:{mku}", "1", ex=86400 * 14)
    except Exception:
        pass


def redis_clear_open_pair(tenant_id: str, market: str) -> None:
    r = _redis_client()
    if r is None:
        return
    tid = str(tenant_id or "default").strip().lower() or "default"
    mku = str(market or "").strip().upper().replace("/", "-")
    if not mku:
        return
    try:
        r.delete(f"paper:open_pair:{tid}:{mku}")
    except Exception:
        pass


def redis_has_open_pair_flag(tenant_id: str, market: str) -> bool:
    r = _redis_client()
    if r is None:
        return False
    tid = str(tenant_id or "default").strip().lower() or "default"
    mku = str(market or "").strip().upper().replace("/", "-")
    if not mku:
        return False
    try:
        return bool(r.exists(f"paper:open_pair:{tid}:{mku}"))
    except Exception:
        return False


def ensure_active_positions_ddl(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS active_positions (
            tenant_id TEXT NOT NULL,
            market TEXT NOT NULL,
            opened_ts_utc TEXT NOT NULL,
            qty REAL NOT NULL DEFAULT 0,
            PRIMARY KEY (tenant_id, market)
        )
        """
    )


def ensure_open_trade_registry_ddl(conn: sqlite3.Connection) -> None:
    """
    Anti-stacking: maximaal één rij met status OPEN per (tenant_id, symbol).
    Dubbele OPEN voor hetzelfde paar → UNIQUE-index violation bij INSERT.
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS open_trade_registry (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tenant_id TEXT NOT NULL,
            symbol TEXT NOT NULL,
            status TEXT NOT NULL,
            opened_ts_utc TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS ux_open_trade_registry_open_symbol
        ON open_trade_registry(tenant_id, symbol)
        WHERE UPPER(TRIM(status)) = 'OPEN'
        """
    )


def delete_open_trade_registry_open(conn: sqlite3.Connection, tenant_id: str, market: str) -> None:
    tid = str(tenant_id or "default").strip().lower() or "default"
    mku = str(market or "").strip().upper().replace("/", "-")
    try:
        ensure_open_trade_registry_ddl(conn)
        conn.execute(
            "DELETE FROM open_trade_registry WHERE tenant_id = ? AND symbol = ? AND UPPER(TRIM(status)) = 'OPEN'",
            (tid, mku),
        )
    except sqlite3.Error:
        pass


def reserve_active_position_slot(
    conn: sqlite3.Connection, tenant_id: str, market: str, ts_utc: str, qty: float
) -> bool:
    """
    INSERT onder BEGIN IMMEDIATE (caller zet transaction).
    Returns False bij duplicate (IntegrityError).
    """
    tid = str(tenant_id or "default").strip().lower() or "default"
    mku = str(market or "").strip().upper().replace("/", "-")
    ensure_active_positions_ddl(conn)
    try:
        conn.execute(
            "INSERT INTO active_positions (tenant_id, market, opened_ts_utc, qty) VALUES (?, ?, ?, ?)",
            (tid, mku, str(ts_utc), float(qty)),
        )
        return True
    except sqlite3.IntegrityError:
        return False


def update_active_position_qty(conn: sqlite3.Connection, tenant_id: str, market: str, qty: float) -> None:
    tid = str(tenant_id or "default").strip().lower() or "default"
    mku = str(market or "").strip().upper().replace("/", "-")
    conn.execute(
        "UPDATE active_positions SET qty = ? WHERE tenant_id = ? AND market = ?",
        (float(qty), tid, mku),
    )


def delete_active_position(conn: sqlite3.Connection, tenant_id: str, market: str) -> None:
    tid = str(tenant_id or "default").strip().lower() or "default"
    mku = str(market or "").strip().upper().replace("/", "-")
    conn.execute("DELETE FROM active_positions WHERE tenant_id = ? AND market = ?", (tid, mku))
    delete_open_trade_registry_open(conn, tid, mku)


def reconcile_active_positions_with_wallet(conn: sqlite3.Connection, tenant_id: str, wallet: dict[str, Any]) -> None:
    """Verwijder DB-rijen zonder open lots; update qty waar wallet nog wel lots heeft."""
    from core.trading_engine import has_active_paper_position_for_ticker

    tid = str(tenant_id or "default").strip().lower() or "default"
    ensure_active_positions_ddl(conn)
    rows = list(conn.execute("SELECT market, qty FROM active_positions WHERE tenant_id = ?", (tid,)).fetchall() or [])
    for row in rows:
        mk = str(row["market"])
        if not has_active_paper_position_for_ticker(wallet, mk):
            delete_active_position(conn, tid, mk)
        else:
            obm = wallet.get("open_lots_by_market") if isinstance(wallet.get("open_lots_by_market"), dict) else {}
            lots = obm.get(mk) if isinstance(obm.get(mk), list) else []
            total = sum(float(x.get("qty") or 0) for x in lots if isinstance(x, dict))
            update_active_position_qty(conn, tid, mk, total)


def merge_duplicate_open_lots_in_wallet(wallet: dict[str, Any]) -> list[str]:
    """
    Per markt: maximaal één lot in ``open_lots_by_market`` (gewogen gemiddelde entry, vroegste entry_ts).
    Returns lijst gemergde markten (uppercase) voor logging.
    """
    if not isinstance(wallet, dict):
        return []
    obm = wallet.setdefault("open_lots_by_market", {})
    if not isinstance(obm, dict):
        return []
    merged_mk: list[str] = []
    for mku, lots in list(obm.items()):
        mk = str(mku).strip().upper().replace("/", "-")
        if not mk or not isinstance(lots, list) or len(lots) <= 1:
            continue
        valid = [x for x in lots if isinstance(x, dict) and float(x.get("qty") or 0) > 1e-12]
        if len(valid) <= 1:
            obm[mk] = valid
            continue
        total_qty = sum(float(x.get("qty") or 0) for x in valid)
        if total_qty <= 1e-12:
            obm.pop(mk, None)
            continue
        cost = sum(float(x.get("qty") or 0) * float(x.get("entry_price") or 0) for x in valid)
        avg_entry = cost / total_qty
        ts_vals: list[str] = []
        for x in valid:
            ts_vals.append(str(x.get("entry_ts_utc") or ""))
        ts_vals = [t for t in ts_vals if t.strip()]
        earliest = min(ts_vals) if ts_vals else datetime.now(UTC).isoformat()
        base = dict(valid[0])
        base["qty"] = total_qty
        base["entry_price"] = avg_entry
        base["entry_ts_utc"] = earliest
        base["market"] = mk
        obm[mk] = [base]
        merged_mk.append(mk)
    return merged_mk


def wallet_open_lot_count(wallet: dict[str, Any] | None, market: str) -> int:
    if not isinstance(wallet, dict):
        return 0
    mku = str(market or "").strip().upper().replace("/", "-")
    obm = wallet.get("open_lots_by_market") if isinstance(wallet.get("open_lots_by_market"), dict) else {}
    lots = obm.get(mku)
    if not isinstance(lots, list):
        return 0
    return sum(1 for x in lots if isinstance(x, dict) and float(x.get("qty") or 0) > 1e-12)
