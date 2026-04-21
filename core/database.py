"""
SQLite trade-ledger reads (zelfde bestand/kolommen als app.services.paper_engine).
"""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from app.datetime_util import UTC


def _trade_db_path(db_path: str | None = None) -> Path:
    raw = str(db_path or os.getenv("TRADE_HISTORY_DB_PATH", "data/trade_history.db")).strip()
    p = Path(raw)
    if not p.is_absolute():
        p = Path.cwd() / p
    return p


def _parse_sort_ts(value: str | None) -> float:
    s = str(value or "").strip()
    if not s:
        return 0.0
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt.timestamp()
    except Exception:
        return 0.0


def get_all_trades(
    *,
    db_path: str | None = None,
    tenant_id: str | None = None,
    limit: int = 50_000,
) -> list[dict[str, Any]]:
    """
    Alle gesloten round-trips (trade_history) + uitgevoerde BUY/SELL-events (trade_events),
    ongefilterd op munt, gesorteerd op tijd (nieuwste eerst). Kolom `pair` = Bitvavo-markt (bv. ETH-EUR).
    """
    if tenant_id is None:
        from app.services.state import current_tenant_id

        tenant_id = str(current_tenant_id() or "default").strip().lower() or "default"
    tid = str(tenant_id or "default").strip().lower() or "default"
    path = _trade_db_path(db_path)
    if not path.exists():
        return []
    lim = int(max(1, min(int(limit or 50000), 100_000)))
    merged: list[dict[str, Any]] = []
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        for r in conn.execute(
            """
            SELECT market, coin, entry_ts_utc, exit_ts_utc, entry_price, exit_price, qty,
                   fees_eur, pnl_eur, pnl_pct, outcome,
                   COALESCE(ledger_context, '') AS ledger_context,
                   exit_ts_utc AS sort_ts
            FROM trade_history
            WHERE tenant_id = ?
            ORDER BY datetime(exit_ts_utc) DESC, id DESC
            LIMIT ?
            """,
            (tid, lim),
        ):
            merged.append(
                {
                    "row_type": "ROUND_TRIP",
                    "pair": str(r["market"]),
                    "coin": str(r["coin"]),
                    "entry_ts_utc": str(r["entry_ts_utc"]),
                    "exit_ts_utc": str(r["exit_ts_utc"]),
                    "entry_price": float(r["entry_price"]),
                    "exit_price": float(r["exit_price"]),
                    "qty": float(r["qty"]),
                    "fees_eur": float(r["fees_eur"]),
                    "pnl_eur": float(r["pnl_eur"]),
                    "pnl_pct": float(r["pnl_pct"]),
                    "outcome": str(r["outcome"]),
                    "ledger_context": str(r["ledger_context"] or ""),
                    "sort_ts": str(r["sort_ts"]),
                }
            )
        for r in conn.execute(
            """
            SELECT ts_utc, market, action, signal, status, price, qty, sentiment_score, pnl_eur, reason
            FROM trade_events
            WHERE tenant_id = ?
              AND UPPER(action) IN ('BUY', 'SELL')
            ORDER BY datetime(ts_utc) DESC, id DESC
            LIMIT ?
            """,
            (tid, lim),
        ):
            mk = str(r["market"])
            base = mk.split("-", 1)[0] if "-" in mk else mk
            merged.append(
                {
                    "row_type": "EVENT",
                    "pair": mk,
                    "coin": base,
                    "entry_ts_utc": str(r["ts_utc"]),
                    "exit_ts_utc": str(r["ts_utc"]),
                    "entry_price": float(r["price"]),
                    "exit_price": float(r["price"]),
                    "qty": float(r["qty"]),
                    "fees_eur": 0.0,
                    "pnl_eur": float(r["pnl_eur"]),
                    "pnl_pct": 0.0,
                    "outcome": str(r["status"]),
                    "ledger_context": str(r["reason"] or ""),
                    "sort_ts": str(r["ts_utc"]),
                    "action": str(r["action"]),
                    "signal": str(r["signal"]),
                }
            )
    merged.sort(key=lambda row: _parse_sort_ts(str(row.get("sort_ts") or "")), reverse=True)
    for row in merged:
        row.pop("sort_ts", None)
    return merged[:lim]
