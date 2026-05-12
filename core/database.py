"""
SQLite trade-ledger reads (zelfde bestand/kolommen als app.services.paper_engine).

Paper-reset: bij ``full_environment_reset=True`` wist ``PaperTradeManager.reset_paper_account``
alle rijen in ``trade_history``, ``trade_events``, ``paper_ledger_reset_batches``, ``wallet_state``,
plus optioneel legacy-tabellen ``active_positions``, ``paper_trade_history``, ``closed_trades`` (DELETE
als de tabel bestaat). Daarna één verse wallet met ``cash``/``equity`` = startkapitaal (open lots leeg).

Micro-PPO na eerste gesloten trade: ``trading_core._paper_trade_closed_listener`` gebruikt env
``RL_MICRO_UPDATE_STEPS_ON_CLOSE`` (default 1024) en ``RL_MIN_EXPERIENCES_FOR_MICRO_TRAIN`` (default 1).
"""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from app.datetime_util import UTC


def _trade_db_path(db_path: str | None = None) -> Path:
    raw = str(db_path or os.getenv("TRADE_HISTORY_DB_PATH", "data/database.db")).strip()
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


def _is_open_buy_event_row(row: dict[str, Any]) -> bool:
    """Paper trade_events: uitgevoerde BUY met status ``opened`` (open positie)."""
    if str(row.get("row_type") or "").upper() != "EVENT":
        return False
    if str(row.get("action") or "").upper() != "BUY":
        return False
    oc = str(row.get("outcome") or "").strip().upper()
    return oc == "OPENED"


def _merge_open_buy_event_group(grp: list[dict[str, Any]]) -> dict[str, Any]:
    """Naar één logische positie: totale qty, gemiddelde entry, vroegste entry-tijd."""
    if len(grp) == 1:
        return grp[0]
    chrono = sorted(grp, key=lambda r: _parse_sort_ts(str(r.get("entry_ts_utc") or "")))
    total_qty = 0.0
    cost = 0.0
    for r in chrono:
        try:
            q = float(r.get("qty") or 0.0)
            ep = float(r.get("entry_price") or 0.0)
        except (TypeError, ValueError):
            continue
        if q > 1e-12 and ep > 1e-12:
            total_qty += q
            cost += q * ep
    avg_px = cost / total_qty if total_qty > 1e-12 else float(chrono[-1].get("entry_price") or 0.0)
    newest = max(
        chrono,
        key=lambda r: _parse_sort_ts(str(r.get("entry_ts_utc") or "")),
    )
    base = dict(newest)
    base["qty"] = total_qty
    base["entry_price"] = avg_px
    base["exit_price"] = avg_px
    base["entry_ts_utc"] = str(chrono[0].get("entry_ts_utc") or "")
    base["exit_ts_utc"] = str(chrono[-1].get("exit_ts_utc") or chrono[-1].get("entry_ts_utc") or "")
    prev_ctx = " · ".join(
        str(r.get("ledger_context") or "").strip()
        for r in chrono
        if str(r.get("ledger_context") or "").strip()
    )
    note = f"[samengevoegd {len(grp)}× BUY → 1 open positie]"
    base["ledger_context"] = (note + (" · " + prev_ctx if prev_ctx else ""))[:500]
    return base


def _collapse_duplicate_open_buy_events(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Voorkomt dubbele ACTIVE-rijen wanneer eerdere runs per ongeluk meerdere BUY-events voor dezelfde
    markt opsloegen (ledger-regels blijven bestaan in SQLite). UI toont dan één samengestelde positie.
    """
    if len(rows) <= 1:
        return rows
    buckets: dict[str, list[dict[str, Any]]] = {}
    rest: list[dict[str, Any]] = []
    for row in rows:
        if _is_open_buy_event_row(row):
            mk = str(row.get("market") or row.get("pair") or "").strip().upper().replace("/", "-")
            if not mk:
                rest.append(row)
                continue
            buckets.setdefault(mk, []).append(row)
        else:
            rest.append(row)
    collapsed: list[dict[str, Any]] = []
    for _mk, grp in buckets.items():
        if len(grp) <= 1:
            collapsed.extend(grp)
        else:
            collapsed.append(_merge_open_buy_event_group(grp))
    out = rest + collapsed
    out.sort(
        key=lambda row: _parse_sort_ts(str(row.get("exit_ts_utc") or row.get("entry_ts_utc") or "")),
        reverse=True,
    )
    return out


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
                    "market": str(r["market"]),
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
                    "market": mk,
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
                    "sentiment_score": float(r["sentiment_score"]),
                }
            )
    merged.sort(key=lambda row: _parse_sort_ts(str(row.get("sort_ts") or "")), reverse=True)
    for row in merged:
        row.pop("sort_ts", None)
    merged = _collapse_duplicate_open_buy_events(merged)
    return merged[:lim]
