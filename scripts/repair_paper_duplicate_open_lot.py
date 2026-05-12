#!/usr/bin/env python3
"""
Eénmalige reparatie: dubbele open lots voor één paar (default ETH-EUR) → behoud vroegste ~€100-lot,
restitueer notional van verwijderde lots op cash, sync active_positions.qty.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

UTC = timezone.utc


def _parse_ts(s: str) -> datetime:
    s = str(s or "").strip()
    if not s:
        return datetime.max.replace(tzinfo=UTC)
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt.astimezone(UTC)
    except Exception:
        return datetime.max.replace(tzinfo=UTC)


def _lot_notional(lot: dict) -> float:
    try:
        return max(0.0, float(lot.get("qty", 0) or 0.0)) * max(0.0, float(lot.get("entry_price", 0) or 0.0))
    except (TypeError, ValueError):
        return 0.0


def pick_keep_lot(lots: list[dict], target_eur: float = 100.0, band: float = 25.0) -> dict | None:
    valid = [x for x in lots if isinstance(x, dict) and float(x.get("qty") or 0) > 1e-12]
    if not valid:
        return None
    valid.sort(key=lambda x: _parse_ts(str(x.get("entry_ts_utc") or "")))
    for lot in valid:
        n = _lot_notional(lot)
        if abs(n - target_eur) <= band:
            return lot
    return valid[0]


def main() -> int:
    ap = argparse.ArgumentParser(description="Repair duplicate open lots for one paper market.")
    ap.add_argument("--db", default="data/database.db", help="SQLite path (paper wallet_state)")
    ap.add_argument("--tenant", default="default")
    ap.add_argument("--market", default="ETH-EUR")
    ap.add_argument("--target-eur", type=float, default=100.0)
    args = ap.parse_args()
    db_path = Path(args.db)
    if not db_path.is_absolute():
        db_path = Path.cwd() / db_path
    if not db_path.exists():
        print(f"DB not found: {db_path}")
        return 1
    mku = str(args.market or "").strip().upper().replace("/", "-")
    tid = str(args.tenant or "default").strip().lower() or "default"

    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT snapshot_json FROM wallet_state WHERE tenant_id = ?",
            (tid,),
        ).fetchone()
        if not row:
            print(f"No wallet_state for tenant={tid}")
            return 1
        w = json.loads(row["snapshot_json"])
        if not isinstance(w, dict):
            print("Invalid wallet JSON")
            return 1
        obm = w.setdefault("open_lots_by_market", {})
        if not isinstance(obm, dict):
            print("No open_lots_by_market")
            return 1
        lots = obm.get(mku)
        if not isinstance(lots, list) or len(lots) <= 1:
            print(f"No duplicates for {mku} (count={len(lots) if isinstance(lots, list) else 0})")
            return 0
        keep = pick_keep_lot(lots, target_eur=float(args.target_eur))
        if keep is None:
            return 1
        removed = [x for x in lots if isinstance(x, dict) and x is not keep and float(x.get("qty") or 0) > 1e-12]
        refund = sum(_lot_notional(x) for x in removed)
        try:
            w["cash"] = float(w.get("cash", 0.0) or 0.0) + refund
        except (TypeError, ValueError):
            w["cash"] = refund
        obm[mku] = [keep]
        flat: list[dict] = []
        for mk in sorted(obm.keys()):
            for lot in obm.get(mk) or []:
                if isinstance(lot, dict):
                    flat.append(lot)
        w["open_lots"] = flat
        pbm = w.setdefault("position_by_market", {})
        if isinstance(pbm, dict):
            tq = sum(max(0.0, float(x.get("qty", 0) or 0.0)) for x in obm[mku] if isinstance(x, dict))
            if tq > 1e-12:
                pbm[mku] = tq
            else:
                pbm.pop(mku, None)
        total = sum(max(0.0, float(q or 0.0)) for q in pbm.values()) if isinstance(pbm, dict) else 0.0
        w["position_qty"] = float(total) if total > 1e-12 else 0.0
        nz = [str(k).upper() for k, q in (pbm or {}).items() if float(q or 0.0) > 1e-12]
        if len(nz) == 1:
            w["position_symbol"] = nz[0]
        elif not nz:
            w["position_symbol"] = None
        else:
            w["position_symbol"] = None
        sb = float(w.get("starting_balance_eur") or 1000.0)
        w.setdefault("starting_balance_eur", round(sb, 2))
        w.setdefault("paper_anchor_equity_eur", float(w["starting_balance_eur"]))

        snap = json.dumps(w, ensure_ascii=False)
        conn.execute(
            "UPDATE wallet_state SET snapshot_json = ?, updated_ts_utc = ? WHERE tenant_id = ?",
            (snap, datetime.now(UTC).isoformat(), tid),
        )
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
        conn.execute("DELETE FROM active_positions WHERE tenant_id = ? AND market = ?", (tid, mku))
        kq = float(keep.get("qty") or 0.0)
        if kq > 1e-12:
            conn.execute(
                "INSERT INTO active_positions (tenant_id, market, opened_ts_utc, qty) VALUES (?, ?, ?, ?)",
                (tid, mku, str(keep.get("entry_ts_utc") or datetime.now(UTC).isoformat()), kq),
            )
        conn.commit()
        print(
            f"Repaired {mku}: kept 1 lot, refunded €{refund:.2f} to cash, active_positions reset. tenant={tid}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
