#!/usr/bin/env python3
"""
Ruim dubbele **open** paper-posities op per **basis** (Bitvavo heeft alleen *-EUR; bv. dubbele ETH-keys → één behouden).

- Houdt het markt-paar met de vroegste lot-``entry_ts_utc`` (min over alle lots van dat paar).
- Restitueert ``qty * entry_price`` van verwijderde lots naar ``cash``.
- Verwijdert overbodige rijen in ``active_positions`` voor dezelfde tenant.
- Schrijft ``wallet_state`` opnieuw weg.

Gebruik (host, vanuit repo-root):
  PYTHONPATH=. python3 scripts/repair_paper_duplicate_base_currency.py
  PYTHONPATH=. python3 scripts/repair_paper_duplicate_base_currency.py --db data/database.db --tenant default
"""
from __future__ import annotations

import argparse
import json
import os
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


def _base(mk: str) -> str:
    m = str(mk or "").strip().upper().replace("/", "-")
    return m.split("-", 1)[0].strip() if "-" in m else m


def _market_earliest_lot_ts(obm: dict, mkt: str) -> datetime:
    lots = obm.get(mkt)
    if not isinstance(lots, list):
        return datetime.max.replace(tzinfo=UTC)
    best = datetime.max.replace(tzinfo=UTC)
    for lot in lots:
        if not isinstance(lot, dict):
            continue
        if float(lot.get("qty") or 0) <= 1e-12:
            continue
        t = _parse_ts(str(lot.get("entry_ts_utc") or ""))
        if t < best:
            best = t
    return best


def _lot_cost_eur(lot: dict) -> float:
    try:
        return max(0.0, float(lot.get("qty", 0) or 0.0)) * max(0.0, float(lot.get("entry_price", 0) or 0.0))
    except (TypeError, ValueError):
        return 0.0


def _flatten_open_lots(obm: dict) -> list[dict]:
    flat: list[dict] = []
    for mk in sorted(obm.keys()):
        for lot in obm.get(mk) or []:
            if isinstance(lot, dict):
                flat.append(lot)
    return flat


def _rebuild_position_by_market(w: dict) -> None:
    obm = w.get("open_lots_by_market") if isinstance(w.get("open_lots_by_market"), dict) else {}
    pbm: dict[str, float] = {}
    for mk, lots in obm.items():
        mku = str(mk).strip().upper().replace("/", "-")
        if not isinstance(lots, list):
            continue
        tq = sum(max(0.0, float(x.get("qty", 0) or 0.0)) for x in lots if isinstance(x, dict))
        if tq > 1e-12:
            pbm[mku] = tq
    w["position_by_market"] = pbm
    total = sum(pbm.values())
    w["position_qty"] = float(total) if total > 1e-12 else 0.0
    nz = [k for k, q in pbm.items() if float(q or 0) > 1e-12]
    if len(nz) == 1:
        w["position_symbol"] = nz[0]
    elif not nz:
        w["position_symbol"] = None
    else:
        w["position_symbol"] = None
    w["open_lots"] = _flatten_open_lots(obm)


def main() -> int:
    ap = argparse.ArgumentParser(description="Verwijder dubbele open paper-posities per basisvaluta.")
    ap.add_argument("--db", default="", help="SQLite (default: env TRADE_HISTORY_DB_PATH of data/database.db)")
    ap.add_argument("--tenant", default="default")
    args = ap.parse_args()
    raw_db = str(args.db or "").strip() or str(os.getenv("TRADE_HISTORY_DB_PATH", "data/database.db")).strip()
    db_path = Path(raw_db).expanduser()
    if not db_path.is_absolute():
        db_path = Path.cwd() / db_path
    if not db_path.exists():
        print(f"DB not found: {db_path}")
        return 1
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
            return 0

        # Groepeer markten met positieve lots per basis
        base_to_markets: dict[str, list[str]] = {}
        for mkt, lots in list(obm.items()):
            if not isinstance(lots, list):
                continue
            has = any(
                isinstance(x, dict) and float(x.get("qty") or 0) > 1e-12 for x in lots
            )
            if not has:
                continue
            mku = str(mkt).strip().upper().replace("/", "-")
            b = _base(mku)
            if not b:
                continue
            base_to_markets.setdefault(b, []).append(mku)

        removed_markets: list[str] = []
        refund = 0.0
        for base, markets in base_to_markets.items():
            if len(markets) <= 1:
                continue
            # Kies markt met vroegste lot-timestamp
            markets.sort(key=lambda m: _market_earliest_lot_ts(obm, m))
            keeper = markets[0]
            for drop in markets[1:]:
                lots = obm.get(drop)
                if isinstance(lots, list):
                    for lot in lots:
                        if isinstance(lot, dict):
                            refund += _lot_cost_eur(lot)
                obm.pop(drop, None)
                removed_markets.append(drop)
                print(f"  verwijder open markt {drop} (basis {base}; behouden {keeper})")

        if not removed_markets:
            print("Geen dubbele basis-valuta open posities gevonden.")
            return 0

        try:
            w["cash"] = float(w.get("cash", 0.0) or 0.0) + refund
        except (TypeError, ValueError):
            w["cash"] = refund
        _rebuild_position_by_market(w)
        try:
            from app.services.portfolio_qty import implied_equity_eur_from_wallet

            w["equity"] = float(implied_equity_eur_from_wallet(w))
        except Exception:
            try:
                w["equity"] = float(w.get("cash", 0.0) or 0.0)
            except (TypeError, ValueError):
                w["equity"] = 0.0

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
        for mk in removed_markets:
            conn.execute("DELETE FROM active_positions WHERE tenant_id = ? AND market = ?", (tid, mk))
        conn.commit()
        print(
            f"OK: tenant={tid} — {len(removed_markets)} markt(en) verwijderd, "
            f"€{refund:.2f} terug naar cash. Herstart worker zodat geheugen-wallet opnieuw laadt."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
