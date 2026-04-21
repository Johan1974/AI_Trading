"""
Bestand: app/services/paper_engine.py
Relatief pad: ./app/services/paper_engine.py
Functie: Paper trading manager met virtuele wallet, trade history opslag en sentiment/outcome analytics.
"""

from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime

from app.datetime_util import UTC
from pathlib import Path
from typing import Any

from app.services.state import current_tenant_id

@dataclass
class PaperConfig:
    starting_balance_eur: float = 10000.0
    fee_rate: float = 0.0015
    db_path: str = "data/trade_history.db"


class PaperTradeManager:
    def __init__(self, config: PaperConfig) -> None:
        self.config = config
        self._db_file = Path(config.db_path)
        if not self._db_file.is_absolute():
            self._db_file = Path.cwd() / self._db_file
        self._db_file.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self._wallets: dict[str, dict[str, Any]] = {}
        self._ensure_wallet_for_tenant(self._tenant_id())

    @property
    def wallet(self) -> dict[str, Any]:
        return self._ensure_wallet_for_tenant(self._tenant_id())

    @wallet.setter
    def wallet(self, value: dict[str, Any]) -> None:
        tid = self._tenant_id()
        if isinstance(value, dict) and value:
            self._wallets[tid] = value
            self._migrate_multi_asset_wallet(self._wallets[tid])
        else:
            self._wallets[tid] = self._init_wallet(self.config.starting_balance_eur)

    def _init_wallet(self, initial_balance: float) -> dict[str, Any]:
        return {
            "cash": float(initial_balance),
            "equity": float(initial_balance),
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

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_file)
        conn.row_factory = sqlite3.Row
        return conn

    def _tenant_id(self) -> str:
        return str(current_tenant_id() or "default").strip().lower() or "default"

    def _ensure_wallet_for_tenant(self, tenant_id: str) -> dict[str, Any]:
        tid = str(tenant_id or "default").strip().lower() or "default"
        if tid not in self._wallets:
            self._wallets[tid] = self._init_wallet(self.config.starting_balance_eur)
            self._restore_wallet_state(tenant_id=tid)
        self._migrate_multi_asset_wallet(self._wallets[tid])
        return self._wallets[tid]

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS trade_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tenant_id TEXT NOT NULL DEFAULT 'default',
                    market TEXT NOT NULL,
                    coin TEXT NOT NULL,
                    entry_ts_utc TEXT NOT NULL,
                    exit_ts_utc TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL NOT NULL,
                    qty REAL NOT NULL,
                    sentiment_score REAL NOT NULL,
                    headlines_json TEXT NOT NULL,
                    fees_eur REAL NOT NULL,
                    pnl_eur REAL NOT NULL,
                    pnl_pct REAL NOT NULL,
                    outcome TEXT NOT NULL,
                    ai_thought TEXT NOT NULL DEFAULT ''
                )
                """
            )
            cols = conn.execute("PRAGMA table_info(trade_history)").fetchall()
            names = {str(row["name"]) for row in cols}
            if "tenant_id" not in names:
                conn.execute("ALTER TABLE trade_history ADD COLUMN tenant_id TEXT NOT NULL DEFAULT 'default'")
            if "ai_thought" not in names:
                conn.execute("ALTER TABLE trade_history ADD COLUMN ai_thought TEXT NOT NULL DEFAULT ''")
            if "ledger_context" not in names:
                conn.execute("ALTER TABLE trade_history ADD COLUMN ledger_context TEXT NOT NULL DEFAULT ''")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS trade_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tenant_id TEXT NOT NULL DEFAULT 'default',
                    ts_utc TEXT NOT NULL,
                    market TEXT NOT NULL,
                    action TEXT NOT NULL,
                    signal TEXT NOT NULL,
                    status TEXT NOT NULL,
                    price REAL NOT NULL,
                    qty REAL NOT NULL,
                    sentiment_score REAL NOT NULL,
                    pnl_eur REAL NOT NULL,
                    reason TEXT NOT NULL DEFAULT ''
                )
                """
            )
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
            cols_ws = conn.execute("PRAGMA table_info(wallet_state)").fetchall()
            names_ws = {str(row["name"]) for row in cols_ws}
            if names_ws and "tenant_id" not in names_ws:
                # Legacy wallet_state (één globale snapshot) mist tenant_id én UNIQUE nodig voor UPSERT.
                rows: list[sqlite3.Row] = []
                if "snapshot_json" in names_ws and "updated_ts_utc" in names_ws:
                    order_by = "datetime(updated_ts_utc) DESC, id DESC" if "id" in names_ws else "datetime(updated_ts_utc) DESC"
                    rows = list(
                        conn.execute(f"SELECT snapshot_json, updated_ts_utc FROM wallet_state ORDER BY {order_by}").fetchall()
                    )
                conn.execute("DROP TABLE wallet_state")
                conn.execute(
                    """
                    CREATE TABLE wallet_state (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        tenant_id TEXT NOT NULL UNIQUE,
                        snapshot_json TEXT NOT NULL,
                        updated_ts_utc TEXT NOT NULL
                    )
                    """
                )
                if rows:
                    snap = str(rows[0]["snapshot_json"] or "{}")
                    uts = str(rows[0]["updated_ts_utc"] or "")
                    conn.execute(
                        """
                        INSERT INTO wallet_state (tenant_id, snapshot_json, updated_ts_utc)
                        VALUES ('default', ?, ?)
                        """,
                        (snap, uts),
                    )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS optimizer_state (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tenant_id TEXT NOT NULL DEFAULT 'default',
                    ts_utc TEXT NOT NULL,
                    settings_json TEXT NOT NULL
                )
                """
            )
            cols_ev = conn.execute("PRAGMA table_info(trade_events)").fetchall()
            names_ev = {str(row["name"]) for row in cols_ev}
            if "tenant_id" not in names_ev:
                conn.execute("ALTER TABLE trade_events ADD COLUMN tenant_id TEXT NOT NULL DEFAULT 'default'")
            cols_opt = conn.execute("PRAGMA table_info(optimizer_state)").fetchall()
            names_opt = {str(row["name"]) for row in cols_opt}
            if "tenant_id" not in names_opt:
                conn.execute("ALTER TABLE optimizer_state ADD COLUMN tenant_id TEXT NOT NULL DEFAULT 'default'")

    def _coin_from_market(self, market: str) -> str:
        upper = (market or "").upper()
        if "-" in upper:
            return upper.split("-", 1)[0]
        return upper

    def _migrate_multi_asset_wallet(self, w: dict[str, Any]) -> None:
        """Normalize legacy single-book snapshots to per-market FIFO queues."""
        if not isinstance(w, dict):
            return
        obm_raw = w.setdefault("open_lots_by_market", {})
        if not isinstance(obm_raw, dict):
            w["open_lots_by_market"] = {}
        obm: dict[str, Any] = w["open_lots_by_market"]
        pbm_raw = w.setdefault("position_by_market", {})
        if not isinstance(pbm_raw, dict):
            w["position_by_market"] = {}
        pbm: dict[str, Any] = w["position_by_market"]
        lp_raw = w.setdefault("last_prices_by_market", {})
        if not isinstance(lp_raw, dict):
            w["last_prices_by_market"] = {}
        lp: dict[str, Any] = w["last_prices_by_market"]
        sym = str(w.get("position_symbol") or "").strip().upper()
        old_lots = w.get("open_lots")
        if isinstance(old_lots, list) and old_lots:
            for lot in old_lots:
                if not isinstance(lot, dict):
                    continue
                lot2 = dict(lot)
                lm = str(lot2.get("market") or sym or "UNKNOWN").strip().upper()
                lot2["market"] = lm
                bucket = obm.setdefault(lm, [])
                if isinstance(bucket, list):
                    bucket.append(lot2)
            w["open_lots"] = []
        self._recompute_position_by_market_from_lots(w)
        has_lots = any(isinstance(v, list) and len(v) > 0 for v in obm.values())
        if not has_lots and sym and float(w.get("position_qty", 0.0) or 0.0) > 1e-12:
            qty = float(w.get("position_qty", 0.0) or 0.0)
            ep = float(w.get("avg_entry_price") or w.get("last_price") or 0.0)
            ts_guess = datetime.now(UTC).isoformat()
            hist = w.get("history")
            if isinstance(hist, list) and hist:
                last_row = hist[-1]
                if isinstance(last_row, dict) and last_row.get("ts"):
                    ts_guess = str(last_row["ts"])
            obm[sym] = [
                {
                    "qty": qty,
                    "entry_price": ep,
                    "entry_ts_utc": ts_guess,
                    "sentiment_score": 0.0,
                    "headlines": [],
                    "ai_thought": "",
                    "ledger_context": "",
                    "market": sym,
                }
            ]
            self._recompute_position_by_market_from_lots(w)
        if sym and w.get("last_price") is not None:
            try:
                lp.setdefault(sym, float(w["last_price"]))
            except (TypeError, ValueError):
                pass
        self._sync_aggregate_position_fields(w)

    def _recompute_position_by_market_from_lots(self, w: dict[str, Any] | None = None) -> None:
        wallet = w if isinstance(w, dict) else self.wallet
        obm = wallet.setdefault("open_lots_by_market", {})
        pbm = wallet.setdefault("position_by_market", {})
        if not isinstance(obm, dict) or not isinstance(pbm, dict):
            return
        pbm.clear()
        empty_keys: list[str] = []
        for mk, lots in obm.items():
            mku = str(mk).strip().upper()
            if not isinstance(lots, list):
                empty_keys.append(mku)
                continue
            tq = sum(max(0.0, float(x.get("qty", 0.0) or 0.0)) for x in lots if isinstance(x, dict))
            if tq > 1e-12:
                pbm[mku] = tq
            else:
                empty_keys.append(mku)
        for k in empty_keys:
            obm.pop(k, None)
        self._sync_aggregate_position_fields(wallet)

    def _sync_aggregate_position_fields(self, wallet: dict[str, Any]) -> None:
        pbm = wallet.get("position_by_market") if isinstance(wallet.get("position_by_market"), dict) else {}
        total = sum(max(0.0, float(q or 0.0)) for q in pbm.values())
        wallet["position_qty"] = float(total) if total > 1e-12 else 0.0
        nonzero = [str(k).upper() for k, q in pbm.items() if float(q or 0.0) > 1e-12]
        if len(nonzero) == 1:
            wallet["position_symbol"] = nonzero[0]
        elif not nonzero:
            wallet["position_symbol"] = None
        else:
            wallet["position_symbol"] = None
        flat: list[dict[str, Any]] = []
        obm = wallet.get("open_lots_by_market") if isinstance(wallet.get("open_lots_by_market"), dict) else {}
        for mk in sorted(obm.keys()):
            for lot in obm.get(mk) or []:
                if isinstance(lot, dict):
                    flat.append(lot)
        wallet["open_lots"] = flat

    def position_qty_for_market(self, market: str) -> float:
        w = self.wallet
        mku = str(market or "").strip().upper()
        pbm = w.get("position_by_market") if isinstance(w.get("position_by_market"), dict) else {}
        q = float(pbm.get(mku, 0.0) or 0.0)
        if q > 1e-12:
            return q
        if str(w.get("position_symbol") or "").upper() == mku:
            return float(w.get("position_qty", 0.0) or 0.0)
        return 0.0

    def _append_wallet_history(
        self,
        market: str,
        action: str,
        price: float,
        sentiment_score: float | None = None,
        pnl_eur: float | None = None,
    ) -> None:
        row = {
            "ts": datetime.now(UTC).isoformat(),
            "market": market,
            "action": action,
            "price": round(float(price), 8),
            "entry_price": round(float(price), 8),
            "sentiment_score": round(float(sentiment_score or 0.0), 4),
            "pnl_eur": round(float(pnl_eur or 0.0), 4),
            "cash": round(float(self.wallet["cash"]), 2),
            "equity": round(float(self.wallet["equity"]), 2),
            "position_qty": round(float(self.wallet["position_qty"]), 8),
            "position_symbol": self.wallet["position_symbol"],
            "realized_pnl_eur": round(float(self.wallet["realized_pnl_eur"]), 2),
        }
        self.wallet["history"].append(row)
        self.wallet["history"] = self.wallet["history"][-1000:]

    def _recompute_equity(self, mark_price: float | None = None, mark_market: str | None = None) -> None:
        lp_map = self.wallet.setdefault("last_prices_by_market", {})
        if not isinstance(lp_map, dict):
            self.wallet["last_prices_by_market"] = {}
            lp_map = self.wallet["last_prices_by_market"]
        if mark_market and mark_price is not None:
            try:
                fv = float(mark_price)
                if fv > 0:
                    lp_map[str(mark_market).strip().upper()] = fv
            except (TypeError, ValueError):
                pass
        if mark_price is not None:
            try:
                self.wallet["last_price"] = float(mark_price)
            except (TypeError, ValueError):
                pass
        cash = float(self.wallet["cash"])
        pbm = self.wallet.get("position_by_market") if isinstance(self.wallet.get("position_by_market"), dict) else {}
        total_mv = 0.0
        for mkt, qv in pbm.items():
            q = float(qv or 0.0)
            if q <= 1e-12:
                continue
            mku = str(mkt).strip().upper()
            px_loc = float(lp_map.get(mku, 0.0) or 0.0)
            if px_loc <= 0 and mku == str(self.wallet.get("position_symbol") or "").strip().upper():
                px_loc = float(self.wallet.get("last_price") or 0.0)
            total_mv += q * px_loc
        self.wallet["equity"] = cash + total_mv
        start = float(self.config.starting_balance_eur)
        if start > 0:
            self.wallet["realized_pnl_pct"] = (float(self.wallet["realized_pnl_eur"]) / start) * 100.0

    def _persist_wallet_state(self) -> None:
        ts = datetime.now(UTC).isoformat()
        tenant_id = self._tenant_id()
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO wallet_state (tenant_id, snapshot_json, updated_ts_utc)
                VALUES (?, ?, ?)
                ON CONFLICT(tenant_id) DO UPDATE SET snapshot_json=excluded.snapshot_json, updated_ts_utc=excluded.updated_ts_utc
                """,
                (tenant_id, json.dumps(self.wallet), ts),
            )

    def _restore_wallet_state(self, tenant_id: str | None = None) -> None:
        tenant_id = str(tenant_id or self._tenant_id() or "default").strip().lower() or "default"
        try:
            with self._conn() as conn:
                row = conn.execute(
                    "SELECT snapshot_json FROM wallet_state WHERE tenant_id = ?",
                    (tenant_id,),
                ).fetchone()
            if row is None:
                return
            data = json.loads(str(row["snapshot_json"] or "{}"))
            if isinstance(data, dict) and data:
                self._wallets[tenant_id] = data
        except Exception:
            return

    def _record_closed_trade(
        self,
        market: str,
        entry_ts_utc: str,
        exit_ts_utc: str,
        entry_price: float,
        exit_price: float,
        qty: float,
        sentiment_score: float,
        headlines: list[str],
        fees_eur: float,
        pnl_eur: float,
        ai_thought: str = "",
        ledger_context: str = "",
    ) -> None:
        tenant_id = self._tenant_id()
        pnl_pct = 0.0
        if entry_price > 0:
            pnl_pct = ((exit_price - entry_price) / entry_price) * 100.0
        outcome = "profit" if pnl_eur >= 0 else "loss"
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO trade_history (
                    tenant_id, market, coin, entry_ts_utc, exit_ts_utc, entry_price, exit_price, qty,
                    sentiment_score, headlines_json, fees_eur, pnl_eur, pnl_pct, outcome, ai_thought, ledger_context
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    tenant_id,
                    market,
                    self._coin_from_market(market),
                    entry_ts_utc,
                    exit_ts_utc,
                    float(entry_price),
                    float(exit_price),
                    float(qty),
                    float(sentiment_score),
                    json.dumps(headlines[:3]),
                    float(fees_eur),
                    float(pnl_eur),
                    float(pnl_pct),
                    outcome,
                    ai_thought or "",
                    str(ledger_context or "")[: 500],
                ),
            )

    def _record_trade_event(
        self,
        ts_utc: str,
        market: str,
        action: str,
        signal: str,
        status: str,
        price: float,
        qty: float,
        sentiment_score: float,
        pnl_eur: float = 0.0,
        reason: str = "",
    ) -> None:
        tenant_id = self._tenant_id()
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO trade_events (
                    tenant_id, ts_utc, market, action, signal, status, price, qty, sentiment_score, pnl_eur, reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    tenant_id,
                    str(ts_utc),
                    str(market),
                    str(action),
                    str(signal),
                    str(status),
                    float(price),
                    float(qty),
                    float(sentiment_score),
                    float(pnl_eur),
                    str(reason or ""),
                ),
            )

    def _close_lots_fifo(
        self,
        market: str,
        qty_to_sell: float,
        exit_price: float,
        ts_utc: str,
        ledger_context_exit: str = "",
    ) -> dict[str, Any]:
        total_realized = 0.0
        total_fees = 0.0
        closed = 0.0
        total_cost = 0.0
        total_gross = 0.0
        mku = str(market or "").strip().upper()
        obm = self.wallet.setdefault("open_lots_by_market", {})
        open_lots = obm.setdefault(mku, [])
        if not isinstance(open_lots, list):
            open_lots = []
            obm[mku] = open_lots
        while qty_to_sell > 1e-12 and open_lots:
            lot = open_lots[0]
            lot_qty = float(lot["qty"])
            close_qty = min(lot_qty, qty_to_sell)
            entry_price = float(lot["entry_price"])
            gross = close_qty * exit_price
            cost = close_qty * entry_price
            fee = gross * float(self.config.fee_rate)
            pnl_eur = gross - cost - fee
            total_cost += cost
            total_gross += gross
            total_realized += pnl_eur
            total_fees += fee
            closed += close_qty
            entry_ctx = str(lot.get("ledger_context") or "").strip()
            exit_ctx = str(ledger_context_exit or "").strip()
            merged_ctx = " · ".join([p for p in (entry_ctx, exit_ctx) if p])[: 500]
            self._record_closed_trade(
                market=market,
                entry_ts_utc=str(lot["entry_ts_utc"]),
                exit_ts_utc=ts_utc,
                entry_price=entry_price,
                exit_price=exit_price,
                qty=close_qty,
                sentiment_score=float(lot["sentiment_score"]),
                headlines=list(lot["headlines"]),
                fees_eur=fee,
                pnl_eur=pnl_eur,
                ai_thought=str(lot.get("ai_thought") or ""),
                ledger_context=merged_ctx,
            )
            lot["qty"] = lot_qty - close_qty
            qty_to_sell -= close_qty
            if float(lot["qty"]) <= 1e-12:
                open_lots.pop(0)
        if not open_lots:
            obm.pop(mku, None)
        self._recompute_position_by_market_from_lots(self.wallet)
        return {
            "qty_closed": closed,
            "realized_pnl_eur": total_realized,
            "fees_eur": total_fees,
            "gross_eur": total_gross,
            "cost_eur": total_cost,
        }

    def process_signal(
        self,
        market: str,
        signal: str,
        price: float,
        size_fraction: float,
        sentiment_score: float,
        news_headlines: list[str],
        ai_thought: str = "",
        ledger_context: str = "",
    ) -> dict[str, Any]:
        ts_utc = datetime.now(UTC).isoformat()
        signal = signal.upper()
        px = float(max(0.0, price))
        size = max(0.0, min(1.0, size_fraction))
        mkt_u = str(market or "").strip().upper()
        self.wallet["last_price"] = px

        if px <= 0 or signal not in {"BUY", "SELL"}:
            self._recompute_equity(px, mkt_u)
            self._append_wallet_history(
                market=market,
                action="HOLD",
                price=px,
                sentiment_score=sentiment_score,
            )
            self._record_trade_event(
                ts_utc=ts_utc,
                market=market,
                action="HOLD",
                signal=signal,
                status="skipped",
                price=px,
                qty=0.0,
                sentiment_score=sentiment_score,
                reason="invalid_price_or_signal",
            )
            self._persist_wallet_state()
            return {
                "status": "skipped",
                "signal": signal,
                "reason": "invalid_price_or_signal",
                "wallet": self.wallet,
            }

        start_equity = float(self.wallet["equity"])
        notional = max(0.0, start_equity * size)

        if signal == "BUY":
            fee = notional * float(self.config.fee_rate)
            if self.wallet["cash"] < (notional + fee) or notional <= 0:
                self._recompute_equity(px, mkt_u)
                self._append_wallet_history(
                    market=market,
                    action="BUY_REJECTED",
                    price=px,
                    sentiment_score=sentiment_score,
                )
                self._record_trade_event(
                    ts_utc=ts_utc,
                    market=market,
                    action="BUY_REJECTED",
                    signal=signal,
                    status="rejected",
                    price=px,
                    qty=0.0,
                    sentiment_score=sentiment_score,
                    reason="insufficient_cash",
                )
                return {
                    "status": "rejected",
                    "signal": signal,
                    "reason": "insufficient_cash",
                    "wallet": self.wallet,
                }
            qty = notional / px
            if qty <= 1e-12 or px <= 0:
                self._record_trade_event(
                    ts_utc=ts_utc,
                    market=market,
                    action="BUY_REJECTED",
                    signal=signal,
                    status="rejected",
                    price=px,
                    qty=0.0,
                    sentiment_score=sentiment_score,
                    reason="invalid_qty_or_entry_price",
                )
                return {
                    "status": "rejected",
                    "signal": signal,
                    "reason": "invalid_qty_or_entry_price",
                    "wallet": self.wallet,
                }
            self.wallet["cash"] -= (notional + fee)
            obm = self.wallet.setdefault("open_lots_by_market", {})
            bucket = obm.setdefault(mkt_u, [])
            if not isinstance(bucket, list):
                bucket = []
                obm[mkt_u] = bucket
            bucket.append(
                {
                    "qty": qty,
                    "entry_price": px,
                    "entry_ts_utc": ts_utc,
                    "sentiment_score": sentiment_score,
                    "headlines": news_headlines[:3],
                    "ai_thought": ai_thought,
                    "ledger_context": str(ledger_context or "")[: 500],
                    "market": mkt_u,
                }
            )
            self._recompute_position_by_market_from_lots(self.wallet)
            self.wallet["trades_count"] += 1
            self._recompute_equity(px, mkt_u)
            self._append_wallet_history(
                market=market,
                action="BUY",
                price=px,
                sentiment_score=sentiment_score,
            )
            self._record_trade_event(
                ts_utc=ts_utc,
                market=market,
                action="BUY",
                signal="BUY",
                status="opened",
                price=px,
                qty=qty,
                sentiment_score=sentiment_score,
                pnl_eur=-fee,
            )
            self._persist_wallet_state()
            return {
                "status": "opened",
                "signal": "BUY",
                "entry_price": px,
                "qty": qty,
                "fee_eur": fee,
                "wallet": self.wallet,
            }

        pos_here = self.position_qty_for_market(mkt_u)
        if pos_here <= 1e-12:
            self._recompute_equity(px, mkt_u)
            self._append_wallet_history(
                market=market,
                action="INVENTORY_ERROR",
                price=px,
                sentiment_score=sentiment_score,
            )
            self._record_trade_event(
                ts_utc=ts_utc,
                market=market,
                action="INVENTORY_ERROR",
                signal=signal,
                status="critical_blocked",
                price=px,
                qty=0.0,
                sentiment_score=sentiment_score,
                reason="inventory_error_no_ownership",
            )
            self._persist_wallet_state()
            return {
                "status": "critical_blocked",
                "signal": signal,
                "reason": "inventory_error_no_ownership",
                "wallet": self.wallet,
            }

        qty_target = notional / px if notional > 0 else pos_here
        qty_to_sell = min(float(pos_here), max(0.0, qty_target))
        close_result = self._close_lots_fifo(
            market=market,
            qty_to_sell=qty_to_sell,
            exit_price=px,
            ts_utc=ts_utc,
            ledger_context_exit=str(ledger_context or "")[: 500],
        )
        proceeds_after_fee = (close_result["qty_closed"] * px) - close_result["fees_eur"]
        self.wallet["cash"] += max(0.0, proceeds_after_fee)
        self.wallet["realized_pnl_eur"] += close_result["realized_pnl_eur"]
        if close_result["realized_pnl_eur"] >= 0:
            self.wallet["wins"] += 1
        else:
            self.wallet["losses"] += 1
        self.wallet["trades_count"] += 1
        self._recompute_equity(px, mkt_u)
        self._append_wallet_history(
            market=market,
            action="SELL",
            price=px,
            sentiment_score=sentiment_score,
            pnl_eur=float(close_result["realized_pnl_eur"]),
        )
        self._record_trade_event(
            ts_utc=ts_utc,
            market=market,
            action="SELL",
            signal="SELL",
            status="closed",
            price=px,
            qty=float(close_result["qty_closed"]),
            sentiment_score=sentiment_score,
            pnl_eur=float(close_result["realized_pnl_eur"]),
        )
        self._persist_wallet_state()
        return {
            "status": "closed",
            "signal": "SELL",
            "exit_price": px,
            "entry_price": (float(close_result["cost_eur"]) / max(1e-12, float(close_result["qty_closed"]))),
            "qty_closed": close_result["qty_closed"],
            "realized_pnl_eur": close_result["realized_pnl_eur"],
            "fee_eur": close_result["fees_eur"],
            "wallet": self.wallet,
        }

    def analytics(self) -> dict[str, Any]:
        tenant_id = self._tenant_id()
        with self._conn() as conn:
            top_losses = conn.execute(
                "SELECT sentiment_score, pnl_eur FROM trade_history WHERE tenant_id = ? ORDER BY pnl_eur ASC LIMIT 10",
                (tenant_id,),
            ).fetchall()
            top_wins = conn.execute(
                "SELECT sentiment_score, pnl_eur FROM trade_history WHERE tenant_id = ? ORDER BY pnl_eur DESC LIMIT 10",
                (tenant_id,),
            ).fetchall()
            coin_rollup = conn.execute(
                """
                SELECT coin,
                       COUNT(*) AS trades,
                       AVG(sentiment_score) AS avg_sentiment,
                       AVG(pnl_eur) AS avg_pnl_eur
                FROM trade_history
                WHERE tenant_id = ?
                GROUP BY coin
                ORDER BY trades DESC
                LIMIT 10
                """,
                (tenant_id,),
            ).fetchall()
            wl = conn.execute(
                "SELECT outcome, COUNT(*) AS cnt FROM trade_history WHERE tenant_id = ? GROUP BY outcome",
                (tenant_id,),
            ).fetchall()
            sentiment_buckets = conn.execute(
                """
                SELECT
                  CASE
                    WHEN sentiment_score > 0.2 THEN 'positive'
                    WHEN sentiment_score < -0.2 THEN 'negative'
                    ELSE 'neutral'
                  END AS bucket,
                  AVG(pnl_eur) AS avg_pnl_eur,
                  COUNT(*) AS samples
                FROM trade_history
                WHERE tenant_id = ?
                GROUP BY bucket
                """
            , (tenant_id,)).fetchall()
            extrema = conn.execute(
                """
                SELECT
                  COALESCE(MAX(pnl_eur), 0.0) AS max_win_eur,
                  COALESCE(MIN(pnl_eur), 0.0) AS max_loss_eur
                FROM trade_history
                WHERE tenant_id = ?
                """
            , (tenant_id,)).fetchone()
            hold_row = conn.execute(
                """
                SELECT AVG((julianday(exit_ts_utc) - julianday(entry_ts_utc)) * 24.0) AS avg_hold_hours
                FROM trade_history
                WHERE tenant_id = ?
                  AND
                  entry_ts_utc IS NOT NULL AND exit_ts_utc IS NOT NULL
                  AND length(trim(entry_ts_utc)) > 0 AND length(trim(exit_ts_utc)) > 0
                """
            , (tenant_id,)).fetchone()

        def _avg(rows: list[sqlite3.Row]) -> float:
            if not rows:
                return 0.0
            return float(sum(float(r["sentiment_score"]) for r in rows) / len(rows))

        wins = sum(int(r["cnt"]) for r in wl if str(r["outcome"]) == "profit")
        losses = sum(int(r["cnt"]) for r in wl if str(r["outcome"]) == "loss")
        total = max(1, wins + losses)
        max_win_eur = float(extrema["max_win_eur"] or 0.0) if extrema is not None else 0.0
        max_loss_eur = float(extrema["max_loss_eur"] or 0.0) if extrema is not None else 0.0
        raw_hold = hold_row["avg_hold_hours"] if hold_row is not None else None
        avg_hold_hours = float(raw_hold) if raw_hold is not None else 0.0
        if math.isnan(avg_hold_hours):
            avg_hold_hours = 0.0
        realized_eur = round(float(self.wallet.get("realized_pnl_eur", 0.0) or 0.0), 2)
        realized_pct = round(float(self.wallet.get("realized_pnl_pct", 0.0) or 0.0), 2)
        return {
            "performance_summary": {
                "total_pnl_eur": realized_eur,
                "total_pnl_pct": realized_pct,
                "win_rate_pct": round((wins / total) * 100.0, 2),
                "wins": int(wins),
                "losses": int(losses),
                "closed_trades": int(wins + losses),
                "max_win_eur": round(max_win_eur, 2),
                "max_loss_eur": round(max_loss_eur, 2),
                "avg_hold_hours": round(avg_hold_hours, 2),
            },
            "sentiment_correlation": {
                "avg_sentiment_top_10_losses": round(_avg(top_losses), 4),
                "avg_sentiment_top_10_wins": round(_avg(top_wins), 4),
                "sample_losses": len(top_losses),
                "sample_wins": len(top_wins),
            },
            "win_loss_ratio": {
                "wins": wins,
                "losses": losses,
                "win_rate_pct": round((wins / total) * 100.0, 2),
            },
            "sentiment_vs_outcome": [
                {
                    "bucket": str(r["bucket"]),
                    "avg_pnl_eur": round(float(r["avg_pnl_eur"] or 0.0), 4),
                    "samples": int(r["samples"] or 0),
                }
                for r in sentiment_buckets
            ],
            "coin_rollup": [
                {
                    "coin": str(r["coin"]),
                    "trades": int(r["trades"] or 0),
                    "avg_sentiment": round(float(r["avg_sentiment"] or 0.0), 4),
                    "avg_pnl_eur": round(float(r["avg_pnl_eur"] or 0.0), 4),
                }
                for r in coin_rollup
            ],
        }

    def recent_trades(self, limit: int = 50) -> list[dict[str, Any]]:
        tenant_id = self._tenant_id()
        with self._conn() as conn:
            event_rows = conn.execute(
                """
                SELECT ts_utc, market, action, signal, status, price, qty, sentiment_score, pnl_eur, reason
                FROM trade_events
                WHERE tenant_id = ?
                ORDER BY datetime(ts_utc) DESC, id DESC
                LIMIT ?
                """,
                (tenant_id, int(max(1, limit))),
            ).fetchall()
        if event_rows:
            output: list[dict[str, Any]] = []
            for r in event_rows:
                output.append(
                    {
                        "type": str(r["action"] or r["signal"] or "HOLD"),
                        "market": str(r["market"]),
                        "coin": self._coin_from_market(str(r["market"])),
                        "entry_ts_utc": str(r["ts_utc"]),
                        "exit_ts_utc": str(r["ts_utc"]),
                        "entry_price": float(r["price"]),
                        "exit_price": float(r["price"]),
                        "qty": float(r["qty"]),
                        "sentiment_score": float(r["sentiment_score"]),
                        "headlines": [],
                        "fees_eur": 0.0,
                        "pnl_eur": float(r["pnl_eur"]),
                        "pnl_pct": 0.0,
                        "outcome": "profit" if float(r["pnl_eur"]) >= 0 else "loss",
                        "ai_thought": "",
                        "status": str(r["status"]),
                        "reason": str(r["reason"]),
                    }
                )
            return output
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT market, coin, entry_ts_utc, exit_ts_utc, entry_price, exit_price, qty,
                       sentiment_score, headlines_json, fees_eur, pnl_eur, pnl_pct, outcome, ai_thought
                FROM trade_history
                WHERE tenant_id = ?
                ORDER BY datetime(exit_ts_utc) DESC, id DESC
                LIMIT ?
                """,
                (tenant_id, int(max(1, limit))),
            ).fetchall()
        output: list[dict[str, Any]] = []
        for r in rows:
            output.append(
                {
                    "type": "SELL",
                    "market": str(r["market"]),
                    "coin": str(r["coin"]),
                    "entry_ts_utc": str(r["entry_ts_utc"]),
                    "exit_ts_utc": str(r["exit_ts_utc"]),
                    "entry_price": float(r["entry_price"]),
                    "exit_price": float(r["exit_price"]),
                    "qty": float(r["qty"]),
                    "sentiment_score": float(r["sentiment_score"]),
                    "headlines": json.loads(str(r["headlines_json"])),
                    "fees_eur": float(r["fees_eur"]),
                    "pnl_eur": float(r["pnl_eur"]),
                    "pnl_pct": float(r["pnl_pct"]),
                    "outcome": str(r["outcome"]),
                    "ai_thought": str(r["ai_thought"] or ""),
                }
            )
        return output

    def record_critical_block(self, market: str, signal: str, reason: str, price: float = 0.0) -> None:
        ts_utc = datetime.now(UTC).isoformat()
        self._record_trade_event(
            ts_utc=ts_utc,
            market=str(market),
            action="INVENTORY_ERROR",
            signal=str(signal).upper(),
            status="critical_blocked",
            price=float(price or 0.0),
            qty=0.0,
            sentiment_score=0.0,
            pnl_eur=0.0,
            reason=str(reason or "critical_block"),
        )

    def round_trip_ledger(self, limit: int = 500) -> list[dict[str, Any]]:
        tenant_id = self._tenant_id()
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT market, coin, entry_ts_utc, exit_ts_utc, entry_price, exit_price, qty, fees_eur, pnl_eur, pnl_pct, outcome,
                       COALESCE(ledger_context, '') AS ledger_context
                FROM trade_history
                WHERE tenant_id = ?
                ORDER BY datetime(exit_ts_utc) DESC, id DESC
                LIMIT ?
                """,
                (tenant_id, int(max(1, limit))),
            ).fetchall()
        out: list[dict[str, Any]] = []
        for r in rows:
            out.append(
                {
                    "open_time_utc": str(r["entry_ts_utc"]),
                    "close_time_utc": str(r["exit_ts_utc"]),
                    "market": str(r["market"]),
                    "coin": str(r["coin"]),
                    "entry_price": float(r["entry_price"]),
                    "exit_price": float(r["exit_price"]),
                    "qty": float(r["qty"]),
                    "fees_eur": float(r["fees_eur"]),
                    "pnl_eur": float(r["pnl_eur"]),
                    "pnl_pct": float(r["pnl_pct"]),
                    "outcome": str(r["outcome"]),
                    "type": "ROUND_TRIP",
                    "status": "CLOSED",
                    "ledger_context": str(r["ledger_context"] or ""),
                }
            )
        # Also expose active open positions (per market) so global ledger remains complete.
        obm = self.wallet.get("open_lots_by_market") if isinstance(self.wallet.get("open_lots_by_market"), dict) else {}
        active_lots: list[tuple[str, dict[str, Any]]] = []
        if obm:
            for mk, lots in obm.items():
                if not isinstance(lots, list):
                    continue
                for lot in lots:
                    if isinstance(lot, dict):
                        active_lots.append((str(mk).upper(), lot))
        else:
            for lot in list(self.wallet.get("open_lots") or []):
                if isinstance(lot, dict):
                    lm = str(lot.get("market") or self.wallet.get("position_symbol") or "UNKNOWN").upper()
                    active_lots.append((lm, lot))
        for market, lot in active_lots:
            if not isinstance(lot, dict):
                continue
            qty = float(lot.get("qty") or 0.0)
            entry_price = float(lot.get("entry_price") or 0.0)
            if qty <= 0 or entry_price <= 0:
                continue
            out.append(
                {
                    "open_time_utc": str(lot.get("entry_ts_utc") or ""),
                    "close_time_utc": "",
                    "market": market,
                    "coin": self._coin_from_market(market),
                    "entry_price": entry_price,
                    "exit_price": None,
                    "qty": qty,
                    "fees_eur": 0.0,
                    "pnl_eur": 0.0,
                    "pnl_pct": 0.0,
                    "outcome": "active",
                    "type": "ACTIVE",
                    "status": "ACTIVE",
                    "ledger_context": str(lot.get("ledger_context") or "")[: 500],
                }
            )
        out.sort(key=lambda x: str(x.get("close_time_utc") or x.get("open_time_utc") or ""), reverse=True)
        out = out[: int(max(1, limit))]
        return out

    def elite8_audit_metrics(self, elite_markets: list[str], window_hours: int = 24) -> dict[str, dict[str, float]]:
        if not elite_markets:
            return {}
        unique = [str(m or "").upper() for m in elite_markets if str(m or "").strip()]
        if not unique:
            return {}
        placeholders = ",".join(["?"] * len(unique))
        window_expr = f"-{int(max(1, window_hours))} hours"
        tenant_id = self._tenant_id()
        with self._conn() as conn:
            rows = conn.execute(
                f"""
                SELECT market,
                       SUM(CASE WHEN pnl_eur > 0 THEN pnl_eur ELSE 0 END) AS gross_profit,
                       ABS(SUM(CASE WHEN pnl_eur < 0 THEN pnl_eur ELSE 0 END)) AS gross_loss,
                       SUM(CASE WHEN pnl_eur > 0 THEN 1 ELSE 0 END) AS wins,
                       SUM(CASE WHEN pnl_eur <= 0 THEN 1 ELSE 0 END) AS losses
                FROM trade_history
                WHERE tenant_id = ?
                  AND datetime(exit_ts_utc) >= datetime('now', ?)
                  AND UPPER(market) IN ({placeholders})
                GROUP BY market
                """,
                (tenant_id, window_expr, *unique),
            ).fetchall()
        out: dict[str, dict[str, float]] = {}
        for r in rows:
            gp = float(r["gross_profit"] or 0.0)
            gl = float(r["gross_loss"] or 0.0)
            wins = int(r["wins"] or 0)
            losses = int(r["losses"] or 0)
            closed = max(1, wins + losses)
            out[str(r["market"]).upper()] = {
                "profit_factor": gp if gl <= 1e-9 else gp / gl,
                "win_rate": (wins / closed) * 100.0,
                "wins": float(wins),
                "losses": float(losses),
            }
        for mk in unique:
            out.setdefault(mk, {"profit_factor": 0.0, "win_rate": 0.0, "wins": 0.0, "losses": 0.0})
        return out

    def save_optimizer_state(self, settings: dict[str, Any]) -> None:
        ts_utc = datetime.now(UTC).isoformat()
        tenant_id = self._tenant_id()
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO optimizer_state (tenant_id, ts_utc, settings_json) VALUES (?, ?, ?)",
                (tenant_id, ts_utc, json.dumps(settings or {})),
            )

    def load_latest_optimizer_state(self) -> dict[str, Any]:
        tenant_id = self._tenant_id()
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT settings_json
                FROM optimizer_state
                WHERE tenant_id = ?
                ORDER BY datetime(ts_utc) DESC, id DESC
                LIMIT 1
                """
            , (tenant_id,)).fetchone()
        if row is None:
            return {}
        try:
            payload = json.loads(str(row["settings_json"] or "{}"))
            return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}
