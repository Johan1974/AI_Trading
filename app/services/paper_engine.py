"""
Bestand: app/services/paper_engine.py
Relatief pad: ./app/services/paper_engine.py
Functie: Paper trading manager met virtuele wallet, trade history opslag en sentiment/outcome analytics.
"""

from __future__ import annotations

import copy
import json
import math
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime

from app.datetime_util import UTC
from pathlib import Path
from typing import Any

from app.services.state import current_tenant_id

@dataclass
class PaperConfig:
    starting_balance_eur: float = 1000.0
    fee_rate: float = 0.0015
    db_path: str = "data/database.db"


class PaperTradeManager:
    def __init__(self, config: PaperConfig) -> None:
        self.config = config
        self._db_file = Path(config.db_path)
        if not self._db_file.is_absolute():
            self._db_file = Path.cwd() / self._db_file
        self._db_file.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self._wallets: dict[str, dict[str, Any]] = {}
        self._trade_closed_hooks: list[Any] = []
        self._ensure_wallet_for_tenant(self._tenant_id())

    def register_trade_closed_listener(self, fn: Any) -> None:
        """Sync callback: (market, realized_pnl_eur, reward_pct, trades_count) na afgeronde SELL-close."""
        if callable(fn) and fn not in self._trade_closed_hooks:
            self._trade_closed_hooks.append(fn)

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
        bal = round(float(initial_balance), 2)
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

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_file)
        conn.row_factory = sqlite3.Row
        return conn

    def _tenant_id(self) -> str:
        try:
            tid = current_tenant_id()
        except (LookupError, RuntimeError, Exception):
            tid = "default"
        return str(tid or "default").strip().lower() or "default"

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
            if "brain_state_json" not in names:
                conn.execute("ALTER TABLE trade_history ADD COLUMN brain_state_json TEXT NOT NULL DEFAULT ''")
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
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS predict_rl_feature_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tenant_id TEXT NOT NULL DEFAULT 'default',
                    ts_utc TEXT NOT NULL,
                    market TEXT NOT NULL,
                    snapshot_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_predict_rl_snap_ts ON predict_rl_feature_snapshots (tenant_id, ts_utc)"
            )
            cols_ev = conn.execute("PRAGMA table_info(trade_events)").fetchall()
            names_ev = {str(row["name"]) for row in cols_ev}
            if "tenant_id" not in names_ev:
                conn.execute("ALTER TABLE trade_events ADD COLUMN tenant_id TEXT NOT NULL DEFAULT 'default'")
            if "brain_state_json" not in names_ev:
                conn.execute("ALTER TABLE trade_events ADD COLUMN brain_state_json TEXT NOT NULL DEFAULT ''")
            cols_opt = conn.execute("PRAGMA table_info(optimizer_state)").fetchall()
            names_opt = {str(row["name"]) for row in cols_opt}
            if "tenant_id" not in names_opt:
                conn.execute("ALTER TABLE optimizer_state ADD COLUMN tenant_id TEXT NOT NULL DEFAULT 'default'")
            from core.paper_open_guard import ensure_active_positions_ddl, ensure_open_trade_registry_ddl

            ensure_active_positions_ddl(conn)
            ensure_open_trade_registry_ddl(conn)

    def _brain_state_for_persist(self, brain_state_json: str | None) -> str:
        if isinstance(brain_state_json, str) and brain_state_json.strip():
            return brain_state_json.strip()[:500000]
        try:
            from app.services.brain_state_capture import capture_brain_state_json

            return capture_brain_state_json()
        except Exception:
            return ""

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
        has_lots_in_obm = any(isinstance(v, list) and v for v in obm.values())
        if isinstance(old_lots, list) and old_lots and not has_lots_in_obm:
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
        if sym:
            _ep_for_sym = 0.0
            _sym_lots = obm.get(sym)
            if isinstance(_sym_lots, list) and _sym_lots:
                try:
                    _ep_for_sym = float(_sym_lots[0].get("entry_price") or 0.0)
                except (TypeError, ValueError):
                    pass
            if _ep_for_sym <= 0 and w.get("last_price") is not None:
                try:
                    _ep_for_sym = float(w["last_price"])
                except (TypeError, ValueError):
                    pass
            if _ep_for_sym > 0:
                lp.setdefault(sym, _ep_for_sym)
        try:
            sb = float(w.get("starting_balance_eur") or 0.0)
        except (TypeError, ValueError):
            sb = 0.0
        if sb <= 0:
            sb = float(self.config.starting_balance_eur)
        w.setdefault("starting_balance_eur", round(sb, 2))
        if w.get("paper_anchor_equity_eur") is None:
            w["paper_anchor_equity_eur"] = float(w["starting_balance_eur"])
        else:
            try:
                w["paper_anchor_equity_eur"] = max(1.0, float(w["paper_anchor_equity_eur"]))
            except (TypeError, ValueError):
                w["paper_anchor_equity_eur"] = float(w["starting_balance_eur"])
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
        self.wallet["history"] = self.wallet["history"][-200:]

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
        try:
            from app.services.portfolio_qty import normalize_paper_base_qty
        except Exception:
            normalize_paper_base_qty = None  # type: ignore[assignment]
        anchor = float(
            self.wallet.get("paper_anchor_equity_eur")
            or self.wallet.get("starting_balance_eur")
            or self.config.starting_balance_eur
            or max(cash, 1.0)
        )
        try:
            eq_hint = max(anchor, cash, float(self.wallet.get("equity", 0.0) or 0.0), 100.0)
        except (TypeError, ValueError):
            eq_hint = max(anchor, cash, 100.0)
        pbm = self.wallet.get("position_by_market") if isinstance(self.wallet.get("position_by_market"), dict) else {}
        obm = self.wallet.get("open_lots_by_market") if isinstance(self.wallet.get("open_lots_by_market"), dict) else {}
        total_mv = 0.0
        for mkt, qv in pbm.items():
            q_raw = float(qv or 0.0)
            if q_raw <= 1e-12:
                continue
            mku = str(mkt).strip().upper().replace("/", "-")
            px_loc = float(lp_map.get(mku, 0.0) or 0.0)
            if px_loc <= 0 and mku == str(self.wallet.get("position_symbol") or "").strip().upper().replace("/", "-"):
                px_loc = float(self.wallet.get("last_price") or 0.0)
            ep_hint: float | None = None
            lots = obm.get(mku) if isinstance(obm.get(mku), list) else None
            if lots is None and isinstance(obm, dict):
                for ok, ov in obm.items():
                    if str(ok).strip().upper().replace("/", "-") == mku and isinstance(ov, list):
                        lots = ov
                        break
            if isinstance(lots, list):
                for lot in lots:
                    if isinstance(lot, dict):
                        try:
                            ep_hint = float(lot.get("entry_price") or 0.0) or None
                        except (TypeError, ValueError):
                            ep_hint = None
                        if ep_hint and ep_hint > 0:
                            break
            if px_loc <= 0 and ep_hint and ep_hint > 0:
                px_loc = ep_hint
            elif px_loc > 0 and ep_hint and ep_hint > 0:
                ratio = px_loc / ep_hint
                if ratio > 50 or ratio < 0.02:
                    px_loc = ep_hint
            if normalize_paper_base_qty is not None:
                q = float(normalize_paper_base_qty(mku, q_raw, entry_price=ep_hint, equity_eur=eq_hint))
            else:
                q = q_raw
            total_mv += q * px_loc
        self.wallet["equity"] = cash + total_mv
        start = float(self.config.starting_balance_eur)
        if start > 0:
            self.wallet["realized_pnl_pct"] = (float(self.wallet["realized_pnl_eur"]) / start) * 100.0

    def _recompute_equity_for_wallet_dict(self, w: dict[str, Any], mark_price: float | None = None, mark_market: str | None = None) -> None:
        """Zelfde als ``_recompute_equity`` maar op een los wallet-dict (SQLite-restore / sanitize)."""
        if not isinstance(w, dict):
            return
        lp_map = w.setdefault("last_prices_by_market", {})
        if not isinstance(lp_map, dict):
            w["last_prices_by_market"] = {}
            lp_map = w["last_prices_by_market"]
        if mark_market and mark_price is not None:
            try:
                fv = float(mark_price)
                if fv > 0:
                    lp_map[str(mark_market).strip().upper()] = fv
            except (TypeError, ValueError):
                pass
        if mark_price is not None:
            try:
                w["last_price"] = float(mark_price)
            except (TypeError, ValueError):
                pass
        cash = float(w.get("cash", 0.0) or 0.0)
        try:
            from app.services.portfolio_qty import normalize_paper_base_qty
        except Exception:
            normalize_paper_base_qty = None  # type: ignore[assignment]
        anchor = float(
            w.get("paper_anchor_equity_eur")
            or w.get("starting_balance_eur")
            or self.config.starting_balance_eur
            or max(cash, 1.0)
        )
        try:
            eq_hint = max(anchor, cash, float(w.get("equity", 0.0) or 0.0), 100.0)
        except (TypeError, ValueError):
            eq_hint = max(anchor, cash, 100.0)
        pbm = w.get("position_by_market") if isinstance(w.get("position_by_market"), dict) else {}
        obm = w.get("open_lots_by_market") if isinstance(w.get("open_lots_by_market"), dict) else {}
        total_mv = 0.0
        for mkt, qv in pbm.items():
            q_raw = float(qv or 0.0)
            if q_raw <= 1e-12:
                continue
            mku = str(mkt).strip().upper().replace("/", "-")
            px_loc = float(lp_map.get(mku, 0.0) or 0.0)
            if px_loc <= 0 and mku == str(w.get("position_symbol") or "").strip().upper().replace("/", "-"):
                px_loc = float(w.get("last_price") or 0.0)
            ep_hint: float | None = None
            lots = obm.get(mku) if isinstance(obm.get(mku), list) else None
            if lots is None and isinstance(obm, dict):
                for ok, ov in obm.items():
                    if str(ok).strip().upper().replace("/", "-") == mku and isinstance(ov, list):
                        lots = ov
                        break
            if isinstance(lots, list):
                for lot in lots:
                    if isinstance(lot, dict):
                        try:
                            ep_hint = float(lot.get("entry_price") or 0.0) or None
                        except (TypeError, ValueError):
                            ep_hint = None
                        if ep_hint and ep_hint > 0:
                            break
            if px_loc <= 0 and ep_hint and ep_hint > 0:
                px_loc = ep_hint
            elif px_loc > 0 and ep_hint and ep_hint > 0:
                ratio = px_loc / ep_hint
                if ratio > 50 or ratio < 0.02:
                    px_loc = ep_hint
            if normalize_paper_base_qty is not None:
                q = float(normalize_paper_base_qty(mku, q_raw, entry_price=ep_hint, equity_eur=eq_hint))
            else:
                q = q_raw
            total_mv += q * px_loc
        w["equity"] = cash + total_mv
        start = float(self.config.starting_balance_eur)
        if start > 0:
            w["realized_pnl_pct"] = (float(w.get("realized_pnl_eur", 0.0) or 0.0) / start) * 100.0

    def _sanitize_wallet_snapshot_from_sqlite(self, tenant_id: str) -> None:
        """
        Na ``wallet_state``-load: console-log van ruwe JSON, BTC-satoshi fix, spook-posities wissen
        als equity/notional boven harde cap uitkomt (paper ~€9.5k).
        """
        w = self._wallets.get(tenant_id)
        if not isinstance(w, dict) or not w:
            return
        start = float(self.config.starting_balance_eur or 1000.0)
        cap_mult = float(os.getenv("PAPER_EQUITY_HARD_CAP_MULT", "1.35") or 1.35)
        cap = max(start * cap_mult, 5000.0)

        try:
            raw_blob = {
                "cash": w.get("cash"),
                "equity": w.get("equity"),
                "position_qty": w.get("position_qty"),
                "position_symbol": w.get("position_symbol"),
                "open_lots_by_market": w.get("open_lots_by_market"),
                "position_by_market": w.get("position_by_market"),
            }
            s = json.dumps(raw_blob, default=str)
            print(f"[PAPER][DB-RAW] tenant={tenant_id} wallet_state snapshot (max 8000 chars):\n{s[:8000]}")
        except Exception as exc:
            print(f"[PAPER][DB-RAW] tenant={tenant_id} serialize failed: {exc}")

        try:
            from app.services.portfolio_qty import implied_equity_eur_from_wallet, normalize_paper_base_qty
        except Exception:
            normalize_paper_base_qty = None  # type: ignore[assignment]
            implied_equity_eur_from_wallet = None  # type: ignore[assignment]

        if normalize_paper_base_qty is not None:
            obm = w.get("open_lots_by_market") if isinstance(w.get("open_lots_by_market"), dict) else {}
            try:
                eq_hint = float(w.get("equity", 0.0) or 0.0)
            except (TypeError, ValueError):
                eq_hint = None
            for mkt, lots in list(obm.items()):
                if not isinstance(lots, list):
                    continue
                mku = str(mkt).strip().upper()
                for lot in lots:
                    if not isinstance(lot, dict):
                        continue
                    try:
                        rq = float(lot.get("qty", 0.0) or 0.0)
                    except (TypeError, ValueError):
                        continue
                    try:
                        ep = float(lot.get("entry_price") or 0.0) or None
                    except (TypeError, ValueError):
                        ep = None
                    if mku.startswith("BTC") and rq >= 100_000.0:
                        old = rq
                        rq = rq / 1e8
                        print(f"[PAPER][UNIT] {mku}: qty lijkt satoshis zonder /1e8 in DB — {old} -> {rq}")
                    lot["qty"] = normalize_paper_base_qty(mku, rq, entry_price=ep, equity_eur=eq_hint)
            pbm = w.setdefault("position_by_market", {})
            if not isinstance(pbm, dict):
                w["position_by_market"] = {}
                pbm = w["position_by_market"]
            for mkt, qv in list(pbm.items()):
                try:
                    qq = float(qv or 0.0)
                except (TypeError, ValueError):
                    continue
                mku = str(mkt).strip().upper()
                if mku.startswith("BTC") and qq >= 100_000.0:
                    print(f"[PAPER][UNIT] {mku} position_by_market: satoshi /1e8 — {qq} -> {qq / 1e8}")
                    qq = qq / 1e8
                pbm[mkt] = normalize_paper_base_qty(mku, qq, entry_price=None, equity_eur=eq_hint)

        self._migrate_multi_asset_wallet(w)
        from core.paper_open_guard import merge_duplicate_open_lots_in_wallet, reconcile_active_positions_with_wallet

        merged_mk = merge_duplicate_open_lots_in_wallet(w)
        if merged_mk:
            print(
                f"[DEDUP] open_lots_by_market: samengevoegd naar één lot per paar: {', '.join(merged_mk)}",
                flush=True,
            )
        self._recompute_position_by_market_from_lots(w)
        self._recompute_equity_for_wallet_dict(w)
        try:
            with self._conn() as conn:
                reconcile_active_positions_with_wallet(conn, tenant_id, w)
        except Exception as exc:
            print(f"[PAPER] active_positions reconcile na sanitize: {exc}", flush=True)
        if merged_mk:
            try:
                self._persist_wallet_state()
            except Exception as exc:
                print(f"[PAPER] persist na lot-dedup: {exc}", flush=True)

        implied = 0.0
        if implied_equity_eur_from_wallet is not None:
            try:
                implied = float(implied_equity_eur_from_wallet(w))
            except Exception:
                implied = 0.0
        try:
            eq_r = float(w.get("equity", 0.0) or 0.0)
        except (TypeError, ValueError):
            eq_r = 0.0

        ghost = False
        if eq_r > cap or implied > cap:
            ghost = True
        obm2 = w.get("open_lots_by_market") if isinstance(w.get("open_lots_by_market"), dict) else {}
        for mkt, lots in obm2.items():
            if not isinstance(lots, list):
                continue
            for lot in lots:
                if not isinstance(lot, dict):
                    continue
                try:
                    qn = float(lot.get("qty", 0.0) or 0.0)
                    ep = float(lot.get("entry_price") or 0.0) or 0.0
                except (TypeError, ValueError):
                    continue
                if qn > 0 and ep > 0 and (qn * ep) > cap:
                    ghost = True
                    break
            if ghost:
                break

        if ghost:
            try:
                cash_v = float(w.get("cash", start) or 0.0)
            except (TypeError, ValueError):
                cash_v = start
            print(
                f"[PAPER][GHOST] tenant={tenant_id} open posities gewist (eq_r={eq_r:.2f} implied={implied:.2f} "
                f"cap={cap:.2f} EUR start={start:.2f})"
            )
            w["open_lots_by_market"] = {}
            w["open_lots"] = []
            w["position_by_market"] = {}
            w["position_qty"] = 0.0
            w["position_symbol"] = None
            w["equity"] = max(0.0, min(cash_v, cap))
            self._migrate_multi_asset_wallet(w)
            self._recompute_position_by_market_from_lots(w)
            self._recompute_equity_for_wallet_dict(w)
            if str(self._tenant_id()) == str(tenant_id):
                try:
                    self._persist_wallet_state()
                except Exception as exc:
                    print(f"[PAPER][GHOST] wallet_state persist na sanitize mislukt: {exc}")

    def _wallet_dict_for_sql_snapshot(self, *, aggressive: bool = False) -> dict[str, Any]:
        """Kopie van wallet voor SQLite JSON; inkorten om SQLITE/python binding limits te vermijden."""
        src = self.wallet if isinstance(self.wallet, dict) else {}
        keep_keys = (
            "cash",
            "equity",
            "position_qty",
            "position_symbol",
            "open_lots_by_market",
            "position_by_market",
            "open_lots",
            "history",
            "realized_pnl_eur",
            "trades_count",
            "starting_balance_eur",
            "paper_anchor_equity_eur",
            "last_trade_utc",
        )
        wallet_snap: dict[str, Any] = {}
        for key in keep_keys:
            if key in src:
                try:
                    wallet_snap[key] = copy.deepcopy(src.get(key))
                except Exception:
                    wallet_snap[key] = src.get(key)
        if not isinstance(wallet_snap.get("open_lots_by_market"), dict):
            wallet_snap["open_lots_by_market"] = {}
        if not isinstance(wallet_snap.get("position_by_market"), dict):
            wallet_snap["position_by_market"] = {}
        if not isinstance(wallet_snap.get("open_lots"), list):
            wallet_snap["open_lots"] = []
        if not isinstance(wallet_snap.get("history"), list):
            wallet_snap["history"] = []
        if aggressive:
            wallet_snap["history"] = []
            obm_a = wallet_snap.get("open_lots_by_market")
            if isinstance(obm_a, dict):
                for lots in obm_a.values():
                    if not isinstance(lots, list):
                        continue
                    for lot in lots:
                        if isinstance(lot, dict):
                            lot["headlines"] = []
                            lot["ai_thought"] = ""
                            lc = str(lot.get("ledger_context") or "")
                            lot["ledger_context"] = lc[:200]
            return wallet_snap

        hist = wallet_snap.get("history")
        if isinstance(hist, list):
            wallet_snap["history"] = hist[-100:]
            for row in wallet_snap["history"]:
                if not isinstance(row, dict):
                    continue
                for k, v in list(row.items()):
                    if isinstance(v, str) and len(v) > 8000:
                        row[k] = v[:8000] + "…"

        _MAX_LOTS = int(os.getenv("WALLET_SNAPSHOT_MAX_LOTS_PER_MARKET", "50") or 50)
        obm = wallet_snap.get("open_lots_by_market")
        if isinstance(obm, dict):
            for mk, lots in list(obm.items()):
                if not isinstance(lots, list):
                    continue
                if len(lots) > _MAX_LOTS:
                    obm[mk] = lots[-_MAX_LOTS:]
                    lots = obm[mk]
                for lot in lots:
                    if not isinstance(lot, dict):
                        continue
                    at = lot.get("ai_thought")
                    if isinstance(at, str) and len(at) > 2000:
                        lot["ai_thought"] = at[:2000] + "…"
                    hl = lot.get("headlines")
                    if isinstance(hl, list):
                        lot["headlines"] = [str(x)[:500] for x in hl[:3]]
                    lc = lot.get("ledger_context")
                    if isinstance(lc, str) and len(lc) > 300:
                        lot["ledger_context"] = lc[:300]
        return wallet_snap

    def _persist_wallet_state(self) -> None:
        ts = datetime.now(UTC).isoformat()
        tenant_id = self._tenant_id()

        last_err: Exception | None = None
        for attempt, aggressive in enumerate((False, True)):
            wallet_snap = self._wallet_dict_for_sql_snapshot(aggressive=aggressive)

            try:
                snap_json = json.dumps(wallet_snap, ensure_ascii=False)
            except Exception as exc:
                print(f"[PAPER] Waarschuwing: Kan wallet state niet serialiseren: {exc}")
                return

            n_chars = len(snap_json)
            if not aggressive and n_chars > 12_000_000:
                print(f"[PAPER] wallet_state snapshot groot ({n_chars} chars); probeer minimale snapshot.", flush=True)
                continue

            try:
                with self._conn() as conn:
                    conn.execute(
                        """
                        INSERT INTO wallet_state (tenant_id, snapshot_json, updated_ts_utc)
                        VALUES (?, ?, ?)
                        ON CONFLICT(tenant_id) DO UPDATE SET snapshot_json=excluded.snapshot_json, updated_ts_utc=excluded.updated_ts_utc
                        """,
                        (tenant_id, snap_json, ts),
                    )
                return
            except (sqlite3.DataError, sqlite3.OperationalError, OverflowError, MemoryError, ValueError) as exc:
                last_err = exc
                msg = str(exc)
                if attempt == 0 and (
                    "INT_MAX" in msg
                    or "too long" in msg.lower()
                    or "Large Objects" in msg
                    or "string or blob too big" in msg.lower()
                ):
                    print(
                        f"[PAPER] wallet_state oversize/bind-fout ({n_chars} chars), herprobeer compact: {exc}",
                        flush=True,
                    )
                    continue
                print(f"[PAPER] wallet_state persist mislukt: {exc}", flush=True)
                return

        if last_err is not None:
            print(f"[PAPER] wallet_state persist faalde na compact-retry: {last_err}", flush=True)

    def force_reconcile_balance_clear_positions(self) -> dict[str, Any]:
        """
        Verwijdert alle open paper-posities (FIFO-lots); equity = cash (+ 0 marktwaarde).
        Wist geen trade_history/trade_events — alleen wallet_state. Gebruik na corrupte spook-notional.
        """
        w = self.wallet
        try:
            cash_before = float(w.get("cash", 0.0) or 0.0)
        except (TypeError, ValueError):
            cash_before = 0.0
        try:
            eq_before = float(w.get("equity", 0.0) or 0.0)
        except (TypeError, ValueError):
            eq_before = 0.0
        obm = w.get("open_lots_by_market") if isinstance(w.get("open_lots_by_market"), dict) else {}
        cleared_markets = sorted({str(m).strip().upper() for m in obm.keys() if m})
        w["open_lots_by_market"] = {}
        w["open_lots"] = []
        w["position_by_market"] = {}
        w["position_qty"] = 0.0
        w["position_symbol"] = None
        self._migrate_multi_asset_wallet(w)
        self._recompute_position_by_market_from_lots(w)
        self._recompute_equity_for_wallet_dict(w)
        try:
            eq_after = float(w.get("equity", 0.0) or 0.0)
        except (TypeError, ValueError):
            eq_after = 0.0
        self._persist_wallet_state()
        return {
            "ok": True,
            "cash": round(float(cash_before), 2),
            "equity_before": round(float(eq_before), 2),
            "equity_after": round(float(eq_after), 2),
            "cleared_open_markets": cleared_markets,
        }

    def _sql_exec_ignore(self, conn: sqlite3.Connection, sql: str) -> None:
        try:
            conn.execute(sql)
        except sqlite3.Error:
            pass

    def reset_paper_account(self, starting_balance_eur: float, *, full_environment_reset: bool | None = None) -> dict[str, Any]:
        """
        Zet paper-cash/equity terug naar ``starting_balance_eur`` en wist ledger-/wallet-SQL.

        - ``full_environment_reset=True`` (default via env ``PAPER_RESET_FULL_ENVIRONMENT``): DELETE *alle*
          rijen in ``trade_history``, ``trade_events``, ``paper_ledger_reset_batches``, ``wallet_state``;
          ook legacy-tabellen ``active_positions``, ``paper_trade_history``, ``closed_trades`` indien aanwezig.
          Geheugen-cache ``_wallets`` wordt geleegd — start met nul open posities.
        - ``False``: oude gedrag: archiveer alleen huidige tenant in ``paper_ledger_reset_batches``, daarna DELETE per tenant.
        """
        tid = self._tenant_id()
        ts = datetime.now(UTC).isoformat()
        start_bal = round(float(starting_balance_eur), 2)
        if start_bal <= 0:
            start_bal = round(float(self.config.starting_balance_eur), 2)
        if full_environment_reset is None:
            full_environment_reset = str(os.getenv("PAPER_RESET_FULL_ENVIRONMENT", "1")).strip().lower() in (
                "1",
                "true",
                "yes",
                "on",
            )
        hist_n = 0
        ev_n = 0
        with self._conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS paper_ledger_reset_batches (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tenant_id TEXT NOT NULL,
                    reset_ts_utc TEXT NOT NULL,
                    trade_history_json TEXT NOT NULL,
                    trade_events_json TEXT NOT NULL
                )
                """
            )
            if full_environment_reset:
                try:
                    hist_n = int(conn.execute("SELECT COUNT(*) FROM trade_history").fetchone()[0] or 0)
                except sqlite3.Error:
                    hist_n = 0
                try:
                    ev_n = int(conn.execute("SELECT COUNT(*) FROM trade_events").fetchone()[0] or 0)
                except sqlite3.Error:
                    ev_n = 0
                conn.execute("DELETE FROM trade_history")
                conn.execute("DELETE FROM trade_events")
                self._sql_exec_ignore(conn, "DELETE FROM paper_ledger_reset_batches")
                conn.execute("DELETE FROM wallet_state")
                for legacy in ("active_positions", "paper_trade_history", "closed_trades", "open_trade_registry"):
                    self._sql_exec_ignore(conn, f"DELETE FROM {legacy}")
                self._wallets.clear()
            else:
                hist_rows = conn.execute("SELECT * FROM trade_history WHERE tenant_id = ?", (tid,)).fetchall()
                ev_rows = conn.execute("SELECT * FROM trade_events WHERE tenant_id = ?", (tid,)).fetchall()
                hist_n = len(hist_rows)
                ev_n = len(ev_rows)
                if hist_rows or ev_rows:
                    hist_json = json.dumps([{k: r[k] for k in r.keys()} for r in hist_rows], default=str)
                    ev_json = json.dumps([{k: r[k] for k in r.keys()} for r in ev_rows], default=str)
                    conn.execute(
                        """
                        INSERT INTO paper_ledger_reset_batches (tenant_id, reset_ts_utc, trade_history_json, trade_events_json)
                        VALUES (?, ?, ?, ?)
                        """,
                        (tid, ts, hist_json, ev_json),
                    )
                conn.execute("DELETE FROM trade_history WHERE tenant_id = ?", (tid,))
                conn.execute("DELETE FROM trade_events WHERE tenant_id = ?", (tid,))
                try:
                    conn.execute("DELETE FROM open_trade_registry WHERE tenant_id = ?", (tid,))
                except sqlite3.Error:
                    pass
        self._wallets[tid] = self._init_wallet(start_bal)
        self._migrate_multi_asset_wallet(self._wallets[tid])
        self._persist_wallet_state()
        return {
            "tenant_id": tid,
            "starting_balance_eur": start_bal,
            "reset_ts_utc": ts,
            "full_environment_reset": bool(full_environment_reset),
            "archived_trade_history_rows": hist_n,
            "archived_trade_events_rows": ev_n,
        }

    def sync_wallet_from_db(self) -> dict[str, Any]:
        """Herlaad wallet uit wallet_state SQLite (voor portal/sync-wallet knop)."""
        tid = self._tenant_id()
        try:
            with self._conn() as conn:
                row = conn.execute(
                    "SELECT snapshot_json, updated_ts_utc FROM wallet_state WHERE tenant_id = ?",
                    (tid,),
                ).fetchone()
            if row is None:
                return {"ok": False, "reason": "no_snapshot"}
            snap = json.loads(str(row["snapshot_json"] or "{}"))
            if not isinstance(snap, dict) or not snap:
                return {"ok": False, "reason": "empty_snapshot"}
            self._wallets[tid] = snap
            self._sanitize_wallet_snapshot_from_sqlite(tid)
            return {
                "ok": True,
                "cash": float(snap.get("cash") or 0.0),
                "equity": float(snap.get("equity") or snap.get("cash") or 0.0),
                "open_markets": list((snap.get("open_lots_by_market") or {}).keys()),
                "updated_ts_utc": str(row["updated_ts_utc"] or ""),
            }
        except Exception as exc:
            return {"ok": False, "reason": str(exc)}

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
            raw_json = str(row["snapshot_json"] or "{}")
            print(f"[PAPER][DB-RAW] tenant={tenant_id} wallet_state.snapshot_json length={len(raw_json)} chars")
            data = json.loads(raw_json)
            if isinstance(data, dict) and data:
                self._wallets[tenant_id] = data
                self._sanitize_wallet_snapshot_from_sqlite(tenant_id)
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
        brain_state_json: str | None = None,
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
                    sentiment_score, headlines_json, fees_eur, pnl_eur, pnl_pct, outcome, ai_thought, ledger_context,
                    brain_state_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    self._brain_state_for_persist(brain_state_json),
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
        brain_state_json: str | None = None,
    ) -> None:
        tenant_id = self._tenant_id()
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO trade_events (
                    tenant_id, ts_utc, market, action, signal, status, price, qty, sentiment_score, pnl_eur, reason,
                    brain_state_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    self._brain_state_for_persist(brain_state_json),
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
            try:
                from core.paper_open_guard import delete_active_position, redis_clear_open_pair

                tid = self._tenant_id()
                with self._conn() as conn:
                    delete_active_position(conn, tid, mku)
                redis_clear_open_pair(tid, mku)
            except Exception as exc:
                print(f"[PAPER] active_positions clear na flat close {mku}: {exc}", flush=True)
        else:
            try:
                from core.paper_open_guard import update_active_position_qty

                tid = self._tenant_id()
                rem = sum(float(x.get("qty") or 0) for x in open_lots if isinstance(x, dict))
                with self._conn() as conn:
                    update_active_position_qty(conn, tid, mku, rem)
            except Exception as exc:
                print(f"[PAPER] active_positions qty-update {mku}: {exc}", flush=True)
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
        mkt_u = str(market or "").strip().upper().replace("/", "-")
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

        start_equity = float(self.wallet.get("equity", 0.0) or 0.0)
        anchor_eq = float(
            self.wallet.get("paper_anchor_equity_eur")
            or self.wallet.get("starting_balance_eur")
            or self.config.starting_balance_eur
            or start_equity
            or 1000.0
        )
        anchor_eq = max(1.0, anchor_eq)
        if signal == "BUY":
            notional = max(0.0, anchor_eq * size)
        else:
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
            from core.risk_manager import paper_iron_lockdown_buy_ok

            ok_iron, why_iron = paper_iron_lockdown_buy_ok(
                self.wallet, mkt_u, px, notional, float(self.config.fee_rate)
            )
            if not ok_iron:
                self._recompute_equity(px, mkt_u)
                self._record_trade_event(
                    ts_utc=ts_utc,
                    market=market,
                    action="BUY_REJECTED",
                    signal=signal,
                    status="rejected",
                    price=px,
                    qty=0.0,
                    sentiment_score=sentiment_score,
                    reason=why_iron,
                )
                self._persist_wallet_state()
                return {
                    "status": "rejected",
                    "signal": signal,
                    "reason": why_iron,
                    "wallet": self.wallet,
                }
            from app.services.reporting import format_skip_buy_open_pair_log
            from core.paper_open_guard import (
                can_open_position_sqlite,
                delete_active_position,
                has_active_position_for_base_sqlite,
                ensure_active_positions_ddl,
                merge_duplicate_open_lots_in_wallet,
                paper_buy_serialized,
                redis_acquire_buy_lock,
                redis_clear_open_pair,
                redis_has_open_pair_flag,
                redis_mark_open_pair,
                redis_release_buy_lock,
                reserve_active_position_slot,
                strict_block_duplicate_log,
                update_active_position_qty,
                wallet_open_lot_count,
            )
            from core.trading_engine import (
                has_active_paper_position_for_base_currency,
                has_active_paper_position_for_ticker,
                paper_base_currency_from_market,
            )

            with paper_buy_serialized(self._db_file, self._tenant_id(), mkt_u):
                tid = self._tenant_id()
                if not redis_acquire_buy_lock(tid, mkt_u):
                    msg = strict_block_duplicate_log(mkt_u) + " (Redis buy-lock bezet)"
                    print(msg, flush=True)
                    self._recompute_equity(px, mkt_u)
                    self._record_trade_event(
                        ts_utc=ts_utc,
                        market=market,
                        action="BUY_REJECTED",
                        signal=signal,
                        status="rejected",
                        price=px,
                        qty=0.0,
                        sentiment_score=sentiment_score,
                        reason="duplicate_buy_lock",
                    )
                    self._persist_wallet_state()
                    return {
                        "status": "rejected",
                        "signal": signal,
                        "reason": "duplicate_buy_lock",
                        "wallet": self.wallet,
                    }
                slot_reserved = False
                try:
                    self._restore_wallet_state(tid)
                    self._migrate_multi_asset_wallet(self.wallet)
                    mdedup = merge_duplicate_open_lots_in_wallet(self.wallet)
                    if mdedup:
                        self._recompute_position_by_market_from_lots(self.wallet)
                        self._recompute_equity(px, mkt_u)
                    if redis_has_open_pair_flag(tid, mkt_u) and not has_active_paper_position_for_ticker(self.wallet, market):
                        redis_clear_open_pair(tid, mkt_u)

                    _one_open = str(os.getenv("PAPER_ENFORCE_ONE_OPEN_TRADE_PER_PAIR", "1")).strip().lower() in (
                        "1",
                        "true",
                        "yes",
                        "on",
                    )
                    if _one_open and has_active_paper_position_for_ticker(self.wallet, market):
                        msg = format_skip_buy_open_pair_log(market)
                        print(msg, flush=True)
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
                            reason="open_position_exists",
                        )
                        self._persist_wallet_state()
                        return {
                            "status": "rejected",
                            "signal": signal,
                            "reason": "open_position_exists",
                            "wallet": self.wallet,
                        }
                    _one_base = str(os.getenv("PAPER_ENFORCE_ONE_OPEN_TRADE_PER_BASE", "1")).strip().lower() in (
                        "1",
                        "true",
                        "yes",
                        "on",
                    )
                    if _one_base and has_active_paper_position_for_base_currency(self.wallet, market):
                        b = paper_base_currency_from_market(mkt_u) or "?"
                        msg = (
                            f"[SKIP] Max 1 open positie per coin ({b}) op Bitvavo (EUR-paren); "
                            f"sluit eerst het openstaande {b}-EUR."
                        )
                        print(msg, flush=True)
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
                            reason="open_position_exists_base_currency",
                        )
                        self._persist_wallet_state()
                        return {
                            "status": "rejected",
                            "signal": signal,
                            "reason": "open_position_exists_base_currency",
                            "wallet": self.wallet,
                        }
                    if wallet_open_lot_count(self.wallet, mkt_u) >= 1:
                        msg = strict_block_duplicate_log(mkt_u)
                        print(msg, flush=True)
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
                            reason="duplicate_open_lots_wallet",
                        )
                        self._persist_wallet_state()
                        return {
                            "status": "rejected",
                            "signal": signal,
                            "reason": "duplicate_open_lots_wallet",
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
                    try:
                        with self._conn() as conn:
                            conn.execute("BEGIN IMMEDIATE")
                            ensure_active_positions_ddl(conn)
                            if not can_open_position_sqlite(conn, tid, mkt_u):
                                conn.rollback()
                                msg = strict_block_duplicate_log(mkt_u)
                                print(msg, flush=True)
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
                                    reason="duplicate_active_positions_sql_precheck",
                                )
                                self._persist_wallet_state()
                                return {
                                    "status": "rejected",
                                    "signal": signal,
                                    "reason": "duplicate_active_positions_sql_precheck",
                                    "wallet": self.wallet,
                                }
                            if _one_base and has_active_position_for_base_sqlite(conn, tid, mkt_u):
                                conn.rollback()
                                b = paper_base_currency_from_market(mkt_u) or "?"
                                msg = (
                                    f"[BLOCK] SQLite: al een actieve {b}-positie (Bitvavo EUR-paren). "
                                    f"Max 1 open trade per coin."
                                )
                                print(msg, flush=True)
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
                                    reason="duplicate_active_positions_base_sql",
                                )
                                self._persist_wallet_state()
                                return {
                                    "status": "rejected",
                                    "signal": signal,
                                    "reason": "duplicate_active_positions_base_sql",
                                    "wallet": self.wallet,
                                }
                            if not reserve_active_position_slot(conn, tid, mkt_u, ts_utc, float(qty)):
                                conn.rollback()
                                msg = strict_block_duplicate_log(mkt_u)
                                print(msg, flush=True)
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
                                    reason="duplicate_active_positions_sql",
                                )
                                self._persist_wallet_state()
                                return {
                                    "status": "rejected",
                                    "signal": signal,
                                    "reason": "duplicate_active_positions_sql",
                                    "wallet": self.wallet,
                                }
                            from core.paper_open_guard import ensure_open_trade_registry_ddl

                            ensure_open_trade_registry_ddl(conn)
                            try:
                                conn.execute(
                                    """
                                    INSERT INTO open_trade_registry (tenant_id, symbol, status, opened_ts_utc)
                                    VALUES (?, ?, 'OPEN', ?)
                                    """,
                                    (tid, mkt_u, ts_utc),
                                )
                            except sqlite3.IntegrityError:
                                conn.rollback()
                                msg = (
                                    f"[BLOCK] Anti-stacking: er staat al een OPEN-registratie voor {mkt_u} "
                                    f"(unieke index tenant+symbol+OPEN)."
                                )
                                print(msg, flush=True)
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
                                    reason="anti_stacking_open_registry_unique",
                                )
                                self._persist_wallet_state()
                                return {
                                    "status": "rejected",
                                    "signal": signal,
                                    "reason": "anti_stacking_open_registry_unique",
                                    "wallet": self.wallet,
                                }
                            conn.commit()
                            slot_reserved = True
                    except Exception as exc:
                        print(f"[PAPER] active_positions reserve mislukt voor {mkt_u}: {exc}", flush=True)
                        self._record_trade_event(
                            ts_utc=ts_utc,
                            market=market,
                            action="BUY_REJECTED",
                            signal=signal,
                            status="rejected",
                            price=px,
                            qty=0.0,
                            sentiment_score=sentiment_score,
                            reason="active_positions_reserve_error",
                        )
                        self._persist_wallet_state()
                        return {
                            "status": "rejected",
                            "signal": signal,
                            "reason": "active_positions_reserve_error",
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
                            "headlines": [str(h)[:2000] for h in (news_headlines or [])[:3]],
                            "ai_thought": str(ai_thought or "")[:8000],
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
                    try:
                        with self._conn() as conn:
                            update_active_position_qty(conn, tid, mkt_u, float(qty))
                    except Exception as exc:
                        print(f"[PAPER] active_positions qty sync na BUY {mkt_u}: {exc}", flush=True)
                    redis_mark_open_pair(tid, mkt_u)
                    return {
                        "status": "opened",
                        "signal": "BUY",
                        "entry_price": px,
                        "qty": qty,
                        "fee_eur": fee,
                        "wallet": self.wallet,
                    }
                except Exception as exc:
                    if slot_reserved:
                        try:
                            with self._conn() as c2:
                                delete_active_position(c2, tid, mkt_u)
                        except Exception:
                            pass
                    print(f"[PAPER] BUY-interne fout voor {mkt_u}: {exc}", flush=True)
                    return {
                        "status": "rejected",
                        "signal": "BUY",
                        "reason": "buy_internal_error",
                        "wallet": self.wallet,
                    }
                finally:
                    redis_release_buy_lock(tid, mkt_u)

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
            self.wallet["wins"] = int(self.wallet.get("wins", 0)) + 1
        else:
            self.wallet["losses"] = int(self.wallet.get("losses", 0)) + 1
        self.wallet["trades_count"] = int(self.wallet.get("trades_count", 0)) + 1
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
        if float(close_result.get("qty_closed") or 0.0) > 1e-12:
            cost_e = float(close_result.get("cost_eur", 0.0) or 0.0)
            rew_pct = (
                (float(close_result["realized_pnl_eur"]) / max(1e-9, cost_e)) * 100.0 if cost_e > 1e-9 else 0.0
            )
            tc = int(self.wallet.get("trades_count", 0) or 0)
            for fn in self._trade_closed_hooks:
                try:
                    fn(
                        market=mkt_u,
                        realized_pnl_eur=float(close_result["realized_pnl_eur"]),
                        reward_pct=float(rew_pct),
                        trades_count=tc,
                    )
                except Exception:
                    pass
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
        # Always read active lots from wallet_state SQLite (ground truth); in-memory wallet
        # is authoritative only on the worker and can be stale on portal after trades re-open.
        obm: dict = {}
        try:
            with self._conn() as _conn:
                _row = _conn.execute(
                    "SELECT snapshot_json FROM wallet_state WHERE tenant_id = ?",
                    (self._tenant_id(),),
                ).fetchone()
            if _row:
                _snap = json.loads(str(_row["snapshot_json"] or "{}"))
                obm = _snap.get("open_lots_by_market") if isinstance(_snap.get("open_lots_by_market"), dict) else {}
        except Exception:
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
        lpm = self.wallet.get("last_prices_by_market") if isinstance(self.wallet.get("last_prices_by_market"), dict) else {}
        for market, lot in active_lots:
            if not isinstance(lot, dict):
                continue
            qty = float(lot.get("qty") or 0.0)
            entry_price = float(lot.get("entry_price") or 0.0)
            if qty <= 0 or entry_price <= 0:
                continue
            mku = str(market).upper()
            mark = float(lpm.get(mku) or lpm.get(market) or 0.0)
            live_pct = 0.0
            live_eur = 0.0
            if mark > 0 and entry_price > 0:
                live_pct = ((mark - entry_price) / entry_price) * 100.0
                live_eur = qty * (mark - entry_price)
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
                    "live_pnl_pct": float(live_pct),
                    "live_pnl_eur": float(live_eur),
                    "mark_price": float(mark),
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
                       SUM(CASE WHEN pnl_eur < 0 THEN 1 ELSE 0 END) AS losses
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
            closed = wins + losses
            if closed <= 0:
                out[str(r["market"]).upper()] = {
                    "profit_factor": 1.0,
                    "win_rate": 50.0,
                    "wins": 0.0,
                    "losses": 0.0,
                }
                continue
            pf = 1.0 if (gl <= 1e-9 and gp <= 1e-9) else (gp / gl if gl > 1e-9 else gp)
            out[str(r["market"]).upper()] = {
                "profit_factor": float(pf),
                "win_rate": (wins / closed) * 100.0,
                "wins": float(wins),
                "losses": float(losses),
            }
        for mk in unique:
            out.setdefault(mk, {"profit_factor": 1.0, "win_rate": 50.0, "wins": 0.0, "losses": 0.0})
        return out

    def record_predict_rl_feature_snapshot(self, snapshot: dict[str, Any]) -> None:
        """Persisteert RL-featurevectors bij /predict (worker) voor offline analyse."""
        if not isinstance(snapshot, dict):
            return
        tenant_id = self._tenant_id()
        ts_utc = str(snapshot.get("ts_utc") or datetime.now(UTC).isoformat())
        market = str(snapshot.get("market") or "").strip().upper().replace("/", "-")
        if not market:
            return
        raw = json.dumps(snapshot, ensure_ascii=False, default=str)
        max_chars = int(os.getenv("RL_PREDICT_SNAPSHOT_MAX_CHARS", "1800000") or 1800000)
        if len(raw) > max_chars:
            raw = raw[: max_chars - 80] + "\n…[truncated RL_PREDICT_SNAPSHOT_MAX_CHARS]"
        keep = int(os.getenv("RL_PREDICT_SNAPSHOT_KEEP", "200") or 200)
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO predict_rl_feature_snapshots (tenant_id, ts_utc, market, snapshot_json)
                VALUES (?, ?, ?, ?)
                """,
                (tenant_id, ts_utc, market, raw),
            )
            conn.execute(
                """
                DELETE FROM predict_rl_feature_snapshots
                WHERE tenant_id = ? AND market = ? AND id NOT IN (
                    SELECT id FROM predict_rl_feature_snapshots
                    WHERE tenant_id = ? AND market = ?
                    ORDER BY id DESC LIMIT ?
                )
                """,
                (tenant_id, market, tenant_id, market, keep),
            )

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
