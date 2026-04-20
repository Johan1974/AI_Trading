"""
Bestand: app/services/paper_engine.py
Relatief pad: ./app/services/paper_engine.py
Functie: Paper trading manager met virtuele wallet, trade history opslag en sentiment/outcome analytics.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime

from app.datetime_util import UTC
from pathlib import Path
from typing import Any


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
        self.wallet = self._init_wallet(config.starting_balance_eur)

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
        }

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_file)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS trade_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
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
            if "ai_thought" not in names:
                conn.execute("ALTER TABLE trade_history ADD COLUMN ai_thought TEXT NOT NULL DEFAULT ''")

    def _coin_from_market(self, market: str) -> str:
        upper = (market or "").upper()
        if "-" in upper:
            return upper.split("-", 1)[0]
        return upper

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

    def _recompute_equity(self, mark_price: float | None) -> None:
        px = float(mark_price if mark_price is not None else self.wallet.get("last_price") or 0.0)
        position_value = float(self.wallet["position_qty"]) * px
        self.wallet["equity"] = float(self.wallet["cash"]) + position_value
        start = float(self.config.starting_balance_eur)
        if start > 0:
            self.wallet["realized_pnl_pct"] = (float(self.wallet["realized_pnl_eur"]) / start) * 100.0

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
    ) -> None:
        pnl_pct = 0.0
        if entry_price > 0:
            pnl_pct = ((exit_price - entry_price) / entry_price) * 100.0
        outcome = "profit" if pnl_eur >= 0 else "loss"
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO trade_history (
                    market, coin, entry_ts_utc, exit_ts_utc, entry_price, exit_price, qty,
                    sentiment_score, headlines_json, fees_eur, pnl_eur, pnl_pct, outcome, ai_thought
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
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
                ),
            )

    def _close_lots_fifo(self, market: str, qty_to_sell: float, exit_price: float, ts_utc: str) -> dict[str, Any]:
        total_realized = 0.0
        total_fees = 0.0
        closed = 0.0
        open_lots = self.wallet["open_lots"]
        while qty_to_sell > 1e-12 and open_lots:
            lot = open_lots[0]
            lot_qty = float(lot["qty"])
            close_qty = min(lot_qty, qty_to_sell)
            entry_price = float(lot["entry_price"])
            gross = close_qty * exit_price
            cost = close_qty * entry_price
            fee = gross * float(self.config.fee_rate)
            pnl_eur = gross - cost - fee
            total_realized += pnl_eur
            total_fees += fee
            closed += close_qty
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
            )
            lot["qty"] = lot_qty - close_qty
            qty_to_sell -= close_qty
            if float(lot["qty"]) <= 1e-12:
                open_lots.pop(0)
        return {"qty_closed": closed, "realized_pnl_eur": total_realized, "fees_eur": total_fees}

    def process_signal(
        self,
        market: str,
        signal: str,
        price: float,
        size_fraction: float,
        sentiment_score: float,
        news_headlines: list[str],
        ai_thought: str = "",
    ) -> dict[str, Any]:
        ts_utc = datetime.now(UTC).isoformat()
        signal = signal.upper()
        px = float(max(0.0, price))
        size = max(0.0, min(1.0, size_fraction))
        self.wallet["last_price"] = px

        if px <= 0 or signal not in {"BUY", "SELL"}:
            self._recompute_equity(px)
            self._append_wallet_history(
                market=market,
                action="HOLD",
                price=px,
                sentiment_score=sentiment_score,
            )
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
                self._recompute_equity(px)
                self._append_wallet_history(
                    market=market,
                    action="BUY_REJECTED",
                    price=px,
                    sentiment_score=sentiment_score,
                )
                return {
                    "status": "rejected",
                    "signal": signal,
                    "reason": "insufficient_cash",
                    "wallet": self.wallet,
                }
            qty = notional / px
            self.wallet["cash"] -= (notional + fee)
            self.wallet["position_qty"] += qty
            self.wallet["position_symbol"] = market
            self.wallet["open_lots"].append(
                {
                    "qty": qty,
                    "entry_price": px,
                    "entry_ts_utc": ts_utc,
                    "sentiment_score": sentiment_score,
                    "headlines": news_headlines[:3],
                    "ai_thought": ai_thought,
                }
            )
            self.wallet["trades_count"] += 1
            self._recompute_equity(px)
            self._append_wallet_history(
                market=market,
                action="BUY",
                price=px,
                sentiment_score=sentiment_score,
            )
            return {
                "status": "opened",
                "signal": "BUY",
                "entry_price": px,
                "qty": qty,
                "fee_eur": fee,
                "wallet": self.wallet,
            }

        if self.wallet["position_qty"] <= 0 or self.wallet["position_symbol"] != market:
            self._recompute_equity(px)
            self._append_wallet_history(
                market=market,
                action="SELL_REJECTED",
                price=px,
                sentiment_score=sentiment_score,
            )
            return {
                "status": "rejected",
                "signal": signal,
                "reason": "no_position",
                "wallet": self.wallet,
            }

        qty_target = notional / px if notional > 0 else self.wallet["position_qty"]
        qty_to_sell = min(float(self.wallet["position_qty"]), max(0.0, qty_target))
        close_result = self._close_lots_fifo(
            market=market,
            qty_to_sell=qty_to_sell,
            exit_price=px,
            ts_utc=ts_utc,
        )
        proceeds_after_fee = (close_result["qty_closed"] * px) - close_result["fees_eur"]
        self.wallet["cash"] += max(0.0, proceeds_after_fee)
        self.wallet["position_qty"] -= close_result["qty_closed"]
        if self.wallet["position_qty"] <= 1e-12:
            self.wallet["position_qty"] = 0.0
            self.wallet["position_symbol"] = None
        self.wallet["realized_pnl_eur"] += close_result["realized_pnl_eur"]
        if close_result["realized_pnl_eur"] >= 0:
            self.wallet["wins"] += 1
        else:
            self.wallet["losses"] += 1
        self.wallet["trades_count"] += 1
        self._recompute_equity(px)
        self._append_wallet_history(
            market=market,
            action="SELL",
            price=px,
            sentiment_score=sentiment_score,
            pnl_eur=float(close_result["realized_pnl_eur"]),
        )
        return {
            "status": "closed",
            "signal": "SELL",
            "exit_price": px,
            "qty_closed": close_result["qty_closed"],
            "realized_pnl_eur": close_result["realized_pnl_eur"],
            "fee_eur": close_result["fees_eur"],
            "wallet": self.wallet,
        }

    def analytics(self) -> dict[str, Any]:
        with self._conn() as conn:
            top_losses = conn.execute(
                "SELECT sentiment_score, pnl_eur FROM trade_history ORDER BY pnl_eur ASC LIMIT 10"
            ).fetchall()
            top_wins = conn.execute(
                "SELECT sentiment_score, pnl_eur FROM trade_history ORDER BY pnl_eur DESC LIMIT 10"
            ).fetchall()
            coin_rollup = conn.execute(
                """
                SELECT coin,
                       COUNT(*) AS trades,
                       AVG(sentiment_score) AS avg_sentiment,
                       AVG(pnl_eur) AS avg_pnl_eur
                FROM trade_history
                GROUP BY coin
                ORDER BY trades DESC
                LIMIT 10
                """
            ).fetchall()
            wl = conn.execute(
                "SELECT outcome, COUNT(*) AS cnt FROM trade_history GROUP BY outcome"
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
                GROUP BY bucket
                """
            ).fetchall()

        def _avg(rows: list[sqlite3.Row]) -> float:
            if not rows:
                return 0.0
            return float(sum(float(r["sentiment_score"]) for r in rows) / len(rows))

        wins = sum(int(r["cnt"]) for r in wl if str(r["outcome"]) == "profit")
        losses = sum(int(r["cnt"]) for r in wl if str(r["outcome"]) == "loss")
        total = max(1, wins + losses)
        return {
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
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT market, coin, entry_ts_utc, exit_ts_utc, entry_price, exit_price, qty,
                       sentiment_score, headlines_json, fees_eur, pnl_eur, pnl_pct, outcome, ai_thought
                FROM trade_history
                ORDER BY id DESC
                LIMIT ?
                """,
                (int(max(1, limit)),),
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
