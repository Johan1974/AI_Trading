"""
Bestand: app/exchanges/bitvavo_manager.py
Relatief pad: ./app/exchanges/bitvavo_manager.py
Functie: Rate-limit manager voor Bitvavo die headers uitleest en auto-throttle toepast.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime

from app.datetime_util import UTC
from pathlib import Path
import sqlite3
import time
from typing import Any


@dataclass
class RateLimitState:
    limit: int = 1000
    remaining: int = 1000
    reset_at_ms: int = 0
    used_weight_local: int = 0


@dataclass
class CircuitBreakerState:
    is_open: bool = False
    half_open: bool = False
    open_until_ms: int = 0
    failure_threshold: int = 5
    failure_window_seconds: int = 120
    open_duration_seconds: int = 60


class BitvavoRateLimitManager:
    def __init__(
        self,
        throttle_threshold: float = 0.80,
        failure_threshold: int = 5,
        failure_window_seconds: int = 120,
        open_duration_seconds: int = 60,
    ) -> None:
        self.throttle_threshold = throttle_threshold
        self.state = RateLimitState()
        self.circuit_breaker = CircuitBreakerState(
            failure_threshold=failure_threshold,
            failure_window_seconds=failure_window_seconds,
            open_duration_seconds=open_duration_seconds,
        )
        self.default_request_weight = 1
        self.endpoint_weights: dict[tuple[str, str], int] = {
            ("GET", "/v2/balance"): 1,
            ("GET", "/v2/ticker/price"): 1,
            ("GET", "/v2/book"): 1,
            ("POST", "/v2/order"): 1,
            ("DELETE", "/v2/order"): 1,
        }
        self._failure_timestamps_ms: deque[int] = deque()
        self._hourly_usage: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"requests": 0, "weight": 0, "errors": 0, "last_status": None}
        )

    def _now_ms(self) -> int:
        return int(time.time() * 1000)

    def _hour_key(self) -> str:
        return datetime.now(UTC).strftime("%Y-%m-%dT%H:00:00Z")

    def _prune_failures(self, now_ms: int) -> None:
        window_ms = self.circuit_breaker.failure_window_seconds * 1000
        while self._failure_timestamps_ms and (now_ms - self._failure_timestamps_ms[0]) > window_ms:
            self._failure_timestamps_ms.popleft()

    def _log_usage(self, weight: int, status: str) -> None:
        row = self._hourly_usage[self._hour_key()]
        row["requests"] += 1
        row["weight"] += max(0, int(weight))
        row["last_status"] = status
        if status.startswith("error_"):
            row["errors"] += 1

    def _is_circuit_open(self, now_ms: int) -> bool:
        if self.circuit_breaker.open_until_ms > now_ms:
            self.circuit_breaker.is_open = True
            return True
        if self.circuit_breaker.is_open:
            self.circuit_breaker.is_open = False
            self.circuit_breaker.half_open = True
        self.circuit_breaker.is_open = False
        return False

    def resolve_weight(self, method: str, path: str) -> int:
        return self.endpoint_weights.get((method.upper(), path), self.default_request_weight)

    def before_request(self, method: str, path: str) -> int:
        now_ms = self._now_ms()
        if self._is_circuit_open(now_ms):
            wait_s = max(0.0, (self.circuit_breaker.open_until_ms - now_ms) / 1000.0)
            raise RuntimeError(f"Bitvavo circuit breaker OPEN. Network blocked for {wait_s:.1f}s to prevent bans.")

        weight = self.resolve_weight(method, path)
        limit = max(1, self.state.limit)
        used_ratio = (self.state.used_weight_local + weight) / limit
        if used_ratio >= self.throttle_threshold:
            self.sleep_until_reset(fallback_seconds=1.0)
            self.state.used_weight_local = 0
        return weight

    def update_from_headers(self, headers: Any, assumed_weight: int) -> None:
        limit = headers.get("bitvavo-ratelimit-limit")
        remaining = headers.get("bitvavo-ratelimit-remaining")
        reset_at = headers.get("bitvavo-ratelimit-resetat")

        if limit is not None:
            self.state.limit = int(limit)
        if remaining is not None:
            self.state.remaining = int(remaining)
        if reset_at is not None:
            self.state.reset_at_ms = int(reset_at)

        self.state.used_weight_local += assumed_weight
        if remaining is not None and limit is not None:
            self.state.used_weight_local = max(0, self.state.limit - self.state.remaining)

        if self.state.limit > 0:
            used_ratio = (self.state.limit - self.state.remaining) / self.state.limit
            if used_ratio >= self.throttle_threshold:
                self.sleep_until_reset(fallback_seconds=1.0)
                self.state.used_weight_local = 0

    def record_success(self, assumed_weight: int) -> None:
        self._log_usage(weight=assumed_weight, status="ok")
        now_ms = self._now_ms()
        self._prune_failures(now_ms)
        
        if self.circuit_breaker.half_open:
            self.circuit_breaker.half_open = False
            self._failure_timestamps_ms.clear()
            
        if not self._is_circuit_open(now_ms):
            if len(self._failure_timestamps_ms) <= 1:
                self._failure_timestamps_ms.clear()

    def record_failure(self, assumed_weight: int, status_code: int | None = None) -> None:
        status = f"error_{status_code}" if status_code is not None else "error_network"
        self._log_usage(weight=assumed_weight, status=status)
        now_ms = self._now_ms()
        
        if self.circuit_breaker.half_open:
            self.circuit_breaker.half_open = False
            self.circuit_breaker.open_until_ms = now_ms + (self.circuit_breaker.open_duration_seconds * 1000)
            self.circuit_breaker.is_open = True
            return
            
        self._failure_timestamps_ms.append(now_ms)
        self._prune_failures(now_ms)

        if len(self._failure_timestamps_ms) >= self.circuit_breaker.failure_threshold:
            self.circuit_breaker.open_until_ms = (
                now_ms + (self.circuit_breaker.open_duration_seconds * 1000)
            )
            self.circuit_breaker.is_open = True

    def handle_429(self, headers: Any, attempt: int) -> None:
        reset_at = (headers or {}).get("bitvavo-ratelimit-resetat")
        if reset_at:
            self.state.reset_at_ms = int(reset_at)
            self.sleep_until_reset(fallback_seconds=1.0)
        else:
            delay = min(30.0, 0.5 * (2**attempt))
            time.sleep(delay)

    def sleep_until_reset(self, fallback_seconds: float = 1.0) -> None:
        now_ms = int(time.time() * 1000)
        if self.state.reset_at_ms > now_ms:
            wait_s = max(0.0, (self.state.reset_at_ms - now_ms) / 1000.0)
            time.sleep(wait_s)
            return
        time.sleep(fallback_seconds)

    def snapshot(self) -> dict[str, Any]:
        now_ms = self._now_ms()
        self._prune_failures(now_ms)
        return {
            "rate_limit": {
                "limit": self.state.limit,
                "remaining": self.state.remaining,
                "reset_at_ms": self.state.reset_at_ms,
                "used_weight_local": self.state.used_weight_local,
            },
            "circuit_breaker": {
                "is_open": self._is_circuit_open(now_ms),
                "open_until_ms": self.circuit_breaker.open_until_ms,
                "failure_threshold": self.circuit_breaker.failure_threshold,
                "failure_window_seconds": self.circuit_breaker.failure_window_seconds,
                "open_duration_seconds": self.circuit_breaker.open_duration_seconds,
                "recent_failures": len(self._failure_timestamps_ms),
            },
            "hourly_usage": dict(self._hourly_usage),
        }


class BitvavoPaperManager:
    """
    Simpele paper-mode opslag voor balance/order calls zonder live exchange request.
    """

    def __init__(self, db_path: str = "data/paper_bitvavo.db", start_balance_eur: float = 10000.0) -> None:
        path = Path(db_path)
        if not path.is_absolute():
            path = Path.cwd() / path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = str(path)
        self.start_balance_eur = start_balance_eur
        self._init_db()
        self._ensure_eur_wallet()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS paper_balances (
                    symbol TEXT PRIMARY KEY,
                    available REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS paper_orders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts_utc TEXT NOT NULL,
                    market TEXT NOT NULL,
                    side TEXT NOT NULL,
                    amount_quote_eur REAL NOT NULL,
                    status TEXT NOT NULL
                )
                """
            )

    def _ensure_eur_wallet(self) -> None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT available FROM paper_balances WHERE symbol = 'EUR'"
            ).fetchone()
            if row is None:
                conn.execute(
                    "INSERT INTO paper_balances(symbol, available) VALUES('EUR', ?)",
                    (float(self.start_balance_eur),),
                )

    def get_balance(self) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT symbol, available FROM paper_balances ORDER BY symbol ASC"
            ).fetchall()
        return [{"symbol": str(r["symbol"]), "available": float(r["available"])} for r in rows]

    def place_market_order(self, market: str, side: str, amount_quote: float) -> dict[str, Any]:
        base, quote = market.split("-", 1)
        amount_quote = max(0.0, float(amount_quote))
        with self._conn() as conn:
            eur_row = conn.execute(
                "SELECT available FROM paper_balances WHERE symbol = ?", (quote.upper(),)
            ).fetchone()
            quote_available = float(eur_row["available"]) if eur_row else 0.0
            status = "executed"

            if side.upper() == "BUY":
                if quote_available < amount_quote:
                    status = "rejected_insufficient_balance"
                else:
                    conn.execute(
                        "UPDATE paper_balances SET available = ? WHERE symbol = ?",
                        (quote_available - amount_quote, quote.upper()),
                    )
                    base_row = conn.execute(
                        "SELECT available FROM paper_balances WHERE symbol = ?", (base.upper(),)
                    ).fetchone()
                    base_available = float(base_row["available"]) if base_row else 0.0
                    if base_row:
                        conn.execute(
                            "UPDATE paper_balances SET available = ? WHERE symbol = ?",
                            (base_available + (amount_quote / 100000.0), base.upper()),
                        )
                    else:
                        conn.execute(
                            "INSERT INTO paper_balances(symbol, available) VALUES(?, ?)",
                            (base.upper(), amount_quote / 100000.0),
                        )
            elif side.upper() == "SELL":
                base_row = conn.execute(
                    "SELECT available FROM paper_balances WHERE symbol = ?", (base.upper(),)
                ).fetchone()
                base_available = float(base_row["available"]) if base_row else 0.0
                sell_qty_proxy = amount_quote / 100000.0
                if base_available < sell_qty_proxy:
                    status = "rejected_insufficient_balance"
                else:
                    conn.execute(
                        "UPDATE paper_balances SET available = ? WHERE symbol = ?",
                        (base_available - sell_qty_proxy, base.upper()),
                    )
                    conn.execute(
                        "UPDATE paper_balances SET available = ? WHERE symbol = ?",
                        (quote_available + amount_quote, quote.upper()),
                    )
            else:
                status = "rejected_invalid_side"

            conn.execute(
                """
                INSERT INTO paper_orders(ts_utc, market, side, amount_quote_eur, status)
                VALUES (?, ?, ?, ?, ?)
                """,
                (datetime.now(UTC).isoformat(), market.upper(), side.upper(), amount_quote, status),
            )
        return {
            "status": status,
            "market": market.upper(),
            "side": side.upper(),
            "amount_quote_eur": amount_quote,
        }
