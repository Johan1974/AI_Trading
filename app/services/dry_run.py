"""
Bestand: app/services/dry_run.py
Relatief pad: ./app/services/dry_run.py
Functie: Decorator voor dry-run trade logging en berekening van dagelijkse fictieve PnL.
"""

from __future__ import annotations

import csv
import os
from datetime import datetime

from app.datetime_util import UTC
from functools import wraps
from pathlib import Path
from typing import Any, Callable


DRY_RUN_FEE_RATE = 0.0015  # 0.15%
LOG_COLUMNS = [
    "ts_utc",
    "source",
    "mode",
    "market",
    "side",
    "signal",
    "reference_price",
    "amount_quote_eur",
    "fee_rate",
    "status",
]


def _log_path() -> Path:
    raw = os.getenv("TRADES_LOG_PATH", "trades_log.csv")
    path = Path(raw)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def _append_log_row(row: dict[str, Any]) -> None:
    path = _log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=LOG_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k) for k in LOG_COLUMNS})


def dry_run_trade_logger(source: str) -> Callable:
    """
    Decorator die trades logt en in dry-run mode echte API execution overslaat.
    Verwacht dat de wrapped functie een dict met minimaal 'status' en tradevelden retourneert.
    """

    def _decorator(func: Callable) -> Callable:
        @wraps(func)
        def _wrapper(*args, **kwargs):
            dry_run = str(os.getenv("DRY_RUN", "true")).strip().lower() in {"1", "true", "yes", "on"}
            ts = datetime.now(UTC).isoformat()

            if dry_run:
                result = {
                    "status": "dry_run_simulated",
                    "mode": "dry_run",
                }
                # Behoud consistente velden voor log + callers.
                result.update(
                    {
                        "market": kwargs.get("market") or kwargs.get("ticker") or "UNKNOWN",
                        "side": (kwargs.get("side") or "").upper(),
                        "signal": kwargs.get("signal", ""),
                        "reference_price": kwargs.get("price") or kwargs.get("reference_price") or 0.0,
                        "amount_quote_eur": kwargs.get("amount_quote")
                        or kwargs.get("amount_quote_eur")
                        or 0.0,
                        "fee_rate": DRY_RUN_FEE_RATE,
                    }
                )
            else:
                result = func(*args, **kwargs)
                if not isinstance(result, dict):
                    result = {"status": "executed", "raw": str(result)}
                result["mode"] = "live"

            _append_log_row(
                {
                    "ts_utc": ts,
                    "source": source,
                    "mode": result.get("mode", "unknown"),
                    "market": result.get("market", kwargs.get("market", kwargs.get("ticker", "UNKNOWN"))),
                    "side": str(result.get("side", kwargs.get("side", ""))).upper(),
                    "signal": result.get("signal", kwargs.get("signal", "")),
                    "reference_price": result.get("reference_price", kwargs.get("price", 0.0)),
                    "amount_quote_eur": result.get("amount_quote_eur", kwargs.get("amount_quote", 0.0)),
                    "fee_rate": result.get("fee_rate", DRY_RUN_FEE_RATE),
                    "status": result.get("status", "unknown"),
                }
            )
            return result

        return _wrapper

    return _decorator


def calculate_daily_fictive_pnl(date_utc: str) -> dict[str, Any]:
    """
    Berekent fictieve dag-PnL op basis van trades_log.csv.
    Verwacht pairings op basis van gemiddelde kostprijs en verwerkt 0.15% fee per trade.
    """

    path = _log_path()
    if not path.exists():
        return {
            "date_utc": date_utc,
            "trades": 0,
            "realized_pnl_eur": 0.0,
            "unrealized_pnl_eur": 0.0,
            "net_pnl_eur": 0.0,
            "fees_paid_eur": 0.0,
            "position_qty": 0.0,
        }

    target = datetime.fromisoformat(f"{date_utc}T00:00:00+00:00").date()
    rows: list[dict[str, Any]] = []
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            ts_raw = row.get("ts_utc")
            if not ts_raw:
                continue
            try:
                ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
            except ValueError:
                continue
            if ts.date() == target:
                rows.append(row)

    position_qty = 0.0
    avg_entry = 0.0
    fees_paid = 0.0
    realized_pnl = 0.0
    last_price = 0.0

    for row in rows:
        side = str(row.get("side", "")).upper()
        price = float(row.get("reference_price") or 0.0)
        amount_quote = float(row.get("amount_quote_eur") or 0.0)
        fee_rate = float(row.get("fee_rate") or DRY_RUN_FEE_RATE)
        if price <= 0 or amount_quote <= 0:
            continue
        qty = amount_quote / price
        fee = amount_quote * fee_rate
        fees_paid += fee
        last_price = price

        if side == "BUY":
            new_cost = (avg_entry * position_qty) + amount_quote + fee
            position_qty += qty
            avg_entry = new_cost / position_qty if position_qty > 0 else 0.0
        elif side == "SELL":
            sell_qty = min(position_qty, qty)
            if sell_qty > 0:
                gross = sell_qty * price
                cost = sell_qty * avg_entry
                realized_pnl += gross - cost - fee
                position_qty -= sell_qty
                if position_qty <= 1e-12:
                    position_qty = 0.0
                    avg_entry = 0.0

    unrealized_pnl = 0.0
    if position_qty > 0 and last_price > 0:
        unrealized_pnl = (position_qty * last_price) - (position_qty * avg_entry)

    net = realized_pnl + unrealized_pnl
    return {
        "date_utc": date_utc,
        "trades": len(rows),
        "realized_pnl_eur": round(realized_pnl, 2),
        "unrealized_pnl_eur": round(unrealized_pnl, 2),
        "net_pnl_eur": round(net, 2),
        "fees_paid_eur": round(fees_paid, 2),
        "position_qty": round(position_qty, 8),
        "log_path": str(path),
    }
