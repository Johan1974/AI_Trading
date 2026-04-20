"""
Bestand: app/services/market_scanner.py
Relatief pad: ./app/services/market_scanner.py
Functie: Scant Bitvavo markten en filtert actieve pairs op 24u volume.
"""

from __future__ import annotations

import os
from typing import Any

import requests

from app.exchanges.bitvavo import BitvavoClient


class MarketScanner:
    def __init__(
        self,
        min_volume_eur: float = 500000.0,
        base_url: str = "https://api.bitvavo.com",
    ) -> None:
        self.min_volume_eur = min_volume_eur
        self.base_url = base_url.rstrip("/")

    def fetch_active_pairs(self) -> list[dict[str, Any]]:
        markets_resp = requests.get(f"{self.base_url}/v2/markets", timeout=20)
        markets_resp.raise_for_status()
        markets_data = markets_resp.json()

        ticker_resp = requests.get(f"{self.base_url}/v2/ticker/24h", timeout=20)
        ticker_resp.raise_for_status()
        ticker_data = ticker_resp.json()

        ticker_map: dict[str, dict[str, Any]] = {
            str(row.get("market")): row for row in ticker_data if isinstance(row, dict) and row.get("market")
        }

        rows: list[dict[str, Any]] = []
        for market in markets_data:
            if not isinstance(market, dict):
                continue
            symbol = str(market.get("market", ""))
            status = str(market.get("status", ""))
            if not symbol or status != "trading":
                continue

            ticker_row = ticker_map.get(symbol, {})
            volume_quote = float(ticker_row.get("volumeQuote", 0.0) or 0.0)
            if volume_quote < self.min_volume_eur:
                continue

            rows.append(
                {
                    "market": symbol,
                    "base": str(market.get("base", "")),
                    "quote": str(market.get("quote", "")),
                    "status": status,
                    "volume_quote_24h": round(volume_quote, 2),
                    "last_price": float(ticker_row.get("last", 0.0) or 0.0),
                    "price_change_pct_24h": float(ticker_row.get("priceChangePercentage", 0.0) or 0.0),
                }
            )

        rows.sort(key=lambda x: x["volume_quote_24h"], reverse=True)
        return rows


def check_pair_balance_from_vault(
    market: str,
    min_quote_balance: float,
    min_base_balance: float,
) -> dict[str, Any]:
    read_key = (os.environ.get("BITVAVO_KEY_READ") or "").strip()
    read_secret = (os.environ.get("BITVAVO_SECRET_READ") or "").strip()
    if not read_key or not read_secret:
        return {
            "market": market,
            "available": False,
            "reason": "read_credentials_missing",
        }

    base_asset, quote_asset = market.split("-", 1)
    client = BitvavoClient(api_key=read_key, api_secret=read_secret)
    balances = client.get_balance()

    def _available(asset: str) -> float:
        for row in balances:
            if str(row.get("symbol", "")).upper() == asset.upper():
                return float(row.get("available", 0.0) or 0.0)
        return 0.0

    quote_available = _available(quote_asset)
    base_available = _available(base_asset)
    enough_quote = quote_available >= min_quote_balance
    enough_base = base_available >= min_base_balance
    return {
        "market": market,
        "available": True,
        "quote_asset": quote_asset,
        "base_asset": base_asset,
        "quote_available": round(quote_available, 8),
        "base_available": round(base_available, 8),
        "min_quote_required": min_quote_balance,
        "min_base_required": min_base_balance,
        "sufficient_for_buy": enough_quote,
        "sufficient_for_sell": enough_base,
        "can_trade": bool(enough_quote or enough_base),
    }
