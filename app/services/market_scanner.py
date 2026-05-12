"""
Bestand: app/services/market_scanner.py
Relatief pad: ./app/services/market_scanner.py
Functie: Scant Bitvavo markten en filtert actieve pairs op 24u volume.
"""

from __future__ import annotations

import os
import time
from typing import Any

import requests

from app.exchanges.bitvavo import BitvavoClient


_GLOBAL_TOP_BASE_CACHE: dict[str, Any] = {"t": 0.0, "n": 0, "bases": frozenset()}


def fetch_global_top_base_symbols(limit: int = 20) -> frozenset[str]:
    """
    Uppercase base-symbolen (BTC, ETH, …) in wereldwijde top ``limit`` op marktkapitalisatie.
    Volgorde: CoinMarketCap listings (als key gezet), anders CoinGecko ``/coins/markets`` (zelfde bron als scanner).
    Cache ~30 min om refresh-loops niet te spammen.
    """
    n = max(5, min(100, int(limit)))
    now = time.monotonic()
    c = _GLOBAL_TOP_BASE_CACHE
    if int(c.get("n") or 0) == n and now - float(c.get("t") or 0.0) < 1800.0 and c.get("bases"):
        return c["bases"]

    bases: set[str] = set()
    key = str(os.getenv("COINMARKETCAP_KEY") or os.getenv("CMC_API_KEY") or "").strip()
    if key:
        try:
            resp = requests.get(
                "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest",
                params={"start": 1, "limit": n, "convert": "EUR"},
                headers={"X-CMC_PRO_API_KEY": key, "Accept": "application/json"},
                timeout=18,
            )
            if resp.status_code == 200:
                payload = resp.json() if resp.content else {}
                data = payload.get("data") if isinstance(payload, dict) else None
                if isinstance(data, list):
                    for row in data:
                        if not isinstance(row, dict):
                            continue
                        sym = str(row.get("symbol") or "").strip().upper()
                        if sym:
                            bases.add(sym)
        except Exception:
            bases = set()

    if not bases:
        try:
            resp = requests.get(
                "https://api.coingecko.com/api/v3/coins/markets",
                params={
                    "vs_currency": "eur",
                    "order": "market_cap_desc",
                    "per_page": n,
                    "page": 1,
                    "sparkline": "false",
                },
                timeout=22,
            )
            if resp.status_code == 200 and isinstance(resp.json(), list):
                for row in resp.json():
                    if not isinstance(row, dict):
                        continue
                    sym = str(row.get("symbol") or "").strip().upper()
                    if sym:
                        bases.add(sym)
        except Exception:
            bases = set()

    out = frozenset(bases)
    _GLOBAL_TOP_BASE_CACHE["t"] = now
    _GLOBAL_TOP_BASE_CACHE["n"] = n
    _GLOBAL_TOP_BASE_CACHE["bases"] = out
    return out


def bitvavo_exclude_bases() -> set[str]:
    """Comma-gescheiden ``BITVAVO_EXCLUDE_BASES`` (bv. ``JUP,ICP,HBAR``) — hoofdletters genegeerd."""
    raw = str(os.getenv("BITVAVO_EXCLUDE_BASES", "") or "").strip()
    if not raw:
        return set()
    return {x.strip().upper() for x in raw.split(",") if x.strip()}


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
            base_u = str(market.get("base", "") or "").strip().upper()
            if base_u and base_u in bitvavo_exclude_bases():
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

    def fetch_top_eur_by_volume(self, limit: int) -> list[dict[str, Any]]:
        """
        Top ``limit`` EUR-markten op Bitvavo naar 24h ``volumeQuote`` (EUR-notatie),
        status ``trading``, met ``min_volume_eur`` filter. Zelfde bron als ``fetch_active_pairs`` maar alleen *-EUR.
        """
        n = max(1, min(200, int(limit)))
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
            quote = str(market.get("quote", "")).upper()
            if not symbol or status != "trading" or quote != "EUR":
                continue
            base_u = str(market.get("base", "") or "").strip().upper()
            if base_u and base_u in bitvavo_exclude_bases():
                continue

            ticker_row = ticker_map.get(symbol, {})
            volume_quote = float(ticker_row.get("volumeQuote", 0.0) or 0.0)
            if volume_quote < self.min_volume_eur:
                continue

            rows.append(
                {
                    "market": symbol,
                    "base": str(market.get("base", "")),
                    "quote": quote,
                    "status": status,
                    "volume_quote_24h": round(volume_quote, 2),
                    "last_price": float(ticker_row.get("last", 0.0) or 0.0),
                    "price_change_pct_24h": float(ticker_row.get("priceChangePercentage", 0.0) or 0.0),
                }
            )

        rows.sort(key=lambda x: x["volume_quote_24h"], reverse=True)
        out = rows[:n]
        for i, r in enumerate(out, start=1):
            r["selection_reason"] = f"Bitvavo EUR #{i} op 24h volume (volumeQuote)"
            r["list_profile"] = "bitvavo_eur_volume_top"
        return out


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
