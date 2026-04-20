"""
Bestand: app/exchanges/bitvavo.py
Relatief pad: ./app/exchanges/bitvavo.py
Functie: Bitvavo client met signed API-calls voor balance en market orders.
"""

import hashlib
import hmac
import json
import os
import time
from typing import Any
from urllib import error, request

from app.exchanges.base import ExchangeClient
from app.exchanges.bitvavo_manager import BitvavoPaperManager, BitvavoRateLimitManager
from app.services.dry_run import DRY_RUN_FEE_RATE

DEFAULT_RATE_LIMIT_MANAGER = BitvavoRateLimitManager()
DEFAULT_PAPER_MANAGER = BitvavoPaperManager(
    db_path=os.getenv("PAPER_BITVAVO_DB_PATH", "data/paper_bitvavo.db"),
    start_balance_eur=float(os.getenv("PAPER_START_BALANCE_EUR", "10000")),
)


class BitvavoClient(ExchangeClient):
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        base_url: str = "https://api.bitvavo.com",
        throttle_threshold: float = 0.80,
        rate_limit_manager: BitvavoRateLimitManager | None = None,
    ) -> None:
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url.rstrip("/")
        self.rate_limit_manager = rate_limit_manager or DEFAULT_RATE_LIMIT_MANAGER
        self.rate_limit_manager.throttle_threshold = throttle_threshold
        self.paper_manager = DEFAULT_PAPER_MANAGER

    def _live_mode_enabled(self) -> bool:
        return str(os.getenv("LIVE_MODE", "false")).strip().lower() in {"1", "true", "yes", "on"}

    def _signed_request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        max_retries: int = 5,
    ) -> Any:
        method = method.upper()
        weight = self.rate_limit_manager.before_request(method=method, path=path)

        body = json.dumps(payload or {}, separators=(",", ":")) if method != "GET" else ""

        attempt = 0
        while True:
            ts = str(int(time.time() * 1000))
            prehash = f"{ts}{method}{path}{body}"
            signature = hmac.new(
                self.api_secret.encode("utf-8"),
                prehash.encode("utf-8"),
                hashlib.sha256,
            ).hexdigest()

            req = request.Request(
                f"{self.base_url}{path}",
                method=method,
                data=body.encode("utf-8") if body else None,
            )
            req.add_header("Bitvavo-Access-Key", self.api_key)
            req.add_header("Bitvavo-Access-Signature", signature)
            req.add_header("Bitvavo-Access-Timestamp", ts)
            req.add_header("Bitvavo-Access-Window", "10000")
            req.add_header("Content-Type", "application/json")
            try:
                with request.urlopen(req, timeout=20) as resp:
                    self.rate_limit_manager.update_from_headers(resp.headers, assumed_weight=weight)
                    text = resp.read().decode("utf-8", errors="replace")
                    self.rate_limit_manager.record_success(assumed_weight=weight)
                    return json.loads(text) if text else {}
            except error.HTTPError as exc:
                body_text = exc.read().decode("utf-8", errors="replace")
                self.rate_limit_manager.update_from_headers(exc.headers or {}, assumed_weight=weight)
                self.rate_limit_manager.record_failure(assumed_weight=weight, status_code=exc.code)
                if exc.code == 429 and attempt < max_retries:
                    self.rate_limit_manager.handle_429(headers=exc.headers or {}, attempt=attempt)
                    attempt += 1
                    continue
                raise RuntimeError(f"Bitvavo HTTP {exc.code}: {body_text}") from exc

    def get_balance(self) -> list[dict[str, Any]]:
        if not self._live_mode_enabled():
            return self.paper_manager.get_balance()
        data = self._signed_request("GET", "/v2/balance")
        return data if isinstance(data, list) else []

    def place_market_order(
        self,
        market: str,
        side: str,
        amount_quote: float,
        client_order_id: str | None = None,
    ) -> dict[str, Any]:
        if not self._live_mode_enabled():
            result = self.paper_manager.place_market_order(
                market=market,
                side=side,
                amount_quote=amount_quote,
            )
            result.setdefault("fee_rate", DRY_RUN_FEE_RATE)
            return result
        payload = {
            "market": market,
            "side": side.lower(),
            "orderType": "market",
            "amountQuote": f"{amount_quote:.8f}",
        }
        if client_order_id:
            payload["clientOrderId"] = client_order_id
        data = self._signed_request("POST", "/v2/order", payload=payload)
        result = data if isinstance(data, dict) else {"raw": data}
        result.setdefault("market", market)
        result.setdefault("side", side.upper())
        result.setdefault("amount_quote_eur", float(amount_quote))
        result.setdefault("fee_rate", DRY_RUN_FEE_RATE)
        result.setdefault("status", "executed")
        return result


def global_rate_limit_status() -> dict[str, Any]:
    return DEFAULT_RATE_LIMIT_MANAGER.snapshot()
