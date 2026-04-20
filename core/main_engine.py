"""
Bestand: core/main_engine.py
Relatief pad: ./core/main_engine.py
Functie: Centrale integratie-loop die elke minuut paper trading cycli triggert.
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime

from app.datetime_util import UTC
from typing import Any, Awaitable, Callable

from app.services.coinmarketcap import CoinMarketCapService
from app.services.news_service import CryptoCompareNewsService
from app.services.rss_engine import RssEngineService


class MainIntegrationEngine:
    def __init__(
        self,
        run_cycle: Callable[[str, int], dict[str, Any]],
        state: dict[str, Any],
        cmc_service: CoinMarketCapService,
        news_service: CryptoCompareNewsService,
        rss_engine: RssEngineService,
    ) -> None:
        self._run_cycle = run_cycle
        self._state = state
        self._cmc_service = cmc_service
        self._news_service = news_service
        self._rss_engine = rss_engine
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._loop())

    async def _loop(self) -> None:
        while True:
            try:
                pair = str(self._state.get("selected_market") or "BTC-EUR").upper()
                lookback = int(self._state.get("lookback_days", 400) or 400)
                # Trigger API+RSS fetch paths and keep feed hot.
                crypto_key = str(self._state.get("cryptocompare_key") or os.getenv("CRYPTOCOMPARE_KEY") or "")
                self._news_service.fetch_latest_news(api_key=crypto_key, limit=30)
                self._rss_engine.fetch_unprocessed_articles(limit=40)

                # Poll CMC every 30 minutes via service TTL.
                cmc_key = self._state.get("cmc_api_key")
                cmc = self._cmc_service.fetch_global_metrics(api_key=str(cmc_key or ""))
                self._state["cmc_metrics"] = cmc

                outcome = await asyncio.to_thread(self._run_cycle, pair, lookback)
                self._state["last_engine_cycle"] = {
                    "ok": True,
                    "pair": pair,
                    "ts": datetime.now(UTC).isoformat(),
                    "result": outcome,
                }
                print(f"[ENGINE] Paper cycle completed for {pair}")
            except Exception as exc:
                self._state["last_engine_cycle"] = {
                    "ok": False,
                    "ts": datetime.now(UTC).isoformat(),
                    "error": str(exc),
                }
                print(f"[ENGINE] Cycle failed: {exc}")
            await asyncio.sleep(60)

