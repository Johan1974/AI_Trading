"""
Bestand: core/engine.py
Relatief pad: ./core/engine.py
Functie: Centrale trading loop met ingest, decision, paper exec en periodieke RL update.
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timedelta

from app.datetime_util import UTC
from typing import Any, Callable

from app.services.coinmarketcap import CoinMarketCapService
from app.services.news_service import CryptoCompareNewsService
from app.services.rss_engine import RssEngineService


def _torch_device_tag() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return f"cuda:{int(torch.cuda.current_device())}"
        return "cpu"
    except Exception:
        return "unknown"


class TradingEngine:
    def __init__(
        self,
        run_cycle: Callable[[str, int], dict[str, Any]],
        online_update: Callable[[str, int, int, dict[str, Any] | None], None] | None,
        state: dict[str, Any],
        cmc_service: CoinMarketCapService,
        news_service: CryptoCompareNewsService,
        rss_engine: RssEngineService,
        interval_minutes: int = 1,
        episode_hours: int = 24,
    ) -> None:
        self._run_cycle = run_cycle
        self._online_update = online_update
        self._state = state
        self._cmc_service = cmc_service
        self._news_service = news_service
        self._rss_engine = rss_engine
        self._interval_minutes = 5 if int(interval_minutes) >= 5 else 1
        self._episode_hours = max(1, int(episode_hours))
        self._task: asyncio.Task[None] | None = None
        self._last_online_update_at: datetime | None = None

    async def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._loop())
            print(
                f"[ENGINE] Started trading loop (interval={self._interval_minutes}m, episode={self._episode_hours}h)"
            )

    async def _loop(self) -> None:
        while True:
            try:
                print(f"[ENGINE] Tick | torch_device={_torch_device_tag()}")
                pair = str(self._state.get("selected_market") or "BTC-EUR").upper()
                lookback = int(self._state.get("lookback_days", 400) or 400)
                crypto_key = str(self._state.get("cryptocompare_key") or os.getenv("CRYPTOCOMPARE_KEY") or "")
                cmc_key = str(
                    self._state.get("cmc_api_key")
                    or os.getenv("COINMARKETCAP_KEY")
                    or os.getenv("CMC_API_KEY")
                    or ""
                )

                # 1) Data ingest
                api_news = self._news_service.fetch_latest_news(api_key=crypto_key, limit=30)
                rss_news = self._rss_engine.fetch_unprocessed_articles(limit=40)
                # CMC guard: service TTL is set to 20m in main bootstrap.
                cmc = self._cmc_service.fetch_global_metrics(api_key=cmc_key, force=False)
                self._state["cmc_metrics"] = cmc

                # 2/3/4) Sentiment + state + action + paper execution
                outcome = await asyncio.to_thread(self._run_cycle, pair, lookback)
                self._state["last_engine_cycle"] = {
                    "ok": True,
                    "pair": pair,
                    "ts": datetime.now(UTC).isoformat(),
                    "news_api_count": len(api_news),
                    "news_rss_count": len(rss_news),
                    "result": outcome,
                }

                st = self._state.get("rl_last_state", {}) if isinstance(self._state.get("rl_last_state"), dict) else {}
                dec = self._state.get("rl_last_decision", {}) if isinstance(self._state.get("rl_last_decision"), dict) else {}
                snapshot = outcome.get("portfolio_snapshot", {}) if isinstance(outcome, dict) else {}
                order = outcome.get("order", {}) if isinstance(outcome, dict) else {}
                signal = str((order.get("signal") if isinstance(order, dict) else None) or dec.get("action") or "HOLD")
                conf = float(dec.get("confidence") or 0.0) * 100.0
                qty = float(order.get("amount_base", 0.0) or 0.0) if isinstance(order, dict) else 0.0
                px = float(order.get("price", 0.0) or 0.0) if isinstance(order, dict) else 0.0
                pp = self._state.get("paper_portfolio") if isinstance(self._state.get("paper_portfolio"), dict) else {}
                snap_wallet = snapshot.get("wallet") if isinstance(snapshot, dict) else None
                if isinstance(snap_wallet, dict):
                    eq = float(snap_wallet.get("equity", pp.get("equity", 0.0)) or 0.0)
                    cash = float(snap_wallet.get("cash", pp.get("cash", 0.0)) or 0.0)
                elif isinstance(snapshot, dict) and "equity" in snapshot:
                    eq = float(snapshot.get("equity", 0.0) or 0.0)
                    cash = float(snapshot.get("cash", 0.0) or 0.0)
                else:
                    eq = float(pp.get("equity", 0.0) or 0.0)
                    cash = float(pp.get("cash", 0.0) or 0.0)
                print(
                    f"[INFO] BTC Dom: {float(st.get('btc_dominance_pct', 0.0)):.2f}% | "
                    f"Sentiment: {float(st.get('sentiment_score', 0.0)):.2f} | "
                    f"RSI: {float(st.get('rsi_14', 0.0)):.2f}"
                )
                print(f"[DECISION] RL Agent kiest {signal} (Confidence: {conf:.0f}%)")
                if qty > 0 and px > 0:
                    print(f"[PAPER] {signal}: {qty:.6f} {pair.split('-')[0]} @ EUR {px:,.2f}")
                else:
                    print(f"[PAPER] Actie: {signal} | Equity: EUR {eq:.2f} | Cash: EUR {cash:.2f}")

                # 5) Online learning per episode (bv. 24h)
                now = datetime.now(UTC)
                if self._online_update is not None:
                    due = (
                        self._last_online_update_at is None
                        or (now - self._last_online_update_at) >= timedelta(hours=self._episode_hours)
                    )
                    if due:
                        timesteps = int(self._state.get("episode_train_steps", 3000) or 3000)
                        await asyncio.to_thread(self._online_update, pair, lookback, timesteps, cmc)
                        self._last_online_update_at = now
                        print(f"[RL-BRAIN] Online PPO update voltooid voor {pair} ({timesteps} timesteps)")
            except Exception as exc:
                self._state["last_engine_cycle"] = {
                    "ok": False,
                    "ts": datetime.now(UTC).isoformat(),
                    "error": str(exc),
                }
                print(f"[ENGINE] Cycle failed: {exc}")
            await asyncio.sleep(self._interval_minutes * 60)

