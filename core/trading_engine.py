"""
Multi-asset elite execution loop: één sweep over alle actieve Elite-markten per interval,
ongeacht de in de UI geselecteerde ticker (die blijft voor charts/UI).

Strikt één actieve paper-positie per ticker: BUY-blokkade en logging gebeuren in de paper-cycle
(`_run_paper_cycle_sync`) via `has_active_paper_position_for_ticker` op de wallet (SQLite wallet_state).
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timedelta
from typing import Any, Callable

from app.datetime_util import UTC
from app.services.coinmarketcap import CoinMarketCapService
from app.services.news_service import CryptoCompareNewsService
from app.services.rss_engine import RssEngineService

from core.risk_management import allocation_snapshot, elite_equal_weight_enabled

SKIP_BUY_ACTIVE_POSITION_LOG = (
    "SKIP BUY: {ticker} already has an active position. No overlapping trades allowed."
)


def has_active_paper_position_for_ticker(wallet: dict[str, Any] | None, ticker: str) -> bool:
    """
    True als er voor deze markt (bv. BTC-EUR) al open qty of open lots bestaan.
    Bron: paper `wallet` snapshot (persisteert naar SQLite `wallet_state` / trade DB), niet aparte trading_data.db.
    """
    if not isinstance(wallet, dict):
        return False
    mku = str(ticker or "").strip().upper()
    if not mku:
        return False
    pbm = wallet.get("position_by_market") if isinstance(wallet.get("position_by_market"), dict) else {}
    if float(pbm.get(mku, 0.0) or 0.0) > 1e-12:
        return True
    obm = wallet.get("open_lots_by_market") if isinstance(wallet.get("open_lots_by_market"), dict) else {}
    lots = obm.get(mku)
    if isinstance(lots, list):
        for lot in lots:
            if isinstance(lot, dict) and float(lot.get("qty", 0.0) or 0.0) > 1e-12:
                return True
    if str(wallet.get("position_symbol") or "").strip().upper() == mku and float(wallet.get("position_qty", 0.0) or 0.0) > 1e-12:
        return True
    return False


DEFAULT_ELITE_PAIRS = [
    "BTC-EUR",
    "ETH-EUR",
    "SOL-EUR",
    "XRP-EUR",
    "ADA-EUR",
    "DOT-EUR",
    "AVAX-EUR",
    "LINK-EUR",
]


def elite_execution_pairs(state: dict[str, Any]) -> list[str]:
    """Actieve Elite-lijst uit state, TICKERS env, of vaste default (max 8)."""
    rows = state.get("active_markets") or []
    out: list[str] = []
    if isinstance(rows, list):
        for m in rows:
            if isinstance(m, dict) and m.get("market"):
                mk = str(m["market"]).strip().upper()
                if mk and mk not in out:
                    out.append(mk)
        if out:
            return out[:8]
    env = str(os.getenv("TICKERS", "")).strip()
    if env:
        tickers = [t.strip().upper() for t in env.split(",") if t.strip()]
        if tickers:
            return tickers[:8]
    return list(DEFAULT_ELITE_PAIRS)


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
            sweep = int(os.getenv("ENGINE_ELITE_SWEEP_SEC", "60") or 60)
            print(
                f"[ENGINE] Started elite parallel sweep (sweep_interval={sweep}s, "
                f"rl_episode={self._episode_hours}h, ui_interval_hint={self._interval_minutes}m)"
            )

    async def _loop(self) -> None:
        sweep_sec = max(15, int(os.getenv("ENGINE_ELITE_SWEEP_SEC", "60") or 60))
        while True:
            try:
                print(f"[ENGINE] Elite sweep | torch_device={_torch_device_tag()}")
                pairs = elite_execution_pairs(self._state)
                lookback = int(self._state.get("lookback_days", 400) or 400)
                crypto_key = str(self._state.get("cryptocompare_key") or os.getenv("CRYPTOCOMPARE_KEY") or "")
                cmc_key = str(
                    self._state.get("cmc_api_key")
                    or os.getenv("COINMARKETCAP_KEY")
                    or os.getenv("CMC_API_KEY")
                    or ""
                )

                api_news = self._news_service.fetch_latest_news(api_key=crypto_key, limit=30)
                rss_news = self._rss_engine.fetch_unprocessed_articles(limit=40)
                cmc = self._cmc_service.fetch_global_metrics(api_key=cmc_key, force=False)
                self._state["cmc_metrics"] = cmc

                if elite_equal_weight_enabled():
                    w_pf = self._state.get("paper_portfolio") if isinstance(self._state.get("paper_portfolio"), dict) else {}
                    snap_a = allocation_snapshot(w_pf, float(w_pf.get("equity") or 0.0))
                    print(f"[ALLOC] {snap_a.get('summary', '')} (equal-weight cap {snap_a.get('slot_pct', 12.5)}% per coin)")

                if not pairs:
                    pairs = ["BTC-EUR"]
                last_pair = str(self._state.get("selected_market") or pairs[0]).upper()
                for pair in pairs:
                    last_pair = pair
                    outcome = await asyncio.to_thread(self._run_cycle, pair, lookback)
                    if not isinstance(outcome, dict):
                        outcome = {}
                    self._state["last_engine_cycle"] = {
                        "ok": True,
                        "pair": pair,
                        "elite_sweep_pairs": pairs,
                        "ts": datetime.now(UTC).isoformat(),
                        "news_api_count": len(api_news),
                        "news_rss_count": len(rss_news),
                        "result": outcome,
                    }

                    st = (
                        self._state.get("rl_last_state", {})
                        if isinstance(self._state.get("rl_last_state"), dict)
                        else {}
                    )
                    dec = (
                        self._state.get("rl_last_decision", {})
                        if isinstance(self._state.get("rl_last_decision"), dict)
                        else {}
                    )
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
                        f"[INFO] {pair} | BTC Dom: {float(st.get('btc_dominance_pct', 0.0)):.2f}% | "
                        f"Sentiment: {float(st.get('sentiment_score', 0.0)):.2f} | "
                        f"RSI: {float(st.get('rsi_14', 0.0)):.2f}"
                    )
                    print(f"[DECISION] {pair} RL → {signal} (Confidence: {conf:.0f}%)")
                    if qty > 0 and px > 0:
                        print(f"[PAPER] {pair} {signal}: {qty:.6f} {pair.split('-')[0]} @ EUR {px:,.2f}")
                    else:
                        print(f"[PAPER] {pair} {signal} | Equity: EUR {eq:.2f} | Cash: EUR {cash:.2f}")

                now = datetime.now(UTC)
                if self._online_update is not None:
                    due = (
                        self._last_online_update_at is None
                        or (now - self._last_online_update_at) >= timedelta(hours=self._episode_hours)
                    )
                    if due:
                        timesteps = int(self._state.get("episode_train_steps", 3000) or 3000)
                        await asyncio.to_thread(self._online_update, last_pair, lookback, timesteps, cmc)
                        self._last_online_update_at = now
                        print(f"[RL-BRAIN] Online PPO update voltooid voor {last_pair} ({timesteps} timesteps)")
            except Exception as exc:
                self._state["last_engine_cycle"] = {
                    "ok": False,
                    "ts": datetime.now(UTC).isoformat(),
                    "error": str(exc),
                }
                print(f"[ENGINE] Cycle failed: {exc}")
            await asyncio.sleep(sweep_sec)
