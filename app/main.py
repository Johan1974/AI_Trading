"""
Bestand: app/main.py
Relatief pad: ./app/main.py
Functie: FastAPI-app voor marktdata- en nieuwsgebaseerde voorspellingen plus activiteitstracking voor de portal.
"""

import asyncio
import io
import os
import json
import shutil
from datetime import datetime, timedelta

from app.datetime_util import UTC
from pathlib import Path
from typing import Any

import numpy as np
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from agent.trader import Trader, TraderConfig
from app.ai.sentiment.finbert_sentiment import FinBertSentimentAnalyzer
from app.schemas.prediction import PredictionResponse
from app.ai.judge.weighted_judge import adjust_weights
from app.exchanges.bitvavo import global_rate_limit_status
from app.services.coinmarketcap import CoinMarketCapService
from app.services.dry_run import calculate_daily_fictive_pnl
from app.services.execution import build_paper_order
from app.services.fear_greed import FearGreedService
from app.services.ingestion import fetch_market_data, fetch_news_articles
from app.services.macro_calendar import MacroCalendarService
from app.services.market_scanner import MarketScanner, check_pair_balance_from_vault
from app.services.news_mapping import NewsMappingService
from app.services.news_engine import NewsEngineService
from app.services.news_service import CryptoCompareNewsService
from app.services.rss_engine import RssEngineService
from app.services.paper_engine import PaperConfig, PaperTradeManager
from app.services.risk import RiskManager, compute_risk_controls, signal_from_expected_return
from app.rl.agent_rl import RLAgentService
from app.rl.data import build_rl_training_frame, fetch_bitvavo_historical_candles
from app.services.signal_engine import SignalEngine
from app.services.state import STATE, append_event
from app.services.system_stats import get_system_stats
from app.services.telegram_notifier import TelegramNotifier
from app.services.whale_watcher import WhaleWatcherService
from core.engine import TradingEngine
from core.risk_manager import RiskManager as CoreRiskManager, risk_profile_dict

load_dotenv()


def _genesis_log_torch_device() -> None:
    """Logt het actieve compute-device; run_bot.sh wacht op `[DEVICE] Using device: cuda:0`."""
    try:
        import torch

        if torch.cuda.is_available():
            name = str(torch.cuda.get_device_name(0) or "CUDA GPU")
            print(f"[DEVICE] Using device: cuda:0 ({name})")
        else:
            print("[DEVICE] Using device: cpu (CUDA niet beschikbaar)")
    except Exception as exc:
        print(f"[DEVICE] Torch device-check mislukt: {exc}")


def _genesis_require_gpu_or_raise() -> None:
    """In Docker (/.dockerenv) standaard GPU verplicht; lokaal alleen met GENESIS_REQUIRE_GPU=1."""
    in_container = Path("/.dockerenv").exists()
    default_flag = "1" if in_container else "0"
    flag = str(os.getenv("GENESIS_REQUIRE_GPU", default_flag)).strip().lower()
    if flag not in ("1", "true", "yes", "on"):
        _genesis_log_torch_device()
        return
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CRITICAL: GPU NIET GEVONDEN!")
    _genesis_log_torch_device()


app = FastAPI(title="AI Trading Bot", version="1.0.0")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")
_FINBERT = FinBertSentimentAnalyzer()
SIGNAL_ENGINE = SignalEngine(sentiment=_FINBERT)
_max_trade_frac = float(os.getenv("RISK_MAX_TRADE_EQUITY_PCT", "10")) / 100.0
RISK_MANAGER = RiskManager(max_budget_fraction_per_trade=max(0.001, min(1.0, _max_trade_frac)))
CORE_RISK = CoreRiskManager()
MARKET_SCANNER = MarketScanner(min_volume_eur=float(os.getenv("MIN_24H_VOLUME_EUR", "500000")))
NEWS_MAPPING = NewsMappingService(sentiment=_FINBERT)
FEAR_GREED = FearGreedService()
WHALE_WATCHER = WhaleWatcherService()
CMC_SERVICE = CoinMarketCapService(ttl_seconds=1200)
MACRO_CALENDAR = MacroCalendarService()
NEWS_SERVICE = CryptoCompareNewsService(ttl_seconds=60)
RSS_ENGINE = RssEngineService()
NEWS_ENGINE = NewsEngineService(
    api_service=NEWS_SERVICE,
    rss_service=RSS_ENGINE,
    freshness_minutes=15,
)
PAPER_MANAGER = PaperTradeManager(
    PaperConfig(
        starting_balance_eur=float(os.getenv("PAPER_START_BALANCE_EUR", "10000")),
        fee_rate=0.0015,
        db_path=os.getenv("TRADE_HISTORY_DB_PATH", "data/trade_history.db"),
    )
)
RL_AGENT = RLAgentService(model_dir=os.getenv("RL_MODEL_DIR", "artifacts/rl"))
TRADER = Trader(
    TraderConfig(
        initial_capital_eur=float(os.getenv("PAPER_START_BALANCE_EUR", "10000")),
        lookback_days=int(os.getenv("LOOKBACK_DAYS", "400")),
        model_dir=os.getenv("RL_MODEL_DIR", "artifacts/rl"),
    ),
    agent_service=RL_AGENT,
)
MAIN_ENGINE = TradingEngine(
    run_cycle=lambda ticker, lookback_days: run_paper_cycle(ticker=ticker, lookback_days=lookback_days),
    online_update=lambda pair, lookback_days, timesteps, cmc: RL_AGENT.online_update(
        pair=pair,
        lookback_days=max(1, int(lookback_days // 16) or 1),
        total_timesteps=timesteps,
        cmc_metrics=cmc,
    ),
    state=STATE,
    cmc_service=CMC_SERVICE,
    news_service=NEWS_SERVICE,
    rss_engine=RSS_ENGINE,
    interval_minutes=int(os.getenv("ENGINE_LOOP_MINUTES", "1")),
    episode_hours=int(os.getenv("RL_EPISODE_HOURS", "24")),
)
TELEGRAM = TelegramNotifier(
    token=os.getenv("TELEGRAM_TOKEN"),
    chat_id=os.getenv("TELEGRAM_CHAT_ID"),
    enabled=True,
)
STORAGE_ROOT = Path.home() / "AI_Trading" / "storage"
STORAGE_STATS_PATH = STORAGE_ROOT / "stats.json"
LOGS_DIR = STORAGE_ROOT / "logs"
BOT_LOG_PATH = LOGS_DIR / "bot_execution.log"
TEE_INSTALLED = False


def _append_bot_log_line(text: str) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).isoformat()
    with BOT_LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(f"[{ts}] {text.rstrip()}\n")


class _TeeTextStream(io.TextIOBase):
    def __init__(self, base: Any) -> None:
        self._base = base
        self._buffer = ""

    def write(self, s: str) -> int:
        text = str(s)
        self._base.write(text)
        self._base.flush()
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line.strip():
                _append_bot_log_line(line)
        return len(text)

    def flush(self) -> None:
        self._base.flush()
        if self._buffer.strip():
            _append_bot_log_line(self._buffer)
            self._buffer = ""

    def isatty(self) -> bool:
        return bool(getattr(self._base, "isatty", lambda: False)())


def _install_stdout_stderr_tee() -> None:
    global TEE_INSTALLED
    if TEE_INSTALLED:
        return
    import sys

    sys.stdout = _TeeTextStream(sys.stdout)  # type: ignore[assignment]
    sys.stderr = _TeeTextStream(sys.stderr)  # type: ignore[assignment]
    TEE_INSTALLED = True


def _tail_lines(path: Path, limit: int = 200) -> list[str]:
    if not path.exists():
        return []
    with path.open("rb") as fh:
        fh.seek(0, os.SEEK_END)
        file_size = fh.tell()
        block = 4096
        data = b""
        while file_size > 0 and data.count(b"\n") <= (limit + 2):
            read_size = block if file_size >= block else file_size
            file_size -= read_size
            fh.seek(file_size)
            data = fh.read(read_size) + data
        lines = data.decode("utf-8", errors="replace").splitlines()
        return lines[-limit:]


def _prune_runtime_logs(max_age_minutes: int = 60, max_bytes: int = 50 * 1024 * 1024) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    if not BOT_LOG_PATH.exists():
        BOT_LOG_PATH.touch(exist_ok=True)
        return
    cutoff = datetime.now(UTC) - timedelta(minutes=max_age_minutes)
    raw = BOT_LOG_PATH.read_text(encoding="utf-8", errors="replace")
    lines = raw.splitlines()
    kept: list[str] = []
    for line in lines:
        if line.startswith("[") and "]" in line:
            stamp = line[1 : line.index("]")]
            try:
                ts = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=UTC)
                if ts >= cutoff:
                    kept.append(line)
                continue
            except Exception:
                pass
        kept.append(line)
    text = "\n".join(kept)
    encoded = text.encode("utf-8")
    if len(encoded) > max_bytes:
        encoded = encoded[-max_bytes:]
        text = encoded.decode("utf-8", errors="ignore")
        if "\n" in text:
            text = text.split("\n", 1)[1]
    BOT_LOG_PATH.write_text(text + ("\n" if text else ""), encoding="utf-8")


async def _background_log_pruner() -> None:
    while True:
        try:
            _prune_runtime_logs(max_age_minutes=60)
        except Exception as exc:
            print(f"[SYSTEM] Log prune task warning: {exc}")
        await asyncio.sleep(600)


async def _background_rss_poller() -> None:
    while True:
        try:
            RSS_ENGINE.fetch_unprocessed_articles(limit=100)
        except Exception as exc:
            print(f"[SYSTEM] RSS poll warning: {exc}")
        await asyncio.sleep(60 if not NEWS_SERVICE.last_fetch_ok else 300)


_install_stdout_stderr_tee()


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


LIVE_MODE = _env_flag("LIVE_MODE", default=False)


def _validate_mode_configuration() -> None:
    # Als LIVE_MODE niet is gezet, blijft default paper mode actief.
    if not LIVE_MODE:
        return

    missing = [
        name
        for name in ("BITVAVO_KEY_TRADE", "BITVAVO_SECRET_TRADE")
        if not os.getenv(name, "").strip()
    ]
    if missing:
        raise RuntimeError(
            f"LIVE_MODE=true maar ontbrekende variabelen in vault: {', '.join(missing)}"
        )


_validate_mode_configuration()


def _coin_from_ticker(ticker: str) -> str:
    upper = (ticker or "").upper()
    if "-" in upper:
        return upper.split("-", 1)[0]
    if upper.endswith("EUR"):
        return upper[:-3]
    if upper.endswith("USD"):
        return upper[:-3]
    return upper


def _build_news_insights(
    ticker: str,
    news_articles: list[dict[str, Any]],
    sentiment_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    target_coin = _coin_from_ticker(ticker)
    insights: list[dict[str, Any]] = []
    for idx, article in enumerate(news_articles[:5]):
        scored = sentiment_items[idx] if idx < len(sentiment_items) else {}
        title = str(article.get("title") or "Untitled")
        description = str(article.get("description") or "")
        text = f"{title} {description}".upper()
        impacts_target = target_coin in text
        insights.append(
            {
                "ts": article.get("publishedAt"),
                "headline": title,
                "source": (article.get("source") or {}).get("name"),
                "ticker_tag": target_coin,
                "impacts_ticker": impacts_target,
                "finbert_score": scored.get("signed_score", 0.0),
                "finbert_confidence": scored.get("confidence", 0.0),
                "finbert_label": scored.get("label", "neutral"),
            }
        )
    return insights


def _signal_news_articles(ticker: str, news_query: str, news_api_key: str | None) -> list[dict[str, Any]]:
    rows = NEWS_ENGINE.fetch_fresh_news(cryptocompare_key=os.getenv("CRYPTOCOMPARE_KEY"), limit=100)
    if rows:
        for row in rows[:40]:
            STATE["news_lag_history"].append(
                {
                    "ts": datetime.now(UTC).isoformat(),
                    "source": str((row.get("source") or {}).get("name") or "unknown"),
                    "news_lag_sec": int(row.get("news_lag_sec") or 0),
                }
            )
        STATE["news_lag_history"] = STATE["news_lag_history"][-300:]
        return rows
    return fetch_news_articles(news_query, news_api_key)


def _genesis_check_bitvavo() -> bool:
    try:
        resp = requests.get("https://api.bitvavo.com/v2/ticker/price", params={"market": "BTC-EUR"}, timeout=10)
        ok = resp.status_code == 200
        print(f"[GENESIS] Check 1 Bitvavo koersdata: {'OK' if ok else 'FAIL'}")
        return ok
    except Exception as exc:
        print(f"[GENESIS] Check 1 Bitvavo koersdata: FAIL ({exc})")
        return False


def _genesis_check_cryptocompare() -> bool:
    try:
        rows = NEWS_SERVICE.fetch_latest_news(api_key=os.getenv("CRYPTOCOMPARE_KEY"), limit=5, force=True)
        ok = bool(rows)
        print(f"[GENESIS] Check 2 CryptoCompare nieuws: {'OK' if ok else 'FAIL'}")
        return ok
    except Exception as exc:
        print(f"[GENESIS] Check 2 CryptoCompare nieuws: FAIL ({exc})")
        return False


def _genesis_check_rss() -> bool:
    try:
        ok = bool(RSS_ENGINE.feeds_healthy())
        print(f"[GENESIS] Check 4 RSS feeds: {'OK' if ok else 'FAIL'}")
        return ok
    except Exception as exc:
        print(f"[GENESIS] Check 4 RSS feeds: FAIL ({exc})")
        return False


def _genesis_check_coinmarketcap() -> bool:
    key = str(os.getenv("COINMARKETCAP_KEY") or os.getenv("CMC_API_KEY") or "").strip()
    if not key:
        print("[GENESIS] Check 3 CoinMarketCap context: FAIL (COINMARKETCAP_KEY missing)")
        return False
    try:
        resp = requests.get(
            "https://pro-api.coinmarketcap.com/v1/global-metrics/quotes/latest",
            headers={"X-CMC_PRO_API_KEY": key, "Accept": "application/json"},
            timeout=10,
        )
        ok = resp.status_code == 200
        print(f"[GENESIS] Check 3 CoinMarketCap context: {'OK' if ok else 'FAIL'}")
        return ok
    except Exception as exc:
        print(f"[GENESIS] Check 3 CoinMarketCap context: FAIL ({exc})")
        return False


def _refresh_cmc_metrics(force: bool = False) -> dict[str, Any]:
    payload = CMC_SERVICE.fetch_global_metrics(
        api_key=os.getenv("COINMARKETCAP_KEY") or os.getenv("CMC_API_KEY"),
        force=force,
    )
    STATE["cmc_metrics"] = payload
    return payload


async def _run_genesis_and_prepare_rl() -> None:
    check1 = _genesis_check_bitvavo()
    check2 = _genesis_check_cryptocompare()
    check3 = _genesis_check_coinmarketcap()
    check4 = _genesis_check_rss()
    if not (check1 and check2 and check3 and check4):
        print("[GENESIS] Not all checks passed; skip RL genesis warmup.")
        return
    print("Genesis Check: Alle 4 API's verbonden.")
    try:
        pair = str(STATE.get("selected_market") or os.getenv("DEFAULT_TICKER", "BTC-EUR")).upper()
        lookback = int(os.getenv("LOOKBACK_DAYS", "400"))
        _refresh_cmc_metrics(force=True)
        await asyncio.to_thread(TRADER.initialize, pair)
        print("[GENESIS] All checks OK. RL agent ready in paper trading mode.")
    except Exception as exc:
        print(f"[GENESIS] RL warmup failed: {exc}")


def _register_signal_marker(
    ticker: str,
    signal: str,
    price: float,
    expected_return_pct: float,
) -> None:
    marker = {
        "ts": datetime.utcnow().isoformat(),
        "ticker": ticker.upper(),
        "signal": signal.upper(),
        "price": float(price),
        "expected_return_pct": float(expected_return_pct),
    }
    STATE["signal_markers"].insert(0, marker)
    STATE["signal_markers"] = STATE["signal_markers"][:300]


def _resolve_execution_price(ticker: str, fallback_price: float) -> float:
    upper = ticker.upper()
    if "-" not in upper:
        return fallback_price
    try:
        resp = requests.get(
            "https://api.bitvavo.com/v2/ticker/price",
            params={"market": upper},
            timeout=10,
        )
        if resp.status_code != 200:
            return fallback_price
        payload = resp.json()
        if isinstance(payload, dict):
            return float(payload.get("price") or fallback_price)
        if isinstance(payload, list) and payload:
            return float(payload[0].get("price") or fallback_price)
    except Exception:
        return fallback_price
    return fallback_price


def _fetch_history_series(pair: str, lookback_days: int) -> tuple[list[str], list[float]]:
    target = pair.upper()
    if "-" in target:
        # Bitvavo pair history for crypto markets.
        interval = "1h" if lookback_days <= 30 else "1d"
        url = f"https://api.bitvavo.com/v2/{target}/candles"
        resp = requests.get(
            url,
            params={"interval": interval, "limit": min(1000, max(100, lookback_days * 3))},
            timeout=15,
        )
        if resp.status_code == 200:
            candles = resp.json()
            if isinstance(candles, list) and candles:
                labels = [datetime.utcfromtimestamp(int(row[0]) / 1000).isoformat() for row in candles]
                prices = [float(row[4]) for row in candles]
                return labels, prices

    # Fallback to yfinance-style history.
    df = fetch_market_data(ticker=target, lookback_days=lookback_days)
    labels = [str(v) for v in df["Date"].astype(str).tolist()]
    prices = [float(v) for v in np.asarray(df["Close"], dtype=float).reshape(-1).tolist()]
    return labels, prices


def _refresh_active_markets_cache() -> None:
    markets = MARKET_SCANNER.fetch_active_pairs()
    STATE["active_markets"] = markets
    if markets and not any(m["market"] == STATE.get("selected_market") for m in markets):
        STATE["selected_market"] = markets[0]["market"]


async def _gpu_cuda_heartbeat() -> None:
    """Elke 5s: console-bevestiging torch.cuda + nvidia-smi snapshot (operators / log-shipping)."""
    while True:
        try:
            import torch

            cuda_ok = bool(torch.cuda.is_available())
            extra = ""
            if cuda_ok:
                extra = f" device={torch.cuda.get_device_name(0)}"
            snap = get_system_stats()
            print(
                "[GPU-HEARTBEAT] "
                f"torch.cuda.is_available()={cuda_ok}{extra} | "
                f"nvidia_gpu_ok={snap.get('gpu_ok')} | "
                f"gpu_sm%={snap.get('gpu_util_pct')} mem_ctrl%={snap.get('gpu_mem_util_pct')} "
                f"effective%={snap.get('gpu_util_effective')} | "
                f"vram_mb={snap.get('vram_used_mb')}/{snap.get('vram_total_mb')}"
            )
        except Exception as exc:
            print(f"[GPU-HEARTBEAT] mislukt: {exc}")
        await asyncio.sleep(5)


@app.on_event("startup")
async def startup_refresh_markets() -> None:
    _genesis_require_gpu_or_raise()
    try:
        _refresh_active_markets_cache()
    except Exception:
        STATE["active_markets"] = []
    try:
        STATE["macro_context"] = MACRO_CALENDAR.fetch_today_macro_context()
        STATE["fear_greed"] = FEAR_GREED.fetch_index()
        STATE["whale_watch"] = WHALE_WATCHER.fetch_exchange_pressure(
            api_key=os.getenv("CRYPTOCOMPARE_KEY"),
            lookback_minutes=60,
        )
    except Exception:
        STATE.setdefault("whale_watch", {"whale_pressure": 0.0})
    _append_bot_log_line("[WHALE-SYNC] Data nu via CryptoCompare feed.")
    STATE["paper_portfolio"] = PAPER_MANAGER.wallet
    STATE["lookback_days"] = int(os.getenv("LOOKBACK_DAYS", "400"))
    STATE["episode_train_steps"] = int(os.getenv("RL_EPISODE_TRAIN_STEPS", "3000"))
    STATE["cryptocompare_key"] = os.getenv("CRYPTOCOMPARE_KEY")
    STATE["cmc_api_key"] = os.getenv("COINMARKETCAP_KEY") or os.getenv("CMC_API_KEY")
    _refresh_cmc_metrics(force=True)
    asyncio.create_task(_background_log_pruner())
    asyncio.create_task(_background_rss_poller())
    asyncio.create_task(_gpu_cuda_heartbeat())
    asyncio.create_task(_run_genesis_and_prepare_rl())
    await MAIN_ENGINE.start()
    TELEGRAM.send_start()
    port = int(os.getenv("PORT", "8000"))
    print(f"[DASHBOARD] Dashboard live op poort {port}")


@app.on_event("shutdown")
async def shutdown_notify() -> None:
    TELEGRAM.send_stop(reason="shutdown")


def generate_prediction(ticker: str, lookback_days: int) -> PredictionResponse:
    df = fetch_market_data(ticker, lookback_days)
    close_prices = df["Close"].values.astype(float)
    close_prices = close_prices.reshape(-1)

    news_api_key = os.getenv("CRYPTOCOMPARE_KEY")
    news_query = "crypto"
    articles = _signal_news_articles(ticker=ticker, news_query=news_query, news_api_key=news_api_key)
    scoring = SIGNAL_ENGINE.evaluate(close_prices=close_prices, news_articles=articles)
    technical_result = scoring["technical"]
    sentiment_result = scoring["sentiment"]
    sentiment_items = scoring.get("sentiment_items", [])
    judge_result = scoring["judge"]

    latest_close = float(close_prices[-1])
    predicted_next_close = latest_close * (1.0 + (technical_result.predicted_return_pct / 100.0))
    expected_return_pct = ((predicted_next_close - latest_close) / latest_close) * 100.0

    # Judge beslist het primaire signaal op basis van technical + sentiment.
    signal = judge_result.signal
    if signal == "HOLD":
        # Conservatieve fallback als judge neutraal is.
        signal = signal_from_expected_return(expected_return_pct)

    prediction = PredictionResponse(
        ticker=ticker.upper(),
        predicted_next_close=round(predicted_next_close, 2),
        latest_close=round(latest_close, 2),
        expected_return_pct=round(expected_return_pct, 2),
        signal=signal,
        news_sentiment=round(sentiment_result.score, 3),
        generated_at=datetime.utcnow().isoformat(),
    )
    STATE["last_scores"] = {
        "technical_score": technical_result.score,
        "technical_predicted_return_pct": technical_result.predicted_return_pct,
        "sentiment_score": sentiment_result.score,
        "sentiment_confidence": sentiment_result.confidence,
        "judge_composite_score": judge_result.composite_score,
        "judge_signal": judge_result.signal,
        "judge_weights": {
            "technical": judge_result.technical_weight,
            "sentiment": judge_result.sentiment_weight,
        },
    }
    STATE["news_insights"] = _build_news_insights(
        ticker=ticker,
        news_articles=articles,
        sentiment_items=sentiment_items,
    )
    for idx, article in enumerate(articles[:20]):
        sent_row = sentiment_items[idx] if idx < len(sentiment_items) else {}
        sentiment = float(sent_row.get("signed_score", 0.0) or 0.0)
        channel = str(article.get("news_channel") or "API").upper()
        source = str((article.get("source") or {}).get("name") or "Unknown")
        title = str(article.get("title") or "Untitled").strip().replace("\n", " ")
        if channel == "RSS":
            print(f"[RSS-FEED] Nieuw artikel gevonden: {title} - Sentiment analyse starten...")
        print(f"[NEWS][{channel}] {source}: {title} | Sentiment: {sentiment:.3f}")
    return prediction


def _estimate_spread_bps_from_recent_range(ticker: str) -> float:
    recent_df = fetch_market_data(ticker, lookback_days=70)
    last_high = float(np.asarray(recent_df["High"], dtype=float).reshape(-1)[-1])
    last_low = float(np.asarray(recent_df["Low"], dtype=float).reshape(-1)[-1])
    last_close = float(np.asarray(recent_df["Close"], dtype=float).reshape(-1)[-1])
    if last_close <= 0:
        return 0.0
    # Proxy voor spread/volatility druk.
    return max(0.0, ((last_high - last_low) / last_close) * 10000.0)


def _read_storage_stats() -> dict[str, Any]:
    if not STORAGE_STATS_PATH.exists():
        return {
            "timestamp": None,
            "size_before": 0,
            "size_after": 0,
            "saved_bytes": 0,
            "history_days": 400,
            "resolution": "Mixed (1s/1m)",
        }
    try:
        payload = json.loads(STORAGE_STATS_PATH.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass
    return {
        "timestamp": None,
        "size_before": 0,
        "size_after": 0,
        "saved_bytes": 0,
        "history_days": 400,
        "resolution": "Mixed (1s/1m)",
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "mode": "live" if LIVE_MODE else "paper",
        "bot_status": STATE.get("bot_status", "running"),
        "cmc_ok": "yes" if bool((STATE.get("cmc_metrics") or {}).get("ok")) else "no",
    }


@app.get("/predict", response_model=PredictionResponse)
def predict(
    ticker: str = Query(default=os.getenv("DEFAULT_TICKER", "BTC-EUR")),
    lookback_days: int = Query(default=int(os.getenv("LOOKBACK_DAYS", "400")), ge=60, le=3000),
) -> PredictionResponse:
    if STATE.get("bot_status") in {"paused", "panic_stop"}:
        raise HTTPException(status_code=423, detail="Bot is paused or panic-stopped.")
    try:
        prediction = generate_prediction(ticker, lookback_days)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    STATE["last_prediction"] = prediction.model_dump()
    risk_controls = compute_risk_controls(prediction.latest_close)
    spread_bps = _estimate_spread_bps_from_recent_range(prediction.ticker)
    wallet = dict(STATE.get("paper_portfolio") or PAPER_MANAGER.wallet)
    equity = float(wallet.get("equity", 10000.0) or 10000.0)
    cash = float(wallet.get("cash", equity) or equity)
    px = float(prediction.latest_close)
    sig = str(prediction.signal or "").upper()
    if sig not in {"BUY", "SELL", "HOLD"}:
        sig = "HOLD"
    size_frac, _quote_eur, size_note = CORE_RISK.calculate_trade_size(
        signal=sig,
        equity=equity,
        cash=cash,
        price=px,
        wallet=wallet,
        market=prediction.ticker,
    )
    risk_decision = RISK_MANAGER.evaluate(
        proposed_signal=sig,
        proposed_size_fraction=size_frac,
        spread_bps=spread_bps,
        sentiment_score=prediction.news_sentiment,
    )
    final_signal = str(risk_decision.adjusted_signal).upper()
    final_frac = float(risk_decision.adjusted_size_fraction)
    if risk_decision.reason == "emergency_exit_negative_sentiment_shock" and final_signal == "SELL":
        final_frac = CORE_RISK.full_exit_size_fraction(
            equity=equity, wallet=wallet, price=px, market=prediction.ticker
        )
    elif final_signal == "BUY":
        ok, why = CORE_RISK.check_safety(
            signal="BUY",
            market=prediction.ticker,
            equity=equity,
            cash=cash,
            price=px,
            wallet=wallet,
            proposed_quote_eur=final_frac * equity,
            fee_rate=float(PAPER_MANAGER.config.fee_rate),
        )
        if not ok:
            final_signal = "HOLD"
            final_frac = 0.0
            size_note = why

    paper_order = build_paper_order(
        signal=final_signal,
        ticker=prediction.ticker,
        price=prediction.latest_close,
        size_fraction=final_frac,
        budget_eur=equity,
    )
    STATE["last_order"] = {
        "risk_controls": risk_controls,
        "risk_decision": {
            "approved": risk_decision.approved,
            "reason": risk_decision.reason,
            "spread_bps": round(spread_bps, 3),
            "max_spread_bps": RISK_MANAGER.max_spread_bps_for_trading,
        },
        "engine_risk": {"sizing_note": size_note, "safety_force_exit": False},
        "order": paper_order,
    }

    append_event(
        {
            "ts": datetime.utcnow().isoformat(),
            "type": "prediction",
            "ticker": prediction.ticker,
            "signal": prediction.signal,
            "expected_return_pct": prediction.expected_return_pct,
        }
    )
    _register_signal_marker(
        ticker=prediction.ticker,
        signal=prediction.signal,
        price=prediction.latest_close,
        expected_return_pct=prediction.expected_return_pct,
    )
    return prediction


@app.post("/paper/run")
def run_paper_cycle(
    ticker: str = Query(default=os.getenv("DEFAULT_TICKER", "BTC-EUR")),
    lookback_days: int = Query(default=int(os.getenv("LOOKBACK_DAYS", "400")), ge=60, le=3000),
) -> dict[str, Any]:
    if STATE.get("bot_status") in {"paused", "panic_stop"}:
        raise HTTPException(status_code=423, detail="Bot is paused or panic-stopped.")
    try:
        prediction = generate_prediction(ticker, lookback_days)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    risk_controls = compute_risk_controls(prediction.latest_close)
    spread_bps = _estimate_spread_bps_from_recent_range(prediction.ticker)
    # Pre-train RL model before first paper cycle for this pair.
    try:
        rl_lookback = int(os.getenv("LOOKBACK_DAYS", "400"))
        TRADER.initialize(pair=prediction.ticker)
    except Exception:
        pass

    # Build latest state for RL decisioning (price-action, volume, sentiment, RSI, EMA-gap).
    rl_action_signal = prediction.signal
    rl_thought = ""
    try:
        end_dt = datetime.now(UTC)
        start_dt = end_dt - timedelta(days=max(30, rl_lookback))
        candles = fetch_bitvavo_historical_candles(
            market=prediction.ticker.upper(),
            interval="1h",
            start_dt=start_dt,
            end_dt=end_dt,
        )
        rl_frame = build_rl_training_frame(
            candles_df=candles,
            market=prediction.ticker.upper(),
            news_query="crypto",
            news_api_key=os.getenv("CRYPTOCOMPARE_KEY"),
            cryptocompare_key=os.getenv("CRYPTOCOMPARE_KEY"),
            cmc_metrics=_refresh_cmc_metrics(),
        )
        last = rl_frame.iloc[-1].to_dict()
        STATE["rl_last_state"] = {
            "fear_greed_score": float(last.get("fear_greed_score", 0.5) or 0.5),
            "btc_dominance_pct": float(last.get("btc_dominance_pct", 0.0) or 0.0),
            "sentiment_score": float(last.get("sentiment_score", 0.0) or 0.0),
            "whale_pressure": float(last.get("whale_pressure", 0.0) or 0.0),
            "macro_volatility_window": bool(float(last.get("macro_volatility_window", 0.0) or 0.0) >= 0.5),
            "rsi_14": float(last.get("rsi_14", 50.0) or 50.0),
            "macd": float(last.get("macd", 0.0) or 0.0),
            "bollinger_width": float(last.get("bollinger_width", 0.0) or 0.0),
            "bollinger_position": float(last.get("bollinger_position", 0.5) or 0.5),
        }
        acct = STATE.get("paper_portfolio", {})
        equity = float(acct.get("equity", 10000.0) or 10000.0)
        cash = float(acct.get("cash", equity) or equity)
        rl_decision_payload = TRADER.decide(
            latest_row=last,
            account={
                "balance_ratio": cash / max(1.0, equity),
                "position": float(acct.get("position_qty", 0.0) or 0.0),
                "pnl_ratio": float(acct.get("realized_pnl_eur", 0.0) or 0.0) / max(1.0, equity),
                "trade_ratio": float(acct.get("trades_count", 0.0) or 0.0) / 10000.0,
            },
        )
        rl_action_signal = str(rl_decision_payload.get("action") or prediction.signal)
        rl_thought = str(rl_decision_payload.get("reasoning") or "")
        STATE["rl_last_decision"] = {
            "action": rl_action_signal,
            "confidence": float(rl_decision_payload.get("confidence") or 0.0),
            "expected_reward_pct": float(rl_decision_payload.get("expected_reward_pct") or 0.0),
            "feature_weights": dict(rl_decision_payload.get("feature_weights") or {}),
            "reasoning": rl_thought,
            "mode": str(rl_decision_payload.get("mode") or "paper"),
            "ts": datetime.utcnow().isoformat(),
        }
    except Exception:
        pass

    wallet = dict(PAPER_MANAGER.wallet)
    live_price = _resolve_execution_price(ticker=prediction.ticker, fallback_price=prediction.latest_close)
    equity = float(wallet.get("equity", 10000.0) or 10000.0)
    cash = float(wallet.get("cash", equity) or equity)

    safety_force, safety_reason = CORE_RISK.hard_exit_for_sl_tp(
        market=prediction.ticker, price=live_price, wallet=wallet
    )
    if safety_force:
        rl_action_signal = "SELL"
        rl_thought = f"{rl_thought}\n[risk:{safety_reason}]".strip()
        if str(safety_reason or "") == "hard_stop_loss":
            RL_AGENT.ingest_paper_stop_loss()

    rl_action_signal = str(rl_action_signal or "").upper()
    if rl_action_signal not in {"BUY", "SELL", "HOLD"}:
        rl_action_signal = "HOLD"

    _rl_ld = STATE.get("rl_last_decision")
    if isinstance(_rl_ld, dict):
        _rl_ld["reasoning"] = rl_thought
        _rl_ld["action"] = rl_action_signal

    if safety_force:
        size_frac = CORE_RISK.full_exit_size_fraction(
            equity=equity, wallet=wallet, price=live_price, market=prediction.ticker
        )
        size_note = str(safety_reason or "safety_exit")
    else:
        size_frac, _quote_eur, size_note = CORE_RISK.calculate_trade_size(
            signal=rl_action_signal,
            equity=equity,
            cash=cash,
            price=live_price,
            wallet=wallet,
            market=prediction.ticker,
        )

    risk_decision = RISK_MANAGER.evaluate(
        proposed_signal=rl_action_signal,
        proposed_size_fraction=size_frac,
        spread_bps=spread_bps,
        sentiment_score=prediction.news_sentiment,
    )
    final_signal = str(risk_decision.adjusted_signal).upper()
    final_frac = float(risk_decision.adjusted_size_fraction)
    if safety_force:
        final_signal = "SELL"
        final_frac = CORE_RISK.full_exit_size_fraction(
            equity=equity, wallet=wallet, price=live_price, market=prediction.ticker
        )
    elif risk_decision.reason == "emergency_exit_negative_sentiment_shock" and final_signal == "SELL":
        final_frac = CORE_RISK.full_exit_size_fraction(
            equity=equity, wallet=wallet, price=live_price, market=prediction.ticker
        )
    elif final_signal == "BUY":
        ok, why = CORE_RISK.check_safety(
            signal="BUY",
            market=prediction.ticker,
            equity=equity,
            cash=cash,
            price=live_price,
            wallet=wallet,
            proposed_quote_eur=final_frac * equity,
            fee_rate=float(PAPER_MANAGER.config.fee_rate),
        )
        if not ok:
            final_signal = "HOLD"
            final_frac = 0.0
            size_note = why

    order = build_paper_order(
        signal=final_signal,
        ticker=prediction.ticker,
        price=prediction.latest_close,
        size_fraction=final_frac,
        budget_eur=equity,
    )
    insights = STATE.get("news_insights", [])
    top_headlines = [str(x.get("headline", "")) for x in insights[:3] if isinstance(x, dict)]
    snapshot = PAPER_MANAGER.process_signal(
        market=prediction.ticker,
        signal=final_signal,
        price=live_price,
        size_fraction=final_frac,
        sentiment_score=prediction.news_sentiment,
        news_headlines=top_headlines,
        ai_thought=rl_thought,
    )
    STATE["paper_portfolio"] = PAPER_MANAGER.wallet
    paper_cycle_seq = int(STATE.get("paper_cycle_seq", 0) or 0) + 1
    STATE["paper_cycle_seq"] = paper_cycle_seq
    if isinstance(snapshot, dict) and str(snapshot.get("status")) in {"opened", "closed"}:
        try:
            if str(snapshot.get("signal", "")).upper() in {"BUY", "SELL"}:
                TELEGRAM.send_trade(
                    market=str(prediction.ticker),
                    signal=str(snapshot.get("signal") or ""),
                    price=float(snapshot.get("entry_price") or snapshot.get("exit_price") or live_price),
                    qty=float(snapshot.get("qty") or snapshot.get("qty_closed") or 0.0),
                    equity=float((snapshot.get("wallet") or {}).get("equity") or STATE["paper_portfolio"].get("equity") or 0.0),
                )
        except Exception:
            pass

    STATE["last_prediction"] = prediction.model_dump()
    STATE["last_order"] = {
        "cycle_seq": paper_cycle_seq,
        "risk_controls": risk_controls,
        "risk_decision": {
            "approved": risk_decision.approved,
            "reason": risk_decision.reason,
            "spread_bps": round(spread_bps, 3),
            "max_spread_bps": RISK_MANAGER.max_spread_bps_for_trading,
        },
        "engine_risk": {
            "sizing_note": size_note,
            "safety_force_exit": bool(safety_force),
            "safety_reason": safety_reason,
            "rl_signal": rl_action_signal,
            "final_signal": final_signal,
            "final_size_fraction": final_frac,
        },
        "order": order,
        "snapshot": snapshot,
    }
    snap_status = str(snapshot.get("status", "")) if isinstance(snapshot, dict) else "n/a"
    print(
        f"[LEDGER] pipeline_tick ticker={prediction.ticker} final_signal={final_signal} "
        f"rl_signal={rl_action_signal} snapshot_status={snap_status} "
        "-> /activity last_order + /api/v1/trades"
    )
    append_event(
        {
            "ts": datetime.utcnow().isoformat(),
            "type": "paper_trade",
            "ticker": prediction.ticker,
            "signal": prediction.signal,
            "expected_return_pct": prediction.expected_return_pct,
        }
    )
    _register_signal_marker(
        ticker=prediction.ticker,
        signal=prediction.signal,
        price=prediction.latest_close,
        expected_return_pct=prediction.expected_return_pct,
    )

    return {
        "prediction": prediction.model_dump(),
        "risk_controls": risk_controls,
        "order": order,
        "portfolio_snapshot": snapshot,
        "ai_thought": rl_thought,
    }


@app.get("/activity")
def activity() -> dict[str, Any]:
    return {
        "mode": "live" if LIVE_MODE else "paper",
        "bot_status": STATE.get("bot_status", "running"),
        "selected_market": STATE.get("selected_market", "BTC-EUR"),
        "active_markets_count": len(STATE.get("active_markets", [])),
        "started_at": STATE["started_at"],
        "last_prediction": STATE["last_prediction"],
        "last_scores": STATE.get("last_scores"),
        "last_order": STATE["last_order"],
        "paper_portfolio": STATE["paper_portfolio"],
        "events": STATE["events"],
        "fear_greed": STATE.get("fear_greed") or {},
        "risk_profile": risk_profile_dict(),
    }


@app.get("/api/v1/news/ticker")
def api_news_ticker() -> list[dict[str, Any]]:
    active_markets = STATE.get("active_markets", [])
    if not active_markets:
        try:
            _refresh_active_markets_cache()
        except Exception:
            pass
    active_markets = STATE.get("active_markets", [])
    news_query = "crypto"
    news_api_key = os.getenv("CRYPTOCOMPARE_KEY")
    combined_articles = _signal_news_articles(
        ticker=STATE.get("selected_market", "BTC-EUR"),
        news_query=news_query,
        news_api_key=news_api_key,
    )
    try:
        return NEWS_MAPPING.get_mapped_news(
            news_query=news_query,
            news_api_key=news_api_key,
            active_markets=active_markets,
            prefetched_articles=combined_articles,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/v1/brain/state-overview")
def api_brain_state_overview() -> dict[str, Any]:
    try:
        STATE["whale_watch"] = WHALE_WATCHER.fetch_exchange_pressure(
            api_key=os.getenv("CRYPTOCOMPARE_KEY"),
            lookback_minutes=120,
        )
    except Exception:
        pass
    try:
        _refresh_cmc_metrics(force=False)
    except Exception:
        pass
    fw = {}
    decision = STATE.get("rl_last_decision")
    if isinstance(decision, dict):
        fw = decision.get("feature_weights", {}) if isinstance(decision.get("feature_weights"), dict) else {}
    last_state = dict(STATE.get("rl_last_state", {}) or {}) if isinstance(STATE.get("rl_last_state"), dict) else {}
    fg_state = STATE.get("fear_greed") if isinstance(STATE.get("fear_greed"), dict) else {}
    if fg_state and fg_state.get("fear_greed_score") is not None and "fear_greed_score" not in last_state:
        last_state["fear_greed_score"] = float(fg_state.get("fear_greed_score") or 0.5)
    cmc = STATE.get("cmc_metrics") if isinstance(STATE.get("cmc_metrics"), dict) else {}
    btc_dom = float(last_state.get("btc_dominance_pct") or 0.0)
    if btc_dom <= 0.0:
        cmc_dom = float(cmc.get("btc_dominance_pct") or 0.0)
        if cmc_dom > 0.0:
            last_state["btc_dominance_pct"] = cmc_dom
        else:
            fb_dom = float(os.getenv("GENESIS_BTC_DOM_FALLBACK", "52.0") or 52.0)
            last_state["btc_dominance_pct"] = max(1.0, min(95.0, fb_dom))
    rsi_raw = last_state.get("rsi_14")
    try:
        rsi_val = float(rsi_raw) if rsi_raw is not None else 0.0
    except (TypeError, ValueError):
        rsi_val = 0.0
    if rsi_raw is None or rsi_val <= 0.0 or rsi_val > 100.0:
        last_state["rsi_14"] = 50.0
    wp_live = float((STATE.get("whale_watch") or {}).get("whale_pressure", 0.0) or 0.0)
    last_state["whale_pressure"] = wp_live
    whales_weight = float(fw.get("whale_pressure", 0.0))
    btc_dom_weight = float(fw.get("btc_dominance_pct", 0.0))
    macro_weight = float(fw.get("macro_volatility_window", 0.0))
    rsi_weight = float(fw.get("rsi_14", 0.0))
    return {
        "state": last_state,
        "weight_focus": {
            "whales": round(whales_weight, 4),
            "btc_dominance": round(btc_dom_weight, 4),
            "macro": round(macro_weight, 4),
            "rsi": round(rsi_weight, 4),
        },
    }


@app.get("/api/v1/history")
def api_history(
    pair: str = Query(default=os.getenv("DEFAULT_TICKER", "BTC-EUR")),
    lookback_days: int = Query(default=180, ge=30, le=1200),
) -> dict[str, Any]:
    target = pair.upper()
    try:
        labels, prices = _fetch_history_series(pair=target, lookback_days=lookback_days)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    markers = [
        m for m in STATE.get("signal_markers", []) if m.get("ticker", "").upper() == target
    ][:200]
    return {
        "pair": target,
        "tv_symbol": f"BITVAVO:{target}" if "-" in target else target,
        "labels": labels,
        "prices": prices,
        "markers": markers,
    }


@app.get("/api/v1/trades")
def api_trades(limit: int = Query(default=25, ge=1, le=200)) -> dict[str, Any]:
    return {"trades": PAPER_MANAGER.recent_trades(limit=limit)}


@app.get("/api/v1/system/storage")
def api_system_storage() -> dict[str, Any]:
    stats = _read_storage_stats()
    total, used, free = shutil.disk_usage(Path.home())
    usage_pct = 0.0 if total <= 0 else (float(used) / float(total)) * 100.0
    return {
        "stats": stats,
        "disk": {
            "total_bytes": int(total),
            "used_bytes": int(used),
            "free_bytes": int(free),
            "usage_pct": round(usage_pct, 2),
        },
    }


@app.get("/api/v1/system/logs")
def api_system_logs(limit: int = Query(default=200, ge=50, le=1000)) -> dict[str, Any]:
    return {"lines": _tail_lines(BOT_LOG_PATH, limit=limit), "path": str(BOT_LOG_PATH)}


@app.get("/api/v1/performance/analytics")
def api_performance_analytics() -> dict[str, Any]:
    analytics = PAPER_MANAGER.analytics()
    suggestions = []
    for row in analytics.get("coin_rollup", [])[:5]:
        suggestions.append(
            adjust_weights(
                coin=str(row.get("coin", "MKT")),
                avg_sentiment_top_losses=float(
                    analytics.get("sentiment_correlation", {}).get("avg_sentiment_top_10_losses", 0.0)
                ),
                avg_sentiment_top_wins=float(
                    analytics.get("sentiment_correlation", {}).get("avg_sentiment_top_10_wins", 0.0)
                ),
            )
        )
    return {
        "wallet": PAPER_MANAGER.wallet,
        "equity_curve": PAPER_MANAGER.wallet.get("history", []),
        "recent_trades": PAPER_MANAGER.recent_trades(limit=50),
        "recent_actions": PAPER_MANAGER.wallet.get("history", [])[-50:],
        "analytics": analytics,
        "weight_adjustment_suggestions": suggestions,
    }


@app.get("/api/v1/brain/reasoning")
def api_brain_reasoning() -> dict[str, Any]:
    return STATE.get("rl_last_decision", {}) if isinstance(STATE.get("rl_last_decision"), dict) else {}


@app.get("/api/v1/brain/feature-importance")
def api_brain_feature_importance() -> dict[str, Any]:
    decision = STATE.get("rl_last_decision", {}) if isinstance(STATE.get("rl_last_decision"), dict) else {}
    return {"feature_weights": decision.get("feature_weights", {})}


@app.get("/api/v1/brain/training-monitor")
def api_brain_training_monitor() -> dict[str, Any]:
    return RL_AGENT.training_monitor()


@app.get("/api/v1/brain/news-lag")
def api_brain_news_lag() -> dict[str, Any]:
    rows = STATE.get("news_lag_history", [])
    return {"items": rows[-120:] if isinstance(rows, list) else []}


@app.get("/terminal/news-insights")
def terminal_news_insights() -> dict[str, Any]:
    return {
        "items": STATE.get("news_insights", []),
        "selected_market": STATE.get("selected_market", "BTC-EUR"),
        "updated_from_prediction": STATE.get("last_prediction", {}).get("generated_at")
        if isinstance(STATE.get("last_prediction"), dict)
        else None,
    }


@app.get("/terminal/chart-points")
def terminal_chart_points(
    ticker: str = Query(default=os.getenv("DEFAULT_TICKER", "BTC-EUR")),
    lookback_days: int = Query(default=120, ge=30, le=600),
) -> dict[str, Any]:
    try:
        df = fetch_market_data(ticker=ticker, lookback_days=lookback_days)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    labels = [str(v) for v in df["Date"].astype(str).tolist()]
    prices = [float(v) for v in np.asarray(df["Close"], dtype=float).reshape(-1).tolist()]
    markers = [
        m
        for m in STATE.get("signal_markers", [])
        if m.get("ticker", "").upper() == ticker.upper()
    ][:200]
    return {
        "ticker": ticker.upper(),
        "labels": labels,
        "prices": prices,
        "markers": markers,
    }


@app.get("/sentiment/current")
def current_sentiment() -> dict[str, Any]:
    scores = STATE.get("last_scores") or {}
    return {
        "sentiment_score": scores.get("sentiment_score"),
        "sentiment_confidence": scores.get("sentiment_confidence"),
        "updated_from_prediction": STATE.get("last_prediction", {}).get("generated_at")
        if isinstance(STATE.get("last_prediction"), dict)
        else None,
    }


@app.get("/bot/status")
def get_bot_status() -> dict[str, str]:
    return {"bot_status": STATE.get("bot_status", "running")}


@app.get("/exchange/rate-limit/status")
def exchange_rate_limit_status() -> dict[str, Any]:
    return global_rate_limit_status()


@app.post("/bot/pause")
def pause_bot() -> dict[str, str]:
    STATE["bot_status"] = "paused"
    append_event({"ts": datetime.utcnow().isoformat(), "type": "bot_status", "status": "paused"})
    TELEGRAM.send_bot_status("paused")
    return {"bot_status": "paused"}


@app.post("/bot/resume")
def resume_bot() -> dict[str, str]:
    STATE["bot_status"] = "running"
    append_event({"ts": datetime.utcnow().isoformat(), "type": "bot_status", "status": "running"})
    TELEGRAM.send_bot_status("running")
    return {"bot_status": "running"}


@app.post("/bot/panic")
def panic_stop() -> dict[str, str]:
    STATE["bot_status"] = "panic_stop"
    append_event({"ts": datetime.utcnow().isoformat(), "type": "bot_status", "status": "panic_stop"})
    TELEGRAM.send_bot_status("panic_stop")
    return {"bot_status": "panic_stop"}


@app.get("/markets/active")
def markets_active(min_volume_eur: float | None = Query(default=None)) -> dict[str, Any]:
    if min_volume_eur is not None:
        scanner = MarketScanner(min_volume_eur=min_volume_eur)
        markets = scanner.fetch_active_pairs()
        return {"markets": markets, "min_volume_eur": min_volume_eur}
    if not STATE.get("active_markets"):
        _refresh_active_markets_cache()
    return {"markets": STATE.get("active_markets", []), "min_volume_eur": MARKET_SCANNER.min_volume_eur}


@app.post("/markets/select")
def market_select(market: str = Query(...)) -> dict[str, Any]:
    if not STATE.get("active_markets"):
        _refresh_active_markets_cache()
    target = market.upper()
    active = STATE.get("active_markets", [])
    if not any(m["market"] == target for m in active):
        raise HTTPException(status_code=400, detail=f"Market {target} not in active filtered list.")
    STATE["selected_market"] = target
    append_event({"ts": datetime.utcnow().isoformat(), "type": "market_select", "market": target})
    return {"selected_market": target}


@app.get("/markets/selected")
def market_selected() -> dict[str, str]:
    return {"selected_market": STATE.get("selected_market", "BTC-EUR")}


@app.get("/vault/balance-check")
def vault_balance_check(market: str | None = Query(default=None)) -> dict[str, Any]:
    target_market = (market or STATE.get("selected_market") or "BTC-EUR").upper()
    min_quote = float(os.getenv("PAIR_MIN_QUOTE_BALANCE", "50"))
    min_base = float(os.getenv("PAIR_MIN_BASE_BALANCE", "0.00001"))
    try:
        return check_pair_balance_from_vault(
            market=target_market,
            min_quote_balance=min_quote,
            min_base_balance=min_base,
        )
    except Exception as exc:
        return {
            "market": target_market,
            "available": False,
            "reason": f"balance_check_failed:{str(exc)}",
        }


@app.get("/dry-run/pnl/daily")
def dry_run_daily_pnl(date_utc: str | None = Query(default=None)) -> dict[str, Any]:
    target_date = date_utc or datetime.utcnow().date().isoformat()
    try:
        return calculate_daily_fictive_pnl(target_date)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.websocket("/ws/trades")
async def ws_trades(websocket: WebSocket) -> None:
    """Live paper-trade rows voor AI Brain-dashboard (polling op server, push naar client)."""
    await websocket.accept()
    try:
        while True:
            rows = PAPER_MANAGER.recent_trades(limit=25)
            await websocket.send_json({"trades": rows})
            await asyncio.sleep(4)
    except WebSocketDisconnect:
        return
    except Exception:
        try:
            await websocket.close()
        except Exception:
            pass


@app.websocket("/ws/logs")
async def ws_logs(websocket: WebSocket) -> None:
    await websocket.accept()
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    BOT_LOG_PATH.touch(exist_ok=True)
    try:
        initial_lines = _tail_lines(BOT_LOG_PATH, limit=200)
        for line in initial_lines:
            await websocket.send_text(line)

        with BOT_LOG_PATH.open("r", encoding="utf-8", errors="replace") as fh:
            fh.seek(0, os.SEEK_END)
            while True:
                where = fh.tell()
                line = fh.readline()
                if not line:
                    await asyncio.sleep(0.8)
                    fh.seek(where)
                    continue
                await websocket.send_text(line.rstrip("\n"))
    except WebSocketDisconnect:
        return
    except Exception:
        try:
            await websocket.close()
        except Exception:
            pass


@app.websocket("/ws/system-stats")
async def ws_system_stats(websocket: WebSocket) -> None:
    """Live CPU/RAM/GPU: elke 2s een vers JSON-pakket (`get_system_stats` / nvidia-smi)."""
    await websocket.accept()
    try:
        while True:
            await websocket.send_json(get_system_stats())
            await asyncio.sleep(2)
    except WebSocketDisconnect:
        return
    except Exception:
        try:
            await websocket.close()
        except Exception:
            pass


@app.get("/", response_class=HTMLResponse)
def portal(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "request": request,
            "default_ticker": os.getenv("DEFAULT_TICKER", "BTC-EUR"),
        },
    )
