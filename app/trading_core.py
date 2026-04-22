"""Trading stack: markt/ML/paper engine — geen FastAPI/Jinja2 (worker + HTTP-shell importeren dit)."""

import asyncio
import io
import logging
import os
import time
import json
import shutil
import requests
import warnings
from datetime import datetime, timedelta

from app.datetime_util import UTC
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

_PORTAL_SLIM = str(os.getenv("AI_TRADING_PROCESS", "") or "").strip().lower() == "portal"

import pytz

from app.schemas.prediction import PredictionResponse
from app.ai.judge.weighted_judge import adjust_weights
from app.exchanges.bitvavo import BitvavoClient, global_rate_limit_status
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
from app.services.rl_metrics_store import merge_historical_training_into_agent
from app.services.risk import RiskManager, compute_risk_controls, signal_from_expected_return
from app.rl.observation_audit import (
    bar_values_from_obs_and_weights,
    log_rl_data_audit_heartbeat,
    merge_feature_weights_for_brain,
)
from app.services.signal_engine import SignalEngine
from app.portal_snapshot import (
    apply_worker_snapshot_to_portal,
    read_worker_portal_snapshot,
    write_worker_portal_snapshot,
)
from app.redis_bridge import (
    SYSTEM_STATS_CHANNEL,
    TRADING_UPDATES_CHANNEL,
    publish_system_stats_update,
    publish_trading_update,
)
from app.services.state import STATE, append_event, current_tenant_id, set_current_tenant
from app.settings import LIVE_MODE, validate_mode_configuration
from app.services.system_stats import collect_system_stats, get_system_stats
from app.services.telegram_notifier import TelegramNotifier
from app.services.jarvis_worker import AITradingPerformanceIntegrityReporter
from app.services.whale_watcher import WhaleWatcherService
from core.notifier import (
    daily_restart_report_loop,
    send_restart_report,
    send_urgent_alert,
    send_watchdog_recovery_telegram,
)
from core.trading_logic import MIN_BUY_CONFIDENCE, MIN_HOLD_MINUTES, should_block_sell_for_min_hold
from core.scanner import DynamicVolatilityScanner
from core.news_engine import apply_social_overlay_to_rl_row, refresh_social_momentum_state
from core.social_engine import (
    build_trade_decision_context,
    format_ledger_social_whale_context,
    refresh_whale_radar_state,
)
from core.trading_engine import (
    SKIP_BUY_ACTIVE_POSITION_LOG,
    TradingEngine,
    has_active_paper_position_for_ticker,
)
from core.risk_manager import RiskManager as CoreRiskManager, risk_profile_dict
from core.risk_management import (
    SKIP_MAX_PORTFOLIO_ALLOCATION_LOG,
    allocation_snapshot,
    apply_equal_weight_buy_fraction_cap,
    market_blocked_by_whale_panic_cooldown,
    whale_danger_zone_for_market,
)
from core.ws_manager import (
    build_brain_ws_wire_payload,
    compact_system_stats,
    elite_lite_rows,
)

if not _PORTAL_SLIM:
    from agent.trader import Trader, TraderConfig
    from app.ai.sentiment.finbert_sentiment import LazyFinBertSentimentAnalyzer
    from app.rl.agent_rl import RLAgentService, get_rl_ppo_device
    from app.rl.data import build_rl_training_frame, fetch_bitvavo_historical_candles
else:
    from app.portal_stubs import (
        PortalNeutralSentiment as LazyFinBertSentimentAnalyzer,
        PortalPaperTrader as Trader,
        PortalRLAgentStub as RLAgentService,
        PortalTraderConfig as TraderConfig,
        portal_get_rl_ppo_device as get_rl_ppo_device,
    )

    def build_rl_training_frame(*_a: Any, **_kw: Any) -> Any:
        raise RuntimeError("RL frame build is not available on the portal-only image")

    def fetch_bitvavo_historical_candles(*_a: Any, **_kw: Any) -> Any:
        raise RuntimeError("RL candle fetch is not available on the portal-only image")


# HF Hub / transformers logging alleen op worker/full image (portal heeft die libs niet).
if not _PORTAL_SLIM:
    warnings.filterwarnings("ignore", message=".*unauthenticated requests to the HF Hub.*")
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)


def _genesis_log_torch_device() -> None:
    """Logt het actieve compute-device; deploy-scripts kunnen op `[DEVICE] Using device: cuda:0` wachten."""
    import importlib.util

    if importlib.util.find_spec("torch") is None:
        if _process_role() == "portal":
            print("[DEVICE] Portal (slim image): geen torch — GPU-log overgeslagen.")
        return
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


def _process_role() -> str:
    """`all` (default): één proces met HTTP + trading. `portal`: alleen FastAPI. `worker`: alleen achtergrond."""
    return str(os.getenv("AI_TRADING_PROCESS", "all") or "all").strip().lower()


def _probe_torch_cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False

_FINBERT = LazyFinBertSentimentAnalyzer()
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
if not _PORTAL_SLIM:
    try:
        merge_historical_training_into_agent(RL_AGENT)
    except Exception:
        pass
TRADER = Trader(
    TraderConfig(
        initial_capital_eur=float(os.getenv("PAPER_START_BALANCE_EUR", "10000")),
        lookback_days=int(os.getenv("LOOKBACK_DAYS", "400")),
        model_dir=os.getenv("RL_MODEL_DIR", "artifacts/rl"),
    ),
    agent_service=RL_AGENT,
)
MAIN_ENGINE = TradingEngine(
    run_cycle=lambda ticker, lookback_days: _run_paper_cycle_sync(ticker=ticker, lookback_days=lookback_days),
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
_storage_env = str(os.getenv("AI_TRADING_STORAGE_ROOT", "")).strip()
STORAGE_ROOT = Path(_storage_env) if _storage_env else (Path.cwd() / "storage")
STORAGE_STATS_PATH = STORAGE_ROOT / "stats.json"
LOGS_DIR = STORAGE_ROOT / "logs"
BOT_LOG_PATH = LOGS_DIR / "bot_execution.log"
TEE_INSTALLED = False
JARVIS_REPORTER: AITradingPerformanceIntegrityReporter | None = None
RESTART_MAIL_TASK: asyncio.Task[Any] | None = None
CONFIG_TICKERS = [
    t.strip().upper()
    for t in str(os.getenv("TICKERS", "")).split(",")
    if t and t.strip()
]
VOL_SCANNER = DynamicVolatilityScanner()
AUDIT_LAST_RUN: str | None = None
AUDIT_LAST_TUNING: dict[str, Any] = {}
AUDIT_REFLECTIONS: list[dict[str, Any]] = []
AUTO_OPT_LAST_RUN: str | None = None
AUTO_OPT_LAST_TUNING: dict[str, Any] = {}
AUTO_OPT_SCORE_HISTORY: list[float] = []
AUTO_OPT_BEST_SETTINGS: dict[str, Any] = {}
ALERT_LAST_SENT_AT: dict[str, str] = {}
# Default > elite sweep interval (~60s) to avoid false positives; raise further if FinBERT/PPO blocks the loop.
WATCHDOG_STALL_LIMIT_SEC = int(os.getenv("WATCHDOG_STALL_LIMIT_SEC", "150") or 150)
TZ = pytz.timezone("Europe/Amsterdam")
MIN_EXPLORATION_EPS = 0.05
PREDICT_QUEUE_MAX = int(os.getenv("PREDICT_QUEUE_MAX", "128") or 128)
PAPER_QUEUE_MAX = int(os.getenv("PAPER_QUEUE_MAX", "128") or 128)
PREDICT_QUEUE: asyncio.Queue[tuple[str, int, str, asyncio.Future[Any]]] = asyncio.Queue(maxsize=PREDICT_QUEUE_MAX)
PAPER_RUN_QUEUE: asyncio.Queue[tuple[str, int, str, asyncio.Future[Any]]] = asyncio.Queue(maxsize=PAPER_QUEUE_MAX)

_BACKGROUND_TASKS: set[asyncio.Task[Any]] = set()

def start_background_task(coro: Any) -> asyncio.Task[Any]:
    task = asyncio.create_task(coro)
    _BACKGROUND_TASKS.add(task)
    task.add_done_callback(_BACKGROUND_TASKS.discard)
    return task

def _balanced_feature_weights() -> dict[str, float]:
    names = list(RL_AGENT.feature_names) if hasattr(RL_AGENT, "feature_names") else []
    if not names:
        return {}
    w = 1.0 / max(1, len(names))
    return {str(k): round(w, 4) for k in names}


def _average_feature_weights(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys: set[str] = set()
    for r in rows:
        keys.update(str(k) for k in r.keys())
    out: dict[str, float] = {}
    for k in keys:
        vals = [float(r.get(k, 0.0) or 0.0) for r in rows]
        out[k] = float(sum(vals) / max(1, len(vals)))
    s = float(sum(max(0.0, v) for v in out.values()))
    if s <= 1e-12:
        return {}
    return {k: round(max(0.0, v) / s, 4) for k, v in out.items()}


def _jarvis_financials_snapshot() -> dict[str, Any]:
    analytics = PAPER_MANAGER.analytics()
    perf = analytics.get("performance_summary", {}) if isinstance(analytics, dict) else {}
    corr = analytics.get("sentiment_correlation", {}) if isinstance(analytics, dict) else {}
    wallet = PAPER_MANAGER.wallet if isinstance(PAPER_MANAGER.wallet, dict) else {}
    start_balance = float(PAPER_MANAGER.config.starting_balance_eur or 10000.0)
    equity = float(wallet.get("equity", start_balance) or start_balance)
    runtime_pnl_eur = equity - start_balance
    runtime_pnl_pct = (runtime_pnl_eur / start_balance) * 100.0 if start_balance > 0 else 0.0
    return {
        "total_pnl_eur": float(perf.get("total_pnl_eur", 0.0) or 0.0),
        "win_rate_pct": float(perf.get("win_rate_pct", 0.0) or 0.0),
        "wins": int(perf.get("wins", 0) or 0),
        "losses": int(perf.get("losses", 0) or 0),
        "avg_sentiment_top_10_losses": float(corr.get("avg_sentiment_top_10_losses", 0.0) or 0.0),
        "avg_sentiment_top_10_wins": float(corr.get("avg_sentiment_top_10_wins", 0.0) or 0.0),
        "start_balance_eur": start_balance,
        "equity_eur": equity,
        "runtime_pnl_eur": runtime_pnl_eur,
        "runtime_pnl_pct": runtime_pnl_pct,
    }


def _jarvis_live_financials_snapshot() -> dict[str, Any]:
    if not LIVE_MODE:
        return {"enabled": False}
    api_key = str(os.getenv("BITVAVO_KEY_TRADE", "")).strip()
    api_secret = str(os.getenv("BITVAVO_SECRET_TRADE", "")).strip()
    if not api_key or not api_secret:
        return {"enabled": True, "available": False, "reason": "missing_live_keys"}
    try:
        client = BitvavoClient(api_key=api_key, api_secret=api_secret)
        balances = client.get_balance()
        eur_available = 0.0
        for row in balances:
            if str(row.get("symbol", "")).upper() == "EUR":
                eur_available = float(row.get("available") or 0.0)
                break
        return {"enabled": True, "available": True, "eur_available": eur_available, "balances_count": len(balances)}
    except Exception as exc:
        return {"enabled": True, "available": False, "reason": str(exc)}


def _maybe_send_urgent_alert(key: str, subject: str, details: str, cooldown_minutes: int = 30) -> None:
    now = datetime.now(UTC)
    last_iso = ALERT_LAST_SENT_AT.get(key)
    if last_iso:
        try:
            last_dt = datetime.fromisoformat(last_iso)
            if (now - last_dt).total_seconds() < max(60, cooldown_minutes * 60):
                return
        except Exception:
            pass
    if send_urgent_alert(subject=subject, details=details):
        ALERT_LAST_SENT_AT[key] = now.isoformat()


def _maybe_send_watchdog_recovery_telegram(engine_age: float, ws_age: float, cooldown_minutes: int = 30) -> None:
    key = "watchdog_stall"
    now = datetime.now(UTC)
    last_iso = ALERT_LAST_SENT_AT.get(key)
    if last_iso:
        try:
            last_dt = datetime.fromisoformat(last_iso)
            if (now - last_dt).total_seconds() < max(60, cooldown_minutes * 60):
                return
        except Exception:
            pass
    if send_watchdog_recovery_telegram(engine_age, ws_age, disable_notification=False):
        ALERT_LAST_SENT_AT[key] = now.isoformat()


def _compute_yesterday_pnl() -> float:
    trades = PAPER_MANAGER.round_trip_ledger(limit=10000)
    local_now = datetime.now(TZ)
    y = (local_now - timedelta(days=1)).date()
    total = 0.0
    for t in trades:
        ts = str(t.get("close_time_utc") or t.get("open_time_utc") or "")
        if not ts:
            continue
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            if dt.astimezone(TZ).date() == y:
                total += float(t.get("pnl_eur") or 0.0)
        except Exception:
            continue
    return round(total, 2)


def _portfolio_distribution_snapshot() -> list[dict[str, Any]]:
    wallet = PAPER_MANAGER.wallet if isinstance(PAPER_MANAGER.wallet, dict) else {}
    equity = float(wallet.get("equity", PAPER_MANAGER.config.starting_balance_eur) or PAPER_MANAGER.config.starting_balance_eur)
    cash = float(wallet.get("cash", 0.0) or 0.0)
    lp = wallet.get("last_prices_by_market") if isinstance(wallet.get("last_prices_by_market"), dict) else {}
    pbm = wallet.get("position_by_market") if isinstance(wallet.get("position_by_market"), dict) else {}
    out = [
        {"asset": "EUR", "qty": cash, "weight_pct": (cash / max(1.0, equity)) * 100.0},
    ]
    if pbm:
        for mkt, qv in pbm.items():
            qty = float(qv or 0.0)
            if qty <= 1e-12:
                continue
            mku = str(mkt).strip().upper()
            px = float(lp.get(mku, 0.0) or 0.0)
            if px <= 0 and mku == str(wallet.get("position_symbol") or "").upper():
                px = float(wallet.get("last_price", 0.0) or 0.0)
            pos_val = max(0.0, qty * px)
            if pos_val > 0:
                out.append({"asset": mku, "qty": qty, "weight_pct": (pos_val / max(1.0, equity)) * 100.0})
        return out
    market = str(wallet.get("position_symbol") or "NONE").upper()
    qty = float(wallet.get("position_qty", 0.0) or 0.0)
    px = float(wallet.get("last_price", 0.0) or 0.0)
    pos_val = max(0.0, qty * px) if market != "NONE" else 0.0
    if market != "NONE" and pos_val > 0:
        out.append({"asset": market, "qty": qty, "weight_pct": (pos_val / max(1.0, equity)) * 100.0})
    return out


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


validate_mode_configuration()


def _coin_from_ticker(ticker: str) -> str:
    upper = (ticker or "").upper()
    if "-" in upper:
        return upper.split("-", 1)[0]
    if upper.endswith("EUR"):
        return upper[:-3]
    if upper.endswith("USD"):
        return upper[:-3]
    return upper


NEWS_ELITE_COIN_ALIASES: dict[str, list[str]] = {
    "BTC": ["BITCOIN"],
    "ETH": ["ETHEREUM", "ETHER"],
    "SOL": ["SOLANA"],
    "EDU": ["OPEN CAMPUS"],
    "ZRO": ["LAYERZERO"],
    "MERL": ["MERLIN"],
    "AAVE": ["AAVE"],
    "PORTAL": ["PORTAL"],
    "GWEI": ["GWEI"],
    "PROM": ["PROM"],
    "HIGH": ["HIGHSTREET", "HIGH"],
}


def _selection_reason_for_market(markets: list[dict[str, Any]], market: str) -> str:
    mku = str(market or "").upper()
    for row in markets or []:
        if isinstance(row, dict) and str(row.get("market", "")).upper() == mku:
            return str(row.get("selection_reason") or "").strip()
    return ""


def _interleave_elite_mapped_news(
    mapped: list[dict[str, Any]],
    elite_markets: list[str],
    max_items: int = 96,
) -> list[dict[str, Any]]:
    """Round-robin high-impact news so all Elite tickers appear in the intelligence ticker."""
    order = [_coin_from_ticker(m) for m in elite_markets if m][:8]
    if not order:
        return list(mapped or [])[:max_items]
    buckets: dict[str, list[dict[str, Any]]] = {b: [] for b in order}
    neutral: list[dict[str, Any]] = []
    for item in mapped or []:
        c = str(item.get("coin") or "MKT").upper()
        if c == "MKT":
            neutral.append(item)
        elif c in buckets:
            buckets[c].append(item)
    out: list[dict[str, Any]] = []
    ni = 0
    while len(out) < max_items:
        progressed = False
        for b in order:
            if buckets[b]:
                out.append(buckets[b].pop(0))
                progressed = True
                if len(out) >= max_items:
                    break
        if neutral and len(out) < max_items and (ni % 2 == 0):
            out.append(neutral.pop(0))
            progressed = True
        ni += 1
        if not progressed:
            break
    return out


def _elite_ai_signals_payload() -> list[dict[str, Any]]:
    elite = [
        str(m.get("market", "")).upper()
        for m in (STATE.get("active_markets") or [])
        if isinstance(m, dict) and m.get("market")
    ][:8]
    decisions = STATE.get("rl_multi_decisions") if isinstance(STATE.get("rl_multi_decisions"), dict) else {}
    out: list[dict[str, Any]] = []
    for mk in elite:
        dec = decisions.get(mk) if isinstance(decisions, dict) else None
        action = ""
        conf = 0.0
        if isinstance(dec, dict):
            action = str(dec.get("action") or "").upper()
            try:
                conf = float(dec.get("confidence") or 0.0)
            except (TypeError, ValueError):
                conf = 0.0
        dz = whale_danger_zone_for_market(STATE, mk)
        danger = bool(dz.get("active"))
        cooldown = market_blocked_by_whale_panic_cooldown(STATE, mk)
        w_pf = STATE.get("paper_portfolio") if isinstance(STATE.get("paper_portfolio"), dict) else {}
        in_pos = has_active_paper_position_for_ticker(w_pf, mk)
        if danger or cooldown:
            state = "panic"
        elif action == "SELL":
            state = "bear"
        elif action in {"BUY", "HOLD"}:
            state = "bull"
        else:
            state = "neutral"
        out.append(
            {
                "market": mk,
                "base": _coin_from_ticker(mk),
                "action": action,
                "confidence": round(conf, 4),
                "state": state,
                "whale_danger": danger,
                "panic_cooldown": cooldown,
                "in_position": bool(in_pos),
            }
        )
    return out


def _whale_panic_cooldowns_payload() -> dict[str, str]:
    """Geeft actieve whale panic cooldowns (ISO timestamps) terug voor de API."""
    cooldowns = STATE.get("whale_panic_cooldown_until")
    return dict(cooldowns) if isinstance(cooldowns, dict) else {}


def _allocation_snapshot_for_activity() -> dict[str, Any]:
    """Geeft de allocatie-snapshot terug voor de /activity API."""
    wallet = STATE.get("paper_portfolio") or PAPER_MANAGER.wallet
    wallet_d = wallet if isinstance(wallet, dict) else {}
    equity = float(wallet_d.get("equity", 10000.0) or 10000.0)
    return allocation_snapshot(wallet_d, equity)


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
    target_coin = _coin_from_ticker(ticker)
    elite_markets = [str(m.get("market", "")).upper() for m in (STATE.get("active_markets") or []) if m.get("market")]
    rows = NEWS_ENGINE.prioritize_for_elite_tickers(
        rows=rows, elite_tickers=elite_markets, coin_aliases=NEWS_ELITE_COIN_ALIASES
    )
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
        mapped = NEWS_MAPPING.get_mapped_news(
            news_query=news_query,
            news_api_key=news_api_key,
            active_markets=STATE.get("active_markets") or [],
            prefetched_articles=rows,
        )
        scoped: list[dict[str, Any]] = []
        for item in mapped:
            coin = str(item.get("coin") or "MKT").upper()
            sent = float(item.get("sentiment", 0.0) or 0.0)
            if coin == target_coin or (coin == "MKT" and abs(sent) >= 0.8):
                scoped.append(
                    {
                        "title": str(item.get("title") or item.get("text") or ""),
                        "description": str(item.get("summary") or ""),
                        "url": item.get("url"),
                        "publishedAt": item.get("published_at"),
                        "source": {"name": str(item.get("source") or "NEWS")},
                        "coin": coin,
                        "mapped_sentiment": sent,
                    }
                )
        if scoped:
            return scoped[:40]
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
    if not str(os.getenv("CRYPTOCOMPARE_KEY") or "").strip():
        print("[GENESIS] Check 2 CryptoCompare nieuws: SKIP (CRYPTOCOMPARE_KEY not set)")
        return False
    try:
        rows = NEWS_SERVICE.fetch_latest_news(api_key=os.getenv("CRYPTOCOMPARE_KEY"), limit=5, force=True)
        ok = bool(rows)
        print(f"[GENESIS] Check 2 CryptoCompare nieuws: {'OK' if ok else 'FAIL (no rows)'}")
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
        print("[GENESIS] Check 3 CoinMarketCap context: SKIP (COINMARKETCAP_KEY / CMC_API_KEY not set)")
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
    if not check1:
        print("[GENESIS] Genesis warmup skipped (Bitvavo connection is REQUIRED).")
        return
    if not (check2 and check3 and check4):
        print("[GENESIS] Warning: Some external APIs (CryptoCompare/CMC/RSS) are failing or missing keys, but proceeding with RL warmup anyway.")
    else:
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


def _sanitize_paper_wallet() -> None:
    """Verwijdert geneste referenties in history om exponentiële RAM-groei te voorkomen."""
    if isinstance(PAPER_MANAGER.wallet.get("history"), list):
        for h in PAPER_MANAGER.wallet["history"]:
            if isinstance(h, dict):
                h.pop("wallet", None)
                h.pop("snap", None)
                h.pop("history", None)
        PAPER_MANAGER.wallet["history"] = PAPER_MANAGER.wallet["history"][-500:]


def _paper_whale_panic_intervention(
    prediction: PredictionResponse,
    spread_bps: float,
    slippage_bps: float,
) -> tuple[str, str]:
    """
    Forced full SELL on held market when elite whale exchange-inflow burst triggers.
    Returns (panic_sold_market_upper_or_empty, note_appended_to_rl_thought).
    """
    from core.risk_management import (
        can_fire_whale_panic_sell,
        record_whale_panic_sell_fired,
        set_whale_panic_cooldown,
        whale_panic_should_force_sell,
    )

    wallet = dict(PAPER_MANAGER.wallet)
    held = str(prediction.ticker or "").strip().upper()
    pbm = wallet.get("position_by_market") if isinstance(wallet.get("position_by_market"), dict) else {}
    qty = float(pbm.get(held, 0.0) or 0.0)
    if qty <= 1e-12:
        qty = float(wallet.get("position_qty", 0.0) or 0.0)
        if str(wallet.get("position_symbol") or "").upper() != held:
            qty = 0.0
    if qty <= 1e-12 or not held:
        return "", ""
    ok, reason = whale_panic_should_force_sell(STATE, held)
    if not ok or not can_fire_whale_panic_sell(STATE, held):
        return "", ""
    fallback_px = float(wallet.get("last_price") or 0.0) or float(prediction.latest_close or 0.0)
    live_price = _resolve_execution_price(ticker=held, fallback_price=fallback_px)
    if live_price <= 0:
        live_price = max(fallback_px, 1e-9)
    equity = float(wallet.get("equity", 10000.0) or 10000.0)
    friction = (float(spread_bps) + float(slippage_bps)) / 10000.0
    exec_price = float(live_price) * max(0.0, 1.0 - friction)
    frac = CORE_RISK.full_exit_size_fraction(equity=equity, wallet=wallet, price=float(live_price), market=held)
    insights = STATE.get("news_insights", [])
    top_headlines = [str(x.get("headline", "")) for x in insights[:3] if isinstance(x, dict)]
    rl_note = f"[WHALE-PANIC] {reason}; preventieve MARKET SELL {held}."
    ledger_ctx = "🚨 PANIC: extreme whale exchange inflow burst"
    snap = PAPER_MANAGER.process_signal(
        market=held,
        signal="SELL",
        price=exec_price,
        size_fraction=frac,
        sentiment_score=float(prediction.news_sentiment or 0.0),
        news_headlines=top_headlines,
        ai_thought=rl_note,
        ledger_context=ledger_ctx,
    )
    _sanitize_paper_wallet()
    STATE["paper_portfolio"] = PAPER_MANAGER.wallet
    set_whale_panic_cooldown(STATE, held)
    record_whale_panic_sell_fired(STATE, held)
    base = held.split("-", 1)[0] if "-" in held else held
    try:
        TELEGRAM.send_whale_panic_mode(coin=str(base or held))
    except Exception:
        pass
    if isinstance(snap, dict) and str(snap.get("status", "")) == "closed":
        append_event(
            {
                "ts": datetime.utcnow().isoformat(),
                "type": "whale_panic_sell",
                "ticker": held,
                "signal": "SELL",
                "reason": reason,
            }
        )
        _register_signal_marker(
            ticker=held,
            signal="SELL",
            price=float(live_price),
            expected_return_pct=float(prediction.expected_return_pct or 0.0),
        )
    snap_st = str(snap.get("status", "")) if isinstance(snap, dict) else ""
    print(f"[WHALE-PANIC] Forced exit {held} reason={reason} snap_status={snap_st}")
    return held, f" Whale panic: positie {held} preventief gesloten."


_BITVAVO_CHART_INTERVALS = frozenset({"1m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d"})


def _fetch_history_series(pair: str, lookback_days: int, interval: str = "5m") -> tuple[list[str], list[float]]:
    target = pair.upper()
    if "-" in target:
        # Bitvavo pair history for crypto markets (portal default: 5m voor “levend” beeld).
        iv = str(interval or "5m").strip().lower()
        if iv not in _BITVAVO_CHART_INTERVALS:
            iv = "5m"
        url = f"https://api.bitvavo.com/v2/{target}/candles"
        resp = requests.get(
            url,
            params={"interval": iv, "limit": 1000},
            timeout=15,
        )
        if resp.status_code == 200:
            candles = resp.json()
            if isinstance(candles, list) and candles:
                parsed: list[tuple[int, float]] = []
                for row in candles:
                    if not isinstance(row, list) or len(row) < 5:
                        continue
                    try:
                        ts = int(row[0])
                        close = float(row[4])
                    except Exception:
                        continue
                    parsed.append((ts, close))
                parsed.sort(key=lambda x: x[0])  # Chronological old -> new for stable X-axis.
                labels = [datetime.utcfromtimestamp(ts / 1000).isoformat() for ts, _ in parsed]
                prices = [px for _, px in parsed]
                return labels, prices

    # Fallback: yfinance/pandas (alleen worker / volledige image; portal gebruikt Bitvavo hierboven).
    if _PORTAL_SLIM:
        raise ValueError(f"Geen Bitvavo-candles voor {target}; controleer markt of worker.")
    df = fetch_market_data(ticker=target, lookback_days=lookback_days)
    if "Date" in df.columns:
        df = df.sort_values("Date")
    labels = [str(v) for v in df["Date"].astype(str).values.flatten().tolist()]
    prices = [float(v) for v in df["Close"].values.flatten().tolist()]
    return labels, prices


def _active_market_row(market: str) -> dict[str, Any] | None:
    mku = str(market or "").upper()
    for m in STATE.get("active_markets") or []:
        if isinstance(m, dict) and str(m.get("market", "")).upper() == mku:
            return m
    return None


def _apply_social_overlay_for_decide(last: dict[str, Any], ticker: str) -> tuple[dict[str, Any], str]:
    sm = STATE.get("social_momentum_by_market")
    sm = sm if isinstance(sm, dict) else {}
    return apply_social_overlay_to_rl_row(dict(last), str(ticker).upper(), sm, _active_market_row(ticker))


async def _social_momentum_refresh_loop() -> None:
    interval_sec = max(120, int(os.getenv("SOCIAL_REFRESH_SEC", "300") or 300))
    while True:
        try:
            await asyncio.to_thread(
                refresh_social_momentum_state,
                state=STATE,
                fetch_fresh_news=lambda k, lim: NEWS_ENGINE.fetch_fresh_news(cryptocompare_key=k, limit=int(lim)),
                cryptocompare_key=os.getenv("CRYPTOCOMPARE_KEY"),
                active_markets=list(STATE.get("active_markets") or []),
            )
            elite = [
                str(m.get("market", "")).upper()
                for m in (STATE.get("active_markets") or [])
                if isinstance(m, dict) and m.get("market")
            ][:8]
            await asyncio.to_thread(
                refresh_whale_radar_state,
                state=STATE,
                cryptocompare_key=os.getenv("CRYPTOCOMPARE_KEY"),
                elite_markets=elite,
            )
        except Exception as exc:
            print(f"[SOCIAL] refresh warning: {exc}")
        await asyncio.sleep(interval_sec)


def _rl_chunk_steps_for_pair(pair: str, base_chunk: int) -> int:
    """BTC/ETH krijgen meer timesteps per RL-chunk (betrouwbaardere patronen)."""
    base_chunk = max(512, int(base_chunk))
    if str(pair or "").upper() in {"BTC-EUR", "ETH-EUR"}:
        mult = float(os.getenv("RL_PRIORITY_PAIR_TRAIN_MULT", "1.65") or 1.65)
        return max(512, int(base_chunk * mult))
    return base_chunk


def _refresh_active_markets_cache() -> None:
    prev_rows = [dict(x) for x in (STATE.get("active_markets") or []) if isinstance(x, dict) and x.get("market")]
    old_elite = [str(m.get("market", "")).upper() for m in prev_rows][:8]

    markets = MARKET_SCANNER.fetch_active_pairs()
    scanner_selected: list[dict[str, Any]] = []
    if str(os.getenv("DYNAMIC_SCANNER_ENABLED", "1")).strip().lower() in {"1", "true", "yes", "on"}:
        try:
            scan = VOL_SCANNER.scan(
                top_volume_count=int(os.getenv("SCANNER_TOP_VOLUME", "30") or 30),
                elite_count=int(os.getenv("SCANNER_ELITE_COUNT", "8") or 8),
            )
            scanner_selected = list(scan.selected_top)
            by_full = {str(m.get("market", "")).upper(): dict(m) for m in markets if m.get("market")}
            merged: list[dict[str, Any]] = []
            for r in scanner_selected:
                mk = str(r.get("market", "")).upper()
                if not mk:
                    continue
                if mk in by_full:
                    row = dict(by_full[mk])
                else:
                    base_q = mk.split("-", 1)
                    row = {
                        "market": mk,
                        "base": base_q[0],
                        "quote": base_q[1] if len(base_q) > 1 else "EUR",
                        "status": "trading",
                        "volume_quote_24h": round(float(r.get("volume_quote_24h", 0.0) or 0.0), 2),
                        "last_price": float(r.get("last_price", 0.0) or 0.0),
                        "price_change_pct_24h": float(r.get("price_change_pct_24h", 0.0) or 0.0),
                    }
                row["is_pillar"] = bool(r.get("is_pillar", False))
                row["pillar_kind"] = str(r.get("pillar_kind", "mover"))
                if r.get("move_pct_4h") is not None:
                    row["move_pct_4h"] = r.get("move_pct_4h")
                if r.get("selection_reason") is not None:
                    row["selection_reason"] = str(r.get("selection_reason") or "")
                if r.get("quality_score") is not None:
                    row["quality_score"] = int(r.get("quality_score") or 0)
                if r.get("momentum_30d_pct") is not None:
                    row["momentum_30d_pct"] = float(r.get("momentum_30d_pct") or 0.0)
                if r.get("passes_quality") is not None:
                    row["passes_quality"] = bool(r.get("passes_quality"))
                merged.append(row)
            markets = merged
        except Exception as exc:
            print(f"[SCANNER] dynamic scan warning: {exc}")
    elif CONFIG_TICKERS:
        wanted = set(CONFIG_TICKERS)
        markets = [m for m in markets if str(m.get("market", "")).upper() in wanted]
        order = {ticker: idx for idx, ticker in enumerate(CONFIG_TICKERS)}
        markets.sort(key=lambda m: order.get(str(m.get("market", "")).upper(), 9999))
        if not markets:
            markets = [
                {
                    "market": ticker,
                    "base": ticker.split("-", 1)[0] if "-" in ticker else ticker,
                    "quote": ticker.split("-", 1)[1] if "-" in ticker else "EUR",
                    "status": "config",
                    "volume_quote_24h": 0.0,
                    "last_price": 0.0,
                    "price_change_pct_24h": 0.0,
                }
                for ticker in CONFIG_TICKERS
            ]
    new_elite = [str(m.get("market", "")).upper() for m in markets if m.get("market")][:8]
    if old_elite and new_elite and old_elite != new_elite:
        feed = STATE.setdefault("scanner_intel_feed", [])
        if not isinstance(feed, list):
            feed = []
            STATE["scanner_intel_feed"] = feed
        for i, nm in enumerate(new_elite):
            om = old_elite[i] if i < len(old_elite) else None
            if not om or om == nm:
                continue
            base_n = _coin_from_ticker(nm)
            base_o = _coin_from_ticker(om)
            reason_n = _selection_reason_for_market(markets, nm)
            msg = (
                f"Scanner: [{base_n}] replaced [{base_o}] — "
                f"{reason_n or 'higher momentum / Elite-8 rotation'}."
            )
            ts = datetime.now(UTC).isoformat()
            feed.append(
                {
                    "headline": msg,
                    "title": msg,
                    "text": msg,
                    "summary": "",
                    "url": "",
                    "source": "Scanner",
                    "published_at": ts,
                    "publishedAt": ts,
                    "coin": base_n,
                    "sentiment": 0.0,
                    "is_urgent": True,
                    "is_scanner_stub": True,
                }
            )
        STATE["scanner_intel_feed"] = feed[-80:]

    STATE["active_markets"] = markets
    STATE["scanner_selected"] = scanner_selected
    if markets and not any(m["market"] == STATE.get("selected_market") for m in markets):
        STATE["selected_market"] = markets[0]["market"]


_rl_bg_training_logged = False


async def _rl_background_training_loop() -> None:
    """Periodieke PPO-updates (SB3 learn in worker threads) + uurlijks model-checkpoint. Zie RL_BG_TRAIN_CONCURRENCY / RL_PPO_DEVICE."""
    import time

    global _rl_bg_training_logged
    last_hourly_save = 0.0
    first_iter = True
    while True:
        enabled = str(os.getenv("RL_BACKGROUND_TRAIN", "0")).strip().lower() in ("1", "true", "yes", "on")
        if not enabled:
            await asyncio.sleep(30)
            continue
        if not _rl_bg_training_logged:
            try:
                conc0 = max(1, int(os.getenv("RL_BG_TRAIN_CONCURRENCY", "1") or 1))
                print(
                    f"Training loop started | SB3 device={get_rl_ppo_device()} | "
                    f"RL_BG_TRAIN_CONCURRENCY={conc0}"
                )
            except Exception as exc:
                print(f"Training loop start device-check mislukt: {exc}")
            _rl_bg_training_logged = True
        interval_sec = max(30, int(os.getenv("RL_TRAIN_INTERVAL_SEC", "120") or 120))
        if not first_iter:
            await asyncio.sleep(interval_sec)
        first_iter = False
        active_pairs = [str(m.get("market", "")).upper() for m in (STATE.get("active_markets") or []) if m.get("market")]
        pair = str(STATE.get("selected_market") or os.getenv("DEFAULT_TICKER", "BTC-EUR")).upper()
        targets = active_pairs[:8] if active_pairs else [pair]
        lookback = int(STATE.get("lookback_days", 400) or 400)
        chunk = max(512, int(os.getenv("RL_TRAIN_CHUNK_STEPS", "1000") or 1000))
        cmc = STATE.get("cmc_metrics") if isinstance(STATE.get("cmc_metrics"), dict) else None
        try:
            conc = max(1, int(os.getenv("RL_BG_TRAIN_CONCURRENCY", "1") or 1))
            sem = asyncio.Semaphore(conc)

            async def _one_pair(tp: str) -> None:
                async with sem:
                    await asyncio.to_thread(
                        RL_AGENT.online_update,
                        tp,
                        max(1, int(lookback // 16) or 1),
                        _rl_chunk_steps_for_pair(tp, chunk),
                        cmc,
                    )

            await asyncio.gather(*[_one_pair(tp) for tp in targets])
            print(
                "[RL-BG] Parallel PPO chunk OK | Elite-"
                f"{len(targets)} markets (not UI-only): "
                f"{', '.join(targets)} | chunk_steps≈{chunk} (pair-weighted for BTC/ETH)."
            )
        except Exception as exc:
            print(f"[RL-BG] Training chunk mislukt: {exc}")
        now = time.time()
        if now - last_hourly_save >= float(os.getenv("RL_CHECKPOINT_INTERVAL_SEC", "3600") or 3600):
            try:
                await asyncio.to_thread(RL_AGENT.save_hourly_checkpoint, pair)
            except Exception as exc:
                print(f"[RL-BG] Checkpoint mislukt: {exc}")
            last_hourly_save = now


async def _scanner_hourly_refresh_loop() -> None:
    interval_sec = max(900, int(os.getenv("SCANNER_INTERVAL_SEC", "3600") or 3600))
    while True:
        try:
            _refresh_active_markets_cache()
            selected = [
                {
                    "market": str(r.get("market", "")),
                    "reason": str(r.get("selection_reason", "")),
                    "quality_score": int(r.get("quality_score", 0) or 0),
                }
                for r in (STATE.get("scanner_selected") or [])
                if r.get("market")
            ]
            print(f"[SCANNER] Scanner update: New Elite-8 selected: {selected}")
        except Exception as exc:
            print(f"[SCANNER] hourly refresh warning: {exc}")
        await asyncio.sleep(interval_sec)


async def _rl_multi_inference_loop() -> None:
    interval_sec = max(20, int(os.getenv("RL_MULTI_INFER_INTERVAL_SEC", "60") or 60))
    while True:
        try:
            active_pairs = [
                str(m.get("market", "")).upper()
                for m in (STATE.get("active_markets") or [])
                if m.get("market")
            ][:8]
            if not active_pairs:
                await asyncio.sleep(interval_sec)
                continue

            async def infer_pair(tp: str) -> tuple[str, dict[str, Any]]:
                end_dt = datetime.now(UTC)
                start_dt = end_dt - timedelta(days=max(30, int(STATE.get("lookback_days", 400) or 400)))
                candles = await asyncio.to_thread(
                    fetch_bitvavo_historical_candles,
                    tp,
                    "1h",
                    start_dt,
                    end_dt,
                )
                rl_frame = await asyncio.to_thread(
                    build_rl_training_frame,
                    candles,
                    tp,
                    "crypto",
                    os.getenv("CRYPTOCOMPARE_KEY"),
                    os.getenv("CRYPTOCOMPARE_KEY"),
                    _refresh_cmc_metrics(),
                )
                last = rl_frame.iloc[-1].to_dict()
                last, _social_note = _apply_social_overlay_for_decide(last, tp)
                acct = STATE.get("paper_portfolio", {})
                equity = float(acct.get("equity", 10000.0) or 10000.0)
                cash = float(acct.get("cash", equity) or equity)
                _tc2 = build_trade_decision_context(tp, STATE)
                
                # Bereken time-in-market (in uren) voor de actieve positie in deze specifieke markt
                position_hours = 0.0
                open_lots = acct.get("open_lots_by_market", {}).get(tp, [])
                if isinstance(open_lots, list) and open_lots:
                    try:
                        open_ts = str(open_lots[0].get("open_time_utc") or "")
                        if open_ts:
                            dt_open = datetime.fromisoformat(open_ts.replace("Z", "+00:00"))
                            if dt_open.tzinfo is None:
                                dt_open = dt_open.replace(tzinfo=UTC)
                            position_hours = (datetime.now(UTC) - dt_open).total_seconds() / 3600.0
                    except Exception:
                        pass

                decision = await asyncio.to_thread(
                    lambda: TRADER.decide(
                        {**last, "ticker_id": float(abs(hash(tp)) % 1000) / 1000.0, "time_in_market_h": round(position_hours, 2)},
                        {
                            "balance_ratio": cash / max(1.0, equity),
                            "position": float(acct.get("position_qty", 0.0) or 0.0),
                            "pnl_ratio": float(acct.get("realized_pnl_eur", 0.0) or 0.0) / max(1.0, equity),
                            "trade_ratio": float(acct.get("trades_count", 0.0) or 0.0) / 10000.0,
                            "position_hours": round(position_hours, 2),
                        },
                        _tc2,
                    )
                )
                return tp, decision

            results = await asyncio.gather(*[infer_pair(tp) for tp in active_pairs], return_exceptions=True)
            out: dict[str, Any] = {}
            for item in results:
                if isinstance(item, Exception):
                    continue
                tp, decision = item
                out[tp] = decision
            STATE["rl_multi_decisions"] = out
            shared = STATE.get("rl_shared_buffer")
            if not isinstance(shared, list):
                shared = []
            ts = datetime.now(UTC).isoformat()
            for tp, decision in out.items():
                shared.append(
                    {
                        "ts": ts,
                        "market": tp,
                        "action": str(decision.get("action") or ""),
                        "confidence": float(decision.get("confidence") or 0.0),
                    }
                )
            STATE["rl_shared_buffer"] = shared[-1000:]
        except Exception as exc:
            print(f"[RL-MULTI] inference warning: {exc}")
        await asyncio.sleep(interval_sec)


async def _audit_engine_loop() -> None:
    global AUDIT_LAST_RUN, AUDIT_LAST_TUNING, AUDIT_REFLECTIONS, CORE_RISK
    interval_sec = max(900, int(os.getenv("AUDIT_INTERVAL_SEC", "3600") or 3600))
    while True:
        try:
            elite_markets = [str(m.get("market", "")).upper() for m in (STATE.get("active_markets") or []) if m.get("market")][:8]
            metrics = PAPER_MANAGER.elite8_audit_metrics(elite_markets=elite_markets, window_hours=24)
            mean_pf = sum(float(v.get("profit_factor", 0.0)) for v in metrics.values()) / max(1, len(metrics))
            mean_wr = sum(float(v.get("win_rate", 0.0)) for v in metrics.values()) / max(1, len(metrics))
            old_threshold = float(STATE.get("decision_threshold", float(os.getenv("RL_ACTION_MIN_CONFIDENCE", "0.55") or 0.55)))
            old_sl = float(STATE.get("stop_loss_pct", float(os.getenv("RISK_STOP_LOSS_PCT", "2.5") or 2.5)))
            delta = 0.05
            if mean_pf >= 1.2 and mean_wr >= 52.0:
                new_threshold = max(0.4, old_threshold - delta)
                new_sl = max(0.5, old_sl - delta)
                reason = "Strong 24h edge (PF/WR) -> more aggressive execution."
            else:
                new_threshold = min(0.95, old_threshold + delta)
                new_sl = min(10.0, old_sl + delta)
                reason = "Weak 24h edge (PF/WR) -> tighter entry and wider protection."
            STATE["decision_threshold"] = round(new_threshold, 4)
            STATE["stop_loss_pct"] = round(new_sl, 4)
            os.environ["RISK_STOP_LOSS_PCT"] = f"{new_sl:.4f}"
            CORE_RISK = CoreRiskManager()
            AUDIT_LAST_RUN = datetime.now(UTC).isoformat()
            AUDIT_LAST_TUNING = {
                "old_decision_threshold": old_threshold,
                "new_decision_threshold": new_threshold,
                "old_stop_loss_pct": old_sl,
                "new_stop_loss_pct": new_sl,
                "reason": reason,
                "per_market": metrics,
            }
            AUDIT_REFLECTIONS.append(
                {
                    "ts": AUDIT_LAST_RUN,
                    "reason": reason,
                    "decision_threshold": round(new_threshold, 4),
                    "stop_loss_pct": round(new_sl, 4),
                }
            )
            AUDIT_REFLECTIONS = AUDIT_REFLECTIONS[-100:]
            print(f"[AUDIT] hourly self-reflection tuned decision_threshold={new_threshold:.2f} stop_loss_pct={new_sl:.2f}")
        except Exception as exc:
            print(f"[AUDIT] engine warning: {exc}")
        await asyncio.sleep(interval_sec)


async def _daily_auto_calibration_loop() -> None:
    interval_sec = max(3600, int(os.getenv("AUTO_CALIBRATION_INTERVAL_SEC", "86400") or 86400))
    while True:
        try:
            elite_markets = [str(m.get("market", "")).upper() for m in (STATE.get("active_markets") or []) if m.get("market")][:8]
            metrics = PAPER_MANAGER.elite8_audit_metrics(elite_markets=elite_markets, window_hours=24)
            if metrics:
                avg_wr = sum(float(v.get("win_rate", 0.0)) for v in metrics.values()) / max(1, len(metrics))
                avg_pf = sum(float(v.get("profit_factor", 0.0)) for v in metrics.values()) / max(1, len(metrics))
                old_threshold = float(STATE.get("decision_threshold", float(os.getenv("RL_ACTION_MIN_CONFIDENCE", "0.55") or 0.55)))
                old_sl = float(STATE.get("stop_loss_pct", float(os.getenv("RISK_STOP_LOSS_PCT", "2.5") or 2.5)))
                dt = 0.05 if (avg_pf < 1.0 or avg_wr < 48.0) else (-0.05 if (avg_pf > 1.25 and avg_wr > 55.0) else 0.0)
                dsl = -0.05 if (avg_pf < 1.0 or avg_wr < 48.0) else (0.05 if (avg_pf > 1.25 and avg_wr > 55.0) else 0.0)
                new_threshold = max(0.50, min(0.95, old_threshold + dt))
                new_sl = max(0.5, min(5.0, old_sl + dsl))
                STATE["decision_threshold"] = round(new_threshold, 4)
                STATE["stop_loss_pct"] = round(new_sl, 4)
                print(
                    f"[AUTO-CAL] 24h calibrated decision_threshold={new_threshold:.2f} "
                    f"stop_loss_pct={new_sl:.2f} (pf={avg_pf:.2f}, wr={avg_wr:.2f}%)"
                )
        except Exception as exc:
            print(f"[AUTO-CAL] warning: {exc}")
        await asyncio.sleep(interval_sec)


async def _health_watchdog_loop() -> None:
    while True:
        try:
            now = datetime.now(UTC)
            lec = STATE.get("last_engine_cycle", {}) if isinstance(STATE.get("last_engine_cycle"), dict) else {}
            engine_ts = str(lec.get("ts") or "")
            ws_ts = str(STATE.get("last_ws_heartbeat_ts") or engine_ts or "")
            api_fail_streak = int(STATE.get("api_fail_streak", 0) or 0)

            def _age_seconds(iso_ts: str) -> float:
                if not iso_ts:
                    return 1e9
                dt = datetime.fromisoformat(iso_ts.replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=UTC)
                return (now - dt.astimezone(UTC)).total_seconds()

            engine_age = _age_seconds(engine_ts)
            ws_age = _age_seconds(ws_ts)
            ws_connections = int(STATE.get("ws_connections", 0) or 0)
            ws_stalled = ws_connections > 0 and ws_age > WATCHDOG_STALL_LIMIT_SEC
            if engine_age > WATCHDOG_STALL_LIMIT_SEC or ws_stalled:
                _maybe_send_watchdog_recovery_telegram(engine_age, ws_age)
                print("[WATCHDOG] stall detected; forcing process restart for auto-recovery.")
                os._exit(1)

            if api_fail_streak >= 3:
                _maybe_send_urgent_alert(
                    key="bitvavo_api_fail",
                    subject="Bitvavo API definitive failure",
                    details=f"Consecutive API/engine failures detected (streak={api_fail_streak}).",
                    cooldown_minutes=60,
                )

            wallet = PAPER_MANAGER.wallet if isinstance(PAPER_MANAGER.wallet, dict) else {}
            equity = float(wallet.get("equity", PAPER_MANAGER.config.starting_balance_eur) or PAPER_MANAGER.config.starting_balance_eur)
            start = float(PAPER_MANAGER.config.starting_balance_eur or 10000.0)
            if start > 0 and equity <= start * 0.97:
                _maybe_send_urgent_alert(
                    key="daily_stop_loss_3pct",
                    subject="Daily stop-loss reached",
                    details=f"Equity dropped to {equity:.2f} EUR; below 3% daily loss threshold from start {start:.2f} EUR.",
                    cooldown_minutes=180,
                )
        except Exception as exc:
            print(f"[WATCHDOG] warning: {exc}")
        await asyncio.sleep(10)


async def _autonomous_improvement_loop() -> None:
    global AUTO_OPT_LAST_RUN, AUTO_OPT_LAST_TUNING, AUTO_OPT_SCORE_HISTORY, AUTO_OPT_BEST_SETTINGS
    interval_sec = max(600, int(os.getenv("AUTO_OPT_INTERVAL_SEC", "3600") or 3600))
    while True:
        try:
            elite_markets = [
                str(m.get("market", "")).upper()
                for m in (STATE.get("active_markets") or [])
                if m.get("market")
            ][:8]
            metrics = PAPER_MANAGER.elite8_audit_metrics(elite_markets=elite_markets, window_hours=24)
            mean_pf = sum(float(v.get("profit_factor", 0.0)) for v in metrics.values()) / max(1, len(metrics))
            mean_wr = sum(float(v.get("win_rate", 0.0)) for v in metrics.values()) / max(1, len(metrics))
            score = float(mean_pf * 100.0 + mean_wr)
            AUTO_OPT_SCORE_HISTORY.append(score)
            AUTO_OPT_SCORE_HISTORY = AUTO_OPT_SCORE_HISTORY[-20:]

            old_eps = max(MIN_EXPLORATION_EPS, float(os.getenv("RL_EXPLORATION_FINAL_EPS", "0.05") or 0.05))
            old_risk_cap = float(os.getenv("RISK_MAX_PER_ASSET_TRADE_PCT", "20") or 20.0)
            old_train_steps = int(os.getenv("RL_TRAIN_CHUNK_STEPS", "1000") or 1000)

            new_eps = old_eps
            new_risk_cap = old_risk_cap
            new_train_steps = old_train_steps
            reason = "Stable regime: keep autonomous optimizer settings."

            if mean_pf < 1.0 or mean_wr < 48.0:
                new_eps = min(0.35, old_eps + 0.02)
                new_risk_cap = max(5.0, old_risk_cap - 1.0)
                new_train_steps = min(5000, old_train_steps + 250)
                reason = "Underperforming 24h edge: increase exploration, reduce risk cap, increase training chunk."
            elif mean_pf > 1.3 and mean_wr > 55.0:
                new_eps = max(MIN_EXPLORATION_EPS, old_eps - 0.01)
                new_risk_cap = min(35.0, old_risk_cap + 1.0)
                new_train_steps = max(800, old_train_steps - 100)
                reason = "Strong 24h edge: reduce exploration, relax risk cap slightly, lower training overhead."

            new_eps = max(MIN_EXPLORATION_EPS, float(new_eps))
            os.environ["RL_EXPLORATION_FINAL_EPS"] = f"{new_eps:.4f}"
            os.environ["RISK_MAX_PER_ASSET_TRADE_PCT"] = f"{new_risk_cap:.2f}"
            os.environ["RL_TRAIN_CHUNK_STEPS"] = str(int(new_train_steps))
            STATE["auto_opt_exploration_eps"] = round(new_eps, 4)
            STATE["auto_opt_risk_cap_pct"] = round(new_risk_cap, 2)
            STATE["auto_opt_train_chunk_steps"] = int(new_train_steps)

            # Opportunistic micro-finetune when regime is weak (BTC/ETH eerst + hogere step-count).
            if (mean_pf < 1.0 or mean_wr < 48.0) and elite_markets:
                boost_steps = max(600, int(new_train_steps // 2))
                cmc = _refresh_cmc_metrics()
                prio = ("BTC-EUR", "ETH-EUR")
                ordered_boost = [m for m in prio if m in elite_markets]
                for m in elite_markets:
                    if m not in ordered_boost:
                        ordered_boost.append(m)
                boost_targets = ordered_boost[:4]
                await asyncio.gather(
                    *[
                        asyncio.to_thread(
                            RL_AGENT.online_update,
                            mk,
                            30,
                            _rl_chunk_steps_for_pair(mk, boost_steps),
                            cmc,
                        )
                        for mk in boost_targets
                    ],
                    return_exceptions=True,
                )

            degraded_two_cycles = (
                len(AUTO_OPT_SCORE_HISTORY) >= 3
                and AUTO_OPT_SCORE_HISTORY[-1] < AUTO_OPT_SCORE_HISTORY[-2] < AUTO_OPT_SCORE_HISTORY[-3]
            )
            best_score = float(AUTO_OPT_BEST_SETTINGS.get("score", -1e18))
            if score > best_score:
                AUTO_OPT_BEST_SETTINGS = {
                    "score": score,
                    "exploration_eps": new_eps,
                    "risk_cap_pct": new_risk_cap,
                    "train_chunk_steps": new_train_steps,
                }
            elif degraded_two_cycles and AUTO_OPT_BEST_SETTINGS:
                # Guardrail rollback: revert to best-known stable optimizer settings.
                new_eps = float(AUTO_OPT_BEST_SETTINGS.get("exploration_eps", new_eps) or new_eps)
                new_risk_cap = float(AUTO_OPT_BEST_SETTINGS.get("risk_cap_pct", new_risk_cap) or new_risk_cap)
                new_train_steps = int(AUTO_OPT_BEST_SETTINGS.get("train_chunk_steps", new_train_steps) or new_train_steps)
                new_eps = max(MIN_EXPLORATION_EPS, float(new_eps))
                os.environ["RL_EXPLORATION_FINAL_EPS"] = f"{new_eps:.4f}"
                os.environ["RISK_MAX_PER_ASSET_TRADE_PCT"] = f"{new_risk_cap:.2f}"
                os.environ["RL_TRAIN_CHUNK_STEPS"] = str(int(new_train_steps))
                STATE["auto_opt_exploration_eps"] = round(new_eps, 4)
                STATE["auto_opt_risk_cap_pct"] = round(new_risk_cap, 2)
                STATE["auto_opt_train_chunk_steps"] = int(new_train_steps)
                reason = f"{reason} Rollback applied to best-known settings after 2-cycle degradation."

            AUTO_OPT_LAST_RUN = datetime.now(UTC).isoformat()
            AUTO_OPT_LAST_TUNING = {
                "reason": reason,
                "score": round(score, 4),
                "mean_profit_factor": round(mean_pf, 4),
                "mean_win_rate": round(mean_wr, 4),
                "old_exploration_eps": old_eps,
                "new_exploration_eps": new_eps,
                "old_risk_cap_pct": old_risk_cap,
                "new_risk_cap_pct": new_risk_cap,
                "old_train_chunk_steps": old_train_steps,
                "new_train_chunk_steps": new_train_steps,
                "history_tail": [round(v, 4) for v in AUTO_OPT_SCORE_HISTORY[-5:]],
                "best_settings": AUTO_OPT_BEST_SETTINGS,
            }
            PAPER_MANAGER.save_optimizer_state(
                {
                    "ts": AUTO_OPT_LAST_RUN,
                    "last_tuning": AUTO_OPT_LAST_TUNING,
                    "best_settings": AUTO_OPT_BEST_SETTINGS,
                    "score_history": AUTO_OPT_SCORE_HISTORY[-20:],
                }
            )
            print(
                "[AUTO-OPT] "
                f"eps={new_eps:.3f} risk_cap={new_risk_cap:.1f}% train_chunk={new_train_steps} "
                f"pf={mean_pf:.3f} wr={mean_wr:.2f}%"
            )
        except Exception as exc:
            print(f"[AUTO-OPT] warning: {exc}")
        await asyncio.sleep(interval_sec)


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


async def _predict_queue_worker() -> None:
    while True:
        ticker, lookback_days, tenant_id, fut = await PREDICT_QUEUE.get()
        try:
            set_current_tenant(tenant_id)
            res = await asyncio.to_thread(generate_prediction, ticker, lookback_days)
            if not fut.done():
                fut.set_result(res)
        except Exception as exc:
            if not fut.done():
                fut.set_exception(exc)
        finally:
            PREDICT_QUEUE.task_done()


async def _paper_run_queue_worker() -> None:
    while True:
        ticker, lookback_days, tenant_id, fut = await PAPER_RUN_QUEUE.get()
        try:
            set_current_tenant(tenant_id)
            res = await asyncio.to_thread(_run_paper_cycle_sync, ticker, lookback_days)
            if not fut.done():
                fut.set_result(res)
        except Exception as exc:
            if not fut.done():
                fut.set_exception(exc)
        finally:
            PAPER_RUN_QUEUE.task_done()


def _bootstrap_state_after_markets_refresh() -> None:
    """Vul STATE vanuit paper-wallet + env; gedeeld door portal- en full-startup."""
    _sanitize_paper_wallet()
    STATE["paper_portfolio"] = PAPER_MANAGER.wallet
    STATE["bot_status"] = "running"
    STATE["last_ws_heartbeat_ts"] = datetime.now(UTC).isoformat()
    STATE["last_engine_cycle"] = {"ok": True, "ts": datetime.now(UTC).isoformat(), "bootstrap": True}
    restored_opt = PAPER_MANAGER.load_latest_optimizer_state()
    if isinstance(restored_opt, dict) and restored_opt:
        _best = restored_opt.get("best_settings") if isinstance(restored_opt.get("best_settings"), dict) else {}
        _hist = restored_opt.get("score_history") if isinstance(restored_opt.get("score_history"), list) else []
        if _best:
            try:
                eps_best = max(
                    MIN_EXPLORATION_EPS,
                    float(_best.get("exploration_eps", os.getenv("RL_EXPLORATION_FINAL_EPS", "0.05")) or 0.05),
                )
                os.environ["RL_EXPLORATION_FINAL_EPS"] = f"{eps_best:.4f}"
                os.environ["RISK_MAX_PER_ASSET_TRADE_PCT"] = f"{float(_best.get('risk_cap_pct', os.getenv('RISK_MAX_PER_ASSET_TRADE_PCT', '20'))):.2f}"
                os.environ["RL_TRAIN_CHUNK_STEPS"] = str(int(_best.get("train_chunk_steps", os.getenv("RL_TRAIN_CHUNK_STEPS", "1000"))))
                STATE["auto_opt_exploration_eps"] = eps_best
                STATE["auto_opt_risk_cap_pct"] = float(_best.get("risk_cap_pct", 20.0))
                STATE["auto_opt_train_chunk_steps"] = int(_best.get("train_chunk_steps", 1000))
            except Exception:
                pass
        if _hist:
            try:
                globals()["AUTO_OPT_SCORE_HISTORY"] = [float(x) for x in _hist][-20:]
            except Exception:
                pass
        try:
            globals()["AUTO_OPT_BEST_SETTINGS"] = {
                "score": float(_best.get("score", -1e18)) if _best else -1e18,
                "exploration_eps": float(_best.get("exploration_eps", 0.05)) if _best else 0.05,
                "risk_cap_pct": float(_best.get("risk_cap_pct", 20.0)) if _best else 20.0,
                "train_chunk_steps": int(_best.get("train_chunk_steps", 1000)) if _best else 1000,
            }
        except Exception:
            pass
    restored_market = str((PAPER_MANAGER.wallet if isinstance(PAPER_MANAGER.wallet, dict) else {}).get("position_symbol") or "")
    if restored_market:
        STATE["selected_market"] = restored_market.upper()
    STATE["lookback_days"] = int(os.getenv("LOOKBACK_DAYS", "400"))
    STATE["episode_train_steps"] = int(os.getenv("RL_EPISODE_TRAIN_STEPS", "3000"))
    STATE["cryptocompare_key"] = os.getenv("CRYPTOCOMPARE_KEY")
    STATE["cmc_api_key"] = os.getenv("COINMARKETCAP_KEY") or os.getenv("CMC_API_KEY")


def _blocking_startup_network_phase() -> None:
    """Scanner + social/whale + macro/CMC in één worker-thread (blokkeert de asyncio-loop niet)."""
    try:
        if not (STATE.get("active_markets") or []):
            _refresh_active_markets_cache()
    except Exception:
        STATE["active_markets"] = []
    try:
        refresh_social_momentum_state(
            state=STATE,
            fetch_fresh_news=lambda k, lim: NEWS_ENGINE.fetch_fresh_news(cryptocompare_key=k, limit=int(lim)),
            cryptocompare_key=os.getenv("CRYPTOCOMPARE_KEY"),
            active_markets=list(STATE.get("active_markets") or []),
        )
        elite0 = [
            str(m.get("market", "")).upper()
            for m in (STATE.get("active_markets") or [])
            if isinstance(m, dict) and m.get("market")
        ][:8]
        refresh_whale_radar_state(
            state=STATE,
            cryptocompare_key=os.getenv("CRYPTOCOMPARE_KEY"),
            elite_markets=elite0,
        )
    except Exception:
        pass
    try:
        STATE["macro_context"] = MACRO_CALENDAR.fetch_today_macro_context()
    except Exception:
        pass
    try:
        STATE["fear_greed"] = FEAR_GREED.fetch_index()
    except Exception:
        pass
    try:
        STATE["whale_watch"] = WHALE_WATCHER.fetch_exchange_pressure(
            api_key=os.getenv("CRYPTOCOMPARE_KEY"),
            lookback_minutes=60,
        )
    except Exception:
        STATE.setdefault("whale_watch", {"whale_pressure": 0.0})
    _refresh_cmc_metrics(force=True)


async def _background_startup_network_bundle() -> None:
    """Draait na server-bind: portal/health eerst bereikbaar; STATE vult daarna aan."""
    try:
        await asyncio.to_thread(_blocking_startup_network_phase)
    except Exception as exc:
        print(f"[STARTUP] Network/scanner bundle mislukt: {exc}")
    _append_bot_log_line("[WHALE-SYNC] Data nu via CryptoCompare feed.")
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
    except Exception:
        cuda_available = False
    wallet = PAPER_MANAGER.wallet if isinstance(PAPER_MANAGER.wallet, dict) else {}
    cash = float(wallet.get("cash", PAPER_MANAGER.config.starting_balance_eur) or PAPER_MANAGER.config.starting_balance_eur)
    equity = float(
        wallet.get("equity", PAPER_MANAGER.config.starting_balance_eur) or PAPER_MANAGER.config.starting_balance_eur
    )
    # Telegram/SMTP is sync; never block the asyncio loop (avoids ERR_EMPTY_RESPONSE on burst GETs).
    await asyncio.to_thread(
        send_restart_report,
        cash,
        equity,
        cuda_available,
        "startup",
        str(Path.cwd() / "ROADMAP.md"),
        list(STATE.get("scanner_selected") or []),
        dict(AUDIT_LAST_TUNING or {}),
        _compute_yesterday_pnl(),
        _portfolio_distribution_snapshot(),
        allocation_snapshot(wallet, equity),
    )
    if isinstance(STATE.get("scanner_selected"), list):
        print(f"[SCANNER] selected_elite={STATE.get('scanner_selected')}")


async def _worker_portal_snapshot_loop() -> None:
    interval_sec = max(1.0, float(os.getenv("PORTAL_SNAPSHOT_INTERVAL_SEC", "2") or 2))
    while True:
        try:
            wire = _brain_ws_wire_payload()
            extras = {"brain_ws": json.loads(json.dumps(wire, default=str))}
            write_worker_portal_snapshot(extras=extras)
        except Exception as exc:
            print(f"[PORTAL-SNAP] worker write mislukt: {exc}")
        await asyncio.sleep(interval_sec)


async def _worker_system_stats_redis_loop() -> None:
    """Portal heeft geen nvidia-smi: worker publiceert echte GPU/host-schijf (compact WS-shape)."""
    interval = max(1.0, float(os.getenv("WORKER_SYSTEM_STATS_REDIS_SEC", "2") or 2))
    while True:
        try:

            def _publish_once() -> None:
                publish_system_stats_update(compact_system_stats(collect_system_stats()))

            await asyncio.to_thread(_publish_once)
        except Exception as exc:
            print(f"[WORKER] system_stats redis publish: {exc}")
        await asyncio.sleep(interval)


async def _portal_system_stats_redis_subscribe_loop() -> None:
    """Vult STATE['_system_stats_ws_payload'] voor `/ws/system-stats` (worker-metrics)."""
    url = str(os.getenv("REDIS_URL", "") or "").strip()
    if not url:
        print("[PORTAL] system_stats: geen REDIS_URL — hardware-meters = portal-container (geen GPU).")
        return
    while True:
        client = None
        pubsub = None
        try:
            import redis.asyncio as aioredis

            client = aioredis.from_url(url, decode_responses=True)
            async with client.pubsub() as pubsub:
                await pubsub.subscribe(SYSTEM_STATS_CHANNEL)
                print(f"[PORTAL] system_stats: Redis subscribe op {SYSTEM_STATS_CHANNEL}")
                async for message in pubsub.listen():
                    if not isinstance(message, dict):
                        continue
                    if message.get("type") != "message":
                        continue
                    raw = message.get("data")
                    if not isinstance(raw, str) or not raw.strip():
                        continue
                    try:
                        parsed = json.loads(raw)
                        if isinstance(parsed, dict) and parsed.get("t") == "system_stats":
                            STATE["_system_stats_ws_payload"] = parsed
                    except Exception:
                        pass
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            print(f"[PORTAL] system_stats redis-sub: {exc} — retry over 3s")
            await asyncio.sleep(3)
        finally:
            if client is not None:
                try:
                    await client.aclose()
                except Exception:
                    pass


def _system_stats_payload_for_websocket() -> dict[str, Any]:
    if _process_role() == "portal":
        cached = STATE.get("_system_stats_ws_payload")
        if isinstance(cached, dict) and cached.get("t") == "system_stats":
            return dict(cached)
    return compact_system_stats(get_system_stats())


async def _portal_snapshot_poll_loop() -> None:
    interval_sec = max(1.0, float(os.getenv("PORTAL_SNAPSHOT_INTERVAL_SEC", "2") or 2))
    while True:
        try:
            blob = read_worker_portal_snapshot()
            if blob:
                apply_worker_snapshot_to_portal(blob)
        except Exception as exc:
            print(f"[PORTAL-SNAP] portal poll mislukt: {exc}")
        await asyncio.sleep(interval_sec)


async def _run_full_trading_startup() -> None:
    global JARVIS_REPORTER, RESTART_MAIL_TASK
    _genesis_require_gpu_or_raise()
    _startup_cuda_flag = _probe_torch_cuda_available()
    # Markten vóór MAIN_ENGINE.start(): anders opent de UI met lege /markets/active en blijft chart/ticker leeg
    # tot de achtergrond-thread klaar is (race met lange engine-startup).
    try:
        await asyncio.to_thread(_refresh_active_markets_cache)
    except Exception as exc:
        print(f"[STARTUP] Eerste active-markets refresh mislukt: {exc}")
        STATE["active_markets"] = []
    start_background_task(_background_startup_network_bundle())
    _bootstrap_state_after_markets_refresh()
    start_background_task(_background_log_pruner())
    start_background_task(_background_rss_poller())
    start_background_task(_gpu_cuda_heartbeat())
    start_background_task(_predict_queue_worker())
    start_background_task(_paper_run_queue_worker())
    start_background_task(_run_genesis_and_prepare_rl())
    start_background_task(_scanner_hourly_refresh_loop())
    start_background_task(_social_momentum_refresh_loop())
    start_background_task(_rl_multi_inference_loop())
    start_background_task(_audit_engine_loop())
    start_background_task(_health_watchdog_loop())
    start_background_task(_autonomous_improvement_loop())
    start_background_task(_daily_auto_calibration_loop())
    if str(os.getenv("RL_BACKGROUND_TRAIN", "0")).strip().lower() in ("1", "true", "yes", "on"):
        start_background_task(_rl_background_training_loop())
    await MAIN_ENGINE.start()
    TELEGRAM.send_start()
    if RESTART_MAIL_TASK is None or RESTART_MAIL_TASK.done():
        RESTART_MAIL_TASK = asyncio.create_task(
            daily_restart_report_loop(
                payload_provider=lambda: {
                    "cash_eur": float(
                        (PAPER_MANAGER.wallet if isinstance(PAPER_MANAGER.wallet, dict) else {}).get(
                            "cash", PAPER_MANAGER.config.starting_balance_eur
                        )
                        or PAPER_MANAGER.config.starting_balance_eur
                    ),
                    "equity_eur": float(
                        (PAPER_MANAGER.wallet if isinstance(PAPER_MANAGER.wallet, dict) else {}).get(
                            "equity", PAPER_MANAGER.config.starting_balance_eur
                        )
                        or PAPER_MANAGER.config.starting_balance_eur
                    ),
                    "cuda_available": bool(_startup_cuda_flag),
                    "scanner_selected": list(STATE.get("scanner_selected") or []),
                    "ai_reflection": dict(AUDIT_LAST_TUNING or {}),
                    "yesterday_pnl_eur": _compute_yesterday_pnl(),
                    "portfolio_distribution": _portfolio_distribution_snapshot(),
                    "allocation_snapshot": allocation_snapshot(
                        (PAPER_MANAGER.wallet if isinstance(PAPER_MANAGER.wallet, dict) else {}),
                        float(
                            (PAPER_MANAGER.wallet if isinstance(PAPER_MANAGER.wallet, dict) else {}).get(
                                "equity", PAPER_MANAGER.config.starting_balance_eur
                            )
                            or PAPER_MANAGER.config.starting_balance_eur
                        ),
                    ),
                },
                roadmap_path=str(Path.cwd() / "ROADMAP.md"),
            )
        )
    JARVIS_REPORTER = AITradingPerformanceIntegrityReporter(
        telegram_notifier=TELEGRAM,
        get_financials=_jarvis_financials_snapshot,
        get_live_financials=_jarvis_live_financials_snapshot,
        get_system_health=get_system_stats,
        get_recent_trades=lambda limit: PAPER_MANAGER.recent_trades(limit=limit),
        get_started_at=lambda: str(STATE.get("started_at") or ""),
        roadmap_path=str(Path.cwd() / "ROADMAP.md"),
    )
    JARVIS_REPORTER.start()
    if _process_role() == "worker":
        start_background_task(_worker_portal_snapshot_loop())
        start_background_task(_worker_system_stats_redis_loop())


async def _portal_startup_only() -> None:
    _genesis_require_gpu_or_raise()
    try:
        await asyncio.to_thread(_refresh_active_markets_cache)
    except Exception as exc:
        print(f"[STARTUP] Eerste active-markets refresh mislukt: {exc}")
        STATE["active_markets"] = []
    start_background_task(_portal_snapshot_poll_loop())
    start_background_task(_portal_system_stats_redis_subscribe_loop())
    _bootstrap_state_after_markets_refresh()
    # Geen _predict_queue_worker: portal mist ML-stack; /predict → 503.
    # Geen _health_watchdog_loop: die vergelijkt last_engine_cycle met worker-ticks; op portal is STATE
    # pas na snapshot-sync vers en anders triggert os._exit (vals positieve stall + Telegram).
    port = int(os.getenv("PORT", "8000"))
    print(f"[PORTAL] FastAPI op poort {port} — UI-state sync via worker-portal snapshot (storage)")


async def run_trading_worker_forever() -> None:
    """Zonder Uvicorn: alle trading/AI-taken + snapshot voor portal."""
    await _run_full_trading_startup()
    print("[WORKER] Trading/AI-loops actief (geen HTTP); portal-state via worker_portal_snapshot.json")
    await asyncio.Future()

def _publish_trading_redis_activity(kind: str, ticker: str) -> None:
    """Publiceert de actuele status naar de trading_updates Redis channel."""
    try:
        payload = {
            "type": "message",
            "kind": kind,
            "ticker": ticker,
            "last_engine_tick_utc": datetime.now(UTC).isoformat(),
            "last_prediction": STATE.get("last_prediction"),
            "paper_portfolio": STATE.get("paper_portfolio") or PAPER_MANAGER.wallet,
            "last_order": STATE.get("last_order"),
            "fear_greed": STATE.get("fear_greed"),
            "risk_profile": risk_profile_dict(CORE_RISK, PAPER_MANAGER.config.starting_balance_eur),
            "elite_ai_signals": _elite_ai_signals_payload(),
            "allocation_snapshot": allocation_snapshot(
                STATE.get("paper_portfolio") or PAPER_MANAGER.wallet,
                float((STATE.get("paper_portfolio") or PAPER_MANAGER.wallet).get("equity", 10000.0) or 10000.0)
            ),
        }
        publish_trading_update(payload)
    except Exception as exc:
        print(f"[REDIS] Error publishing activity: {exc}")

def _run_paper_cycle_sync(ticker: str, lookback_days: int) -> dict[str, Any]:
    """Synchrone paper run voor background thread."""
    prediction = generate_prediction(ticker, lookback_days)
    STATE["last_prediction"] = prediction.model_dump() if hasattr(prediction, "model_dump") else prediction.dict()
    
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
        market=ticker,
    )
    
    spread_bps, slippage_bps = _orderbook_spread_slippage_bps(
        ticker,
        quote_notional_eur=max(0.0, float(equity) * max(0.0, float(size_frac))),
    )
    if spread_bps <= 0.0:
        spread_bps = _estimate_spread_bps_from_recent_range(ticker)
    if slippage_bps <= 0.0:
        slippage_bps = max(0.0, spread_bps * 0.15)
        
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
            equity=equity, wallet=wallet, price=px, market=ticker
        )
    elif final_signal == "BUY":
        ok, why = CORE_RISK.check_safety(
            signal="BUY",
            market=ticker,
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

    # Dynamische Trailing Stop-Loss check via de RiskManager
    is_hard_exit, exit_reason = CORE_RISK.hard_exit_for_sl_tp(
        market=ticker,
        price=px,
        wallet=wallet,
        current_volatility_pct=spread_bps / 100.0,
    )
    if is_hard_exit:
        final_signal = "SELL"
        final_frac = CORE_RISK.full_exit_size_fraction(equity=equity, wallet=wallet, price=px, market=ticker)
        size_note = f"Forced {exit_reason} (Volatiliteit: {spread_bps/100.0:.2f}%)"
        
        # Telegram Notificatie sturen zodra de Trailing Stop wordt geraakt
        if exit_reason == "hard_trailing_stop":
            _maybe_send_urgent_alert(
                key=f"trailing_stop_{ticker}",
                subject=f"Trailing Stop-Loss Geraakt: {ticker}",
                details=f"De dynamische Trailing Stop-Loss voor {ticker} is geactiveerd op een prijs van €{px:.2f}. "
                        f"Huidige volatiliteit: {spread_bps/100.0:.2f}%. De positie wordt direct gesloten om kapitaal veilig te stellen.",
                cooldown_minutes=60,
            )

    forced_panic_market, rl_note = _paper_whale_panic_intervention(prediction, spread_bps, slippage_bps)
    if forced_panic_market == ticker:
        _publish_trading_redis_activity(kind="paper_cycle", ticker=ticker)
        return {"status": "panic_sold_via_intervention", "market": ticker}

    insights = STATE.get("news_insights", [])
    top_headlines = [str(x.get("headline", "")) for x in insights[:3] if isinstance(x, dict)]

    ai_thought = rl_note or f"Paper run {final_signal}"
    if is_hard_exit:
        ai_thought = f"🛡️ [RISK] {size_note}"

    snap = PAPER_MANAGER.process_signal(
        market=ticker,
        signal=final_signal,
        price=px,
        size_fraction=final_frac,
        sentiment_score=float(prediction.news_sentiment),
        news_headlines=top_headlines,
        ai_thought=ai_thought,
        ledger_context=format_ledger_social_whale_context(ticker, STATE),
    )
    
    _sanitize_paper_wallet()
    STATE["paper_portfolio"] = PAPER_MANAGER.wallet
    
    risk_controls = compute_risk_controls(prediction.latest_close)
    paper_order = build_paper_order(
        signal=final_signal,
        ticker=ticker,
        price=px,
        size_fraction=final_frac,
        budget_eur=equity,
    )
    
    STATE["last_order"] = {
        "cycle_seq": int(time.time()),
        "risk_controls": risk_controls,
        "risk_decision": {
            "approved": risk_decision.approved,
            "reason": risk_decision.reason,
            "spread_bps": round(spread_bps, 3),
            "slippage_bps": round(slippage_bps, 3),
            "max_spread_bps": RISK_MANAGER.max_spread_bps_for_trading,
        },
        "engine_risk": {"sizing_note": size_note, "safety_force_exit": bool(is_hard_exit), "final_signal": final_signal},
        "order": paper_order,
        "snap": {k: v for k, v in snap.items() if k not in ("wallet", "history")} if isinstance(snap, dict) else snap,
    }
    
    _publish_trading_redis_activity(kind="paper_cycle", ticker=ticker)
    return snap



def generate_prediction(ticker: str, lookback_days: int) -> PredictionResponse:
    df = fetch_market_data(ticker, lookback_days)
    close_prices = [float(x) for x in df["Close"].values.flatten().tolist()]

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
    highs = recent_df["High"].values.flatten().tolist()
    lows = recent_df["Low"].values.flatten().tolist()
    closes = recent_df["Close"].values.flatten().tolist()
    if not highs or not lows or not closes:
        return 0.0
    last_high = float(highs[-1])
    last_low = float(lows[-1])
    last_close = float(closes[-1])
    if last_close <= 0:
        return 0.0
    # Proxy voor spread/volatility druk.
    return max(0.0, ((last_high - last_low) / last_close) * 10000.0)


def _orderbook_spread_slippage_bps(ticker: str, quote_notional_eur: float) -> tuple[float, float]:
    market = str(ticker or "").upper()
    if "-" not in market:
        return 0.0, 0.0
    try:
        resp = requests.get(
            "https://api.bitvavo.com/v2/book",
            params={"market": market, "depth": 50},
            timeout=8,
        )
        if resp.status_code != 200:
            return 0.0, 0.0
        data = resp.json()
        bids = data.get("bids", []) if isinstance(data, dict) else []
        asks = data.get("asks", []) if isinstance(data, dict) else []
        if not bids or not asks:
            return 0.0, 0.0
        best_bid = float(bids[0][0])
        best_ask = float(asks[0][0])
        mid = (best_bid + best_ask) / 2.0 if (best_bid > 0 and best_ask > 0) else 0.0
        spread_bps = 0.0 if mid <= 0 else ((best_ask - best_bid) / mid) * 10000.0
        remain = max(0.0, float(quote_notional_eur or 0.0))
        notional_consumed = 0.0
        weighted_px_sum = 0.0
        for row in asks:
            if not isinstance(row, list) or len(row) < 2:
                continue
            px = float(row[0]); qty = float(row[1])
            lvl_notional = px * qty
            take = min(remain, lvl_notional)
            if take <= 0:
                continue
            weighted_px_sum += px * take
            notional_consumed += take
            remain -= take
            if remain <= 1e-9:
                break
        if notional_consumed <= 1e-9:
            return max(0.0, spread_bps), 0.0
        vwap = weighted_px_sum / notional_consumed
        slippage_bps = 0.0 if mid <= 0 else max(0.0, ((vwap - mid) / mid) * 10000.0)
        return max(0.0, spread_bps), max(0.0, slippage_bps)
    except Exception:
        return 0.0, 0.0


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

def _brain_ws_payload() -> dict[str, Any]:
    """Snapshot voor `/ws/brain-stats` (zelfde bronnen als REST brain-endpoints)."""
    decision = STATE.get("rl_last_decision")
    if not isinstance(decision, dict):
        decision = {}
    ld = RL_AGENT.last_decision
    policy_fw = ld.feature_weights if ld is not None else None
    fw_policy = merge_feature_weights_for_brain(decision, policy_fw)
    obs = STATE.get("rl_last_observation")
    obs_d = obs if isinstance(obs, dict) else {}
    fw_bar = bar_values_from_obs_and_weights(fw_policy, obs_d)
    return {
        "topic": "brain_stats",
        "training_monitor": RL_AGENT.training_monitor(),
        "feature_weights": fw_bar,
        "feature_weights_policy": fw_policy,
        "rl_observation": obs_d,
        "social_buzz": STATE.get("social_buzz_summary") if isinstance(STATE.get("social_buzz_summary"), dict) else {},
    }


def _brain_ws_wire_payload() -> dict[str, Any]:
    """Compact brain_stats + Elite-8 lite (7 achtergrondmunten: prijs + status)."""
    if _process_role() == "portal":
        cached = STATE.get("_portal_brain_ws")
        if isinstance(cached, dict) and cached:
            return cached
    focus = str(STATE.get("selected_market") or os.getenv("DEFAULT_TICKER", "BTC-EUR")).upper()
    inner = _brain_ws_payload()
    lite = elite_lite_rows(focus, _elite_ai_signals_payload(), list(STATE.get("active_markets") or []))
    return build_brain_ws_wire_payload(
        focus_market=focus,
        training_monitor=inner.get("training_monitor") or {},
        feature_weights=inner.get("feature_weights") or {},
        feature_weights_policy=inner.get("feature_weights_policy") or {},
        rl_observation=inner.get("rl_observation") or {},
        social_buzz=inner.get("social_buzz") or {},
        lite_elite=lite,
    )
