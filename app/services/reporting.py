"""
Nachtelijk train-/paper-rapport: ochtend 08:00 Europe/Amsterdam (+ catch-up na opstart).
Cold-start: send_initial_report() — kort Telegram + HTML e-mail na eerste main-loop start.
Credentials uitsluitend uit .trading_vault (geen hardcoded secrets).
"""

from __future__ import annotations

import json
import hashlib
import logging
import math
import os
import socket
import sys
import smtplib
import sqlite3
from datetime import date, datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formatdate, make_msgid
from html import escape
from pathlib import Path
from typing import Any

import requests

from app.datetime_util import UTC
from app.services.portfolio_qty import implied_equity_eur_from_wallet, normalize_paper_base_qty

_log = logging.getLogger(__name__)

SKIP_BUY_OPEN_PAIR_LOG = "[SKIP] Trade geweigerd: er staat al een positie open voor {pair}"


def format_skip_buy_open_pair_log(pair: str) -> str:
    """Vaste logregel voor BUY-hard-gate: maximaal één open positie per valutapaar."""
    p = str(pair or "").strip().upper().replace("/", "-")
    return SKIP_BUY_OPEN_PAIR_LOG.format(pair=p)


def watchdog_auto_heal_stall_limit_sec() -> int:
    """
    Zelfde drempel als worker ``WATCHDOG_STALL_LIMIT_SEC`` (Auto-Heal / process-restart).
    Default 1200s = 20 minuten — minder restarts tijdens zware RL-training.
    """
    try:
        return max(60, int(os.getenv("WATCHDOG_STALL_LIMIT_SEC", "1200") or 1200))
    except (TypeError, ValueError):
        return 1200


# NOODSITUATIE crash-loop: alle executive Telegram/e-mail (startup + ochtend) uit tot stack stabiel is.
EXECUTIVE_NOTIFICATIONS_EMERGENCY_MUTE = True


def executive_notifications_muted() -> bool:
    """True = geen startup-/ochtendrapport Telegram of SMTP. Noodmute tot code terugzet; heropen met MUTE_EXECUTIVE_NOTIFICATIONS=0."""
    env = str(os.getenv("MUTE_EXECUTIVE_NOTIFICATIONS", "1")).strip().lower()
    if env in ("0", "false", "no", "off"):
        return False
    if EXECUTIVE_NOTIFICATIONS_EMERGENCY_MUTE:
        return True
    return env in ("1", "true", "yes", "on")


def _portfolio_math_fatal_exit_enabled() -> bool:
    return str(os.getenv("PORTFOLIO_MATH_FATAL_EXIT", "1")).strip().lower() in ("1", "true", "yes", "on")


def _fatal_portfolio_math_exit(detail: str, *, repair_context: dict[str, Any] | None = None) -> None:
    """Voorkomt Docker restart-storm: bij fatale portfolio-wiskunde proces stoppen; schrijft repair_request.json."""
    ctx: dict[str, Any] = dict(repair_context or {})
    ctx.setdefault("detail", detail)
    try:
        from app.services.repair_bridge import write_repair_request_if_enabled

        write_repair_request_if_enabled(
            detail=str(ctx.get("detail", detail)),
            trigger=str(ctx.get("trigger", "portfolio_math_fatal")),
            phase=str(ctx.get("phase", "refresh_portfolio_equity_integrity")),
            wallet=ctx.get("wallet") if isinstance(ctx.get("wallet"), dict) else None,
            reported_equity_eur=ctx.get("reported_equity_eur") if "reported_equity_eur" in ctx else None,
            implied_equity_eur=ctx.get("implied_equity_eur") if "implied_equity_eur" in ctx else None,
            rel_deviation_pct=float(ctx["rel_deviation_pct"]) if "rel_deviation_pct" in ctx else None,
            threshold_pct=float(ctx["threshold_pct"]) if "threshold_pct" in ctx else None,
            allocation_snapshot=ctx.get("allocation_snapshot") if isinstance(ctx.get("allocation_snapshot"), dict) else None,
            exception=ctx.get("exception") if isinstance(ctx.get("exception"), BaseException) else None,
            exception_traceback=ctx.get("exception_traceback") if isinstance(ctx.get("exception_traceback"), str) else None,
            extra_state=ctx.get("extra_state") if isinstance(ctx.get("extra_state"), dict) else None,
        )
    except Exception:
        _log.warning("repair_bridge: write failed", exc_info=True)
    if not _portfolio_math_fatal_exit_enabled():
        return
    _log.error("FATAL portfolio math — stopping process (set PORTFOLIO_MATH_FATAL_EXIT=0 to disable): %s", detail)
    sys.exit(1)


def reporting_coerce_btc_qty_to_btc_base(market: str, qty_raw: float) -> float:
    """
    Dwing BTC-hoeveelheid naar bitcoin-eenheid: ruwe satoshi-cache (≥10⁵) altijd /10⁸.
    """
    try:
        q = float(qty_raw)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(q) or q <= 0:
        return q
    m = str(market or "").strip().upper().replace("/", "-")
    base = m.split("-", 1)[0].strip() if "-" in m else m
    if base in ("BTC", "TBTC") and q >= 100_000.0:
        qn = q / 1e8
        _log.warning("reporting: BTC qty satoshi→BTC /1e8 market=%s raw=%s -> %s", market, qty_raw, qn)
        return qn
    return q


def sanitize_wallet_corrupt_positions_vs_cash(wallet: dict[str, Any]) -> int:
    """
    Bij laden: positie verwijderen als notionele waarde > volledige cash (corrupt / verkeerde eenheid).
    Past wallet in-place aan; retourneert aantal gedropte lots.
    """
    if not isinstance(wallet, dict):
        return 0
    try:
        cash = float(wallet.get("cash", 0.0) or 0.0)
    except (TypeError, ValueError):
        cash = 0.0
    if not math.isfinite(cash) or cash <= 0:
        return 0

    lp = wallet.get("last_prices_by_market") if isinstance(wallet.get("last_prices_by_market"), dict) else {}
    obm = wallet.setdefault("open_lots_by_market", {})
    if not isinstance(obm, dict):
        wallet["open_lots_by_market"] = {}
        obm = wallet["open_lots_by_market"]

    pos_sym = str(wallet.get("position_symbol") or "").strip().upper()
    try:
        last_px = float(wallet.get("last_price") or 0.0)
    except (TypeError, ValueError):
        last_px = 0.0

    removed = 0
    for mkt, lots in list(obm.items()):
        if not isinstance(lots, list):
            continue
        mku = str(mkt).strip().upper()
        keep: list[dict[str, Any]] = []
        for lot in lots:
            if not isinstance(lot, dict):
                continue
            try:
                rq = float(lot.get("qty") or 0.0)
            except (TypeError, ValueError):
                continue
            rq = reporting_coerce_btc_qty_to_btc_base(mku, rq)
            try:
                ep = float(lot.get("entry_price") or 0.0)
            except (TypeError, ValueError):
                ep = 0.0
            px = ep if ep > 0.0 else float(lp.get(mku, 0.0) or 0.0)
            if px <= 0.0 and mku == pos_sym and last_px > 0:
                px = last_px
            notional = abs(rq) * px if px > 0.0 and rq != 0.0 else 0.0
            if notional > cash + 1e-6:
                _log.error(
                    "[SANITIZE] corrupt lot dropped: notional %.2f EUR > cash %.2f | market=%s qty=%s px=%s",
                    notional,
                    cash,
                    mku,
                    rq,
                    px,
                )
                removed += 1
                continue
            lot["qty"] = rq
            keep.append(lot)
        obm[mkt] = keep
        if not keep:
            obm.pop(mkt, None)

    _rebuild_position_by_market_and_open_lots_flat(wallet)
    return removed


def _rebuild_position_by_market_and_open_lots_flat(wallet: dict[str, Any]) -> None:
    """Na wijziging van ``open_lots_by_market``: ``position_by_market`` + ``open_lots`` flat lijst."""
    obm = wallet.setdefault("open_lots_by_market", {})
    if not isinstance(obm, dict):
        wallet["open_lots_by_market"] = {}
        obm = wallet["open_lots_by_market"]
    pbm = wallet.setdefault("position_by_market", {})
    if isinstance(pbm, dict):
        pbm.clear()
    for mkt, lots in obm.items():
        if not isinstance(lots, list):
            continue
        tq = 0.0
        for lot in lots:
            if isinstance(lot, dict):
                try:
                    tq += float(lot.get("qty") or 0.0)
                except (TypeError, ValueError):
                    pass
        if tq > 1e-12:
            pbm[str(mkt).strip().upper()] = tq
    flat: list[dict[str, Any]] = []
    for mk in sorted(obm.keys()):
        for lot in obm.get(mk) or []:
            if isinstance(lot, dict):
                flat.append(lot)
    wallet["open_lots"] = flat


def strip_wallet_over_allocation_markets(wallet: dict[str, Any], *, equity_eur: float) -> int:
    """
    Verwijdert markten waar ``allocation_snapshot`` een onmogelijk ruw gewicht t.o.v. equity meldt
    (qty-eenheid, stale prijs, of legacy ``position_by_market`` desync). Past wallet in-place aan.
    """
    if not isinstance(wallet, dict):
        return 0
    try:
        from core.risk_management import allocation_snapshot
    except Exception:
        return 0
    try:
        eq = float(equity_eur)
    except (TypeError, ValueError):
        return 0
    if not math.isfinite(eq) or eq <= 0:
        return 0
    strip_gt = float(os.getenv("PORTFOLIO_STRIP_ALLOCATION_RAW_WEIGHT_GT", "100.02") or 100.02)
    alloc = allocation_snapshot(wallet, eq)
    if not bool(alloc.get("invalid_weight")):
        return 0
    bad: set[str] = set()
    for ln in alloc.get("lines") or []:
        if not isinstance(ln, dict):
            continue
        m = str(ln.get("market") or "").strip().upper()
        if not m:
            continue
        if bool(ln.get("integrity_extreme_weight")):
            bad.add(m)
            continue
        try:
            raw = float(ln.get("weight_pct_raw") or 0.0)
        except (TypeError, ValueError):
            raw = 0.0
        if raw > strip_gt:
            bad.add(m)
    if not bad:
        obm0 = wallet.get("open_lots_by_market") if isinstance(wallet.get("open_lots_by_market"), dict) else {}
        if obm0:
            bad = {str(k).strip().upper() for k in obm0}
            _log.warning(
                "[SANITIZE] invalid_weight zonder strippable lines — alle open_lots-markten verwijderd (%d)",
                len(bad),
            )
        else:
            pbm = wallet.get("position_by_market") if isinstance(wallet.get("position_by_market"), dict) else {}
            if pbm:
                bad = {str(k).strip().upper() for k in pbm}
                _log.warning(
                    "[SANITIZE] invalid_weight zonder open_lots — leegmaken position_by_market (%d keys)",
                    len(bad),
                )
    if not bad:
        return 0
    obm = wallet.setdefault("open_lots_by_market", {})
    if not isinstance(obm, dict):
        wallet["open_lots_by_market"] = {}
        obm = wallet["open_lots_by_market"]
    removed = 0
    for m in bad:
        if m in obm:
            obm.pop(m, None)
            removed += 1
            _log.warning("[SANITIZE] over-allocation market stripped: %s (equity_eur=%.2f)", m, eq)
        pbm2 = wallet.get("position_by_market") if isinstance(wallet.get("position_by_market"), dict) else None
        if isinstance(pbm2, dict) and m in pbm2:
            pbm2.pop(m, None)
    ps = str(wallet.get("position_symbol") or "").strip().upper()
    if ps in bad:
        wallet["position_symbol"] = ""
        try:
            wallet["position_qty"] = 0.0
        except Exception:
            pass
    _rebuild_position_by_market_and_open_lots_flat(wallet)
    return removed


PORTFOLIO_EQUITY_OK_KEY = "portfolio_equity_integrity_ok"
PORTFOLIO_EQUITY_DETAIL_KEY = "portfolio_equity_integrity_detail"
PORTFOLIO_EQUITY_CHECKED_KEY = "portfolio_equity_integrity_checked_utc"


def refresh_portfolio_equity_integrity(
    wallet: dict[str, Any] | None,
    *,
    threshold_pct: float | None = None,
) -> tuple[bool, str]:
    """
    Vergelijk gerapporteerde equity met cash + genormaliseerde open posities * prijs.
    Bij > drempel (default 5%): ERROR log + ``portfolio_equity_integrity_ok`` in system_state op False.
    """
    th = float(threshold_pct if threshold_pct is not None else os.getenv("PORTFOLIO_EQUITY_MISMATCH_PCT", "5") or 5.0)
    th = max(0.1, min(50.0, th))
    if isinstance(wallet, dict):
        sanitize_wallet_corrupt_positions_vs_cash(wallet)

    if not isinstance(wallet, dict):
        detail = "wallet_not_a_dict"
        _log.error("portfolio equity integrity: %s", detail)
        _write_portfolio_integrity_state(False, detail)
        _fatal_portfolio_math_exit(
            detail,
            repair_context={
                "trigger": "wallet_not_a_dict",
                "wallet": None,
                "threshold_pct": th,
            },
        )
        return False, detail

    try:
        reported = float(wallet.get("equity", 0.0) or 0.0)
    except (TypeError, ValueError):
        reported = 0.0
    if not math.isfinite(reported) or reported <= 0:
        _write_portfolio_integrity_state(True, "")
        return True, ""

    try:
        from core.risk_management import allocation_snapshot

        _alloc = allocation_snapshot(wallet, reported)
        if bool(_alloc.get("invalid_weight")):
            if str(os.getenv("PORTFOLIO_STRIP_OVERALLOCATION", "1")).strip().lower() in ("1", "true", "yes", "on"):
                n_strip = strip_wallet_over_allocation_markets(wallet, equity_eur=reported)
                if n_strip > 0:
                    _log.warning(
                        "portfolio integrity: auto-stripped %d over-allocation market(s); re-running allocation check",
                        n_strip,
                    )
                    _alloc = allocation_snapshot(wallet, reported)
            if bool(_alloc.get("invalid_weight")):
                detail = "allocation_weight_pct_raw_exceeds_100"
                _log.error(
                    "portfolio integrity FAILED: %s lines=%s",
                    detail,
                    _alloc.get("lines"),
                )
                _write_portfolio_integrity_state(False, detail)
                _fatal_portfolio_math_exit(
                    detail,
                    repair_context={
                        "trigger": "allocation_invalid_weight",
                        "wallet": wallet,
                        "reported_equity_eur": reported,
                        "allocation_snapshot": dict(_alloc),
                        "threshold_pct": th,
                    },
                )
                return False, detail
    except (OverflowError, ZeroDivisionError) as exc:
        detail = f"allocation_numeric_overflow:{exc}"
        _log.error("portfolio integrity FAILED: %s", detail)
        _write_portfolio_integrity_state(False, detail)
        _fatal_portfolio_math_exit(
            detail,
            repair_context={
                "trigger": "allocation_numeric_overflow",
                "wallet": wallet,
                "reported_equity_eur": reported,
                "exception": exc,
                "threshold_pct": th,
            },
        )
        return False, detail
    except Exception as exc:
        _log.warning("allocation weight integrity check skipped: %s", exc)

    implied = implied_equity_eur_from_wallet(wallet)
    if not math.isfinite(implied) or implied < 0:
        detail = f"implied_equity_invalid implied={implied}"
        _log.error("portfolio equity integrity: %s reported=%.2f", detail, reported)
        _write_portfolio_integrity_state(False, detail)
        _fatal_portfolio_math_exit(
            detail,
            repair_context={
                "trigger": "implied_equity_invalid",
                "wallet": wallet,
                "reported_equity_eur": reported,
                "implied_equity_eur": implied if math.isfinite(implied) else None,
                "threshold_pct": th,
            },
        )
        return False, detail

    rel_pct = abs(reported - implied) / max(reported, 1e-9) * 100.0
    if rel_pct > th:
        detail = (
            f"equity_mismatch rel_pct={rel_pct:.2f}% threshold={th:.2f}% "
            f"reported_eur={reported:.2f} implied_cash_plus_positions_eur={implied:.2f}"
        )
        _log.error("portfolio equity integrity FAILED: %s", detail)
        _write_portfolio_integrity_state(False, detail)
        _fatal_portfolio_math_exit(
            detail,
            repair_context={
                "trigger": "equity_mismatch",
                "wallet": wallet,
                "reported_equity_eur": reported,
                "implied_equity_eur": implied,
                "rel_deviation_pct": rel_pct,
                "threshold_pct": th,
            },
        )
        return False, detail

    _write_portfolio_integrity_state(True, "")
    return True, ""


def _write_portfolio_integrity_state(ok: bool, detail: str) -> None:
    try:
        st = read_system_state()
        st[PORTFOLIO_EQUITY_OK_KEY] = bool(ok)
        st[PORTFOLIO_EQUITY_DETAIL_KEY] = str(detail or "")[:2000]
        st[PORTFOLIO_EQUITY_CHECKED_KEY] = datetime.now(UTC).isoformat()
        write_system_state(st)
    except Exception as exc:
        _log.warning("portfolio integrity state write failed: %s", exc)


def portfolio_math_integrity_allows_sends() -> bool:
    """Alleen ``portfolio_equity_integrity_ok`` uit system_state (zonder mute-flag)."""
    try:
        st = read_system_state()
        if PORTFOLIO_EQUITY_OK_KEY not in st:
            return True
        return bool(st.get(PORTFOLIO_EQUITY_OK_KEY))
    except Exception:
        return True


def portfolio_equity_reports_allowed() -> bool:
    """Executive sends (Jarvis, …): mute OF mislukte portfolio-wiskunde."""
    if executive_notifications_muted():
        return False
    return portfolio_math_integrity_allows_sends()

# NOODSITUATIE (crash-loop): cold-start pad volledig uit. Zet op False na stabilisatie.
STARTUP_COLD_START_REPORT_DISABLED = True

_PERSISTENT_CRASH_LOGGER_INITIALIZED = False


def persistent_crash_log_path() -> Path:
    """Zelfde locatie als andere hub-logs: overleeft container-restarts op gemount volume."""
    if Path("/.dockerenv").exists():
        return Path("/app/logs") / "persistent_crash.log"
    return Path.cwd() / "_logs_hub" / "persistent_crash.log"


class _UtcIsoMsFormatter(logging.Formatter):
    """Exacte UTC-timestamp per logregel (onderscheidt opeenvolgende crashes)."""

    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        return datetime.fromtimestamp(record.created, tz=UTC).isoformat(timespec="milliseconds")


def log_persistent_crash_error(component: str) -> None:
    """
    Append ERROR + volledige traceback naar persistent_crash.log.
    Alleen aanroepen vanuit een actieve except-handler (exc_info).
    """
    global _PERSISTENT_CRASH_LOGGER_INITIALIZED
    path = persistent_crash_log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    lg = logging.getLogger("ai_trading.persistent_crash")
    if not _PERSISTENT_CRASH_LOGGER_INITIALIZED:
        lg.handlers.clear()
        lg.setLevel(logging.ERROR)
        fh = logging.FileHandler(path, encoding="utf-8", mode="a")
        fh.setLevel(logging.ERROR)
        fh.setFormatter(
            _UtcIsoMsFormatter("%(asctime)s | %(levelname)s | %(message)s", datefmt=None)
        )
        lg.addHandler(fh)
        lg.propagate = False
        _PERSISTENT_CRASH_LOGGER_INITIALIZED = True
    lg.error("%s", component, exc_info=True)


def _absurd_pnl_pct_threshold() -> float:
    try:
        return max(1e3, float(os.getenv("REPORTING_ABSURD_PNL_PCT_THRESHOLD", "1000000") or 1_000_000))
    except (TypeError, ValueError):
        return 1_000_000.0


def safe_finance_float(value: Any, *, label: str, context: str = "") -> float:
    """Voorkomt NaN/Inf in portfolio/PnL — logt ERROR en geeft 0.0."""
    try:
        x = float(value)
    except (TypeError, ValueError):
        _log.error("portfolio/PnL invalid float label=%s value=%r ctx=%s", label, value, context)
        return 0.0
    if not math.isfinite(x):
        _log.error("portfolio/PnL non-finite label=%s value=%r ctx=%s", label, value, context)
        return 0.0
    return x


def _int_part_nl_thousands(n: int) -> str:
    """Geheel ≥0 met duizendtallen gescheiden door '.' (NL)."""
    s = str(max(0, int(n)))
    out: list[str] = []
    for i, ch in enumerate(reversed(s)):
        if i and i % 3 == 0:
            out.append(".")
        out.append(ch)
    return "".join(reversed(out))


def _display_float(value: Any) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return 0.0
    return x if math.isfinite(x) else 0.0


def format_eur_nl(value: Any) -> str:
    """Nederlandse getalnotatie: 1.234,56 (zonder €)."""
    x = _display_float(value)
    neg = x < 0.0
    av = abs(round(x, 2))
    tot = int(round(av * 100.0 + 1e-9))
    whole, cents = divmod(tot, 100)
    body = f"{_int_part_nl_thousands(whole)},{cents:02d}"
    return f"-{body}" if neg else body


def format_eur_nl_currency(value: Any) -> str:
    """Eurobedrag met NL-notatie, bv. € 1.234,56 of -€ 1.234,56."""
    x = _display_float(value)
    neg = x < 0.0
    av = abs(round(x, 2))
    tot = int(round(av * 100.0 + 1e-9))
    whole, cents = divmod(tot, 100)
    body = f"{_int_part_nl_thousands(whole)},{cents:02d}"
    return f"-€ {body}" if neg else f"€ {body}"


def safe_pnl_percent(
    *,
    pnl_eur: float,
    cost_eur: float,
    context: str,
) -> float:
    """PnL % voor rapportage/Telegram; bij corrupte of extreme waarden ERROR loggen, geen crash."""
    pnl_eur = safe_finance_float(pnl_eur, label="pnl_eur", context=context)
    cost_eur = safe_finance_float(cost_eur, label="cost_eur", context=context)
    if cost_eur <= 1e-12:
        return 0.0
    pct = (pnl_eur / cost_eur) * 100.0
    if not math.isfinite(pct):
        _log.error("PnL %% non-finite ctx=%s pnl_eur=%s cost_eur=%s", context, pnl_eur, cost_eur)
        return 0.0
    cap = _absurd_pnl_pct_threshold()
    if abs(pct) > cap:
        _log.error(
            "PnL %% absurd (>|%.0f|) ctx=%s pct=%s pnl_eur=%s cost_eur=%s — clamp voor weergave",
            cap,
            context,
            pct,
            pnl_eur,
            cost_eur,
        )
        return max(-cap, min(cap, pct))
    return pct

try:
    import fcntl
except ImportError:  # pragma: no cover — Windows
    fcntl = None  # type: ignore[misc, assignment]

_INITIAL_REPORT_LOCK_FD: int | None = None

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    ZoneInfo = None  # type: ignore[misc, assignment]


def _ams_tz() -> Any:
    if ZoneInfo is None:
        return UTC
    return ZoneInfo("Europe/Amsterdam")


def amsterdam_now() -> datetime:
    return datetime.now(_ams_tz())


def resolve_vault_paths() -> list[Path]:
    raw = str(os.getenv("TRADING_VAULT_PATH", "") or "").strip()
    out: list[Path] = []
    if raw:
        out.append(Path(raw).expanduser())
    out.append(Path.cwd().parent / ".trading_vault")
    out.append(Path.home() / ".trading_vault")
    seen: set[str] = set()
    uniq: list[Path] = []
    for p in out:
        key = str(p.resolve()) if p.exists() else str(p)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(p)
    return uniq


def parse_trading_vault_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        _log.warning("vault read failed %s: %s", path, exc)
        return out
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip().strip("'\"")
        if key:
            out[key] = val
    return out


def load_trading_vault() -> dict[str, str]:
    for p in resolve_vault_paths():
        data = parse_trading_vault_file(p)
        if data:
            _log.info("trading vault loaded from %s (%d keys)", p, len(data))
            return data
    _log.warning("no trading vault file found (tried TRADING_VAULT_PATH, ../.trading_vault, ~/.trading_vault)")
    return {}


def notification_credentials_from_vault(vault: dict[str, str] | None = None) -> dict[str, str]:
    """
    Leest o.a. SMTP_SERVER, SMTP_PORT, EMAIL_RECEIVER, TELEGRAM_CHAT_ID uit de vault,
    met fallback op bestaande PRIVATE_* / TELEGRAM_* / os.environ.
    """
    v = vault if isinstance(vault, dict) else load_trading_vault()

    def _g(*keys: str, default: str = "") -> str:
        for k in keys:
            if v.get(k):
                return str(v[k]).strip()
            ev = os.getenv(k)
            if ev:
                return str(ev).strip()
        return default

    smtp_server = _g("SMTP_SERVER", "PRIVATE_SMTP")
    port_raw = _g("SMTP_PORT") or _g("PRIVATE_PORT") or os.getenv("SMTP_PORT") or os.getenv("PRIVATE_PORT", "587")
    try:
        smtp_port = int(str(port_raw).strip() or "587")
    except (TypeError, ValueError):
        smtp_port = 587
    email_receiver = _g("EMAIL_RECEIVER", "MORNING_REPORT_EMAIL_TO")
    smtp_user = _g("SMTP_USER", "PRIVATE_EMAIL")
    smtp_pass = _g("SMTP_PASS", "PRIVATE_PASS")
    token = _g("TELEGRAM_TOKEN")
    chat_id = _g("TELEGRAM_CHAT_ID")
    tg_on = _g("TELEGRAM_ENABLED") or os.getenv("TELEGRAM_ENABLED", "1")
    return {
        "telegram_token": token,
        "telegram_chat_id": chat_id,
        "telegram_enabled": tg_on,
        "smtp_server": smtp_server,
        "smtp_port": str(smtp_port),
        "smtp_user": smtp_user,
        "smtp_pass": smtp_pass,
        "email_receiver": email_receiver or smtp_user,
    }


def apply_vault_to_os(keys: frozenset[str], *, override: bool = False) -> None:
    """Zet geselecteerde vault-keys in os.environ zodat bestaande send_email / flows ze zien."""
    vault = load_trading_vault()
    for k in keys:
        v = vault.get(k)
        if not v:
            continue
        if not override and os.getenv(k):
            continue
        os.environ[k] = str(v).strip()


def logs_dir() -> Path:
    p = Path("/app/logs") if Path("/.dockerenv").exists() else Path.cwd() / "_logs_hub"
    p.mkdir(parents=True, exist_ok=True)
    return p


def system_state_path() -> Path:
    """Persistente state voor notification throttles / dedupe."""
    return logs_dir() / "system_state.json"


def read_system_state() -> dict[str, Any]:
    p = system_state_path()
    if not p.is_file():
        return {}
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def write_system_state(doc: dict[str, Any]) -> None:
    p = system_state_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    try:
        tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(p)
    except Exception:
        try:
            p.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass


def _parse_iso_dt(s: str) -> datetime | None:
    try:
        dt = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
        return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)
    except Exception:
        return None


def notification_cooldown_due(*, key: str, cooldown_sec: int) -> bool:
    """True = versturen toegestaan; False = cooldown actief."""
    cd = max(1, int(cooldown_sec))
    st = read_system_state()
    raw = st.get(key)
    if not raw:
        return True
    prev = _parse_iso_dt(str(raw))
    if prev is None:
        return True
    return (datetime.now(UTC) - prev.astimezone(UTC)).total_seconds() >= float(cd)


def mark_notification_sent(*, key: str) -> None:
    st = read_system_state()
    st[key] = datetime.now(UTC).isoformat()
    write_system_state(st)


# --- Shutdown Telegram: alleen onverwachte fouten, debounce over workers (system_state) ---

SHUTDOWN_ERROR_TELEGRAM_SENT_KEY = "shutdown_error_telegram_last_sent_utc"


def shutdown_telegram_debounce_sec() -> int:
    return max(1, int(float(os.getenv("TELEGRAM_SHUTDOWN_DEBOUNCE_SEC", "30") or 30)))


def _is_quiet_shutdown_exception(exc: BaseException | None) -> bool:
    """Geen Telegram: normale stop, interrupt, of geannuleerde taken."""
    if exc is None:
        return True
    import asyncio

    return isinstance(exc, (KeyboardInterrupt, SystemExit, asyncio.CancelledError, GeneratorExit))


def try_send_unexpected_shutdown_telegram(exc: BaseException | None) -> bool:
    """
    Één „gestopt door fout”-Telegram (HTML), zonder globale 10-min rate-limit.
    Debounce via ``SHUTDOWN_ERROR_TELEGRAM_SENT_KEY`` (default 30s): tweede poging binnen venster wordt overgeslagen.
    """
    if exc is None or _is_quiet_shutdown_exception(exc):
        return False
    if str(os.getenv("TELEGRAM_SHUTDOWN_ERROR_ENABLED", "1")).strip().lower() not in ("1", "true", "yes", "on"):
        return False
    deb = shutdown_telegram_debounce_sec()
    if not notification_cooldown_due(key=SHUTDOWN_ERROR_TELEGRAM_SENT_KEY, cooldown_sec=deb):
        _log.info("shutdown error telegram skipped (debounce key=%s sec=%s)", SHUTDOWN_ERROR_TELEGRAM_SENT_KEY, deb)
        return False
    vault = load_trading_vault()
    token = (vault.get("TELEGRAM_TOKEN") or os.getenv("TELEGRAM_TOKEN") or "").strip()
    chat = (vault.get("TELEGRAM_CHAT_ID") or os.getenv("TELEGRAM_CHAT_ID") or "").strip()
    if str(vault.get("TELEGRAM_ENABLED") or os.getenv("TELEGRAM_ENABLED", "1")).strip().lower() not in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return False
    if not token or not chat:
        return False
    try:
        import traceback as _tb

        tb_txt = "".join(_tb.format_exception(type(exc), exc, exc.__traceback__))
    except Exception:
        tb_txt = repr(exc)
    tb_txt = tb_txt[-2800:]
    try:
        host = socket.gethostname()
    except OSError:
        host = "unknown"
    ts = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    safe_type = escape(type(exc).__name__)
    safe_msg = escape(str(exc))[:1200]
    safe_tb = escape(tb_txt)
    html = (
        "🛑 <b>AI Trading Bot gestopt (onverwachte fout)</b>\n"
        f"Tijd: <code>{escape(ts)}</code>\n"
        f"Host: <code>{escape(host)}</code>\n"
        f"Type: <code>{safe_type}</code>\n"
        f"Bericht: <code>{safe_msg}</code>\n"
        "<pre>" + safe_tb + "</pre>"
    )
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        r = requests.post(
            url,
            json={
                "chat_id": chat,
                "text": html,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            },
            timeout=12,
        )
        ok = r.status_code == 200
        if ok:
            mark_notification_sent(key=SHUTDOWN_ERROR_TELEGRAM_SENT_KEY)
            _log.error("shutdown error telegram sent (%s)", safe_type)
        return ok
    except Exception as post_exc:
        _log.warning("shutdown error telegram post failed: %s", post_exc)
        return False


def install_unhandled_exception_shutdown_telegram_hooks() -> None:
    """
    Eén hook-set per proces: ``sys.excepthook`` + ``threading.excepthook`` (geen atexit/signal-dubbels).
    Normale FastAPI-shutdown stuurt géén Telegram meer; alleen onverwerkte exceptions.
    """
    mod = sys.modules[__name__]
    if getattr(mod, "_shutdown_telegram_hooks_installed", False):
        return
    setattr(mod, "_shutdown_telegram_hooks_installed", True)

    prev = sys.excepthook

    def _excepthook(exc_type: type, exc: BaseException | None, tb: Any) -> None:
        try:
            if exc is not None:
                try_send_unexpected_shutdown_telegram(exc)
        except Exception:
            pass
        prev(exc_type, exc, tb)

    sys.excepthook = _excepthook

    import threading

    if hasattr(threading, "excepthook"):
        prev_th = threading.excepthook

        def _thread_hook(args: Any) -> None:
            try:
                ev = getattr(args, "exc_value", None)
                if isinstance(ev, BaseException):
                    try_send_unexpected_shutdown_telegram(ev)
            except Exception:
                pass
            if prev_th is not None:
                prev_th(args)

        threading.excepthook = _thread_hook  # type: ignore[assignment]


def _startup_telegram_throttle_lock_path() -> Path:
    return logs_dir() / "startup_telegram_throttle.lock"


def try_begin_startup_telegram_send(*, kind: str) -> bool:
    """
    Mag deze startup-categorie nu één Telegram sturen? (flock + system_state)

    Voorkomt storm bij meerdere Uvicorn-workers en snelle restart:always loops (zelfde soort ≤1 per cooldown).
    Zet ``DISABLE_STARTUP_BRIEFING_TELEGRAM=1`` om alles te skippen.

    kind: ``jarvis_integrity`` | ``restart_audit``
    """
    if executive_notifications_muted():
        _log.info("startup telegram skipped (EXECUTIVE_NOTIFICATIONS_EMERGENCY_MUTE / MUTE_EXECUTIVE_NOTIFICATIONS)")
        return False
    k = str(kind or "").strip().lower()
    if k not in ("jarvis_integrity", "restart_audit"):
        return True
    if str(os.getenv("DISABLE_STARTUP_BRIEFING_TELEGRAM", "0")).strip().lower() in ("1", "true", "yes", "on"):
        _log.info("startup telegram skipped (DISABLE_STARTUP_BRIEFING_TELEGRAM)")
        return False
    # Zelfde stilte als mail-pad bij auto-restart (crash-loop / restart:always).
    if startup_mode() == "auto" and str(os.getenv("EXTENDED_STARTUP_TELEGRAM_ON_AUTO_RESTART", "0")).strip().lower() not in (
        "1",
        "true",
        "yes",
        "on",
    ):
        _log.info("startup telegram skipped (auto_restart silent; zet EXTENDED_STARTUP_TELEGRAM_ON_AUTO_RESTART=1)")
        return False

    per_kind_cd = max(60, int(os.getenv("STARTUP_COOLDOWN_SEC", "3600") or 3600))
    key_kind = "startup_telegram_jarvis_last_utc" if k == "jarvis_integrity" else "startup_telegram_restart_audit_last_utc"

    lock_path = _startup_telegram_throttle_lock_path()
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    import os as _os

    def _allowed(st: dict[str, Any]) -> bool:
        now = datetime.now(UTC)
        prev_k = _parse_iso_dt(str(st.get(key_kind) or ""))
        if prev_k is not None and (now - prev_k.astimezone(UTC)).total_seconds() < float(per_kind_cd):
            return False
        return True

    def _commit(st: dict[str, Any]) -> None:
        ts = datetime.now(UTC).isoformat()
        st[key_kind] = ts
        write_system_state(st)

    if fcntl is None:
        st = read_system_state()
        if not _allowed(st):
            return False
        _commit(st)
        return True

    fd = _os.open(str(lock_path), _os.O_CREAT | _os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        st = read_system_state()
        if not _allowed(st):
            return False
        _commit(st)
        return True
    except OSError as exc:
        _log.warning("startup telegram throttle lock failed: %s", exc)
        return False
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except Exception:
            pass
        try:
            _os.close(fd)
        except Exception:
            pass


def startup_mode() -> str:
    """
    'manual' of 'auto' (silent restart). Wordt gezet door app/main.py.
    Default = manual.
    """
    st = read_system_state()
    mode = str(st.get("startup_mode") or "manual").strip().lower()
    return mode if mode in ("manual", "auto") else "manual"


def initial_report_cold_start_lock_path() -> Path:
    """flock-bestand: slechts één proces per container (multi-uvicorn-worker) stuurt cold-start rapport."""
    return logs_dir() / "initial_report_cold_start.lock"


def _try_acquire_initial_report_cold_start_lock() -> bool:
    """
    Non-blocking exclusive lock; fd blijft open tot proces exit → OS geeft lock vrij.
    Voorkomt dubbele Telegram/mail bij meerdere workers of herhaalde FastAPI-lifespan (zeldzaam).
    """
    global _INITIAL_REPORT_LOCK_FD
    if _INITIAL_REPORT_LOCK_FD is not None:
        return False
    if fcntl is None:
        return True
    path = initial_report_cold_start_lock_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(str(path), os.O_RDWR | os.O_CREAT, 0o644)
    except OSError as exc:
        _log.warning("initial report lock open failed: %s", exc)
        return False
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        try:
            os.close(fd)
        except OSError:
            pass
        return False
    _INITIAL_REPORT_LOCK_FD = fd
    return True


def morning_report_marker_path() -> Path:
    # Expliciet compact marker-bestand voor dedupe per kalenderdag.
    return logs_dir() / "last_morning_report.txt"


def _legacy_morning_report_marker_path() -> Path:
    # Backward compatibility met oudere deploys.
    return logs_dir() / "morning_report_last_sent_ams_date.txt"


def morning_report_dispatch_lock_path() -> Path:
    """Bestandslock voor atomaire morning-report dispatch (multi-process safe)."""
    return logs_dir() / "morning_report_dispatch.lock"


def _try_acquire_morning_report_dispatch_lock() -> int | None:
    """
    Non-blocking lock voor dispatch-if-due.
    Retourneert fd bij succes; caller moet close() doen.
    """
    if fcntl is None:
        return None
    path = morning_report_dispatch_lock_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(str(path), os.O_RDWR | os.O_CREAT, 0o644)
    except OSError as exc:
        _log.warning("morning report lock open failed: %s", exc)
        return None
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        try:
            os.close(fd)
        except OSError:
            pass
        return None
    return fd


def night_baseline_path() -> Path:
    return logs_dir() / "morning_report_night_baseline.json"


def learning_issues_log_path() -> Path:
    return logs_dir() / "learning_issues.log"


def append_learning_issue(line: str) -> None:
    try:
        ts = datetime.now(UTC).isoformat()
        with learning_issues_log_path().open("a", encoding="utf-8") as fp:
            fp.write(f"[{ts}] {str(line).strip()}\n")
    except Exception:
        pass


def read_last_sent_ams_date() -> date | None:
    for p in (morning_report_marker_path(), _legacy_morning_report_marker_path()):
        if not p.is_file():
            continue
        try:
            s = p.read_text(encoding="utf-8").strip()
            return date.fromisoformat(s)
        except Exception:
            continue
    return None


def write_last_sent_ams_date(d: date) -> None:
    txt = d.isoformat()
    morning_report_marker_path().write_text(txt, encoding="utf-8")
    # Keep legacy marker in sync for smooth roll-forward.
    try:
        _legacy_morning_report_marker_path().write_text(txt, encoding="utf-8")
    except Exception:
        pass


def maybe_write_night_baseline(*, wallet: dict[str, Any]) -> None:
    """Rond 23:00 Amsterdam: snapshot equity voor morgenochtend vergelijking."""
    now = amsterdam_now()
    target_h = int(os.getenv("MORNING_REPORT_BASELINE_HOUR", "23") or 23)
    if now.hour != target_h or now.minute > 5:
        return
    payload_path = night_baseline_path()
    try:
        if payload_path.is_file():
            prev = json.loads(payload_path.read_text(encoding="utf-8"))
            if isinstance(prev, dict) and str(prev.get("night_of_ams", "")) == now.date().isoformat():
                return
    except Exception:
        pass
    eq = safe_finance_float(wallet.get("equity", 0.0), label="equity", context="maybe_write_night_baseline")
    cash = safe_finance_float(wallet.get("cash", 0.0), label="cash", context="maybe_write_night_baseline")
    doc = {
        "night_of_ams": now.date().isoformat(),
        "equity_eur": round(eq, 2),
        "cash_eur": round(cash, 2),
        "written_ts_utc": datetime.now(UTC).isoformat(),
    }
    payload_path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    _log.info("night baseline written: %s", doc)


def read_night_baseline_for_morning(now_ams: datetime) -> dict[str, Any] | None:
    """Baseline van de vorige kalenderdag (nacht die net eindigde)."""
    p = night_baseline_path()
    if not p.is_file():
        return None
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(doc, dict):
        return None
    yday = (now_ams.date() - timedelta(days=1)).isoformat()
    if str(doc.get("night_of_ams", "")) == yday:
        return doc
    return None


def reporting_night_window_utc(now_ams: datetime) -> tuple[datetime, datetime]:
    """
    Venster voor statistieken: gisteren 22:00 AMS t/m vandaag 08:00 AMS (rapport ~08:00).
    Overschrijf met MORNING_REPORT_WINDOW_START_HOUR / MORNING_REPORT_WINDOW_END_HOUR.
    """
    start_h = int(os.getenv("MORNING_REPORT_WINDOW_START_HOUR", "22") or 22)
    end_h = int(os.getenv("MORNING_REPORT_WINDOW_END_HOUR", "8") or 8)
    today = now_ams.date()
    yday = today - timedelta(days=1)
    start = datetime(yday.year, yday.month, yday.day, start_h, 0, 0, tzinfo=now_ams.tzinfo)
    end = datetime(today.year, today.month, today.day, end_h, 0, 0, tzinfo=now_ams.tzinfo)
    return start.astimezone(UTC), end.astimezone(UTC)


def _iso_utc(dt: datetime) -> str:
    if dt.tzinfo is None:
        return dt.isoformat(timespec="seconds") + "Z"
    return dt.astimezone(UTC).replace(tzinfo=None).isoformat(timespec="seconds")


def trade_stats_for_window(
    db_path: Path,
    tenant_id: str,
    start_utc: datetime,
    end_utc: datetime,
) -> dict[str, Any]:
    start_s = _iso_utc(start_utc)
    end_s = _iso_utc(end_utc)
    out: dict[str, Any] = {"closed_trades": 0, "avg_pnl_pct": None, "sum_pnl_eur": 0.0, "by_coin": []}
    if not db_path.is_file():
        return out
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
    except OSError:
        return out
    try:
        q = """
            SELECT COUNT(*) AS n,
                   AVG(pnl_pct) AS avgp,
                   SUM(pnl_eur) AS sump
            FROM trade_history
            WHERE tenant_id = ?
              AND replace(replace(exit_ts_utc, 'T', ' '), 'Z', '') >= replace(replace(?, 'T', ' '), 'Z', '')
              AND replace(replace(exit_ts_utc, 'T', ' '), 'Z', '') < replace(replace(?, 'T', ' '), 'Z', '')
        """
        row = conn.execute(q, (tenant_id, start_s, end_s)).fetchone()
        if row:
            out["closed_trades"] = int(row["n"] or 0)
            avgp = float(row["avgp"]) if row["avgp"] is not None else None
            if avgp is not None and not math.isfinite(avgp):
                _log.error("trade_stats avg_pnl_pct non-finite: %s", avgp)
                avgp = None
            out["avg_pnl_pct"] = avgp
            sump = float(row["sump"] or 0.0)
            if not math.isfinite(sump):
                _log.error("trade_stats sum_pnl_eur non-finite: %s", sump)
                sump = 0.0
            out["sum_pnl_eur"] = sump
        rows = conn.execute(
            """
            SELECT coin, SUM(pnl_eur) AS tpnl, COUNT(*) AS cnt
            FROM trade_history
            WHERE tenant_id = ?
              AND replace(replace(exit_ts_utc, 'T', ' '), 'Z', '') >= replace(replace(?, 'T', ' '), 'Z', '')
              AND replace(replace(exit_ts_utc, 'T', ' '), 'Z', '') < replace(replace(?, 'T', ' '), 'Z', '')
            GROUP BY coin
            ORDER BY tpnl DESC
            LIMIT 25
            """,
            (tenant_id, start_s, end_s),
        ).fetchall()
        by_coin: list[dict[str, Any]] = []
        for r in rows:
            raw_t = float(r["tpnl"] or 0.0)
            if not math.isfinite(raw_t):
                _log.error("trade_stats by_coin pnl non-finite coin=%s val=%s", r["coin"], raw_t)
                raw_t = 0.0
            by_coin.append(
                {"coin": str(r["coin"]), "pnl_eur": round(raw_t, 4), "trades": int(r["cnt"] or 0)}
            )
        out["by_coin"] = by_coin
    except Exception as exc:
        _log.warning("trade_stats_for_window: %s", exc)
    finally:
        try:
            conn.close()
        except Exception:
            pass
    return out


def rl_training_stats_for_window(start_utc: datetime, end_utc: datetime) -> dict[str, Any]:
    from app.services.rl_metrics_store import _conn, init_rl_metrics_db

    init_rl_metrics_db()
    chunks = 0
    all_loss: list[float] = []
    reward_tails: list[float] = []
    start_s = start_utc.astimezone(UTC).isoformat()
    end_s = end_utc.astimezone(UTC).isoformat()
    try:
        with _conn() as conn:
            rows = conn.execute(
                "SELECT rewards_json, loss_json FROM rl_training_chunks WHERE ts_utc >= ? AND ts_utc < ? ORDER BY id ASC",
                (start_s, end_s),
            ).fetchall()
    except Exception as exc:
        _log.warning("rl_training_stats: %s", exc)
        return {"chunks": 0, "avg_loss": None, "loss_trend": "n/a", "avg_reward_tail": None}
    for row in rows:
        chunks += 1
        try:
            rj = json.loads(row[0])
            if isinstance(rj, list) and rj:
                reward_tails.append(float(rj[-1]))
        except Exception:
            pass
        try:
            lj = json.loads(row[1])
            if isinstance(lj, list):
                for x in lj:
                    try:
                        v = float(x)
                        if v == v:
                            all_loss.append(v)
                    except (TypeError, ValueError):
                        pass
        except Exception:
            pass
    avg_loss = sum(all_loss) / len(all_loss) if all_loss else None
    avg_r = sum(reward_tails) / len(reward_tails) if reward_tails else None
    trend = "n/a"
    mid = len(all_loss) // 2
    if mid > 2:
        a = sum(all_loss[:mid]) / mid
        b = sum(all_loss[mid:]) / max(1, len(all_loss) - mid)
        if a > b * 1.05:
            trend = "dalend (gem. loss eerste helft > tweede helft)"
        elif b > a * 1.05:
            trend = "stijgend (let op: hogere loss is slechter)"
        else:
            trend = "stabiel"
    return {
        "chunks": chunks,
        "avg_loss": round(avg_loss, 8) if avg_loss is not None else None,
        "loss_trend": trend,
        "avg_reward_tail": round(avg_r, 6) if avg_r is not None else None,
        "loss_samples": len(all_loss),
    }


def top_volume_markets(state: dict[str, Any], limit: int = 20) -> list[str]:
    mks = state.get("active_markets") if isinstance(state.get("active_markets"), list) else []
    out: list[str] = []
    for m in mks[:limit]:
        if isinstance(m, dict) and m.get("market"):
            out.append(str(m["market"]).upper().replace("/", "-"))
    return out


def best_coin_in_top_volume(by_coin: list[dict[str, Any]], top_markets: list[str]) -> tuple[str | None, str]:
    if not by_coin:
        return None, "Geen gesloten trades in het venster."
    top_set = {m.upper() for m in top_markets}
    for row in by_coin:
        c = str(row.get("coin") or "").upper()
        if not c:
            continue
        mk_candidates = [f"{c}-EUR", f"{c}-USDT"]
        if any(m in top_set for m in mk_candidates) or not top_set:
            return c, str(row.get("coin"))
    best = by_coin[0]
    return str(best.get("coin")), str(best.get("coin"))


def intelligence_line(
    state: dict[str, Any],
    best_coin: str | None,
    rl_stats: dict[str, Any],
    exploration_eps: float | None,
) -> str:
    ls = state.get("last_scores") if isinstance(state.get("last_scores"), dict) else {}
    rsi = ls.get("rsi_14")
    eps = exploration_eps
    parts: list[str] = []
    if eps is not None:
        try:
            e = float(eps)
            parts.append(f"exploratie ε≈{e:.3f} ({'relatief greedy' if e < 0.12 else 'nog exploratief'})")
        except (TypeError, ValueError):
            parts.append("exploratie: onbekend")
    if rsi is not None and best_coin:
        try:
            r = float(rsi)
            tag = "oversold" if r < 35 else "overbought" if r > 65 else "neutraal"
            parts.append(f"laatste RSI {r:.1f} ({tag}) — muntfocus {best_coin}")
        except (TypeError, ValueError):
            pass
    elif best_coin:
        parts.append(f"meeste PnL in venster bij {best_coin}")
    if not parts:
        return "Onvoldoende live-metrics voor een korte conclusie; check RL-training chunks in SQLite."
    return "De AI: " + "; ".join(parts) + "."


def build_morning_report_body(
    *,
    state: dict[str, Any],
    wallet: dict[str, Any],
    tenant_id: str,
    trade_db: Path,
    exploration_eps: float | None = None,
) -> str:
    now_ams = amsterdam_now()
    start_utc, end_utc = reporting_night_window_utc(now_ams)
    tstats = trade_stats_for_window(trade_db, tenant_id, start_utc, end_utc)
    rlstats = rl_training_stats_for_window(start_utc, end_utc)
    top20 = top_volume_markets(state, 20)
    _best_sym, best_label = best_coin_in_top_volume(tstats.get("by_coin") or [], top20)
    intel = intelligence_line(state, best_label, rlstats, exploration_eps)
    baseline = read_night_baseline_for_morning(now_ams)
    eq = safe_finance_float(wallet.get("equity", 0.0), label="equity", context="build_morning_report_body")
    cash = safe_finance_float(wallet.get("cash", 0.0), label="cash", context="build_morning_report_body")
    lines: list[str] = [
        f"Ochtendrapport (Europe/Amsterdam {now_ams.strftime('%Y-%m-%d %H:%M')})",
        f"Venster UTC: {start_utc.isoformat()} → {end_utc.isoformat()}",
        "",
        "• Training (PPO / SQLite chunks)",
        f"  - learn-batches in venster: {rlstats.get('chunks', 0)}",
        f"  - gemiddelde loss (train/): {rlstats.get('avg_loss')}",
        f"  - loss-trend: {rlstats.get('loss_trend')}",
        f"  - gem. eind-cumulatieve reward per batch: {rlstats.get('avg_reward_tail')}",
        "",
        "• Paper trades (trade_history)",
        f"  - gesloten round-trips: {tstats.get('closed_trades', 0)}",
        f"  - gemiddelde PnL % per trade: {tstats.get('avg_pnl_pct')}",
        f"  - som PnL EUR: {round(float(tstats.get('sum_pnl_eur') or 0.0), 2)}",
        "",
        "• Markt (top volume-set vs winst)",
        f"  - meest winstgevende munt in venster (top-{len(top20)} focus): {best_label or '—'}",
        "",
        "• Intelligence",
        f"  - {intel}",
        "",
        "• Portfolio",
        f"  - equity nu: €{eq:,.2f} | cash: €{cash:,.2f}",
    ]
    if baseline:
        b_eq = safe_finance_float(baseline.get("equity_eur", 0.0), label="baseline_equity", context="build_morning_report_body")
        delta = eq - b_eq
        lines.append(f"  - baseline nacht ({baseline.get('night_of_ams')}): €{b_eq:,.2f} → Δ €{delta:+,.2f} t.o.v. start nacht")
    else:
        lines.append("  - geen nacht-baseline (snapshot 23:00 AMS ontbreekt); zet MORNING_REPORT_BASELINE_HOUR of wacht vanavond.")
    return "\n".join(lines)


def send_telegram_html(token: str, chat_id: str, html: str) -> bool:
    token = str(token or "").strip()
    chat_id = str(chat_id or "").strip()
    if not token or not chat_id:
        return False
    try:
        # Hard cap: max 1 telegram bericht per 10 minuten, ongeacht herstarts.
        global_sec = max(60, int(float(os.getenv("TELEGRAM_GLOBAL_RATE_LIMIT_SEC", "600") or 600)))
        if not notification_cooldown_due(key="telegram_global_last_sent_utc", cooldown_sec=global_sec):
            _log.info(
                "telegram blocked by global rate-limit: key=%s cooldown_sec=%s",
                "telegram_global_last_sent_utc",
                global_sec,
            )
            return False
        window_sec = max(30, int(float(os.getenv("TELEGRAM_DEDUPE_WINDOW_SEC", "180") or 180)))
        digest = hashlib.sha1(str(html).encode("utf-8", errors="ignore")).hexdigest()[:20]
        dedupe_key = f"telegram_dedupe_{digest}"
        if not notification_cooldown_due(key=dedupe_key, cooldown_sec=window_sec):
            _log.info("telegram blocked by content dedupe: key=%s window_sec=%s", dedupe_key, window_sec)
            return False
    except Exception:
        dedupe_key = ""
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        r = requests.post(
            url,
            json={
                "chat_id": chat_id,
                "text": html,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            },
            timeout=12,
        )
        ok = r.status_code == 200
        if ok and dedupe_key:
            mark_notification_sent(key=dedupe_key)
            mark_notification_sent(key="telegram_global_last_sent_utc")
        return ok
    except Exception as exc:
        _log.warning("telegram send failed: %s", exc)
        return False


def format_trade_execution_telegram_html(snap: dict[str, Any], market: str, execution_price: float) -> str | None:
    """
    🟢 BUY bij status opened; 🔴 SELL bij closed met PnL (EUR + %) — entry uit snap/ledger.
    Returns None als geen notificatie (HOLD/rejected/...).
    """
    if not isinstance(snap, dict):
        return None
    mkt = escape(str(market or "").strip().upper().replace("/", "-"))
    st = str(snap.get("status") or "").lower()
    sig = str(snap.get("signal") or snap.get("action") or "").upper()

    if st == "opened" and sig == "BUY":
        qty = safe_finance_float(snap.get("qty") or 0.0, label="qty", context=f"BUY {mkt}")
        ep = safe_finance_float(snap.get("entry_price") or execution_price, label="entry_price", context=f"BUY {mkt}")
        fee = safe_finance_float(snap.get("fee_eur") or 0.0, label="fee_eur", context=f"BUY {mkt}")
        inleg = qty * ep if qty > 1e-18 and ep > 1e-18 else 0.0
        return (
            f"🟢 <b>BUY</b> {mkt}\n"
            f"Werkelijke inleg: <b>{escape(format_eur_nl_currency(inleg))}</b> <i>(qty × prijs)</i>\n"
            f"Prijs: <code>{ep:.4f}</code> EUR\n"
            f"Qty: <code>{qty:.8f}</code>\n"
            f"Fee: <code>{escape(format_eur_nl_currency(fee))}</code>"
        )

    if st == "closed" and sig == "SELL":
        whale_ctx = str(snap.get("telegram_whale_context") or "").strip()
        whale_block = ""
        if whale_ctx:
            whale_block = f"🚨 <b>WHALE PANIC</b>\n{escape(whale_ctx)}\n\n"
        qty = safe_finance_float(snap.get("qty_closed") or snap.get("qty") or 0.0, label="qty", context=f"SELL {mkt}")
        exit_px = safe_finance_float(snap.get("exit_price") or execution_price, label="exit_price", context=f"SELL {mkt}")
        entry_px = safe_finance_float(snap.get("entry_price") or 0.0, label="entry_price", context=f"SELL {mkt}")
        pnl_eur = safe_finance_float(snap.get("realized_pnl_eur") or 0.0, label="realized_pnl_eur", context=f"SELL {mkt}")
        fees_eur = safe_finance_float(
            snap.get("fees_eur") or snap.get("fee_eur") or 0.0,
            label="fees_eur",
            context=f"SELL {mkt}",
        )
        gross_exit = exit_px * qty
        cost_entry = entry_px * qty if entry_px > 1e-12 and qty > 1e-12 else 0.0
        if not math.isfinite(cost_entry):
            _log.error("cost_entry non-finite SELL %s entry=%s qty=%s", mkt, entry_px, qty)
            cost_entry = 0.0
        realized_calc = gross_exit - cost_entry - fees_eur
        if not math.isfinite(realized_calc):
            realized_calc = pnl_eur
        pnl_pct = safe_pnl_percent(pnl_eur=realized_calc, cost_eur=cost_entry, context=f"telegram SELL {mkt}")
        winst_nl = escape(format_eur_nl_currency(realized_calc))
        pct_txt = escape(f"{pnl_pct:+.2f}%".replace(".", ","))
        return (
            whale_block
            + f"🔴 <b>SELL</b> {mkt}\n"
            f"Gerealiseerde winst: <b>{winst_nl}</b> <i>(verkoop − aankoop − fees)</i>\n"
            f"Entry: <code>{entry_px:.4f}</code> EUR → Exit: <code>{exit_px:.4f}</code> EUR\n"
            f"Qty: <code>{qty:.8f}</code>\n"
            f"Bruto verkoop: <code>{escape(format_eur_nl_currency(gross_exit))}</code> · "
            f"Inleg (kost): <code>{escape(format_eur_nl_currency(cost_entry))}</code> · "
            f"Fees: <code>{escape(format_eur_nl_currency(fees_eur))}</code>\n"
            f"PnL t.o.v. inleg: <b>{pct_txt}</b>"
        )

    return None


def telegram_closed_pnl_market_allowed(market: str) -> bool:
    """
    Voor gesloten SELL: beperk winst-Telegram tot deze markten (standaard BTC-EUR, ETH-EUR).
    Zet ``TELEGRAM_PNL_CLOSE_MARKETS=*`` om alle paren te melden; leeg = alle paren.
    """
    raw = str(os.getenv("TELEGRAM_PNL_CLOSE_MARKETS", "BTC-EUR,ETH-EUR") or "").strip()
    if not raw or raw in ("*", "ALL", "all"):
        return True
    mku = str(market or "").strip().upper().replace("/", "-")
    allow = {x.strip().upper().replace("/", "-") for x in raw.split(",") if x.strip()}
    return mku in allow


def send_trade_execution_telegram(snap: dict[str, Any], market: str, execution_price: float) -> bool:
    """Directe Telegram na paper trade; vault TELEGRAM_TOKEN / TELEGRAM_CHAT_ID."""
    if str(os.getenv("TELEGRAM_PAPER_TRADES", "1")).strip().lower() not in ("1", "true", "yes", "on"):
        return False
    html = format_trade_execution_telegram_html(snap, market, execution_price)
    if not html:
        return False
    st = str(snap.get("status") or "").lower()
    sig = str(snap.get("signal") or snap.get("action") or "").upper()
    if st == "closed" and sig == "SELL" and not telegram_closed_pnl_market_allowed(market):
        _log.info("telegram closed PnL skipped (TELEGRAM_PNL_CLOSE_MARKETS): market=%s", market)
        return False
    vault = load_trading_vault()
    cred = notification_credentials_from_vault(vault)
    if str(cred.get("telegram_enabled", "1")).lower() not in ("1", "true", "yes", "on"):
        return False
    return send_telegram_html(cred["telegram_token"], cred["telegram_chat_id"], html)


def send_smtp_html(
    *,
    smtp_server: str,
    smtp_port: int,
    smtp_user: str,
    smtp_pass: str,
    subject: str,
    html_body: str,
    to_addr: str,
    plain_fallback: str,
) -> bool:
    if not smtp_server or not smtp_user or not smtp_pass or not to_addr:
        return False
    msg = MIMEMultipart("alternative")
    msg["From"] = smtp_user
    msg["To"] = to_addr.strip()
    msg["Subject"] = subject
    msg["Date"] = formatdate(localtime=True)
    msg["Message-ID"] = make_msgid()
    msg.attach(MIMEText(plain_fallback, "plain", "utf-8"))
    msg.attach(MIMEText(html_body, "html", "utf-8"))
    try:
        with smtplib.SMTP(smtp_server, int(smtp_port), timeout=60) as server:
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.send_message(msg)
        return True
    except Exception as exc:
        _log.warning("smtp html send failed: %s", exc)
        return False


def send_smtp_plain(
    *,
    smtp_server: str,
    smtp_port: int,
    smtp_user: str,
    smtp_pass: str,
    subject: str,
    body: str,
    to_addr: str | None,
) -> bool:
    if not smtp_server or not smtp_user or not smtp_pass:
        return False
    to_ = (to_addr or smtp_user).strip()
    msg = MIMEMultipart()
    msg["From"] = smtp_user
    msg["To"] = to_
    msg["Subject"] = subject
    msg["Date"] = formatdate(localtime=True)
    msg["Message-ID"] = make_msgid()
    msg.attach(MIMEText(body, "plain", "utf-8"))
    try:
        with smtplib.SMTP(smtp_server, int(smtp_port), timeout=60) as server:
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.send_message(msg)
        return True
    except Exception as exc:
        _log.warning("smtp morning report failed: %s", exc)
        return False


def plaintext_to_telegram_html(text: str) -> str:
    safe = escape(text)
    return "<b>Ochtendrapport AI Trading</b>\n<pre>" + safe + "</pre>"


_VAULT_KEYS = frozenset(
    {
        "TELEGRAM_TOKEN",
        "TELEGRAM_CHAT_ID",
        "TELEGRAM_ENABLED",
        "SMTP_SERVER",
        "SMTP_PORT",
        "SMTP_USER",
        "SMTP_PASS",
        "EMAIL_RECEIVER",
        "PRIVATE_SMTP",
        "PRIVATE_PORT",
        "PRIVATE_EMAIL",
        "PRIVATE_PASS",
        "MORNING_REPORT_EMAIL_TO",
    }
)

_INITIAL_STARTUP_VAULT_KEYS = _VAULT_KEYS


def run_morning_report_dispatch(
    *,
    state: dict[str, Any],
    wallet: dict[str, Any],
    tenant_id: str,
    trade_db: Path,
    exploration_eps: float | None = None,
) -> dict[str, Any]:
    """
    Laadt vault → bouwt rapport → Telegram + SMTP.
    Gebruikt eigen SMTP-send zodat MORNING_REPORT_EMAIL_TO werkt naast PRIVATE_EMAIL.
    """
    refresh_portfolio_equity_integrity(wallet)
    if executive_notifications_muted():
        _log.warning("ochtendrapport volledig uit (EXECUTIVE_NOTIFICATIONS_EMERGENCY_MUTE)")
        return {
            "telegram_ok": False,
            "email_ok": False,
            "subject": "muted",
            "skipped": True,
            "reason": "executive_notifications_emergency_mute",
        }
    if not portfolio_math_integrity_allows_sends():
        _log.error("ochtendrapport niet verstuurd: portfolio equity gate actief")
        return {
            "telegram_ok": False,
            "email_ok": False,
            "subject": "skipped",
            "skipped": True,
            "reason": "portfolio_equity_integrity_gate",
        }

    apply_vault_to_os(_VAULT_KEYS, override=False)
    vault = load_trading_vault()
    body = build_morning_report_body(
        state=state,
        wallet=wallet,
        tenant_id=tenant_id,
        trade_db=trade_db,
        exploration_eps=exploration_eps,
    )
    subject = f"AI Trading — ochtendrapport {amsterdam_now().strftime('%Y-%m-%d')}"

    token = (vault.get("TELEGRAM_TOKEN") or os.getenv("TELEGRAM_TOKEN") or "").strip()
    chat = (vault.get("TELEGRAM_CHAT_ID") or os.getenv("TELEGRAM_CHAT_ID") or "").strip()
    # User request: ochtendrapport alleen per e-mail. Telegram default uit; opt-in via MORNING_REPORT_TELEGRAM_ENABLED=1.
    tg_on = str(os.getenv("MORNING_REPORT_TELEGRAM_ENABLED", "0")).strip().lower() in ("1", "true", "yes", "on")

    smtp_server = (vault.get("PRIVATE_SMTP") or os.getenv("PRIVATE_SMTP") or "").strip()
    try:
        smtp_port = int(vault.get("PRIVATE_PORT") or os.getenv("PRIVATE_PORT") or 587)
    except (TypeError, ValueError):
        smtp_port = 587
    smtp_user = (vault.get("PRIVATE_EMAIL") or os.getenv("PRIVATE_EMAIL") or "").strip()
    smtp_pass = (vault.get("PRIVATE_PASS") or os.getenv("PRIVATE_PASS") or "").strip()
    mail_to = (vault.get("MORNING_REPORT_EMAIL_TO") or os.getenv("MORNING_REPORT_EMAIL_TO") or smtp_user).strip()

    report_day = amsterdam_now().date().isoformat()
    tg_ok = False
    if tg_on and token and chat:
        # Extra hard guard: morning report telegram max 1x per kalenderdag.
        day_key = f"morning_report_telegram_day_{report_day}"
        if notification_cooldown_due(key=day_key, cooldown_sec=36 * 3600):
            tg_ok = send_telegram_html(token, chat, plaintext_to_telegram_html(body))
            if tg_ok:
                mark_notification_sent(key=day_key)
    mail_ok = send_smtp_plain(
        smtp_server=smtp_server,
        smtp_port=smtp_port,
        smtp_user=smtp_user,
        smtp_pass=smtp_pass,
        subject=subject,
        body=body,
        to_addr=mail_to,
    )
    if not tg_ok and not (token and chat):
        _log.info("telegram skipped (missing token/chat in vault/env)")
    if not mail_ok and not (smtp_server and smtp_user and smtp_pass):
        _log.info("email skipped (missing SMTP in vault/env)")
    if not tg_on:
        _log.info("morning report telegram disabled (MORNING_REPORT_TELEGRAM_ENABLED=0)")
    return {"telegram_ok": tg_ok, "email_ok": mail_ok, "subject": subject}


def collect_open_paper_positions(wallet: dict[str, Any]) -> list[dict[str, Any]]:
    """Actieve paper-lots (vorige sessie) voor HTML-tabel."""
    rows: list[dict[str, Any]] = []
    try:
        eq_hint = float(wallet.get("equity", 0.0) or 0.0)
    except (TypeError, ValueError):
        eq_hint = None
    obm = wallet.get("open_lots_by_market") if isinstance(wallet.get("open_lots_by_market"), dict) else {}
    for mkt, lots in obm.items():
        if not isinstance(lots, list):
            continue
        for lot in lots:
            if not isinstance(lot, dict):
                continue
            _rq = lot.get("qty")
            if _rq is None:
                _qty = None
            else:
                try:
                    ep = float(lot.get("entry_price") or 0.0) or None
                except (TypeError, ValueError):
                    ep = None
                _rqf = safe_finance_float(_rq, label="open_lot_qty", context=str(mkt))
                _rqf = reporting_coerce_btc_qty_to_btc_base(str(mkt), _rqf)
                _qty = normalize_paper_base_qty(str(mkt), _rqf, entry_price=ep, equity_eur=eq_hint)
            rows.append(
                {
                    "market": str(mkt).upper(),
                    "qty": _qty,
                    "entry_price": lot.get("entry_price"),
                    "entry_ts": lot.get("entry_ts_utc") or lot.get("ts_utc"),
                }
            )
    if rows:
        return rows
    for lot in list(wallet.get("open_lots") or []):
        if not isinstance(lot, dict):
            continue
        mkt = str(lot.get("market") or wallet.get("position_symbol") or "—").upper()
        _rq2 = lot.get("qty")
        if _rq2 is None:
            _qty2 = None
        else:
            try:
                ep2 = float(lot.get("entry_price") or 0.0) or None
            except (TypeError, ValueError):
                ep2 = None
            _rq2f = safe_finance_float(_rq2, label="open_lot_qty", context=mkt)
            _rq2f = reporting_coerce_btc_qty_to_btc_base(mkt, _rq2f)
            _qty2 = normalize_paper_base_qty(mkt, _rq2f, entry_price=ep2, equity_eur=eq_hint)
        rows.append(
            {
                "market": mkt,
                "qty": _qty2,
                "entry_price": lot.get("entry_price"),
                "entry_ts": lot.get("entry_ts_utc"),
            }
        )
    pbm = wallet.get("position_by_market") if isinstance(wallet.get("position_by_market"), dict) else {}
    for mkt, qty in pbm.items():
        q_raw = safe_finance_float(qty or 0.0, label="position_qty", context=f"position_by_market {mkt}")
        q_raw = reporting_coerce_btc_qty_to_btc_base(str(mkt), q_raw)
        q = normalize_paper_base_qty(str(mkt), q_raw, entry_price=None, equity_eur=eq_hint)
        if q > 1e-12:
            rows.append({"market": str(mkt).upper(), "qty": q, "entry_price": "—", "entry_ts": "—"})
    return rows


def _fmt_eur(v: float) -> str:
    return f"€{v:,.2f}"


def send_telegram_plain(token: str, chat_id: str, text: str) -> bool:
    token = str(token or "").strip()
    chat_id = str(chat_id or "").strip()
    if not token or not chat_id:
        return False
    try:
        global_sec = max(60, int(float(os.getenv("TELEGRAM_GLOBAL_RATE_LIMIT_SEC", "600") or 600)))
        if not notification_cooldown_due(key="telegram_global_last_sent_utc", cooldown_sec=global_sec):
            _log.info(
                "telegram plain blocked by global rate-limit: key=%s cooldown_sec=%s",
                "telegram_global_last_sent_utc",
                global_sec,
            )
            return False
        window_sec = max(30, int(float(os.getenv("TELEGRAM_DEDUPE_WINDOW_SEC", "180") or 180)))
        digest = hashlib.sha1(str(text).encode("utf-8", errors="ignore")).hexdigest()[:20]
        dedupe_key = f"telegram_dedupe_{digest}"
        if not notification_cooldown_due(key=dedupe_key, cooldown_sec=window_sec):
            _log.info("telegram plain blocked by content dedupe: key=%s window_sec=%s", dedupe_key, window_sec)
            return False
    except Exception:
        dedupe_key = ""
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        r = requests.post(
            url,
            json={"chat_id": chat_id, "text": text},
            timeout=12,
        )
        ok = r.status_code == 200
        if ok and dedupe_key:
            mark_notification_sent(key=dedupe_key)
            mark_notification_sent(key="telegram_global_last_sent_utc")
        return ok
    except Exception as exc:
        _log.warning("telegram plain send failed: %s", exc)
        return False


def build_initial_telegram_plain_text(
    *,
    paper_mode: bool,
    balance_eur: float,
    n_markets: int,
    epsilon: float | None,
    threshold_display: str,
) -> str:
    mode = "Paper" if paper_mode else "Live"
    try:
        ef = float(epsilon) if epsilon is not None else float("nan")
    except (TypeError, ValueError):
        ef = float("nan")
    eps_s = f"{ef:.4f}" if ef == ef else "—"
    thr_s = str(threshold_display or "—")
    bal = _fmt_eur(float(balance_eur))
    n = max(1, int(n_markets))
    return "\n".join(
        [
            f"🚀 Bot Started | Mode: {mode}",
            f"💰 Balance: {bal}",
            f"🎯 Target: Top {n} Markets",
            f"🧠 AI: Epsilon {eps_s} | Threshold {thr_s}",
        ]
    )


def build_initial_email_html(
    *,
    active_markets: list[dict[str, Any]],
    scanner_selected: list[dict[str, Any]],
    system_stats: dict[str, Any],
    open_positions: list[dict[str, Any]],
    paper_mode: bool,
    balance_eur: float,
) -> tuple[str, str]:
    """Returns (html, plain_fallback)."""
    mode = "Paper" if paper_mode else "Live"
    rows_html: list[str] = []
    src = scanner_selected if scanner_selected else active_markets
    for m in src[:40]:
        if not isinstance(m, dict):
            continue
        mk = escape(str(m.get("market") or "—"))
        reason = escape(str(m.get("reason") or m.get("selection_reason") or ""))
        px = m.get("last_price")
        try:
            px_s = escape(f"{float(px):,.4f}") if px is not None else "—"
        except (TypeError, ValueError):
            px_s = "—"
        rows_html.append(f"<tr><td>{mk}</td><td>{reason}</td><td>{px_s}</td></tr>")

    cpu = escape(str(system_stats.get("cpu_pct", "—")))
    ram = escape(str(system_stats.get("ram_pct", "—")))
    disk = escape(str(system_stats.get("disk_pct", "—")))
    gpu_u = escape(str(system_stats.get("gpu_util_pct", "—")))
    gpu_n = escape(str(system_stats.get("gpu_name", "—")))
    vram_u = escape(str(system_stats.get("vram_used_mb", "—")))
    vram_t = escape(str(system_stats.get("vram_total_mb", "—")))

    pos_rows: list[str] = []
    for p in open_positions:
        pos_rows.append(
            "<tr>"
            f"<td>{escape(str(p.get('market','—')))}</td>"
            f"<td>{escape(str(p.get('qty','—')))}</td>"
            f"<td>{escape(str(p.get('entry_price','—')))}</td>"
            f"<td>{escape(str(p.get('entry_ts','—')))}</td>"
            "</tr>"
        )
    if not pos_rows:
        pos_rows.append("<tr><td colspan='4'>Geen open paper-posities.</td></tr>")

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"/><title>Startup audit</title>
<style>body{{font-family:system-ui,sans-serif;background:#111;color:#eee;padding:16px;}}
table{{border-collapse:collapse;width:100%;margin:12px 0;}} th,td{{border:1px solid #444;padding:8px;text-align:left;}}
th{{background:#222;}}</style></head><body>
<h1>AI Trading — cold start audit</h1>
<p><b>Modus:</b> {escape(mode)} &nbsp;|&nbsp; <b>Equity:</b> {escape(_fmt_eur(float(balance_eur)))}</p>
<h2>Actieve / geselecteerde markten</h2>
<table><thead><tr><th>Market</th><th>Reden</th><th>Last price</th></tr></thead><tbody>{''.join(rows_html) or '<tr><td colspan="3">Geen marktdata.</td></tr>'}</tbody></table>
<h2>Hardware</h2>
<table><tbody>
<tr><td>CPU %</td><td>{cpu}</td></tr>
<tr><td>RAM %</td><td>{ram}</td></tr>
<tr><td>Disk %</td><td>{disk}</td></tr>
<tr><td>GPU util %</td><td>{gpu_u}</td></tr>
<tr><td>GPU naam</td><td>{gpu_n}</td></tr>
<tr><td>VRAM MB</td><td>{vram_u} / {vram_t}</td></tr>
</tbody></table>
<h2>Open paper-posities (sessie-hervatting)</h2>
<table><thead><tr><th>Market</th><th>Qty</th><th>Entry</th><th>Ts</th></tr></thead><tbody>{''.join(pos_rows)}</tbody></table>
<p style="color:#888;font-size:12px;">Credentials uit .trading_vault — geen secrets in deze mail.</p>
</body></html>"""
    plain = (
        f"AI Trading cold start audit | Mode: {mode} | Balance: {_fmt_eur(float(balance_eur))}\n"
        "Zie HTML-versie voor tabellen (markten, hardware, open posities)."
    )
    return html, plain


def send_initial_report(snapshot: dict[str, Any]) -> dict[str, Any]:
    """
    Kort Telegram (platte tekst) + uitgebreide HTML-mail. Geen dedup hier — gebruik cold_start_send_initial_report().
    Credentials: .trading_vault (SMTP_SERVER, SMTP_PORT, EMAIL_RECEIVER, TELEGRAM_CHAT_ID, …).
    """
    if str(os.getenv("SKIP_INITIAL_REPORT", "0")).strip().lower() in ("1", "true", "yes", "on"):
        return {"skipped": True, "reason": "SKIP_INITIAL_REPORT"}

    wallet_gate = snapshot.get("wallet") if isinstance(snapshot.get("wallet"), dict) else {}
    refresh_portfolio_equity_integrity(wallet_gate)
    if executive_notifications_muted():
        _log.warning("send_initial_report overgeslagen: EXECUTIVE_NOTIFICATIONS_EMERGENCY_MUTE")
        return {
            "skipped": True,
            "reason": "executive_notifications_emergency_mute",
            "telegram_ok": False,
            "email_ok": False,
        }
    if not portfolio_math_integrity_allows_sends():
        _log.error("send_initial_report overgeslagen: portfolio equity gate (cash+posities vs equity > drempel)")
        return {
            "skipped": True,
            "reason": "portfolio_equity_integrity_gate",
            "telegram_ok": False,
            "email_ok": False,
        }

    apply_vault_to_os(_INITIAL_STARTUP_VAULT_KEYS, override=False)
    vault = load_trading_vault()
    cred = notification_credentials_from_vault(vault)

    paper_mode = not bool(snapshot.get("live_mode"))
    balance_eur = float(snapshot.get("balance_eur") or 0.0)
    active_markets = snapshot.get("active_markets") if isinstance(snapshot.get("active_markets"), list) else []
    scanner_selected = snapshot.get("scanner_selected") if isinstance(snapshot.get("scanner_selected"), list) else []
    n_markets = int(snapshot.get("top_market_count") or 0)
    if n_markets <= 0:
        n_markets = len(scanner_selected) if scanner_selected else min(8, len(active_markets)) if active_markets else 1
    n_markets = max(1, n_markets)
    epsilon = snapshot.get("epsilon")
    if epsilon is not None:
        try:
            epsilon = float(epsilon)
        except (TypeError, ValueError):
            epsilon = None
    thr = snapshot.get("threshold_display")
    thr_s = str(thr if thr is not None else "—")
    wallet = snapshot.get("wallet") if isinstance(snapshot.get("wallet"), dict) else {}
    system_stats = snapshot.get("system_stats") if isinstance(snapshot.get("system_stats"), dict) else {}

    open_pos = collect_open_paper_positions(wallet)

    tg_text = build_initial_telegram_plain_text(
        paper_mode=paper_mode,
        balance_eur=balance_eur,
        n_markets=n_markets,
        epsilon=epsilon,
        threshold_display=thr_s,
    )
    mail_html, mail_plain = build_initial_email_html(
        active_markets=active_markets,
        scanner_selected=scanner_selected,
        system_stats=system_stats,
        open_positions=open_pos,
        paper_mode=paper_mode,
        balance_eur=balance_eur,
    )
    subject = f"AI Trading — cold start audit ({datetime.now(UTC).strftime('%Y-%m-%d %H:%M')} UTC)"

    # NOODSITUATIE: startup Telegram + SMTP volledig uit (crash-loop spam).
    tg_ok = False
    # if str(cred.get("telegram_enabled", "1")).lower() in ("1", "true", "yes", "on"):
    #     tg_ok = send_telegram_plain(cred["telegram_token"], cred["telegram_chat_id"], tg_text)

    mail_ok = False
    # mail_ok = send_smtp_html(
    #     smtp_server=cred["smtp_server"],
    #     smtp_port=int(cred["smtp_port"] or 587),
    #     smtp_user=cred["smtp_user"],
    #     smtp_pass=cred["smtp_pass"],
    #     subject=subject,
    #     html_body=mail_html,
    #     to_addr=cred["email_receiver"],
    #     plain_fallback=mail_plain,
    # )
    _log.info("initial report: telegram_ok=%s email_ok=%s (muted)", tg_ok, mail_ok)
    return {"skipped": True, "reason": "STARTUP_NOTIFICATIONS_MUTED", "telegram_ok": tg_ok, "email_ok": mail_ok, "subject": subject}


def build_initial_startup_snapshot_from_trading_core() -> dict[str, Any]:
    """Snapshot voor send_initial_report — alleen aanroepen vanuit worker/trading_core (lazy import)."""
    import app.trading_core as tc

    from app.services.prediction_ui import trade_confidence_threshold_01
    from app.services.system_stats import get_system_stats
    from app.settings import LIVE_MODE

    st = tc.STATE if isinstance(tc.STATE, dict) else {}
    wallet = dict(st.get("paper_portfolio") or tc.PAPER_MANAGER.wallet or {})
    sc = st.get("scanner_selected") if isinstance(st.get("scanner_selected"), list) else []
    am = st.get("active_markets") if isinstance(st.get("active_markets"), list) else []
    n_top = len(sc) if sc else (min(8, len(am)) if am else 1)
    eq = safe_finance_float(
        wallet.get("equity", wallet.get("cash", 0.0)),
        label="balance_eur",
        context="build_initial_startup_snapshot_from_trading_core",
    )
    eps: float | None = None
    try:
        lr = getattr(tc.RL_AGENT, "last_training_stats", None)
        if isinstance(lr, dict) and lr.get("exploration_final_eps") is not None:
            eps = float(lr["exploration_final_eps"])
            if not math.isfinite(eps):
                _log.error("exploration_final_eps non-finite: %s", eps)
                eps = None
    except Exception:
        pass
    th01 = float(trade_confidence_threshold_01())
    try:
        dt_f = float(st.get("decision_threshold")) if st.get("decision_threshold") is not None else None
    except (TypeError, ValueError):
        dt_f = None
    if dt_f is not None:
        thr_disp = f"state {dt_f:.4f} | RL_conf_01 {th01:.4f}"
    else:
        thr_disp = f"RL_conf_01 {th01:.4f}"
    try:
        stats = get_system_stats()
    except Exception:
        stats = {}
    return {
        "live_mode": bool(LIVE_MODE),
        "balance_eur": eq,
        "active_markets": list(am),
        "scanner_selected": list(sc),
        "top_market_count": max(1, int(n_top)),
        "epsilon": eps,
        "threshold_display": thr_disp,
        "wallet": wallet,
        "system_stats": stats if isinstance(stats, dict) else {},
    }


def cold_start_send_initial_report() -> dict[str, Any]:
    """
    flock: één send per container bij multi-uvicorn-worker. Geen herhaalde sends bij WS-reconnect
    (FastAPI-startup draait niet opnieuw).
    """
    if executive_notifications_muted():
        return {"skipped": True, "reason": "executive_notifications_emergency_mute"}
    if STARTUP_COLD_START_REPORT_DISABLED:
        return {"skipped": True, "reason": "STARTUP_NOTIFICATIONS_MUTED"}
    if str(os.getenv("SKIP_INITIAL_REPORT", "0")).strip().lower() in ("1", "true", "yes", "on"):
        return {"skipped": True, "reason": "SKIP_INITIAL_REPORT"}
    # Silent restart: bij automatische herstart geen uitgebreid startup rapport (voorkomt spam bij crash-loop).
    if startup_mode() == "auto" and str(os.getenv("EXTENDED_REPORT_ON_AUTO_RESTART", "0")).strip().lower() not in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return {"skipped": True, "reason": "auto_restart_silent_mode"}
    if not _try_acquire_initial_report_cold_start_lock():
        return {"skipped": True, "reason": "cold_start_lock_held_or_unavailable"}
    try:
        snap = build_initial_startup_snapshot_from_trading_core()
        out = send_initial_report(snap)
        return {**out, "skipped": False}
    except Exception as exc:
        _log.exception("cold_start_send_initial_report: %s", exc)
        return {"skipped": False, "error": str(exc)}


def should_send_morning_report_now(*, now_ams: datetime, last_sent: date | None) -> bool:
    today = now_ams.date()
    if last_sent == today:
        return False
    hour = now_ams.hour
    minute = now_ams.minute
    # Primair: 08:00–08:30
    if hour == 8 and minute < 30:
        return True
    # Catch-up: eerste start na de nacht (tot uur X)
    catch_until = int(os.getenv("MORNING_REPORT_CATCHUP_UNTIL_HOUR", "13") or 13)
    catch_from = int(os.getenv("MORNING_REPORT_CATCHUP_FROM_HOUR", "8") or 8)
    if catch_from <= hour < catch_until:
        return True
    return False


def dispatch_if_due(
    *,
    state: dict[str, Any],
    wallet: dict[str, Any],
    tenant_id: str,
    trade_db: Path,
    exploration_eps: float | None = None,
) -> dict[str, Any] | None:
    now_ams = amsterdam_now()
    maybe_write_night_baseline(wallet=wallet)
    if executive_notifications_muted():
        return None
    lock_fd: int | None = _try_acquire_morning_report_dispatch_lock()
    if fcntl is not None and lock_fd is None:
        return None
    try:
        # Re-check onder lock om race conditions tussen workers/restarts te voorkomen.
        last = read_last_sent_ams_date()
        if not should_send_morning_report_now(now_ams=now_ams, last_sent=last):
            return None
        out = run_morning_report_dispatch(
            state=state,
            wallet=wallet,
            tenant_id=tenant_id,
            trade_db=trade_db,
            exploration_eps=exploration_eps,
        )
        if isinstance(out, dict) and out.get("skipped"):
            return out
        # Leerproblemen expliciet in _logs_hub voor analyse/optimalisatie.
        try:
            start_utc, end_utc = reporting_night_window_utc(now_ams)
            rlstats = rl_training_stats_for_window(start_utc, end_utc)
            tstats = trade_stats_for_window(trade_db, tenant_id, start_utc, end_utc)
            if int(rlstats.get("chunks", 0) or 0) <= 0:
                append_learning_issue(
                    "No RL learn-batches in report window "
                    f"({start_utc.isoformat()} -> {end_utc.isoformat()}); "
                    f"avg_loss={rlstats.get('avg_loss')} avg_reward_tail={rlstats.get('avg_reward_tail')}"
                )
            if int(tstats.get("closed_trades", 0) or 0) <= 0:
                append_learning_issue(
                    "No closed paper trades in report window "
                    f"({start_utc.isoformat()} -> {end_utc.isoformat()}); "
                    f"sum_pnl_eur={tstats.get('sum_pnl_eur')}"
                )
        except Exception:
            pass
        write_last_sent_ams_date(now_ams.date())
        return out
    finally:
        if lock_fd is not None:
            try:
                os.close(lock_fd)
            except OSError:
                pass
